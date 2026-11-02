# 与 SGA-PDE（Chen 等，Phys. Rev. Research 4, 023174, 2022）开源代码比较。SGA-PDE 代码不改，放在 external/SGA-PDE。
#
# 每个（数据集, 噪声, 种子）只生成一次带噪声的 u，两边用同一份数据：
#   SGA-PDE：按论文默认参数（种群 20、深度 4、宽度 5、100 代、AIC），有限差分求导；每次在单独进程里跑。
#            用一个替身 configure 模块把数据交给它（原 configure.py 里数据集是写死的），并关掉 pdb 断点。
#   本文方法：弱形式 + L1 / IRL1 / 自写 GA / 穷举（与 noise_inverse.py 相同的库和判据）。
# SGA-PDE 的项是符号树，判定方式：把它选出的每一项（系数非零）与常见候选项逐一算余弦相似度，
#   归到最相近的那一项（< 0.9 记为“其他”）；归类后的集合等于真项才算选对。原始字符串一并记录。
# 数据集：SGA-PDE 自带的 Burgers、KdV、Chafee–Infante；本文的 Burgers（Cole–Hopf）和 KdV 双孤子。
#
# 运行：python sga_compare.py            （编排，并行跑全部任务）
#       python sga_compare.py --run job  （单个 SGA 任务，由编排进程调用）

import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import scipy.io as scio

ROOT = Path(__file__).resolve().parent
SGA_DIR = ROOT / "external" / "SGA-PDE"
NOISE = [0.0, 0.01, 0.05]
SEEDS = [0, 1, 2]
SGA_PARAMS = {"num": 20, "depth": 4, "width": 5, "p_var": 0.5, "p_mute": 0.3, "p_cro": 0.5, "p_rep": 1, "gens": 100}
WORKERS = 8
TIMEOUT_S = 1800  # 每次 SGA 运行的时间预算；超时取最后一代快照（记录跑到第几代）


# ---------- 数据集：u 的形状统一为 (nx, nt)，与 SGA-PDE 一致 ----------

def load_dataset(name):
    if name == "sga_burgers":
        d = scio.loadmat(SGA_DIR / "data" / "burgers.mat")
        u, x, t = np.real(d["usol"]), np.squeeze(d["x"]), np.squeeze(d["t"])
        true_strong = {"u ux": -1.0, "uxx": 0.1}
        true_weak = {"(u^2)_x": -0.5, "u_xx": 0.1}
        half = (16, 16)
    elif name == "sga_kdv":
        d = scio.loadmat(SGA_DIR / "data" / "Kdv.mat") if (SGA_DIR / "data" / "Kdv.mat").exists() else scio.loadmat(SGA_DIR / "data" / "KdV.mat")
        u, x, t = np.real(d["uu"]), np.squeeze(d["x"]), np.squeeze(d["tt"])
        true_strong = {"u ux": -1.0, "uxxx": -0.0025}
        true_weak = {"(u^2)_x": -0.5, "u_xxx": -0.0025}
        half = (24, 16)
    elif name == "sga_chafee_infante":
        u = np.load(SGA_DIR / "data" / "chafee_infante_CI.npy")
        x = np.load(SGA_DIR / "data" / "chafee_infante_x.npy")
        t = np.load(SGA_DIR / "data" / "chafee_infante_t.npy")
        true_strong = {"uxx": 1.0, "u": -1.0, "u^3": 1.0}
        true_weak = {"u_xx": 1.0, "u": -1.0, "u^3": 1.0}
        half = (16, 16)
    elif name == "ours_burgers":
        import inverse_irl1 as I
        x, t = np.linspace(-1.0, 3.0, 161), np.linspace(0.0, 1.0, 81)
        tt, xx = np.meshgrid(t, x, indexing="xy")
        u = I.cole_hopf(xx.ravel(), tt.ravel())[0].reshape(xx.shape)
        true_strong = {"u ux": -1.0, "uxx": 0.1}
        true_weak = {"(u^2)_x": -0.5, "u_xx": 0.1}
        half = (8, 8)
    elif name == "ours_kdv":
        import redo_two
        x, t = np.linspace(-10.0, 12.0, 221), np.linspace(0.0, 1.2, 121)
        tt, xx = np.meshgrid(t, x, indexing="xy")
        u = redo_two.two_soliton_fields(xx.ravel(), tt.ravel())[0].reshape(xx.shape)
        true_strong = {"u ux": -6.0, "uxxx": -1.0}
        true_weak = {"(u^2)_x": -3.0, "u_xxx": -1.0}
        half = (10, 10)
    else:
        raise KeyError(name)
    return {"u": np.asarray(u, float), "x": np.asarray(x, float), "t": np.asarray(t, float),
            "true_strong": true_strong, "true_weak": true_weak, "half": half}


DATASETS = ["sga_burgers", "sga_kdv", "sga_chafee_infante", "ours_burgers", "ours_kdv"]


def noisy(u, sigma, seed):
    rng = np.random.default_rng(10_000 + seed)
    return u + sigma * np.std(u) * rng.standard_normal(u.shape)


# ---------- 本文方法：弱形式，复用 noise_inverse.py ----------

def weak_rows(u, x, t, half):
    import noise_inverse as N
    from scipy.signal import correlate
    ut_ = u.T  # (nt, nx)
    dx, dt = x[1] - x[0], t[1] - t[0]
    hx, ht = half
    px, pt = N.bump(hx, dx), N.bump(ht, dt, max_d=1)

    def integ(field, kx, kt):
        ker = np.outer(pt[kt], px[kx])
        return (correlate(field, ker, mode="valid", method="fft") * dx * dt)[::ht, ::hx].ravel()

    a = np.column_stack([(-1) ** k * integ(ut_**m, k, 0) for k, m in N.TERMS])
    b = -integ(ut_, 0, 1)
    return a, b


def run_ours(data, u):
    import noise_inverse as N
    a, b = weak_rows(u, data["x"], data["t"], data["half"])
    names = N.term_names()
    out = {}
    for m in ["L1", "IRL1", "GA", "Exhaustive"]:
        cr = N.Criterion(a, b)
        t0 = time.perf_counter()
        S = N.m_ga(cr, np.random.default_rng(0)) if m == "GA" else {"L1": N.m_l1, "IRL1": N.m_irl1, "Exhaustive": N.m_exhaustive}[m](cr)
        ev = N.evaluate(cr, S, names, data["true_weak"])
        ev["time_s"] = time.perf_counter() - t0
        ev["calls"] = cr.calls
        out[m] = ev
    return out


# ---------- SGA-PDE 单次运行（子进程） ----------

REF_TERMS = ["1", "u", "u^2", "u^3", "ux", "uxx", "uxxx", "u ux", "u uxx", "u uxxx", "u^2 ux", "ux^2", "x", "x u", "x ux"]


def sga_worker(job_path):
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    arr = np.load(job["data"])
    u, x, t = arr["u"], arr["x"], arr["t"]

    import pdb
    pdb.set_trace = lambda *a, **k: None
    # 兼容：SGA-PDE 用 np.array 把数组和字符串混在一个列表里，NumPy ≥ 1.24 要求显式 dtype=object
    _np_array = np.array

    def _array_compat(obj, *a, **k):
        try:
            return _np_array(obj, *a, **k)
        except ValueError:
            if "dtype" in k or a:
                raise
            return _np_array(obj, dtype=object)

    np.array = _array_compat
    import matplotlib
    matplotlib.use("Agg")
    import torch
    import torch.nn as nn

    cfg = types.ModuleType("configure")
    cfg.problem, cfg.seed, cfg.device = "custom", job["seed"], torch.device("cpu")
    cfg.max_epoch, cfg.hidden_dim, cfg.train_ratio, cfg.num_feature, cfg.normal = 100000, 50, 1, 2, True
    cfg.path = "unused.pkl"
    cfg.fine_ratio, cfg.use_metadata, cfg.delete_edges, cfg.aic_ratio = 2, False, False, 1
    cfg.Net = nn.Module

    def divide(up, down, eta=1e-10):
        while np.any(down == 0):
            down += eta
        return up / down

    cfg.divide = divide
    cfg.u, cfg.x, cfg.t = u, x, t
    # setup.py 只用这几行打印诊断信息
    cfg.right_side, cfg.left_side = "right_side = 0*ut", "left_side = ut"
    cfg.right_side_origin, cfg.left_side_origin = "right_side_origin = 0*ut_origin", "left_side_origin = ut_origin"
    sys.modules["configure"] = cfg
    sys.path.insert(0, str(SGA_DIR / "codes"))

    import PDE_find as F

    def stridge_compat(X0, y, lam, maxit, tol, normalize=0, print_results=False):
        # 与 PDE_find.STRidge 逐行相同，只把 `if biginds != []` 改成 len 判断（旧版 NumPy 下两者等价）
        n, d = X0.shape
        X = np.zeros((n, d))
        if normalize != 0:
            Mreg = np.zeros((d, 1))
            for i in range(0, d):
                Mreg[i] = 1.0 / (np.linalg.norm(X0[:, i], normalize))
                X[:, i] = Mreg[i] * X0[:, i]
        else:
            X = X0
        if lam != 0:
            w = np.linalg.lstsq(X.T.dot(X) + lam * np.eye(d), X.T.dot(y), rcond=None)[0]
        else:
            w = np.linalg.lstsq(X, y, rcond=None)[0]
        num_relevant = d
        biginds = np.where(abs(w) > tol)[0]
        for j in range(maxit):
            smallinds = np.where(abs(w) < tol)[0]
            new_biginds = [i for i in range(d) if i not in smallinds]
            if num_relevant == len(new_biginds):
                break
            else:
                num_relevant = len(new_biginds)
            if len(new_biginds) == 0:
                if j == 0:
                    return w
                else:
                    break
            biginds = new_biginds
            w[smallinds] = 0
            if lam != 0:
                w[biginds] = np.linalg.lstsq(X[:, biginds].T.dot(X[:, biginds]) + lam * np.eye(len(biginds)), X[:, biginds].T.dot(y), rcond=None)[0]
            else:
                w[biginds] = np.linalg.lstsq(X[:, biginds], y, rcond=None)[0]
        if len(biginds) != 0:
            w[biginds] = np.linalg.lstsq(X[:, biginds], y, rcond=None)[0]
        if normalize != 0:
            return np.multiply(Mreg, w)
        return w

    F.STRidge = stridge_compat
    import setup as S
    import pde as P
    captured = {}
    orig_train = P.Train

    def train_capture(R, Ut, *a, **k):
        captured["R"] = R
        return orig_train(R, Ut, *a, **k)

    P.Train = train_capture
    import sga as G

    p = job["params"]
    t0 = time.perf_counter()
    model = G.SGA(num=p["num"], depth=p["depth"], width=p["width"], p_var=p["p_var"], p_rep=p["p_rep"],
                  p_mute=p["p_mute"], p_cro=p["p_cro"])
    # 参考项：用 SGA 自己的有限差分导数，保证同口径
    uu, ux, uxx, uxxx, X = S.u, S.ux, S.uxx, S.uxxx, S.x
    ref = {"1": np.ones_like(uu), "u": uu, "u^2": uu**2, "u^3": uu**3, "ux": ux, "uxx": uxx, "uxxx": uxxx,
           "u ux": uu * ux, "u uxx": uu * uxx, "u uxxx": uu * uxxx, "u^2 ux": uu**2 * ux, "ux^2": ux**2,
           "x": X, "x u": X * uu, "x ux": X * ux,
           # sga_compare_extra.py 的两个方程用到（前 35 次运行没有这两项，结果不受影响：那些方程的真项里没有它们）
           "ux/x": ux / X, "(u ux)_x": uu * uxx + ux**2}
    ref = {k: v.reshape(-1) for k, v in ref.items()}
    import copy

    def snapshot(gen):
        # 每代结束写一次当前最优方程：超时被杀时也留下“到第几代为止的最优结果”
        best, best_aic = model.the_best()
        elems = copy.deepcopy(best.elements)
        elems, w = P.evaluate_mse(elems, True)
        R = captured["R"]
        w = np.asarray(w).reshape(-1)
        names = ["u"] + [e.inorder for e in elems]
        terms = []
        for j in range(R.shape[1]):
            if abs(w[j]) < 1e-4:  # 与 SGA-PDE 自己的 concise_visualize 相同的阈值
                continue
            v = R[:, j]
            sims = {k: abs(float(v @ r) / (np.linalg.norm(v) * np.linalg.norm(r) + 1e-300)) for k, r in ref.items()}
            k_best = max(sims, key=sims.get)
            label = k_best if sims[k_best] >= 0.9 else "other"
            scale = float(v @ ref[k_best] / (ref[k_best] @ ref[k_best] + 1e-300)) if label != "other" else 0.0
            terms.append({"sga_term": names[j], "maps_to": label, "cos": sims[k_best], "coef": float(w[j] * scale)})
        res = {"concise": best.concise_visualize(), "aic": float(best_aic), "generations": gen,
               "time_s": time.perf_counter() - t0, "terms": terms}
        tmp = Path(job["out"] + ".tmp")
        tmp.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, job["out"])

    for g in range(1, p["gens"] + 1):
        model.run(1)
        snapshot(g)


def judge_sga(res, true_strong):
    labels = [t["maps_to"] for t in res["terms"]]
    got = {}
    for t in res["terms"]:
        got[t["maps_to"]] = got.get(t["maps_to"], 0.0) + t["coef"]
    exact = sorted(labels) == sorted(true_strong) and "other" not in labels
    err = None
    if exact:
        err = float(np.sqrt(sum((got[k] - v) ** 2 for k, v in true_strong.items()) / sum(v**2 for v in true_strong.values())))
    return {"exact": exact, "support": labels, "coef_rel_err": err,
            "extra": sum(1 for k in labels if k not in true_strong), "missed": len(set(true_strong) - set(labels))}


def main():
    work = Path(tempfile.mkdtemp(prefix="sga_cmp_"))
    jobs, ours = [], {}
    for ds in DATASETS:
        data = load_dataset(ds)
        for sigma in NOISE:
            for sd in (SEEDS if sigma > 0 else SEEDS[:1]):
                key = f"{ds}|{sigma}|{sd}"
                un = noisy(data["u"], sigma, sd)
                ours[key] = run_ours(data, un)
                dpath = work / f"{ds}_{sigma}_{sd}.npz"
                np.savez(dpath, u=un, x=data["x"], t=data["t"])
                jpath = work / f"{ds}_{sigma}_{sd}.json"
                jpath.write_text(json.dumps({"data": str(dpath), "seed": sd, "params": SGA_PARAMS,
                                             "out": str(work / f"{ds}_{sigma}_{sd}_res.json")}), encoding="utf-8")
                jobs.append((key, jpath, data["true_strong"]))
                print("ours", key, {m: (v["exact"], round(v["coef_rel_err"], 4)) for m, v in ours[key].items()}, flush=True)

    env = dict(os.environ, MPLBACKEND="Agg", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")

    def launch(job):
        key, jpath, _ = job
        cwd = work / ("run_" + jpath.stem)
        cwd.mkdir(exist_ok=True)
        t0 = time.time()
        try:
            r = subprocess.run([sys.executable, "-W", "ignore", str(Path(__file__).resolve()), "--run", str(jpath)],
                               cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE, timeout=TIMEOUT_S, text=True, encoding="utf-8", errors="replace")
            status = "ok" if r.returncode == 0 else "error: " + r.stderr[-400:]
        except subprocess.TimeoutExpired:
            out = Path(str(jpath).replace(".json", "_res.json"))
            gen = json.loads(out.read_text(encoding="utf-8"))["generations"] if out.exists() else 0
            status = f"timeout at generation {gen}"
        print(f"sga {key} {status[:60]} {time.time() - t0:.0f}s", flush=True)
        return key, status

    with ThreadPoolExecutor(WORKERS) as ex:
        statuses = dict(ex.map(launch, jobs))

    report = {"python": platform.python_version(), "numpy": np.__version__, "noise": NOISE, "seeds": SEEDS,
              "sga_params": SGA_PARAMS, "sga_source": "github.com/YuntianChen/SGA-PDE（2022-06 版本，代码未改）",
              "judge": "SGA 选出的项按余弦相似度归到常见候选项（< 0.9 记为 other），集合等于真项才算选对",
              "runs": {}}
    for key, jpath, true_strong in jobs:
        out = Path(str(jpath).replace(".json", "_res.json"))
        sga = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
        report["runs"][key] = {"ours": ours[key], "sga_status": statuses[key],
                               "sga_raw": sga, "sga": judge_sga(sga, true_strong) if sga else None}
    Path(ROOT / "sga_compare_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("done", work)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--run":
        sga_worker(sys.argv[2])
    else:
        main()
