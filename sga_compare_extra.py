# 补跑 SGA-PDE 论文里另外两个方程（数据来自 external/SGA-PDE/data，与其 configure.py 相同的网格）：
#   PDE_compound：u_t = (u u_x)_x = ½ (u²)_xx       x ∈ [1,2]（100 点），t ∈ [0,0.5]（251 点）
#   PDE_divide：  u_t = −u_x / x + 0.25 u_xx         x ∈ [1,2]（100 点），t ∈ [0,1]（251 点）
# 本文方法跑两版库：
#   基础库：与 noise_inverse.py 相同（∂ₓᵏ(uᵐ)，13 列）。PDE_divide 的 u_x/x 不在库里，预期找不到。
#   扩展库：基础库 + 4 个含 x 的项 x u、u/x、x u_x、u_x/x（17 列）。
#     弱形式：∫φ·(x u_x) = −∫φ_x·(x u) − ∫φ·u；∫φ·(u_x/x) = −∫φ_x·(u/x) + ∫φ·(u/x²)，都不对数据求导。
# SGA-PDE 与 sga_compare.py 完全相同（同参数、同 30 分钟预算、同一个子进程入口）。
# 输出 sga_compare_extra_result.json。

import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import sga_compare as C

ROOT = Path(__file__).resolve().parent
HALF = (10, 20)
X_TERMS = ["x u", "u/x", "x u_x", "u_x/x"]
TRUTH = {
    "sga_compound": {"weak": {"(u^2)_xx": 0.5}, "strong": [{"(u ux)_x": 1.0}, {"u uxx": 1.0, "ux^2": 1.0}]},
    "sga_divide": {"weak": {"u_x/x": -1.0, "u_xx": 0.25}, "strong": [{"ux/x": -1.0, "uxx": 0.25}]},
}


def load(name):
    f, t_end = {"sga_compound": ("PDE_compound.npy", 0.5), "sga_divide": ("PDE_divide.npy", 1.0)}[name]
    u = np.load(C.SGA_DIR / "data" / f).T  # 与 configure.py 相同：转置成 (nx, nt)
    return {"u": u, "x": np.linspace(1.0, 2.0, 100), "t": np.linspace(0.0, t_end, 251)}


def weak_rows_ext(u, x, t):
    import noise_inverse as N
    from scipy.signal import correlate
    a, b = C.weak_rows(u, x, t, HALF)
    f = u.T  # (nt, nx)
    X = x[None, :]
    dx, dt = x[1] - x[0], t[1] - t[0]
    hx, ht = HALF
    px, pt = N.bump(hx, dx), N.bump(ht, dt, max_d=1)

    def integ(field, kx):
        ker = np.outer(pt[0], px[kx])
        return (correlate(field, ker, mode="valid", method="fft") * dx * dt)[::ht, ::hx].ravel()

    extra = np.column_stack([integ(X * f, 0), integ(f / X, 0),
                             -integ(X * f, 1) - integ(f, 0),
                             -integ(f / X, 1) + integ(f / X**2, 0)])
    return a, np.column_stack([a, extra]), b


def run_ours(a, b, names, true):
    import noise_inverse as N
    out = {}
    for m in ["L1", "IRL1", "GA", "Exhaustive"]:
        cr = N.Criterion(a, b)
        t0 = time.perf_counter()
        S = N.m_ga(cr, np.random.default_rng(0)) if m == "GA" else {"L1": N.m_l1, "IRL1": N.m_irl1, "Exhaustive": N.m_exhaustive}[m](cr)
        if all(k in names for k in true):
            ev = N.evaluate(cr, S, names, true)
        else:  # 真项不在库里：不可能选对，只记录选了什么
            ev = {"support": [names[i] for i in S], "exact": False, "coef_rel_err": None,
                  "coefficients": dict(zip([names[i] for i in S], map(float, cr.fit(S)[1])))}
        ev["time_s"] = time.perf_counter() - t0
        ev["calls"] = cr.calls
        out[m] = ev
    return out


def judge_sga(res, truths):
    best = None
    for tr in truths:
        j = C.judge_sga(res, tr)
        if best is None or j["exact"]:
            best = j
        if j["exact"]:
            break
    return best


def main():
    import noise_inverse as N
    base_names = N.term_names()
    ext_names = base_names + X_TERMS
    work = Path(tempfile.mkdtemp(prefix="sga_extra_"))
    jobs, ours = [], {}
    for ds in TRUTH:
        data = load(ds)
        for sigma in C.NOISE:
            for sd in (C.SEEDS if sigma > 0 else C.SEEDS[:1]):
                key = f"{ds}|{sigma}|{sd}"
                un = C.noisy(data["u"], sigma, sd)
                a_base, a_ext, b = weak_rows_ext(un, data["x"], data["t"])
                ours[key] = {"base": run_ours(a_base, b, base_names, TRUTH[ds]["weak"]),
                             "extended": run_ours(a_ext, b, ext_names, TRUTH[ds]["weak"])}
                dpath = work / f"{ds}_{sigma}_{sd}.npz"
                np.savez(dpath, u=un, x=data["x"], t=data["t"])
                jpath = work / f"{ds}_{sigma}_{sd}.json"
                jpath.write_text(json.dumps({"data": str(dpath), "seed": sd, "params": C.SGA_PARAMS,
                                             "out": str(work / f"{ds}_{sigma}_{sd}_res.json")}), encoding="utf-8")
                jobs.append((key, jpath))
                print("ours", key, {lib: {m: (v["exact"], v["support"]) for m, v in r.items()} for lib, r in ours[key].items()},
                      flush=True)

    env = dict(os.environ, MPLBACKEND="Agg", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")

    def launch(job):
        key, jpath = job
        cwd = work / ("run_" + jpath.stem)
        cwd.mkdir(exist_ok=True)
        t0 = time.time()
        try:
            r = subprocess.run([sys.executable, "-W", "ignore", str(ROOT / "sga_compare.py"), "--run", str(jpath)],
                               cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE, timeout=C.TIMEOUT_S, text=True, encoding="utf-8", errors="replace")
            status = "ok" if r.returncode == 0 else "error: " + r.stderr[-400:]
        except subprocess.TimeoutExpired:
            out = Path(str(jpath).replace(".json", "_res.json"))
            gen = json.loads(out.read_text(encoding="utf-8"))["generations"] if out.exists() else 0
            status = f"timeout at generation {gen}"
        print(f"sga {key} {status[:60]} {time.time() - t0:.0f}s", flush=True)
        return key, status

    with ThreadPoolExecutor(C.WORKERS) as ex:
        statuses = dict(ex.map(launch, jobs))

    report = {"python": platform.python_version(), "numpy": np.__version__, "noise": C.NOISE, "seeds": C.SEEDS,
              "sga_params": C.SGA_PARAMS, "sga_timeout_s": C.TIMEOUT_S, "half_width": HALF,
              "libraries": {"base": base_names, "extended": ext_names}, "truth": TRUTH, "runs": {}}
    for key, jpath in jobs:
        ds = key.split("|")[0]
        out = Path(str(jpath).replace(".json", "_res.json"))
        sga = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
        report["runs"][key] = {"ours": ours[key], "sga_status": statuses[key], "sga_raw": sga,
                               "sga": judge_sga(sga, TRUTH[ds]["strong"]) if sga else None}
    (ROOT / "sga_compare_extra_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("done", work)


if __name__ == "__main__":
    main()
