# KdV 正问题从零重做：不用真解附近起点。tier3_kdv.py 不改，只调用它的数据、库和残差。
#
# 与 tier3_kdv.py 的区别：
#   1. 起点：不再用 0.9 倍真系数。改为 N_STARTS 个随机起点（固定种子，N(0, s²)，s = 真项列的归一化尺度量级，
#      只用到列尺度，不用真解位置），每个起点做无正则 Gauss–Newton，取 PDE+初边值残差最小者 x*。
#   2. 权重：不再用岭回归。迭代重加权 L1：第 0 轮权重全 1（普通 L1），之后 w = 1/(|ξ|+ε)，ξ 来自上一轮 L1 解。
#   3. 每个 L1 子问题用精确 LARS 同伦（lasso_exact.py），Gauss–Newton 线搜索按 L1 目标（½‖r‖² + λΣw|ξ|）接受。
#   4. 每个 λ 两个候选：上一个 λ 的解（热启动链）和 x*；按 L1 目标取小者，不按与真解的距离。
#   定支撑：40 个 λ（λ_max 到 1e-6 λ_max）上 |ξ| > 1e-5 的次数过半。阈值不放宽。
#   不提供解的内部采样：损失只有配点 PDE 残差 + 初边值残差 + L1。
#
# 仍保留的限制（与 tier3 相同，写进 JSON）：波速是预先离散成列的，c=4 那一列在库里，不是连续搜出来的。

import json
import platform
from pathlib import Path

import numpy as np

import tier3_kdv as K
from lasso_exact import lars_lasso_path, lasso_at

SUPPORT_TOL = 1e-5
N_LAMBDA = 40
LAMBDA_LO = 1e-6
N_STARTS = 30
SEED = 0
IRL1_RATIO = 1e-3
IRL1_EPS = 1e-8
IRL1_MAX_ITER = 10
SEED_CHECK = list(range(10))
SWEEP_RATIOS = [1e-2, 1e-3, 1e-4]


class Problem:
    def __init__(self):
        self.names = K.library_spec()
        raw = K.pack()
        self.scale = K.column_scales(raw)
        self.data = K.apply_scale(raw, self.scale)
        self.p = len(self.names)
        r0, j0 = K.residual_and_jacobian(self.data, np.zeros(self.p))
        self.r0, self.j0 = r0, j0
        self.base = float(np.linalg.norm(r0))

    def residual(self, x):
        return K.residual_and_jacobian(self.data, x)

    def rel(self, x):
        return float(np.linalg.norm(self.residual(x)[0]) / self.base)

    def objective(self, x, lam, w):
        r = self.residual(x)[0]
        return float(0.5 * r @ r + lam * np.sum(w * np.abs(x)))

    def lam_max(self, w):
        return float(np.max(np.abs(self.j0.T @ (-self.r0)) / w))


def gn_lsq(pb, start):
    def upd(j, y, xi):
        return np.linalg.lstsq(j, y, rcond=None)[0]

    return K.gauss_newton(pb.data, start, upd)


def gn_l1(pb, start, lam, w, iters=40):
    x = start.copy()
    best = pb.objective(x, lam, w)
    for _ in range(iters):
        r, j = pb.residual(x)
        y = j @ x - r
        prop = lasso_at(lars_lasso_path(j, y, w, lam_min_ratio=0.0), lam)
        step = prop - x
        alpha, accepted = 1.0, None
        for _ls in range(20):
            trial = x + alpha * step
            o = pb.objective(trial, lam, w)
            if o <= best * (1 + 1e-12):
                accepted, best = trial, o
                break
            alpha *= 0.5
        if accepted is None:
            break
        delta = float(np.linalg.norm(accepted - x))
        x = accepted
        if delta < 1e-12 * (1 + np.linalg.norm(x)):
            break
    return x


def multistart(pb, seed):
    rng = np.random.default_rng(seed)
    s = float(np.median(pb.scale))
    fits = []
    for _ in range(N_STARTS):
        x = gn_lsq(pb, rng.normal(size=pb.p) * s)
        fits.append((pb.rel(x), x))
    fits.sort(key=lambda f: f[0])
    return fits[0][1], [f[0] for f in fits]


def irl1_weights(pb, x_star, ratio=IRL1_RATIO):
    w = np.ones(pb.p)
    hist, prev = [], None
    for it in range(IRL1_MAX_ITER):
        z = gn_l1(pb, x_star, pb.lam_max(w) * ratio, w)
        supp = [int(i) for i in np.where(np.abs(z) > SUPPORT_TOL)[0]]
        hist.append({"iter": it, "support_names": [pb.names[i] for i in supp], "relative_residual": pb.rel(z)})
        w = 1.0 / (np.abs(z) + IRL1_EPS)
        if supp == prev:
            break
        prev = supp
    return w, hist


def path_support(pb, x_star, w):
    lam_max = pb.lam_max(w)
    lams = lam_max * np.geomspace(1.0, LAMBDA_LO, N_LAMBDA)
    chain = np.zeros(pb.p)
    counts = np.zeros(pb.p, dtype=int)
    rows = []
    for lam in lams:
        cands = [gn_l1(pb, chain, float(lam), w), gn_l1(pb, x_star, float(lam), w)]
        objs = [pb.objective(c, float(lam), w) for c in cands]
        k = int(np.argmin(objs))
        chain = cands[k]
        counts[np.abs(chain) > SUPPORT_TOL] += 1
        rows.append({"lambda_over_max": float(lam / lam_max), "relative_residual": pb.rel(chain),
                     "picked": ["warm_chain", "multistart_x*"][k]})
    support = [int(i) for i in np.where(counts > N_LAMBDA / 2.0)[0]]
    return counts, support, rows, chain


def refit(pb, support):
    mask = np.zeros(pb.p, dtype=bool)
    mask[support] = True

    def upd(j, y, xi):
        beta = np.zeros(pb.p)
        beta[mask] = np.linalg.lstsq(j[:, mask], y, rcond=None)[0]
        return beta

    start = np.zeros(pb.p)
    start[mask] = 1.0 * pb.scale[mask]  # 只用列尺度，系数从 1（归一化前）起
    x = K.gauss_newton(pb.data, start, upd)
    x[~mask] = 0.0
    return x


def run_once(pb, seed, ratio=IRL1_RATIO):
    x_star, start_res = multistart(pb, seed)
    w, hist = irl1_weights(pb, x_star, ratio)
    counts, support, rows, last = path_support(pb, x_star, w)
    return x_star, start_res, w, hist, counts, support, rows, last


def main():
    deriv = K.check_derivatives()
    pb = Problem()
    xi_true = np.zeros(pb.p)
    xi_true[0] = pb.scale[0]
    exact = pb.rel(xi_true)  # 只用来报告，不进入求解

    x_star, start_res, w, hist, counts, support, rows, last = run_once(pb, SEED)
    passed = support == [0]
    xr = refit(pb, support) if support else np.zeros(pb.p)
    report = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pde": "u_t + 6 u u_x + u_xxx = 0",
        "library": pb.names,
        "derivative_check_max_abs": deriv,
        "exact_solution_relative_residual": exact,
        "starts": f"{N_STARTS} 个随机起点，种子 {SEED}，N(0, median(列尺度)²)。不用真解位置。",
        "multistart_relative_residuals_sorted": start_res,
        "multistart_n_below_1e-8": int(sum(r < 1e-8 for r in start_res)),
        "x_star_physical": (x_star / pb.scale).tolist(),
        "irl1": {"ratio": IRL1_RATIO, "eps": IRL1_EPS, "history": hist},
        "frequency": {pb.names[i]: int(counts[i]) for i in range(pb.p)},
        "support_names": [pb.names[i] for i in support],
        "passed_support": passed,
        "refit_physical": {pb.names[i]: float(xr[i] / pb.scale[i]) for i in support},
        "refit_relative_residual": pb.rel(xr),
        "path": rows,
        "rule": "最终权重下 40 个 λ 上 |ξ|>1e-5 次数过半。阈值不放宽。",
        "limitation": "波速预先离散成列（c=1,2,4,9 等），不是连续搜索。",
    }

    seeds = []
    for sd in SEED_CHECK:
        _, sres, _, _, _, s2, _, _ = run_once(pb, sd)
        seeds.append({"seed": sd, "multistart_n_below_1e-8": int(sum(r < 1e-8 for r in sres)),
                      "support_names": [pb.names[i] for i in s2], "passed": s2 == [0]})
    report["seed_check"] = {"n_passed": sum(s["passed"] for s in seeds), "n": len(seeds), "runs": seeds}
    sweep = []
    for r in SWEEP_RATIOS:
        _, _, _, _, _, s3, _, _ = run_once(pb, SEED, ratio=r)
        sweep.append({"ratio": r, "support_names": [pb.names[i] for i in s3], "passed": s3 == [0]})
    report["irl1_ratio_check"] = sweep

    out = Path(__file__).with_name("kdv_forward_blind_result.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("python", report["python"], "numpy", report["numpy"])
    print(f"exact residual {exact:.2e}  multistart hits {report['multistart_n_below_1e-8']}/{N_STARTS}")
    print("IRL1:", [h["support_names"] for h in hist])
    print("frequency:", {k.split('  ')[0]: v for k, v in report["frequency"].items()})
    print("support:", report["support_names"], "PASSED" if passed else "FAILED")
    print("refit:", report["refit_physical"], f"res {report['refit_relative_residual']:.2e}")
    print(f"seeds passed {report['seed_check']['n_passed']}/{report['seed_check']['n']}")
    print("ratio check:", [(s["ratio"], s["passed"]) for s in sweep])


if __name__ == "__main__":
    main()
