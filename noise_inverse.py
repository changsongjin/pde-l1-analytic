# 反问题加高斯噪声：误差分析，并与遗传算法比较。旧脚本不改，只调用它们的精确解。
#
# 设定（所有方法一致）：
#   1. 数据：精确解 u(x,t) 在均匀网格上取值，加噪声 u + σ·std(u)·N(0,1)，σ ∈ NOISE（相对噪声）。
#   2. 弱形式：不对带噪声的数据求导。方程乘光滑测试函数 φ(x,t) = (1-(x/a)²)^P (1-(t/b)²)^P 后积分，
#      分部积分把导数全部移到 φ 上：∫ ∂ₓᵏ(uᵐ) φ = (-1)ᵏ ∫ uᵐ ∂ₓᵏφ，∫ u_t φ = -∫ u φ_t。
#      测试函数中心按半宽步长排布（相邻重叠一半），每个中心给一行。
#      （先试过逐点求导：Savitzky–Golay 在 1% 噪声下 u_xx 误差已到 45%，见 HANDOFF。）
#   3. 候选库：13 列 {1, u, u², u³} ∪ {∂ₓᵏ(uᵐ): k=1..3, m=1..3}；波动方程目标 u_tt，另加 u_t 列。
#      真方程都在库里：Burgers 的 u u_x = ½(u²)_x，KdV 的 6 u u_x = 3(u²)_x。
#   4. 选模型：所有方法都用同一个判据，在各自给出的候选支撑里取最小。方法只差在“怎么产生候选支撑”。
#      判据（weak SINDy / MSTLS，Messenger & Bortz 2021）：L(S) = ‖A(w_S − w_LS)‖/‖A w_LS‖ + |S|/p，
#      w_S 是在支撑 S 上的最小二乘重拟合，w_LS 是全库最小二乘。无量纲，精确数据和带噪声数据都适用。
#      （先试过 BIC：精确数据时 RSS 趋零，ln RSS 让 BIC 一直偏好多加项，穷举在热方程、波动方程上也选错。）
#   5. 方法：
#      L1      精确 LARS 路径上出现过的全部支撑；
#      IRL1    第 0 轮普通 L1（λ = 1e-3 λ_max），迭代重加权到支撑不变，再用最终权重的 LARS 路径上的全部支撑；
#      STRidge 序贯阈值岭回归（PDE-FIND），阈值扫一遍得到的全部支撑（只作对照）；
#      GA      二进制染色体（每位表示一项选不选），适应度直接是判据 L；
#      穷举    最多 6 项的全部支撑都算判据，作为“判据本身”的上限：它选错说明是判据或数据的问题，不是搜索的问题。
#   指标：选对率（支撑 = 真项）、多选数、漏选数、真项系数相对误差、耗时、目标函数调用次数。

import itertools
import json
import platform
import time
from pathlib import Path

import numpy as np
from numpy.polynomial import polynomial as Poly
from scipy.signal import correlate

import inverse_irl1 as I
import redo_two
from lasso_exact import lars_lasso_path, lasso_at

NOISE = [0.0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1]
N_SEEDS = 20
SUPPORT_TOL = 1e-5
IRL1_RATIO, IRL1_EPS, IRL1_MAX_ITER = 1e-3, 1e-8, 20
GA = {"pop": 40, "gens": 60, "p_cross": 0.9, "elite": 2, "tour": 3}

P_TEST = 6
# 库的列：(k, m) 表示 ∂ₓᵏ(uᵐ)
TERMS = [(0, 0), (0, 1), (0, 2), (0, 3)] + [(k, m) for k in (1, 2, 3) for m in (1, 2, 3)]


def term_name(k, m):
    base = {0: "1", 1: "u", 2: "u^2", 3: "u^3"}[m]
    if k == 0:
        return base
    if m == 1:
        return "u_" + "x" * k
    return f"({base})_" + "x" * k


def term_names(extra_ut=False):
    return [term_name(k, m) for k, m in TERMS] + (["u_t"] if extra_ut else [])


def bump(n_half, h, p=P_TEST, max_d=3):
    """离散测试函数 (1-s²)^p，s ∈ [-1,1]，及其对物理坐标的 0..max_d 阶导数（解析多项式求导）。"""
    s = np.linspace(-1.0, 1.0, 2 * n_half + 1)
    a = n_half * h
    poly = Poly.polypow([1.0, 0.0, -1.0], p)
    out, q = [], poly
    for d in range(max_d + 1):
        out.append(Poly.polyval(s, q) / a**d)
        q = Poly.polyder(q)
    return out


# ---------- 精确解（只取 u，导数一律不从数据求） ----------

def u_heat(k):
    return lambda x, t: np.exp(-((k * np.pi) ** 2) * t) * np.sin(k * np.pi * x)


def u_wave(k):
    return lambda x, t: np.sin(k * np.pi * x) * np.cos(k * np.pi * t)


def u_adv(sigma0):
    return lambda x, t: redo_two.gaussian(x, t, sigma0)[0]


def u_burgers(x, t):
    return I.cole_hopf(x, t)[0]


def u_kdv(x, t):
    return redo_two.two_soliton_fields(x, t)[0]


# 每个情形：若干条轨迹（同一方程不同解），网格，测试函数半宽（x、t 方向格点数），真项与系数
CASES = {
    "heat": {"pde": "u_t = u_xx", "traj": [u_heat(1), u_heat(2)], "x": (0.0, 1.0, 201), "t": (0.0, 0.2, 101),
             "half": (20, 10), "target": "u_t", "true": {"u_xx": 1.0}},
    "wave": {"pde": "u_tt = u_xx", "traj": [u_wave(1), u_wave(2)], "x": (0.0, 1.0, 201), "t": (0.0, 1.0, 201),
             "half": (20, 20), "target": "u_tt", "true": {"u_xx": 1.0}},
    "advection_diffusion": {"pde": "u_t = -0.4 u_x + 0.05 u_xx", "traj": [u_adv(0.5), u_adv(0.9)],
                            "x": (-3.0, 4.0, 281), "t": (0.0, 1.0, 101), "half": (20, 10),
                            "target": "u_t", "true": {"u_x": -0.4, "u_xx": 0.05}},
    "burgers": {"pde": "u_t = -u u_x + 0.1 u_xx", "traj": [u_burgers], "x": (-1.0, 3.0, 321), "t": (0.0, 1.0, 161),
                "half": (16, 16), "target": "u_t", "true": {"(u^2)_x": -0.5, "u_xx": 0.1}},
    "kdv": {"pde": "u_t = -6 u u_x - u_xxx", "traj": [u_kdv], "x": (-10.0, 12.0, 441), "t": (0.0, 1.2, 241),
            "half": (20, 20), "target": "u_t", "true": {"(u^2)_x": -3.0, "u_xxx": -1.0}},
}


def make_data(case, sigma, rng):
    x = np.linspace(*case["x"])
    t = np.linspace(*case["t"])
    dx, dt = x[1] - x[0], t[1] - t[0]
    tt, xx = np.meshgrid(t, x, indexing="ij")  # 形状 (nt, nx)
    hx, ht = case["half"]
    px = bump(hx, dx)
    pt = bump(ht, dt, max_d=2)
    blocks_a, blocks_b = [], []
    for f in case["traj"]:
        u0 = f(xx.ravel(), tt.ravel()).reshape(xx.shape)
        u = u0 + sigma * np.std(u0) * rng.standard_normal(u0.shape)

        def integ(field, kx, kt):
            # ∫∫ field · ∂ₓ^kx ∂ₜ^kt φ，所有中心一次算完；取步长为半宽的中心
            ker = np.outer(pt[kt], px[kx])
            full = correlate(field, ker, mode="valid", method="fft") * dx * dt
            return full[::ht, ::hx].ravel()

        cols = []
        for k, m in TERMS:
            cols.append((-1) ** k * integ(u**m, k, 0))
        if case["target"] == "u_tt":
            cols.append(-integ(u, 0, 1))      # ∫ u_t φ = -∫ u φ_t
            b = integ(u, 0, 2)                # ∫ u_tt φ = ∫ u φ_tt
        else:
            b = -integ(u, 0, 1)               # ∫ u_t φ = -∫ u φ_t
        a = np.column_stack(cols)
        s = np.linalg.norm(b)
        blocks_a.append(a / s)
        blocks_b.append(b / s)
    return np.vstack(blocks_a), np.concatenate(blocks_b)


# ---------- 判据：用 Gram 矩阵快速算任意支撑的重拟合 ----------

class Criterion:
    def __init__(self, a, b):
        self.col = np.maximum(np.linalg.norm(a, axis=0), 1e-300)
        self.an = a / self.col
        self.b = b
        self.n = a.shape[0]
        self.G = self.an.T @ self.an
        self.g = self.an.T @ b
        self.bb = float(b @ b)
        self.calls = 0
        self.cache = {}
        self.p = a.shape[1]
        w_ls = np.linalg.lstsq(self.G, self.g, rcond=None)[0]
        self.rss_ls = max(self.bb - float(w_ls @ self.g), 0.0)
        self.fit_norm = float(np.sqrt(max(w_ls @ self.G @ w_ls, 1e-300)))

    def fit(self, S):
        S = tuple(sorted(S))
        if S in self.cache:
            return self.cache[S]
        self.calls += 1
        if not S:
            rss, coef = self.bb, np.zeros(0)
        else:
            idx = list(S)
            coef = np.linalg.lstsq(self.G[np.ix_(idx, idx)], self.g[idx], rcond=None)[0]
            rss = max(self.bb - 2 * coef @ self.g[idx] + coef @ self.G[np.ix_(idx, idx)] @ coef, 1e-300)
        # 最小二乘的正交性：‖A(w_S − w_LS)‖² = RSS_S − RSS_LS
        loss = np.sqrt(max(rss - self.rss_ls, 0.0)) / self.fit_norm + len(S) / self.p
        self.cache[S] = (loss, coef / self.col[list(S)] if S else coef)
        return self.cache[S]

    def best(self, supports):
        sups = {tuple(sorted(s)) for s in supports}
        return min(sups, key=lambda s: self.fit(s)[0])


# ---------- 各方法产生候选支撑 ----------

def path_supports(an, b, w):
    knots = lars_lasso_path(an, b, w, lam_min_ratio=0.0)
    return [tuple(int(i) for i in np.where(np.abs(k[1]) > SUPPORT_TOL * np.max(np.abs(k[1]) + 1e-300))[0]) for k in knots[1:]]


def m_l1(cr):
    return cr.best(path_supports(cr.an, cr.b, np.ones(cr.an.shape[1])))


def m_irl1(cr):
    p = cr.an.shape[1]
    w = np.ones(p)
    prev = None
    for _ in range(IRL1_MAX_ITER):
        knots = lars_lasso_path(cr.an, cr.b, w, lam_min_ratio=IRL1_RATIO * 0.1)
        beta = lasso_at(knots, knots[0][0] * IRL1_RATIO)
        supp = tuple(np.where(np.abs(beta) > SUPPORT_TOL)[0])
        w = 1.0 / (np.abs(beta) + IRL1_EPS)
        if supp == prev:
            break
        prev = supp
    return cr.best(path_supports(cr.an, cr.b, w))


def m_stridge(cr, lam_ridge=1e-5, n_tol=50, max_it=10):
    p = cr.an.shape[1]
    sups = []
    for tol in np.geomspace(1e-4, 10.0, n_tol):
        S = list(range(p))
        for _ in range(max_it):
            G = cr.G[np.ix_(S, S)] + lam_ridge * np.eye(len(S))
            c = np.linalg.solve(G, cr.g[S])
            keep = [S[i] for i in range(len(S)) if abs(c[i]) >= tol]
            if keep == S or not keep:
                S = keep
                break
            S = keep
        sups.append(tuple(S))
    return cr.best(sups)


def m_ga(cr, rng):
    p = cr.an.shape[1]
    pop = rng.random((GA["pop"], p)) < 0.2
    fit = lambda ind: cr.fit(tuple(np.where(ind)[0]))[0]
    scores = np.array([fit(i) for i in pop])
    for _ in range(GA["gens"]):
        order = np.argsort(scores)
        new = [pop[i].copy() for i in order[: GA["elite"]]]
        while len(new) < GA["pop"]:
            def pick():
                c = rng.choice(GA["pop"], GA["tour"], replace=False)
                return pop[c[np.argmin(scores[c])]]
            a, b = pick(), pick()
            child = np.where(rng.random(p) < 0.5, a, b) if rng.random() < GA["p_cross"] else a.copy()
            child ^= rng.random(p) < 1.0 / p
            new.append(child)
        pop = np.array(new)
        scores = np.array([fit(i) for i in pop])
    return tuple(np.where(pop[np.argmin(scores)])[0])


def m_exhaustive(cr):
    p = cr.an.shape[1]
    best, best_s = np.inf, ()
    for k in range(0, 7):  # 最多 6 项；真方程最多 2 项
        for S in itertools.combinations(range(p), k):
            v = cr.fit(S)[0]
            if v < best:
                best, best_s = v, S
    return best_s


METHODS = ["L1", "IRL1", "STRidge", "GA", "Exhaustive"]


def evaluate(cr, S, names, true):
    tidx = {names.index(k): v for k, v in true.items()}
    Sset, Tset = set(S), set(tidx)
    coef = dict(zip(S, cr.fit(S)[1]))
    err = np.sqrt(sum((coef.get(i, 0.0) - v) ** 2 for i, v in tidx.items()) / sum(v**2 for v in tidx.values()))
    return {"support": [names[i] for i in S], "exact": Sset == Tset, "extra": len(Sset - Tset),
            "missed": len(Tset - Sset), "coef_rel_err": float(err)}


def main():
    report = {"python": platform.python_version(), "numpy": np.__version__, "noise": NOISE, "n_seeds": N_SEEDS,
              "formulation": "弱形式，测试函数 (1-s²)^6，中心步长为半宽",
              "criterion": "L(S) = ||A(w_S - w_LS)|| / ||A w_LS|| + |S|/p（weak SINDy），所有方法共用",
              "ga": GA, "cases": {}}
    for name, case in CASES.items():
        names = term_names(extra_ut=case["target"] == "u_tt")
        rows = []
        for sigma in NOISE:
            agg = {m: {"exact": 0, "extra": 0, "missed": 0, "errs": [], "time": 0.0, "calls": 0, "supports": {}} for m in METHODS}
            seeds = range(N_SEEDS) if sigma > 0 else range(1)
            for sd in seeds:
                rng = np.random.default_rng(1000 * sd + 7)
                a, b = make_data(case, sigma, rng)
                for m in METHODS:
                    cr = Criterion(a, b)
                    t0 = time.perf_counter()
                    if m == "GA":
                        S = m_ga(cr, np.random.default_rng(sd))
                    else:
                        S = {"L1": m_l1, "IRL1": m_irl1, "STRidge": m_stridge, "Exhaustive": m_exhaustive}[m](cr)
                    dt_ = time.perf_counter() - t0
                    ev = evaluate(cr, S, names, case["true"])
                    g = agg[m]
                    g["exact"] += ev["exact"]; g["extra"] += ev["extra"]; g["missed"] += ev["missed"]
                    g["errs"].append(ev["coef_rel_err"]); g["time"] += dt_; g["calls"] += cr.calls
                    key = " + ".join(ev["support"])
                    g["supports"][key] = g["supports"].get(key, 0) + 1
            n = len(seeds)
            row = {"sigma": sigma, "n": n}
            for m in METHODS:
                g = agg[m]
                row[m] = {"exact_rate": g["exact"] / n, "mean_extra": g["extra"] / n, "mean_missed": g["missed"] / n,
                          "coef_rel_err_median": float(np.median(g["errs"])), "coef_rel_err_max": float(np.max(g["errs"])),
                          "mean_time_s": g["time"] / n, "mean_calls": g["calls"] / n,
                          "supports": dict(sorted(g["supports"].items(), key=lambda kv: -kv[1])[:4])}
            rows.append(row)
            print(f"{name:20s} σ={sigma:<6} " + "  ".join(f"{m}:{row[m]['exact_rate']:.2f}/{row[m]['coef_rel_err_median']:.1e}" for m in METHODS), flush=True)
        report["cases"][name] = {"pde": case["pde"], "library": names, "true": case["true"], "grid_x": case["x"],
                                 "grid_t": case["t"], "test_half_width_points": case["half"], "rows": rows}
    out = Path(__file__).with_name("noise_inverse_result.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
