# 反问题：扩大候选库（导师建议“把能找到的都放进去”）。noise_inverse.py 不改，复用它的判据和各方法。
#
# 三个库，看库变大以后各方法的选对率和代价怎么变：
#   L13：原来的 13 列 {1, u, u², u³} ∪ {∂ₓᵏ(uᵐ): k = 1..3, m = 1..3}
#   L26：1 ∪ {∂ₓᵏ(uᵐ): k = 0..4, m = 1..5}
#   L42：L26 ∪ {∂ₓᵏ(u⁶), ∂ₓᵏ(sin u), ∂ₓᵏ(cos u): k = 0..3} ∪ {∂ₓᵏ(x uᵐ): k = 0, 1, m = 1, 2}
# 弱形式与 noise_inverse.py 相同：∫ φ ∂ₓᵏ f = (−1)ᵏ ∫ f ∂ₓᵏφ，含 x 的列只是被积函数多乘一个 x。
# 方程：原来 5 个 + 新增 5 个（mKdV、Fisher、Chafee–Infante、KdV–Burgers、Kuramoto–Sivashinsky）。
#   单条行波同时满足更短的 u_t + c u_x = 0，所以新方程每个给两条轨迹（两种波速，或行波 + 空间均匀解），
#   只有真方程能同时拟合两条。真解与 forward_biglib.py 相同，已用 sympy 代回方程核对。
# 穷举：所有不超过 4 项的支撑（真方程最多 3 项）。L42 时 4 项以内约 12 万个支撑，6 项以内约 600 万个。

import itertools
import json
import platform
import time
from pathlib import Path

import numpy as np
from scipy.signal import correlate

import noise_inverse as N

NOISE = [0.0, 0.01, 0.03, 0.05]
N_SEEDS = 10
EXHAUSTIVE_MAX = 4
METHODS = ["L1", "IRL1", "STRidge", "GA", "Exhaustive"]

POW = {m: (lambda u, m=m: u**m) for m in range(1, 7)}
FUN = {"u": POW[1], "u^2": POW[2], "u^3": POW[3], "u^4": POW[4], "u^5": POW[5], "u^6": POW[6],
       "sin u": np.sin, "cos u": np.cos, "x u": None, "x u^2": None}


def name(k, f):
    if k == 0:
        return f
    return ("u" if f == "u" else f"({f})") + "_" + "x" * k


def library(tag):
    cols = [(0, "1")]
    if tag == "L13":
        cols += [(0, f) for f in ("u", "u^2", "u^3")] + [(k, f) for k in (1, 2, 3) for f in ("u", "u^2", "u^3")]
        return cols
    cols += [(k, f) for k in range(5) for f in ("u", "u^2", "u^3", "u^4", "u^5")]
    if tag == "L42":
        cols += [(k, f) for k in range(4) for f in ("u^6", "sin u", "cos u")]
        cols += [(k, f) for k in (0, 1) for f in ("x u", "x u^2")]
    return cols


def sech(z):
    return 1.0 / np.cosh(z)


KS_K = np.sqrt(209) / 38
KS_A1, KS_A3 = -135 * np.sqrt(209) / 361, 165 * np.sqrt(209) / 361

NEW_CASES = {
    "mkdv": {"pde": "u_t = -6 u² u_x - u_xxx",
             "traj": [lambda x, t: sech(x + 3 - t), lambda x, t: 1.5 * sech(1.5 * (x + 5 - 2.25 * t))],
             "x": (-15.0, 15.0, 301), "t": (0.0, 4.0, 161), "half": (20, 16), "target": "u_t",
             "true": {"(u^3)_x": -2.0, "u_xxx": -1.0}},
    "fisher": {"pde": "u_t = u_xx + u - u²",
               "traj": [lambda x, t: 1 / (1 + np.exp((x - 5 * t / np.sqrt(6)) / np.sqrt(6))) ** 2,
                        lambda x, t: 1 / (1 + 9 * np.exp(-t)) + 0 * x],
               "x": (-15.0, 25.0, 281), "t": (0.0, 4.0, 101), "half": (20, 10), "target": "u_t",
               "true": {"u_xx": 1.0, "u": 1.0, "u^2": -1.0}},
    # 不放 Chafee–Infante / Nagumo：它们的显式解（tanh 扭结、sigmoid 前沿）满足 u_x = 多项式(u)，
    # 于是 u_xx 也恰是 u 的多项式，库里 u_xx 与 u、u²、u³ 线性相关，方程不唯一（试过，穷举也选错）。
    "kdv_burgers": {"pde": "u_t = -u u_x + u_xx - 0.1 u_xxx",
                    "traj": [lambda x, t, c=c, x0=x0: c + 1.2 - 2.4 * np.tanh(x - x0 - c * t) - 1.2 * np.tanh(x - x0 - c * t) ** 2
                             for c, x0 in ((0.5, -2.0), (-1.0, 3.0))],
                    "x": (-10.0, 10.0, 321), "t": (0.0, 4.0, 161), "half": (16, 16), "target": "u_t",
                    "true": {"(u^2)_x": -0.5, "u_xx": 1.0, "u_xxx": -0.1}},
    "kuramoto_sivashinsky": {"pde": "u_t = -u u_x - u_xx - u_xxxx",
                             "traj": [lambda x, t, c=c, x0=x0: c + KS_A1 * np.tanh(KS_K * (x - x0 - c * t)) + KS_A3 * np.tanh(KS_K * (x - x0 - c * t)) ** 3
                                      for c, x0 in ((0.5, -1.5), (1.5, -4.5))],
                             "x": (-25.0, 25.0, 401), "t": (0.0, 6.0, 121), "half": (20, 12), "target": "u_t",
                             "true": {"(u^2)_x": -0.5, "u_xx": -1.0, "u_xxxx": -1.0}},
}
CASES = {**N.CASES, **NEW_CASES}


def make_data(case, cols, sigma, rng):
    x = np.linspace(*case["x"])
    t = np.linspace(*case["t"])
    dx, dt = x[1] - x[0], t[1] - t[0]
    tt, xx = np.meshgrid(t, x, indexing="ij")
    hx, ht = case["half"]
    px = N.bump(hx, dx, max_d=4)
    pt = N.bump(ht, dt, max_d=2)
    blocks_a, blocks_b = [], []
    for f in case["traj"]:
        u0 = f(xx.ravel(), tt.ravel()).reshape(xx.shape)
        u = u0 + sigma * np.std(u0) * rng.standard_normal(u0.shape)

        def integ(field, kx, kt):
            ker = np.outer(pt[kt], px[kx])
            return (correlate(field, ker, mode="valid", method="fft") * dx * dt)[::ht, ::hx].ravel()

        a = []
        for k, fn in cols:
            if fn == "1":
                field = np.ones_like(u)
            elif fn == "x u":
                field = xx * u
            elif fn == "x u^2":
                field = xx * u**2
            else:
                field = FUN[fn](u)
            a.append((-1) ** k * integ(field, k, 0))
        if case["target"] == "u_tt":
            a.append(-integ(u, 0, 1))
            b = integ(u, 0, 2)
        else:
            b = -integ(u, 0, 1)
        a = np.column_stack(a)
        # 与 noise_inverse.py 相同按 ‖b‖ 归一；稳态轨迹 u_t ≡ 0 时 ‖b‖ = 0，改按 ‖A‖ 归一
        s = np.linalg.norm(b)
        if s < 1e-8 * np.linalg.norm(a):
            s = np.linalg.norm(a)
        blocks_a.append(a / s)
        blocks_b.append(b / s)
    return np.vstack(blocks_a), np.concatenate(blocks_b)


def m_exhaustive(cr):
    best, best_s = np.inf, ()
    for k in range(0, EXHAUSTIVE_MAX + 1):
        for Sc in itertools.combinations(range(cr.p), k):
            v = cr.fit(Sc)[0]
            if v < best:
                best, best_s = v, Sc
    return best_s


def evaluate(cr, S, names, true):
    if not all(k in names for k in true):  # 真项不在库里：不可能选对
        return {"support": [names[i] for i in S], "exact": False, "extra": len(S), "missed": len(true), "coef_rel_err": None}
    return N.evaluate(cr, S, names, true)


def main():
    report = {"python": platform.python_version(), "numpy": np.__version__, "noise": NOISE, "n_seeds": N_SEEDS,
              "exhaustive_max_terms": EXHAUSTIVE_MAX, "criterion": "与 noise_inverse.py 相同（weak SINDy / MSTLS），所有方法共用",
              "libraries": {}, "cases": {}}
    t_all = time.time()
    for tag in ("L13", "L26", "L42"):
        cols = library(tag)
        report["libraries"][tag] = [name(k, f) for k, f in cols]
    for cname, case in CASES.items():
        report["cases"][cname] = {"pde": case["pde"], "true": case["true"], "by_library": {}}
        for tag in ("L13", "L26", "L42"):
            cols = library(tag)
            names = [name(k, f) for k, f in cols] + (["u_t"] if case["target"] == "u_tt" else [])
            in_lib = all(k in names for k in case["true"])
            rows = []
            for sigma in NOISE:
                agg = {m: {"exact": 0, "errs": [], "time": 0.0, "calls": 0, "supports": {}} for m in METHODS}
                seeds = range(N_SEEDS) if sigma > 0 else range(1)
                for sd in seeds:
                    a, b = make_data(case, cols, sigma, np.random.default_rng(1000 * sd + 7))
                    for m in METHODS:
                        cr = N.Criterion(a, b)
                        t0 = time.perf_counter()
                        if m == "GA":
                            S = N.m_ga(cr, np.random.default_rng(sd))
                        elif m == "Exhaustive":
                            S = m_exhaustive(cr)
                        else:
                            S = {"L1": N.m_l1, "IRL1": N.m_irl1, "STRidge": N.m_stridge}[m](cr)
                        dt_ = time.perf_counter() - t0
                        ev = evaluate(cr, S, names, case["true"])
                        g = agg[m]
                        g["exact"] += ev["exact"]
                        if ev["coef_rel_err"] is not None:
                            g["errs"].append(ev["coef_rel_err"])
                        g["time"] += dt_
                        g["calls"] += cr.calls
                        key = " + ".join(ev["support"])
                        g["supports"][key] = g["supports"].get(key, 0) + 1
                n = len(seeds)
                row = {"sigma": sigma, "n": n}
                for m in METHODS:
                    g = agg[m]
                    row[m] = {"exact_rate": g["exact"] / n, "coef_rel_err_median": float(np.median(g["errs"])) if g["errs"] else None,
                              "mean_time_s": g["time"] / n, "mean_calls": g["calls"] / n,
                              "supports": dict(sorted(g["supports"].items(), key=lambda kv: -kv[1])[:3])}
                rows.append(row)
                print(f"{cname:22s} {tag} σ={sigma:<5} " + "  ".join(f"{m}:{row[m]['exact_rate']:.2f}" for m in METHODS)
                      + f"  calls GA {row['GA']['mean_calls']:.0f} Ex {row['Exhaustive']['mean_calls']:.0f}"
                      + f"  t IRL1 {row['IRL1']['mean_time_s']*1e3:.1f}ms Ex {row['Exhaustive']['mean_time_s']:.1f}s", flush=True)
            report["cases"][cname]["by_library"][tag] = {"p": len(names), "true_in_library": in_lib, "rows": rows}
    report["wall_time_s"] = time.time() - t_all
    Path(__file__).with_name("noise_inverse_biglib_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
