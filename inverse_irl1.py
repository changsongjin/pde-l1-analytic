# 两条未通过的反问题：粘性 Burgers、KdV 双孤子。旧脚本不改，只调用。
#
# 第一部分 诊断（不改数据）：
#   1. 设计阵的秩：数据能不能唯一确定方程。
#   2. 不可表示条件 |A_j^T A_S (A_S^T A_S)^{-1} s| < w_j：
#      无噪声时，它 > 1 的列在所有足够小的 λ 都会被 L1 选上，与求解器无关。
#   3. 用精确同伦路径（LARS-Lasso）复现旧结果，并检查 KKT，排除“坐标下降没收敛”。
#
# 第二部分 修法：迭代重加权 L1（Candes, Wakin, Boyd 2008）。
#   第 0 轮权重全 1，就是普通 L1；之后 w_j = 1/(|β_j| + ε)，β 来自上一轮 L1 解。
#   不用岭回归、不用最小二乘给权重。每一轮都是精确 L1。
#   定支撑仍用旧规则：40 个 λ（λ_max 到 λ_max·1e-6）上 |β| > 1e-5 的次数过半。阈值不放宽。
#
# Burgers 旧数据（两条 tanh 行波）设计阵秩亏 2：库里存在另外两族精确方程（都比真方程稠密）。
# L1 取最稀疏者，仍可选对；但为排除“靠秩亏碰巧”，另加满秩的新数据： Cole–Hopf 精确解：φ 是 4 个指数之和，3 个激波追赶合并。
# 导数用累积量的解析式，u_t 由 φ_t = ν φ_xx 独立推出，不是从 Burgers 方程倒填。

import json
import platform
from pathlib import Path

import numpy as np

import inverse_burgers
import redo_two
from lasso_exact import kkt_violation, lars_lasso_path, lasso_at

SUPPORT_TOL = 1e-5
N_LAMBDA = 40
LAMBDA_LO = 1e-6
IRL1_RATIO = 1e-3          # 每轮 IRL1 求解所用 λ = λ_max(w) * 这个比例
IRL1_EPS = 1e-8
IRL1_MAX_ITER = 20
SWEEP_RATIOS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
SWEEP_EPS = [1e-8, 1e-6, 1e-4, 1e-2]

NU_B = 0.1
CH_K = [0.0, -5.0, -10.0, -15.0]
CH_X0 = [0.0, 1.0, 0.4, 0.2]
CH_X = (-1.0, 3.0, 161)
CH_T = (0.0, 1.0, 41)


# ---------- 数据 ----------

def burgers_library_names():
    return inverse_burgers.library_spec()


def burgers_columns(u, u_x, u_xx, u_xxx):
    return np.column_stack(
        [u * u_x, u_xx, np.ones_like(u), u, u**2, u**3, u_x, u_xxx, (u**2) * u_x, u * u_xx]
    )


def cole_hopf(x, t, ks=CH_K, x0s=CH_X0, nu=NU_B):
    """u = -2ν ∂x log φ，φ = Σ exp(k_i (x - x0_i) + ν k_i² t)。
    ∂x^m log φ 是 k 在权重 p_i ∝ exp(θ_i) 下的 m 阶累积量。"""
    ks_ = np.asarray(ks, dtype=float)[:, None]
    x0_ = np.asarray(x0s, dtype=float)[:, None]
    theta = ks_ * (x[None, :] - x0_) + nu * ks_**2 * t[None, :]
    theta = theta - theta.max(axis=0)
    e = np.exp(theta)
    p = e / e.sum(axis=0)
    m1, m2, m3, m4 = [(p * ks_**k).sum(axis=0) for k in (1, 2, 3, 4)]
    k1 = m1
    k2 = m2 - m1**2
    k3 = m3 - 3 * m2 * m1 + 2 * m1**3
    k4 = m4 - 4 * m3 * m1 - 3 * m2**2 + 12 * m2 * m1**2 - 6 * m1**4
    u = -2 * nu * k1
    u_x = -2 * nu * k2
    u_xx = -2 * nu * k3
    u_xxx = -2 * nu * k4
    # ∂t log φ = ν φ_xx/φ = ν (k2 + k1²)，u_t = -2ν ∂x 之 = -2ν² (k3 + 2 k1 k2)
    u_t = -2 * nu**2 * (k3 + 2 * k1 * k2)
    return u, u_t, u_x, u_xx, u_xxx


def cole_hopf_fd_check():
    rng = np.random.default_rng(1)
    x = rng.uniform(CH_X[0], CH_X[1], 50)
    t = rng.uniform(CH_T[0], CH_T[1], 50)
    h = 1e-5
    u, u_t, u_x, u_xx, _ = cole_hopf(x, t)
    ut = (cole_hopf(x, t + h)[0] - cole_hopf(x, t - h)[0]) / (2 * h)
    ux = (cole_hopf(x + h, t)[0] - cole_hopf(x - h, t)[0]) / (2 * h)
    uxx = (cole_hopf(x + h, t)[2] - cole_hopf(x - h, t)[2]) / (2 * h)
    return float(max(np.max(np.abs(ut - u_t)), np.max(np.abs(ux - u_x)), np.max(np.abs(uxx - u_xx))))


def data_burgers_two_waves():
    a, b = inverse_burgers.design([0.5, 1.0])
    return a, b, burgers_library_names(), [0, 1], {0: -1.0, 1: NU_B}


def data_burgers_cole_hopf():
    x = np.linspace(*CH_X)
    t = np.linspace(*CH_T)
    xx, tt = np.meshgrid(x, t, indexing="xy")
    u, u_t, u_x, u_xx, u_xxx = cole_hopf(xx.ravel(), tt.ravel())
    return burgers_columns(u, u_x, u_xx, u_xxx), u_t, burgers_library_names(), [0, 1], {0: -1.0, 1: NU_B}


def kdv_names():
    return [
        "u u_x  [真项, 系数 -6]",
        "u_xxx  [真项, 系数 -1]",
        "1",
        "u",
        "u_x",
        "u_xx",
        "u^2",
        "u^2 u_x",
    ]


def data_kdv_two_soliton():
    # 与 redo_two.py / optimize_ux.py 完全相同的网格和库
    x = np.linspace(-10.0, 12.0, 90)
    t = np.linspace(0.0, 1.2, 16)
    xx, tt = np.meshgrid(x, t, indexing="xy")
    u, u_t, u_x, u_xx, u_xxx = redo_two.two_soliton_fields(xx.ravel(), tt.ravel())
    a = np.column_stack([u * u_x, u_xxx, np.ones_like(u), u, u_x, u_xx, u**2, (u**2) * u_x])
    return a, u_t, kdv_names(), [0, 1], {0: -6.0, 1: -1.0}


# ---------- 诊断 ----------

def diagnose(a_raw, b, names, true_idx, true_coef):
    p = a_raw.shape[1]
    col = np.maximum(np.linalg.norm(a_raw, axis=0), 1e-15)
    a = a_raw / col
    xi = np.zeros(p)
    for i, c in true_coef.items():
        xi[i] = c
    exact = float(np.linalg.norm(a_raw @ xi - b) / np.linalg.norm(b))
    _, sv, vt = np.linalg.svd(a, full_matrices=False)
    rank = int(np.sum(sv > sv[0] * 1e-12))
    null = []
    for k in range(rank, p):
        v = vt[k] / col
        v = v / np.max(np.abs(v))
        null.append({names[i]: float(v[i]) for i in range(p) if abs(v[i]) > 1e-9})
    s_idx = list(true_idx)
    gram = a[:, s_idx].T @ a[:, s_idx]
    sign = np.sign(xi[s_idx])
    irrep_plain = np.abs(a.T @ a[:, s_idx] @ np.linalg.solve(gram, sign))
    ridge = np.linalg.solve(a.T @ a + 1e-2 * np.eye(p), a.T @ b)
    w_r = 1.0 / (np.abs(ridge) + 1e-8)
    irrep_ridge = np.abs(a.T @ a[:, s_idx] @ np.linalg.solve(gram, w_r[s_idx] * sign)) / w_r
    return {
        "exact_equation_relative_residual": exact,
        "n_rows": int(a.shape[0]),
        "singular_values_normalized": sv.tolist(),
        "rank": rank,
        "n_columns": p,
        "identifiable": rank == p,
        "exact_null_vectors_physical": null,
        "irrepresentable_plain_l1": {names[i]: float(irrep_plain[i]) for i in range(p) if i not in s_idx},
        "irrepresentable_ridge_adaptive_l1": {names[i]: float(irrep_ridge[i]) for i in range(p) if i not in s_idx},
        "irrepresentable_note": "无噪声时，某列的值 > 1，则所有足够小的 λ 下 L1 都会选上这一列。与求解器无关。",
    }


def frequency_on_path(a, b, w):
    knots = lars_lasso_path(a, b, w, lam_min_ratio=LAMBDA_LO * 0.1)
    lam_max = knots[0][0]
    lams = lam_max * np.geomspace(1.0, LAMBDA_LO, N_LAMBDA)
    counts = np.zeros(a.shape[1], dtype=int)
    worst_kkt = 0.0
    rows = []
    bnorm = float(np.linalg.norm(b))
    for lam in lams:
        beta = lasso_at(knots, float(lam))
        counts[np.abs(beta) > SUPPORT_TOL] += 1
        worst_kkt = max(worst_kkt, kkt_violation(a, b, beta, float(lam), w))
        rows.append((float(lam / lam_max), float(np.linalg.norm(a @ beta - b) / bnorm), beta))
    return counts, worst_kkt, rows, lam_max


def plain_path_record(a, b, names, ux_index):
    counts, kkt, rows, _ = frequency_on_path(a, b, np.ones(a.shape[1]))
    pick = [0, 10, 20, 30, 39]
    return {
        "frequency": {names[i]: int(counts[i]) for i in range(len(names))},
        "max_relative_kkt_violation": kkt,
        "samples": [
            {
                "lambda_over_lambda_max": rows[k][0],
                "relative_residual": rows[k][1],
                "u_x_normalized": float(rows[k][2][ux_index]),
            }
            for k in pick
        ],
        "note": "精确同伦解。无噪声时残差和 u_x 系数都随 λ 线性趋零，λ_min = 1e-6 λ_max 时残差约 1e-6 量级，所以旧的 1e-8 残差门槛在这条路径上本来就到不了。",
    }


def irl1(a, b, ratio=IRL1_RATIO, eps=IRL1_EPS):
    p = a.shape[1]
    w = np.ones(p)
    history = []
    prev = None
    for it in range(IRL1_MAX_ITER):
        knots = lars_lasso_path(a, b, w, lam_min_ratio=ratio * 0.1)
        lam = knots[0][0] * ratio
        beta = lasso_at(knots, lam)
        supp = [int(i) for i in np.where(np.abs(beta) > SUPPORT_TOL)[0]]
        history.append({"iter": it, "support": supp, "kkt": kkt_violation(a, b, beta, lam, w)})
        w = 1.0 / (np.abs(beta) + eps)
        if prev is not None and supp == prev:
            break
        prev = supp
    return w, history


def solve_case(label, a_raw, b, names, true_idx, true_coef, ux_name):
    p = a_raw.shape[1]
    col = np.maximum(np.linalg.norm(a_raw, axis=0), 1e-15)
    a = a_raw / col
    bnorm = float(np.linalg.norm(b))
    ux_index = [i for i, n in enumerate(names) if n.split()[0] == ux_name][0]
    out = {"name": label, "library": names}
    out["diagnosis"] = diagnose(a_raw, b, names, true_idx, true_coef)
    out["plain_l1_exact_path"] = plain_path_record(a, b, names, ux_index)

    w, hist = irl1(a, b)
    counts, kkt, _, _ = frequency_on_path(a, b, w)
    support = [int(i) for i in np.where(counts > N_LAMBDA / 2.0)[0]]
    xi = np.zeros(p)
    if support:
        coef, *_ = np.linalg.lstsq(a_raw[:, support], b, rcond=None)
        xi[support] = coef
    out["irl1"] = {
        "ratio": IRL1_RATIO,
        "eps": IRL1_EPS,
        "history": [{**h, "support_names": [names[i] for i in h["support"]]} for h in hist],
        "final_frequency": {names[i]: int(counts[i]) for i in range(p)},
        "max_relative_kkt_violation_on_path": kkt,
        "support_names": [names[i] for i in support],
        "passed_support": support == list(true_idx),
        "refit_physical": {names[i]: float(xi[i]) for i in support},
        "refit_relative_residual": float(np.linalg.norm(a_raw @ xi - b) / bnorm),
        "rule": "最终权重下 40 个 λ（λ_max 到 1e-6 λ_max）上 |β|>1e-5 次数过半。阈值不放宽。",
    }
    sweep = []
    for r in SWEEP_RATIOS:
        for e in SWEEP_EPS:
            w2, h2 = irl1(a, b, ratio=r, eps=e)
            c2, _, _, _ = frequency_on_path(a, b, w2)
            s2 = [int(i) for i in np.where(c2 > N_LAMBDA / 2.0)[0]]
            sweep.append({"ratio": r, "eps": e, "n_iter": len(h2), "support_names": [names[i] for i in s2], "passed": s2 == list(true_idx)})
    out["irl1_sensitivity"] = {
        "n_settings": len(sweep),
        "n_passed": sum(s["passed"] for s in sweep),
        "settings": sweep,
    }
    return out


def main():
    report = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "solver": "精确 LARS-Lasso 同伦路径（lasso_exact.py），每个 λ 做 KKT 检查",
        "cole_hopf": {"k": CH_K, "x0": CH_X0, "nu": NU_B, "x": CH_X, "t": CH_T, "fd_check_max_abs": cole_hopf_fd_check()},
        "cases": [],
    }
    report["cases"].append(solve_case("反问题 KdV 双孤子（旧数据，不改）", *data_kdv_two_soliton(), "u_x"))
    report["cases"].append(solve_case("反问题 粘性 Burgers Cole–Hopf 激波合并（新数据）", *data_burgers_cole_hopf(), "u_x"))
    report["cases"].append(solve_case("粘性 Burgers 两条行波（旧数据，不改；设计阵秩亏 2）", *data_burgers_two_waves(), "u_x"))

    out = Path(__file__).with_name("inverse_irl1_result.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("python", report["python"], "numpy", report["numpy"])
    print("Cole-Hopf FD check", f"{report['cole_hopf']['fd_check_max_abs']:.2e}")
    for c in report["cases"]:
        d = c["diagnosis"]
        print("==", c["name"])
        print(f"  exact residual {d['exact_equation_relative_residual']:.2e}  rank {d['rank']}/{d['n_columns']}  min sv {d['singular_values_normalized'][-1]:.2e}")
        print("  irrepresentable (plain):", {k.split('  ')[0]: round(v, 3) for k, v in d["irrepresentable_plain_l1"].items()})
        pl = c["plain_l1_exact_path"]
        print("  plain L1 freq:", {k.split('  ')[0]: v for k, v in pl["frequency"].items()}, f"KKT {pl['max_relative_kkt_violation']:.1e}")
        ir = c["irl1"]
        print("  IRL1 iters:", [h["support_names"] for h in ir["history"]])
        print("  IRL1 freq:", {k.split('  ')[0]: v for k, v in ir["final_frequency"].items()})
        print("  support:", ir["support_names"], "PASSED" if ir["passed_support"] else "FAILED")
        print("  refit:", ir["refit_physical"], f"res {ir['refit_relative_residual']:.2e}")
        print(f"  sensitivity {c['irl1_sensitivity']['n_passed']}/{c['irl1_sensitivity']['n_settings']}")


if __name__ == "__main__":
    main()
