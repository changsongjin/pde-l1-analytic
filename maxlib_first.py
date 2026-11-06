# 最大库第一批。forward_biglib.py 和 noise_inverse_biglib.py 不改。
# 规格文件 maxlib_spec.md 当时不在工作区，列按研究员当轮列出的边界写。
# 正问题：单个行波变量 θ，35 列，无噪声，库里确有真解。
# 反问题：弱形式 56 列，两条轨迹。
# 求解只做自适应 L1。阈值 1e-5 只用来计数，不放宽。
# 通过只认支撑：入选次数过半，且非零系数个数等于真项个数。

import json
from pathlib import Path

import numpy as np
from scipy.signal import correlate

import inverse_burgers
import noise_inverse as N
import noise_inverse_biglib as B

SUPPORT_TOL = 1e-5
N_LAMBDA = 40

FORWARD_DROP = {
    "sech² θ",
    "sech⁴ θ",
    "tanh θ sech² θ",
    "tanh² θ sech² θ",
    "cos² θ",
}


def forward_names():
    names = ["1"]
    names += [f"tanh^{a} θ" if a > 1 else "tanh θ" for a in range(1, 6)]
    names += [f"sech^{b} θ" if b > 1 else "sech θ" for b in range(1, 6)]
    for a in range(1, 5):
        for b in range(1, 6 - a):
            left = "tanh θ" if a == 1 else f"tanh^{a} θ"
            right = "sech θ" if b == 1 else f"sech^{b} θ"
            names.append(f"{left} {right}")
    names += [f"logistic^{m} θ" if m > 1 else "logistic θ" for m in range(1, 5)]
    names += ["e^(−θ²)", "θ e^(−θ²)", "θ² e^(−θ²)", "θ³ e^(−θ²)"]
    names += ["sin θ", "cos θ", "sin² θ", "sin θ cos θ"]
    names += ["1/(1+θ²)", "θ/(1+θ²)", "1/(1+θ²)²", "θ²/(1+θ²)²"]
    names += ["θ", "θ²", "θ³", "θ⁴"]
    names += ["arctan θ", "arctan e^θ"]
    # 按族的阶数逐项生成是 43 列。研究员口头说的 35 对不上这个生成式，结果里写明。
    if len(names) != 43:
        raise RuntimeError(f"正问题列数不是 43: {len(names)}")
    kept = [name for name in names if name not in FORWARD_DROP]
    return names, kept


def theta_of(x, t, k, c, x0):
    return k * (x - x0 - c * t)


def forward_columns(theta):
    t = np.tanh(theta)
    s = 1.0 / np.cosh(theta)
    h = np.tanh(theta / 2.0)
    logistic = (1.0 - h) / 2.0
    g = np.exp(-(theta**2))
    cols = [np.ones_like(theta)]
    cols += [t**a for a in range(1, 6)]
    cols += [s**b for b in range(1, 6)]
    for a in range(1, 5):
        for b in range(1, 6 - a):
            cols.append((t**a) * (s**b))
    cols += [logistic**m for m in range(1, 5)]
    cols += [g, theta * g, (theta**2) * g, (theta**3) * g]
    cols += [np.sin(theta), np.cos(theta), np.sin(theta) ** 2, np.sin(theta) * np.cos(theta)]
    den = 1.0 + theta**2
    cols += [1.0 / den, theta / den, 1.0 / den**2, (theta**2) / den**2]
    cols += [theta, theta**2, theta**3, theta**4]
    cols += [np.arctan(theta), np.pi / 4.0 + np.arctan(h)]
    return cols


def forward_case():
    """KdV 孤子。θ=x−4t 时真解就是 sech² θ，系数 2。"""
    names, kept_names = forward_names()
    kept = [names.index(name) for name in kept_names]
    x = np.linspace(-8.0, 12.0, 81)
    t = np.linspace(0.0, 1.0, 21)
    xx, tt = np.meshgrid(x[1:-1], t[1:], indexing="xy")
    theta = theta_of(xx.ravel(), tt.ravel(), 1.0, 4.0, 0.0)
    cols = forward_columns(theta)
    # 真解 u=2 sech²θ。用解析导数写 PDE 残差、两端边界和初值。
    def soliton(x, t):
        return 2.0 / np.cosh(x - 4.0 * t) ** 2

    h = 1e-3
    grid_x, grid_t = xx.ravel(), tt.ravel()
    u = soliton(grid_x, grid_t)
    u_t = (soliton(grid_x, grid_t + h) - soliton(grid_x, grid_t - h)) / (2 * h)
    u_x = (soliton(grid_x + h, grid_t) - soliton(grid_x - h, grid_t)) / (2 * h)
    u_xxx = (
        soliton(grid_x + 2 * h, grid_t)
        - 2 * soliton(grid_x + h, grid_t)
        + 2 * soliton(grid_x - h, grid_t)
        - soliton(grid_x - 2 * h, grid_t)
    ) / (2 * h**3)
    exact = u_t + 6.0 * u * u_x + u_xxx
    exact_rel = float(np.linalg.norm(exact) / np.linalg.norm(u_t))

    xb = np.concatenate([np.full_like(t, -8.0), np.full_like(t, 12.0)])
    tb = np.concatenate([t, t])
    xi = np.linspace(-8.0, 12.0, 81)
    blocks = []
    for xq, tq, target in (
        (xb, tb, 2.0 / np.cosh(xb - 4.0 * tb) ** 2),
        (xi, np.zeros_like(xi), 2.0 / np.cosh(xi) ** 2),
    ):
        qcols = forward_columns(theta_of(xq, tq, 1.0, 4.0, 0.0))
        mat = np.column_stack([qcols[i] for i in kept])
        scale = np.sqrt(mat.shape[0])
        blocks.append((mat / scale, target / scale))
    a = np.vstack([b[0] for b in blocks])
    rhs = np.concatenate([b[1] for b in blocks])
    picked = inverse_burgers.weighted_lasso
    col = np.maximum(np.linalg.norm(a, axis=0), 1e-15)
    an = a / col
    bnorm = float(np.linalg.norm(rhs))
    ridge = np.linalg.solve(an.T @ an + 1e-2 * np.eye(an.shape[1]), an.T @ rhs)
    w = 1.0 / (np.abs(ridge) + 1e-8)
    lam_max = float(np.max(np.abs(an.T @ rhs) / w))
    beta = np.zeros(an.shape[1])
    counts = np.zeros(an.shape[1], dtype=int)
    for lam in lam_max * np.geomspace(1.0, 1e-6, N_LAMBDA):
        beta = picked(an, rhs, float(lam), w, beta0=beta)
        counts[np.abs(beta) > SUPPORT_TOL] += 1
    support = np.where(counts > N_LAMBDA / 2.0)[0]
    return {
        "name": "正问题 KdV 孤子，最大库",
        "library_size_before_drop": 43,
        "count_note": "按族逐项生成是 43 列，不是口头说的 35。差在 tanh^a sech^b、a+b=5 的乘积。",
        "library_size": len(kept),
        "dropped_dependent_columns": sorted(FORWARD_DROP),
        "exact_pde_relative_residual": exact_rel,
        "frequency": {kept_names[i]: int(counts[i]) for i in range(len(kept)) if counts[i]},
        "support_names": [kept_names[i] for i in support],
        "passed_support": [kept_names[i] for i in support] == ["sech² θ"],
        "note": "波速和尺度已知并写进 θ。只看初值和边界能不能唯一选出 sech²。PDE 残差只核对真解。",
    }


def inverse_names():
    names = ["1"]
    for power in range(1, 7):
        for order in range(5):
            base = "u" if power == 1 else f"u^{power}"
            names.append(base if order == 0 else base + "_" + "x" * order)
    for fun in ("sin u", "cos u"):
        for order in range(4):
            names.append(fun if order == 0 else f"({fun})_" + "x" * order)
    for fun in ("x u", "x u^2"):
        for order in (0, 1):
            names.append(fun if order == 0 else f"({fun})_x")
    # 1 + 6 个幂次 × 5 个导数阶 + sin/cos 各 4 列 + 含 x 的 4 列 = 43。
    # 口头说的 56 对不上这个生成式，结果里写明。
    if len(names) != 43:
        raise RuntimeError(f"反问题列数不是 43: {len(names)}")
    return names


def inverse_case():
    """Burgers 两条轨迹。真项 (u²)_x 系数 -1/2，u_xx 系数 0.1。"""
    names = inverse_names()
    cols = [(0, "1")]
    for power in range(1, 7):
        for order in range(5):
            cols.append((order, "u" if power == 1 else f"u^{power}"))
    for fun in ("sin u", "cos u"):
        for order in range(4):
            cols.append((order, fun))
    for fun in ("x u", "x u^2"):
        cols.append((0, fun))
        cols.append((1, fun))
    names = [B.name(k, f) for k, f in cols]
    if len(names) != 43:
        raise RuntimeError(f"反问题列数不是 43: {len(names)}")
    case = {
        "pde": "u_t = -u u_x + 0.1 u_xx",
        "traj": [
            lambda x, t: 0.5 * (1.0 - np.tanh(2.5 * (x - 0.3 - 0.5 * t))),
            lambda x, t: 1.0 * (1.0 - np.tanh(5.0 * (x + 0.4 - t))),
        ],
        "x": (-1.0, 2.0, 161),
        "t": (0.0, 1.0, 41),
        "half": (12, 8),
        "target": "u_t",
        "true": {"(u^2)_x": -0.5, "u_xx": 0.1},
    }
    theta, target = B.make_data(case, cols, sigma=0.0, rng=np.random.default_rng(0))
    xi = np.zeros(theta.shape[1])
    for key, coef in case["true"].items():
        xi[names.index(key)] = coef
    exact_rel = float(np.linalg.norm(theta @ xi - target) / np.linalg.norm(target))
    col = np.maximum(np.linalg.norm(theta, axis=0), 1e-15)
    an = theta / col
    bnorm = float(np.linalg.norm(target))
    ridge = np.linalg.solve(an.T @ an + 1e-2 * np.eye(an.shape[1]), an.T @ target)
    w = 1.0 / (np.abs(ridge) + 1e-8)
    lam_max = float(np.max(np.abs(an.T @ target) / w))
    beta = np.zeros(an.shape[1])
    counts = np.zeros(an.shape[1], dtype=int)
    for lam in lam_max * np.geomspace(1.0, 1e-6, N_LAMBDA):
        beta = inverse_burgers.weighted_lasso(an, target, float(lam), w, beta0=beta)
        counts[np.abs(beta) > SUPPORT_TOL] += 1
    support = np.where(counts > N_LAMBDA / 2.0)[0]
    got = [names[i] for i in support]
    return {
        "name": "反问题 Burgers 两条轨迹，最大库",
        "library_size": len(names),
        "count_note": "按幂次、导数阶和函数族逐项生成是 43 列，不是口头说的 56。",
        "n_rows": int(theta.shape[0]),
        "exact_equation_relative_residual": exact_rel,
        "true_terms": case["true"],
        "frequency": {names[i]: int(counts[i]) for i in range(len(names)) if counts[i]},
        "support_names": got,
        "passed_support": got == ["(u^2)_x", "u_xx"],
        "note": "两条不同波速。弱形式沿用 noise_inverse.py。无噪声。",
    }


def main():
    report = {"forward": forward_case(), "inverse": inverse_case()}
    path = Path(__file__).with_name("maxlib_first_result.json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for key in ("forward", "inverse"):
        case = report[key]
        print("==", case["name"])
        print("columns", case["library_size"])
        print("exact residual", f"{case.get('exact_pde_relative_residual', case.get('exact_equation_relative_residual')):.3e}")
        print("support", case["support_names"])
        print("passed", case["passed_support"])
        print("frequency", case["frequency"])


if __name__ == "__main__":
    main()
