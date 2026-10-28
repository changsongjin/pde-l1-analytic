# 重做两条反问题。batch5.py 不改。
# 平流扩散：宽度按 σ²(t)=σ0²+2νt 增长。
# KdV 双孤子：导数用 τ 的幂级数，不用差分。
# 真方程相对残差大于 1e-8 就停，不报支撑。

import json
from pathlib import Path

import numpy as np

SUPPORT_TOL = 1e-5
C = 0.4
NU = 0.05
SIGMAS = (0.5, 0.9)


def soft(rho, thr):
    if rho > thr:
        return rho - thr
    if rho < -thr:
        return rho + thr
    return 0.0


def weighted_lasso(a, b, lam, w, beta0=None):
    n, p = a.shape
    beta = np.zeros(p) if beta0 is None else beta0.copy()
    col2 = np.sum(a * a, axis=0)
    r = b - a @ beta
    for _ in range(8000):
        max_delta = 0.0
        for j in range(p):
            if col2[j] < 1e-18:
                continue
            r += a[:, j] * beta[j]
            new = soft(float(a[:, j] @ r), lam * w[j]) / col2[j]
            max_delta = max(max_delta, abs(new - beta[j]))
            r -= a[:, j] * new
            beta[j] = new
        if max_delta < 1e-12:
            break
    return beta


def linear_select(a_raw, b, names, true_index):
    p = a_raw.shape[1]
    col = np.maximum(np.linalg.norm(a_raw, axis=0), 1e-15)
    a = a_raw / col
    bnorm = float(np.linalg.norm(b))
    ridge = np.linalg.solve(a.T @ a + 1e-2 * np.eye(p), a.T @ b)
    w = 1.0 / (np.abs(ridge) + 1e-8)
    lam_max = float(np.max(np.abs(a.T @ b) / w))
    lams = lam_max * np.geomspace(1.0, 1e-6, 40)
    beta = np.zeros(p)
    counts = np.zeros(p, dtype=int)
    for lam in lams:
        beta = weighted_lasso(a, b, float(lam), w, beta0=beta)
        counts[np.abs(beta) > SUPPORT_TOL] += 1
    support = np.where(counts > len(lams) / 2.0)[0]
    xi = np.zeros(p)
    if len(support):
        coef, _, _, _ = np.linalg.lstsq(a_raw[:, support], b, rcond=None)
        xi[support] = coef
    return {
        "library": names,
        "n_rows": int(a_raw.shape[0]),
        "frequency": {names[i]: int(counts[i]) for i in range(p)},
        "support_indices": [int(i) for i in support],
        "support_names": [names[i] for i in support],
        "passed_support": list(support) == list(true_index),
        "refit_physical": xi.tolist(),
        "refit_relative_residual": float(np.linalg.norm(a_raw @ xi - b) / bnorm),
        "n_lambda": len(lams),
        "rule": "40 个 λ 上入选次数过半。阈值 1e-5。自适应 L1。",
    }


def gaussian(x, t, sigma0, x0=0.0):
    s = sigma0**2 + 2.0 * NU * t
    xi = x - x0 - C * t
    u = (2.0 * np.pi * s) ** -0.5 * np.exp(-(xi**2) / (2.0 * s))
    u_x = u * (-xi / s)
    u_xx = u * (xi**2 / s**2 - 1.0 / s)
    u_xxx = u_x * (xi**2 / s**2 - 1.0 / s) + u * (2.0 * xi / s**2 - 2.0 * xi**3 / s**3 + 2.0 * xi / s**2)
    # ∂t log u = -ν/s + c ξ/s + ν ξ²/s²，已手算并与 -c u_x + ν u_xx 对齐
    u_t = u * (-NU / s + C * xi / s + NU * xi**2 / s**2)
    return u, u_t, u_x, u_xx, u_xxx


def advection():
    names = [
        "u_x  [真项, 系数 -0.4]",
        "u_xx  [真项, 系数 0.05]",
        "1",
        "u",
        "u^2",
        "u u_x",
        "u_xxx",
    ]
    x = np.linspace(-3.0, 4.0, 141)
    t = np.linspace(0.0, 1.0, 21)
    xx, tt = np.meshgrid(x, t, indexing="xy")
    xx, tt = xx.ravel(), tt.ravel()
    blocks_a, blocks_b = [], []
    exact_num = 0.0
    exact_den = 0.0
    for sigma0 in SIGMAS:
        u, u_t, u_x, u_xx, u_xxx = gaussian(xx, tt, sigma0)
        residual = u_t + C * u_x - NU * u_xx
        exact_num += float(np.dot(residual, residual))
        exact_den += float(np.dot(u_t, u_t))
        theta = np.column_stack([u_x, u_xx, np.ones_like(u), u, u**2, u * u_x, u_xxx])
        scale = np.linalg.norm(u_t)
        blocks_a.append(theta / scale)
        blocks_b.append(u_t / scale)
    exact_rel = float(np.sqrt(exact_num / exact_den))
    out = {
        "name": "反问题 平流扩散",
        "pde": "u_t = -0.4 u_x + 0.05 u_xx",
        "exact": "两个高斯，σ0=0.5 和 0.9，σ²(t)=σ0²+2νt，中心 x=ct",
        "exact_equation_relative_residual": exact_rel,
        "derivative": "解析式",
    }
    if exact_rel > 1e-8:
        out["stopped"] = True
        out["reason"] = "真方程残差大于 1e-8，不报支撑"
        return out
    picked = linear_select(np.vstack(blocks_a), np.concatenate(blocks_b), names, [0, 1])
    out.update(picked)
    out["stopped"] = False
    out["refit_of_true_terms"] = {
        names[0]: picked["refit_physical"][0],
        names[1]: picked["refit_physical"][1],
    }
    return out


def series_mul(a, b):
    n = a.shape[0]
    c = np.zeros_like(a)
    for k in range(n):
        acc = np.zeros(a.shape[1])
        for j in range(k + 1):
            acc += a[j] * b[k - j]
        c[k] = acc
    return c


def derivs_of_log(tau):
    order = tau.shape[0] - 1
    f = np.zeros_like(tau)
    f[0] = np.log(tau[0])
    for m in range(1, order + 1):
        acc = tau[m].copy()
        binom = 1.0
        for j in range(1, m):
            binom *= (m - j) / j
            acc = acc - binom * tau[j] * f[m - j]
        f[m] = acc / tau[0]
    return f


def derivs_of_quotient(num, den):
    order = den.shape[0] - 1
    r = np.zeros_like(den)
    for m in range(order + 1):
        acc = num[m].copy()
        binom = 1.0
        for j in range(1, m + 1):
            binom *= (m - j + 1) / j
            acc = acc - binom * den[j] * r[m - j]
        r[m] = acc / den[0]
    return r


def exp_series(a, x, t, x0, order=5):
    theta = a * (x - x0) - (a**3) * t
    e = np.exp(theta)
    return np.stack([e * a**k for k in range(order + 1)], axis=0)


def two_soliton_fields(x, t):
    a1, a2 = 1.0, 2.0
    x01, x02 = -2.0, -8.0
    alpha = ((a1 - a2) / (a1 + a2)) ** 2
    order = 5
    e1 = exp_series(a1, x, t, x01, order)
    e2 = exp_series(a2, x, t, x02, order)
    inter = np.stack(
        [alpha * np.exp(a1 * (x - x01) + a2 * (x - x02) - (a1**3 + a2**3) * t) * (a1 + a2) ** k
         for k in range(order + 1)],
        axis=0,
    )
    tau = e1 + e2 + inter
    tau[0] += 1.0
    f = derivs_of_log(tau)
    # ∂t exp(θi) = -a_i³ exp(θi)，和 ∂x 可交换
    t1 = -a1**3 * e1
    t2 = -a2**3 * e2
    t12 = -(a1**3 + a2**3) * inter
    tau_t = t1 + t2 + t12
    r = derivs_of_quotient(tau_t, tau)
    u = 2.0 * f[2]
    u_x = 2.0 * f[3]
    u_xx = 2.0 * f[4]
    u_xxx = 2.0 * f[5]
    u_t = 2.0 * r[2]
    return u, u_t, u_x, u_xx, u_xxx


def kdv():
    names = [
        "u u_x  [真项, 系数 -6]",
        "u_xxx  [真项, 系数 -1]",
        "1",
        "u",
        "u_x",
        "u_xx",
        "u^2",
        "u^2 u_x",
    ]
    x = np.linspace(-10.0, 12.0, 90)
    t = np.linspace(0.0, 1.2, 16)
    xx, tt = np.meshgrid(x, t, indexing="xy")
    u, u_t, u_x, u_xx, u_xxx = two_soliton_fields(xx.ravel(), tt.ravel())
    residual = u_t + 6.0 * u * u_x + u_xxx
    exact_rel = float(np.linalg.norm(residual) / np.linalg.norm(u_t))
    out = {
        "name": "反问题 KdV 双孤子",
        "pde": "u_t = -6 u u_x - u_xxx",
        "exact": "振幅参数 1 和 2，u = 2 ∂xx log τ",
        "exact_equation_relative_residual": exact_rel,
        "derivative": "τ 的幂级数求导，不是差分",
    }
    if exact_rel > 1e-8:
        out["stopped"] = True
        out["reason"] = "真方程残差大于 1e-8，不报支撑"
        return out
    theta = np.column_stack(
        [u * u_x, u_xxx, np.ones_like(u), u, u_x, u_xx, u**2, (u**2) * u_x]
    )
    picked = linear_select(theta, u_t, names, [0, 1])
    out.update(picked)
    out["stopped"] = False
    out["refit_of_true_terms"] = {
        names[0]: picked["refit_physical"][0],
        names[1]: picked["refit_physical"][1],
    }
    return out


def main():
    cases = [advection(), kdv()]
    out = Path(__file__).with_name("redo_two_result.json")
    out.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    for case in cases:
        print("==", case["name"])
        print("exact residual", f"{case['exact_equation_relative_residual']:.3e}")
        print("stopped", case["stopped"])
        if case["stopped"]:
            print(case["reason"])
            continue
        print("support", case["support_names"])
        print("passed", case["passed_support"])
        print("refit", case["refit_of_true_terms"])
        print("refit residual", f"{case['refit_relative_residual']:.3e}")
        print("frequency", case["frequency"])


if __name__ == "__main__":
    main()
