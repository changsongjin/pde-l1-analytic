# 第一档：热方程，无解的采样，自适应 L1。
# 只看支撑是否只剩真项，以及丢掉 L1 之后系数是否回到 1。
# 默认值都写在 DEFAULTS，结果里原样打印。

import json
from pathlib import Path

import numpy as np

DEFAULTS = {
    "pde": "u_t = u_xx",
    "domain_x": [0.0, 1.0],
    "domain_t": [0.0, 1.0],
    "ic": "sin(pi x)",
    "bc": "u(0,t)=u(1,t)=0",
    "n_x_pde": 25,
    "n_t": 15,
    "n_x_ic": 51,
    "block_scale": "每个残差块除以 sqrt(行数)，PDE / 边界 / 初值三块总权重相同",
    "column_normalize": True,
    "ridge_alpha": 1e-2,
    "adaptive_gamma": 1.0,
    "adaptive_eps": 1e-8,
    "lambda_path": "50 个，从 lambda_max 几何降到 lambda_max * 1e-6",
    "fit_tol": 1e-6,
    "support_tol": 1e-5,
    "cd_max_iter": 8000,
    "cd_tol": 1e-10,
    "solver": "自适应 L1：先岭回归得权重，再加权 L1。不是序贯阈值，不是普通 Lasso。",
    "no_solution_samples": True,
}

PI = np.pi


def library_spec():
    names = ["exp(-pi^2 t) sin(pi x)  [真项]"]
    for k in range(2, 6):
        names.append(f"exp(-({k}pi)^2 t) sin({k}pi x)")
    names += [
        "1",
        "x",
        "x^2",
        "t",
        "x t",
        "x^2 t",
        "cos(pi x)",
        "cos(2 pi x)",
        "exp(-pi^2 t) cos(pi x)",
    ]
    return names


def eval_library(x, t):
    """返回 Phi, Phi_t, Phi_xx，形状 (n, p)。解析导数，不用数值差分。"""
    x = np.asarray(x, dtype=float).ravel()
    t = np.asarray(t, dtype=float).ravel()
    cols = []
    dts = []
    dxxs = []

    def add(phi, dt, dxx):
        cols.append(phi)
        dts.append(dt)
        dxxs.append(dxx)

    # 真项以及 k=2..5：都满足热方程和齐次边界，只有初值不同
    for k in range(1, 6):
        decay = np.exp(-((k * PI) ** 2) * t)
        phi = decay * np.sin(k * PI * x)
        add(phi, -((k * PI) ** 2) * phi, -((k * PI) ** 2) * phi)

    ones = np.ones_like(x)
    zeros = np.zeros_like(x)
    add(ones, zeros, zeros)
    add(x.copy(), zeros, zeros)
    add(x**2, zeros, 2.0 * ones)
    add(t.copy(), ones, zeros)
    add(x * t, x.copy(), zeros)
    add((x**2) * t, x**2, 2.0 * t)
    add(np.cos(PI * x), zeros, -(PI**2) * np.cos(PI * x))
    add(np.cos(2 * PI * x), zeros, -((2 * PI) ** 2) * np.cos(2 * PI * x))
    decay = np.exp(-(PI**2) * t)
    cos = decay * np.cos(PI * x)
    add(cos, -(PI**2) * cos, -(PI**2) * cos)

    return np.column_stack(cols), np.column_stack(dts), np.column_stack(dxxs)


def build_system():
    n_x = DEFAULTS["n_x_pde"]
    n_t = DEFAULTS["n_t"]
    x_all = np.linspace(0.0, 1.0, n_x)
    t_all = np.linspace(0.0, 1.0, n_t)
    x_int = x_all[1:-1]
    t_pde = t_all[1:]

    xx, tt = np.meshgrid(x_int, t_pde, indexing="xy")
    phi, dt, dxx = eval_library(xx.ravel(), tt.ravel())
    a_pde = dt - dxx
    b_pde = np.zeros(a_pde.shape[0])

    t_bc = np.concatenate([t_all, t_all])
    x_bc = np.concatenate([np.zeros_like(t_all), np.ones_like(t_all)])
    phi_bc, _, _ = eval_library(x_bc, t_bc)
    b_bc = np.zeros(phi_bc.shape[0])

    x_ic = np.linspace(0.0, 1.0, DEFAULTS["n_x_ic"])
    phi_ic, _, _ = eval_library(x_ic, np.zeros_like(x_ic))
    b_ic = np.sin(PI * x_ic)

    def scale(a, b):
        s = np.sqrt(a.shape[0])
        return a / s, b / s

    blocks = [scale(a_pde, b_pde), scale(phi_bc, b_bc), scale(phi_ic, b_ic)]
    a = np.vstack([b[0] for b in blocks])
    rhs = np.concatenate([b[1] for b in blocks])
    counts = {
        "pde": int(a_pde.shape[0]),
        "bc": int(phi_bc.shape[0]),
        "ic": int(phi_ic.shape[0]),
    }
    return a, rhs, counts


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
    for _ in range(DEFAULTS["cd_max_iter"]):
        max_delta = 0.0
        for j in range(p):
            r += a[:, j] * beta[j]
            rho = float(a[:, j] @ r)
            new = soft(rho, lam * w[j]) / col2[j]
            max_delta = max(max_delta, abs(new - beta[j]))
            r -= a[:, j] * new
            beta[j] = new
        if max_delta < DEFAULTS["cd_tol"]:
            break
    return beta


def main():
    names = library_spec()
    a_raw, b, counts = build_system()
    n, p = a_raw.shape
    assert p == len(names)

    col_scale = np.linalg.norm(a_raw, axis=0)
    col_scale = np.maximum(col_scale, 1e-15)
    a = a_raw / col_scale

    # 岭回归只用来生成自适应权重，不决定支撑
    gram = a.T @ a + DEFAULTS["ridge_alpha"] * np.eye(p)
    beta_ridge = np.linalg.solve(gram, a.T @ b)
    w = 1.0 / (np.abs(beta_ridge) + DEFAULTS["adaptive_eps"]) ** DEFAULTS["adaptive_gamma"]

    corr = np.abs(a.T @ b) / w
    lam_max = float(np.max(corr))
    if lam_max <= 0:
        raise RuntimeError("lambda_max 为 0，右端项和库没有相关性")
    lams = lam_max * np.geomspace(1.0, 1e-6, 50)

    bnorm = float(np.linalg.norm(b))
    rows = []
    beta = np.zeros(p)
    for lam in lams:
        beta = weighted_lasso(a, b, float(lam), w, beta0=beta)
        resid = float(np.linalg.norm(a @ beta - b) / bnorm)
        support = np.where(np.abs(beta) > DEFAULTS["support_tol"])[0]
        rows.append((float(lam), resid, support, beta.copy()))

    good = [r for r in rows if r[1] <= DEFAULTS["fit_tol"]]
    if good:
        good.sort(key=lambda r: (len(r[2]), -r[0]))
        chosen = good[0]
        rule = "相对残差 <= 1e-6 的 λ 里，支撑最短；一样短则取更大的 λ"
    else:
        chosen = min(rows, key=lambda r: (r[1], -r[0]))
        rule = "没有任何 λ 达到 1e-6，退而取相对残差最小的 λ"

    lam, resid, support, beta = chosen
    xi_l1 = beta / col_scale

    # 支撑只由加权 L1 决定。丢掉 L1 之后的最小二乘只读系数。
    if len(support) == 0:
        xi_refit = np.zeros(p)
        refit_resid = 1.0
    else:
        coef, _, _, _ = np.linalg.lstsq(a_raw[:, support], b, rcond=None)
        xi_refit = np.zeros(p)
        xi_refit[support] = coef
        refit_resid = float(np.linalg.norm(a_raw @ xi_refit - b) / bnorm)

    true_index = 0
    passed = list(support) == [true_index]
    coef_back = float(xi_refit[true_index]) if passed else None

    # 沿 λ 路径，支撑恰好是真项的区间，用来看默认规则是不是只碰到一次
    stable = [r[0] for r in rows if list(r[2]) == [true_index]]

    report = {
        "defaults": DEFAULTS,
        "rows": counts,
        "library": names,
        "selection_rule": rule,
        "lambda": lam,
        "lambda_max": lam_max,
        "relative_residual_l1": resid,
        "relative_residual_refit": refit_resid,
        "support_indices": [int(i) for i in support],
        "support_names": [names[i] for i in support],
        "passed_support": passed,
        "refit_coefficient_of_true_term": coef_back,
        "coefficient_back_to_1": bool(passed and abs(coef_back - 1.0) < 1e-6),
        "lambdas_with_exact_true_support": stable,
        "xi_adaptive_l1": xi_l1.tolist(),
        "xi_refit": xi_refit.tolist(),
        "beta_ridge_normalized": beta_ridge.tolist(),
    }

    out = Path(__file__).with_name("tier1_heat_result.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("rows", counts, "p", p)
    print("rule:", rule)
    print(f"lambda_max={lam_max:.6e}  lambda={lam:.6e}")
    print(f"L1 relative residual={resid:.3e}")
    print(f"refit relative residual={refit_resid:.3e}")
    print("support:", [names[i] for i in support])
    print("passed_support:", passed)
    print("refit coef of true term:", coef_back)
    print(f"lambdas with only the true term: {len(stable)} / {len(rows)}")
    print("--- coefficients (adaptive L1 | refit) ---")
    for i, name in enumerate(names):
        mark = "*" if i in set(support) else " "
        print(f"{mark} {xi_l1[i]:+.6e}  {xi_refit[i]:+.6e}  {name}")


if __name__ == "__main__":
    main()
