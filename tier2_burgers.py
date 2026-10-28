# 第二档：粘性 Burgers，无内部解采样，自适应 L1。
# 残差对系数是二次的。每一步把残差线性化，子问题仍是加权 L1。
# 不是序贯阈值，也不是普通 Lasso。
# 默认值写在 DEFAULTS，结果里原样打印。

import json
from pathlib import Path

import numpy as np

DEFAULTS = {
    "pde": "u_t + u u_x = nu u_xx",
    "nu": 0.1,
    "wave_speed": 0.5,
    "front_x0": 0.3,
    "domain_x": [0.0, 1.0],
    "domain_t": [0.0, 0.8],
    "exact": "u = c (1 - tanh(c/(2 nu) * (x - x0 - c t)))",
    "n_x_pde": 41,
    "n_t": 21,
    "n_x_ic": 81,
    "block_scale": "每个残差块除以 sqrt(行数)，PDE / 边界 / 初值三块总权重相同",
    "column_normalize": True,
    "ridge_alpha": 1e-2,
    "adaptive_gamma": 1.0,
    "adaptive_eps": 1e-8,
    "lambda_path": "40 个，从 lambda_max 几何降到 lambda_max * 1e-6",
    "fit_tol": 1e-6,
    "support_tol": 1e-5,
    "gn_max_iter": 25,
    "gn_tol": 1e-10,
    "cd_max_iter": 5000,
    "cd_tol": 1e-10,
    "solver": "自适应 L1：先岭回归（Gauss-Newton）得权重，再加权 L1。二次残差在每步线性化。不是序贯阈值。",
    "no_solution_samples": True,
    "ic_bc": "初值和两端边界用这条行波的精确值，不是内部采样",
}

NU = DEFAULTS["nu"]
C = DEFAULTS["wave_speed"]
X0 = DEFAULTS["front_x0"]


def wave_parts(x, t, speed):
    """返回 φ, φ_t, φ_x, φ_xx。φ = speed * (1 - tanh(k ξ))，k = speed / (2 ν)。"""
    k = speed / (2.0 * NU)
    xi = np.asarray(x, dtype=float) - X0 - speed * np.asarray(t, dtype=float)
    th = np.tanh(k * xi)
    s2 = 1.0 / np.cosh(k * xi) ** 2
    phi = speed * (1.0 - th)
    phi_x = -speed * k * s2
    phi_t = -(speed) * phi_x
    phi_xx = 2.0 * speed * (k**2) * s2 * th
    return phi, phi_t, phi_x, phi_xx


def sech2_parts(x, t):
    k = C / (2.0 * NU)
    xi = np.asarray(x, dtype=float) - X0 - C * np.asarray(t, dtype=float)
    th = np.tanh(k * xi)
    s2 = 1.0 / np.cosh(k * xi) ** 2
    phi = s2
    phi_x = -2.0 * k * s2 * th
    phi_t = -C * phi_x
    phi_xx = 2.0 * (k**2) * s2 * (2.0 * th**2 - s2)
    return phi, phi_t, phi_x, phi_xx


def gaussian_parts(x, t, width):
    xi = np.asarray(x, dtype=float) - X0 - C * np.asarray(t, dtype=float)
    phi = np.exp(-((xi / width) ** 2))
    phi_x = phi * (-2.0 * xi / width**2)
    phi_t = -C * phi_x
    phi_xx = phi * ((-2.0 * xi / width**2) ** 2 - 2.0 / width**2)
    return phi, phi_t, phi_x, phi_xx


def library_spec():
    return [
        "c(1-tanh(k(x-x0-c t)))  c=0.5  [真项]",
        "c(1-tanh(k(x-x0-c t)))  c=0.25",
        "c(1-tanh(k(x-x0-c t)))  c=1",
        "sech^2(k(x-x0-c t))  c=0.5",
        "1",
        "x",
        "t",
        "x t",
        "sin(pi x)",
        "exp(-((x-x0-c t)/0.3)^2)",
    ]


def eval_library(x, t):
    cols = []

    def add(parts):
        cols.append(parts)

    add(wave_parts(x, t, C))
    add(wave_parts(x, t, 0.25))
    add(wave_parts(x, t, 1.0))
    add(sech2_parts(x, t))
    n = np.asarray(x, dtype=float).size
    z = np.zeros(n)
    ones = np.ones(n)
    xx = np.asarray(x, dtype=float).ravel()
    tt = np.asarray(t, dtype=float).ravel()
    add((ones, z, z, z))
    add((xx.copy(), z, ones.copy(), z))
    add((tt.copy(), ones.copy(), z, z))
    add((xx * tt, xx.copy(), tt.copy(), z))
    sx = np.sin(np.pi * xx)
    add((sx, z, np.pi * np.cos(np.pi * xx), -(np.pi**2) * sx))
    add(gaussian_parts(x, t, 0.3))
    phi = np.column_stack([c[0] for c in cols])
    dt = np.column_stack([c[1] for c in cols])
    dx = np.column_stack([c[2] for c in cols])
    dxx = np.column_stack([c[3] for c in cols])
    return phi, dt, dx, dxx


def grids():
    x_all = np.linspace(0.0, 1.0, DEFAULTS["n_x_pde"])
    t_all = np.linspace(0.0, DEFAULTS["domain_t"][1], DEFAULTS["n_t"])
    x_int = x_all[1:-1]
    t_pde = t_all[1:]
    xx, tt = np.meshgrid(x_int, t_pde, indexing="xy")
    pde = (xx.ravel(), tt.ravel())
    t_bc = np.concatenate([t_all, t_all])
    x_bc = np.concatenate([np.zeros_like(t_all), np.ones_like(t_all)])
    bc = (x_bc, t_bc)
    x_ic = np.linspace(0.0, 1.0, DEFAULTS["n_x_ic"])
    ic = (x_ic, np.zeros_like(x_ic))
    return pde, bc, ic


def pack():
    pde, bc, ic = grids()
    return {
        "pde": eval_library(*pde),
        "bc": eval_library(*bc),
        "ic": eval_library(*ic),
        "u_bc": wave_parts(*bc, C)[0],
        "u_ic": wave_parts(*ic, C)[0],
        "counts": {
            "pde": int(pde[0].size),
            "bc": int(bc[0].size),
            "ic": int(ic[0].size),
        },
    }


def column_scales(data):
    """用线性块（φ_t - ν φ_xx、边界、初值）的列范数归一化。"""
    phi, dt, dx, dxx = data["pde"]
    a_pde = dt - NU * dxx
    blocks = [a_pde, data["bc"][0], data["ic"][0]]
    scales = []
    for block in blocks:
        scales.append(np.linalg.norm(block, axis=0) / np.sqrt(block.shape[0]))
    scale = np.sqrt(np.sum(np.vstack(scales) ** 2, axis=0))
    return np.maximum(scale, 1e-15)


def apply_scale(data, scale):
    out = {}
    for key in ("pde", "bc", "ic"):
        out[key] = tuple(arr / scale for arr in data[key])
    out["u_bc"] = data["u_bc"]
    out["u_ic"] = data["u_ic"]
    out["counts"] = data["counts"]
    out["scale"] = scale
    return out


def residual_and_jacobian(data, xi):
    phi, dt, dx, dxx = data["pde"]
    u = phi @ xi
    u_t = dt @ xi
    u_x = dx @ xi
    u_xx = dxx @ xi
    r_pde = u_t + u * u_x - NU * u_xx
    # ∂/∂ξ_m (u u_x) = φ_m u_x + u φ_m_x
    j_pde = dt + u_x[:, None] * phi + u[:, None] * dx - NU * dxx

    r_bc = data["bc"][0] @ xi - data["u_bc"]
    j_bc = data["bc"][0]
    r_ic = data["ic"][0] @ xi - data["u_ic"]
    j_ic = data["ic"][0]

    def scale(r, j):
        s = np.sqrt(r.size)
        return r / s, j / s

    blocks_r = []
    blocks_j = []
    for r, j in ((r_pde, j_pde), (r_bc, j_bc), (r_ic, j_ic)):
        rr, jj = scale(r, j)
        blocks_r.append(rr)
        blocks_j.append(jj)
    return np.concatenate(blocks_r), np.vstack(blocks_j)


def soft(rho, thr):
    if rho > thr:
        return rho - thr
    if rho < -thr:
        return rho + thr
    return 0.0


def weighted_lasso(a, y, lam, w, beta0=None):
    n, p = a.shape
    beta = np.zeros(p) if beta0 is None else beta0.copy()
    col2 = np.sum(a * a, axis=0)
    r = y - a @ beta
    for _ in range(DEFAULTS["cd_max_iter"]):
        max_delta = 0.0
        for j in range(p):
            if col2[j] < 1e-18:
                continue
            r += a[:, j] * beta[j]
            rho = float(a[:, j] @ r)
            new = soft(rho, lam * w[j]) / col2[j]
            max_delta = max(max_delta, abs(new - beta[j]))
            r -= a[:, j] * new
            beta[j] = new
        if max_delta < DEFAULTS["cd_tol"]:
            break
    return beta


def gauss_newton(data, xi0, update):
    """update(J, y, xi) 返回新的系数。残差上升时把步长减半。"""
    xi = xi0.copy()
    r, _ = residual_and_jacobian(data, xi)
    best = float(np.linalg.norm(r))
    for _ in range(DEFAULTS["gn_max_iter"]):
        r, j = residual_and_jacobian(data, xi)
        y = j @ xi - r
        proposed = update(j, y, xi)
        step = proposed - xi
        accepted = None
        trial_norm = None
        alpha = 1.0
        for _ls in range(12):
            trial = xi + alpha * step
            rt, _ = residual_and_jacobian(data, trial)
            trial_norm = float(np.linalg.norm(rt))
            if trial_norm <= best * (1.0 + 1e-8):
                accepted = trial
                best = trial_norm
                break
            alpha *= 0.5
        if accepted is None:
            break
        delta = float(np.linalg.norm(accepted - xi))
        xi = accepted
        if delta < DEFAULTS["gn_tol"] * (1.0 + np.linalg.norm(xi)):
            break
    return xi


def main():
    names = library_spec()
    raw = pack()
    scale = column_scales(raw)
    data = apply_scale(raw, scale)
    p = len(names)

    # 精确解自检：物理系数 1 对应归一化系数 scale[0]
    xi_true = np.zeros(p)
    xi_true[0] = scale[0]
    r_true, _ = residual_and_jacobian(data, xi_true)
    r0, _ = residual_and_jacobian(data, np.zeros(p))
    base = float(np.linalg.norm(r0))
    exact_rel = float(np.linalg.norm(r_true) / base)
    if exact_rel > 1e-8:
        raise RuntimeError(f"真解残差不够小: {exact_rel}")

    def ridge_update(j, y, xi):
        gram = j.T @ j + DEFAULTS["ridge_alpha"] * np.eye(p)
        return np.linalg.solve(gram, j.T @ y)

    beta_ridge = gauss_newton(data, np.zeros(p), ridge_update)
    w = 1.0 / (np.abs(beta_ridge) + DEFAULTS["adaptive_eps"]) ** DEFAULTS["adaptive_gamma"]

    # λ_max 用零点处的雅可比，保证 λ_max 把系数压成 0
    r_at0, j_at0 = residual_and_jacobian(data, np.zeros(p))
    y0 = -r_at0
    corr = np.abs(j_at0.T @ y0) / w
    lam_max = float(np.max(corr))
    if lam_max <= 0:
        raise RuntimeError("lambda_max 为 0")
    lams = lam_max * np.geomspace(1.0, 1e-6, 40)

    rows = []
    xi = np.zeros(p)
    for lam in lams:
        def l1_update(j, y, current, lam=lam, start=xi):
            return weighted_lasso(j, y, float(lam), w, beta0=start)

        xi = gauss_newton(data, xi, l1_update)
        rr, _ = residual_and_jacobian(data, xi)
        resid = float(np.linalg.norm(rr) / base)
        support = np.where(np.abs(xi) > DEFAULTS["support_tol"])[0]
        rows.append((float(lam), resid, support, xi.copy()))

    good = [r for r in rows if r[1] <= DEFAULTS["fit_tol"]]
    if good:
        good.sort(key=lambda r: (len(r[2]), -r[0]))
        chosen = good[0]
        rule = "相对残差 <= 1e-6 的 λ 里，支撑最短；一样短则取更大的 λ"
    else:
        chosen = min(rows, key=lambda r: (r[1], -r[0]))
        rule = "没有任何 λ 达到 1e-6，退而取相对残差最小的 λ"

    lam, resid, support, xi = chosen
    xi_phys_l1 = xi / scale

    if len(support) == 0:
        xi_refit = np.zeros(p)
        refit_resid = 1.0
    else:
        mask = np.zeros(p, dtype=bool)
        mask[support] = True

        def refit_update(j, y, current):
            beta = current.copy()
            js = j[:, mask]
            coef, _, _, _ = np.linalg.lstsq(js, y, rcond=None)
            beta[mask] = coef
            beta[~mask] = 0.0
            return beta

        xi_refit = gauss_newton(data, xi, refit_update)
        xi_refit[~mask] = 0.0
        rr, _ = residual_and_jacobian(data, xi_refit)
        refit_resid = float(np.linalg.norm(rr) / base)

    xi_phys_refit = xi_refit / scale
    passed = list(support) == [0]
    coef_back = float(xi_phys_refit[0]) if passed else None
    stable = [r[0] for r in rows if list(r[2]) == [0]]

    # 列相关，只作记录，不参与选择
    phi_ic = data["ic"][0]
    corr = np.corrcoef(phi_ic.T)
    off = np.abs(corr - np.eye(p))

    report = {
        "defaults": DEFAULTS,
        "rows": raw["counts"],
        "library": names,
        "exact_solution_relative_residual": exact_rel,
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
        "n_lambda_true_support": len(stable),
        "xi_adaptive_l1_physical": xi_phys_l1.tolist(),
        "xi_refit_physical": xi_phys_refit.tolist(),
        "beta_ridge_normalized": beta_ridge.tolist(),
        "beta_ridge_physical": (beta_ridge / scale).tolist(),
        "max_offdiag_corr_on_ic": float(np.max(off)),
    }
    out = Path(__file__).with_name("tier2_burgers_result.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("exact relative residual", f"{exact_rel:.3e}")
    print("rows", raw["counts"], "p", p)
    print("rule:", rule)
    print(f"lambda_max={lam_max:.6e}  lambda={lam:.6e}")
    print(f"L1 relative residual={resid:.3e}")
    print(f"refit relative residual={refit_resid:.3e}")
    print("support:", [names[i] for i in support])
    print("passed_support:", passed)
    print("refit coef of true term:", coef_back)
    print(f"lambdas with only the true term: {len(stable)} / {len(rows)}")
    print("max off-diag corr", float(np.max(off)))
    print("--- physical coefficients (adaptive L1 | refit | ridge) ---")
    ridge_phys = beta_ridge / scale
    for i, name in enumerate(names):
        mark = "*" if i in set(int(s) for s in support) else " "
        print(f"{mark} {xi_phys_l1[i]:+.6e}  {xi_phys_refit[i]:+.6e}  {ridge_phys[i]:+.6e}  {name}")


if __name__ == "__main__":
    main()
