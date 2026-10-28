# Chafee–Infante：u_t = u_xx + u - u^3。
# 稳态真解 tanh(x/sqrt(2))。不给内部采样。自适应 L1 定支撑。
# 论文 PDF 未打开，系数按这条标准形式，不是从原文抄的系数。
# 残差含 u^3，对系数是三次。每一步线性化，子问题仍是加权 L1。

import json
from pathlib import Path

import numpy as np

K = 1.0 / np.sqrt(2.0)

DEFAULTS = {
    "pde": "u_t = u_xx + u - u^3",
    "source": "DLGA-PDE / SGA-PDE 测过 Chafee-Infante。PDF 未打开。系数用标准形式，不是原文核对值。",
    "exact": "u = tanh(x/sqrt(2))，不随时间变化",
    "domain_x": [-6.0, 6.0],
    "domain_t": [0.0, 1.0],
    "n_x_pde": 41,
    "n_t": 11,
    "n_x_ic": 81,
    "block_scale": "每个残差块除以 sqrt(行数)，PDE / 边界 / 初值三块总权重相同",
    "column_normalize": True,
    "ridge_alpha": 1e-2,
    "adaptive_gamma": 1.0,
    "adaptive_eps": 1e-8,
    "lambda_path": "40 个，从 lambda_max 几何降到 lambda_max * 1e-6",
    "fit_tol": 1e-6,
    "support_tol": 1e-5,
    "gn_max_iter": 30,
    "gn_tol": 1e-10,
    "cd_max_iter": 5000,
    "cd_tol": 1e-10,
    "solver": "自适应 L1：先岭回归（Gauss-Newton）得权重，再加权 L1。u^3 在每步线性化。不是序贯阈值。",
    "starts": "先只从零出发。若停在错的盆地，结果里会写明，不偷偷换起点。",
    "no_solution_samples": True,
    "ic_bc": "初值和两端边界用 tanh(x/sqrt(2)) 的精确值，不是内部采样",
}


def tanh_wave(x, t, k, speed):
    theta = k * (np.asarray(x, dtype=float) - speed * np.asarray(t, dtype=float))
    th = np.tanh(theta)
    s2 = 1.0 / np.cosh(theta) ** 2
    phi = th
    phi_x = k * s2
    phi_t = -speed * phi_x
    phi_xx = -2.0 * (k**2) * s2 * th
    return phi, phi_t, phi_xx


def sech2_steady(x, t, k):
    theta = k * np.asarray(x, dtype=float)
    th = np.tanh(theta)
    s2 = 1.0 / np.cosh(theta) ** 2
    phi = s2
    phi_xx = (k**2) * 2.0 * s2 * (2.0 * th**2 - s2)
    z = np.zeros_like(phi)
    return phi, z, phi_xx


def library_spec():
    return [
        "tanh(x/sqrt(2))  [真项]",
        "tanh(x)",
        "tanh(x/2)",
        "tanh((x-t)/sqrt(2))",
        "sech^2(x/sqrt(2))",
        "1",
        "x",
        "sin(pi x/6)",
        "exp(-(x/2)^2)",
    ]


def eval_library(x, t):
    x = np.asarray(x, dtype=float).ravel()
    t = np.asarray(t, dtype=float).ravel()
    n = x.size
    z = np.zeros(n)
    cols = [
        tanh_wave(x, t, K, 0.0),
        tanh_wave(x, t, 1.0, 0.0),
        tanh_wave(x, t, 0.5, 0.0),
        tanh_wave(x, t, K, 1.0),
        sech2_steady(x, t, K),
        (np.ones(n), z, z),
        (x.copy(), z, z),
        (
            np.sin(np.pi * x / 6.0),
            z,
            -((np.pi / 6.0) ** 2) * np.sin(np.pi * x / 6.0),
        ),
        (
            np.exp(-((x / 2.0) ** 2)),
            z,
            np.exp(-((x / 2.0) ** 2)) * ((x / 2.0) ** 2 - 0.5) * 2.0 / 1.0,
        ),
    ]
    # exp(-(x/2)^2): d²/dx² = d/dx [exp * (-x/2)] = exp*((-x/2)^2) + exp*(-1/2)
    # = exp * (x^2/4 - 1/2). 上面写错了，这里改掉。
    g = np.exp(-((x / 2.0) ** 2))
    cols[8] = (g, z, g * ((x**2) / 4.0 - 0.5))
    return tuple(np.column_stack([c[i] for c in cols]) for i in range(3))


def check_derivatives():
    rng = np.random.default_rng(0)
    x = rng.uniform(-5.0, 5.0, 20)
    t = rng.uniform(0.0, 1.0, 20)
    h = 1e-4
    makers = [
        lambda xx, tt: tanh_wave(xx, tt, K, 0.0),
        lambda xx, tt: tanh_wave(xx, tt, K, 1.0),
        lambda xx, tt: sech2_steady(xx, tt, K),
    ]
    worst = 0.0
    for maker in makers:
        phi, phi_t, phi_xx = maker(x, t)
        pt = (maker(x, t + h)[0] - maker(x, t - h)[0]) / (2 * h)
        pxx = (maker(x + h, t)[0] - 2 * maker(x, t)[0] + maker(x - h, t)[0]) / h**2
        worst = max(worst, float(np.max(np.abs(pt - phi_t))), float(np.max(np.abs(pxx - phi_xx))))
    g = np.exp(-((x / 2.0) ** 2))
    gxx = g * ((x**2) / 4.0 - 0.5)
    fd = (np.exp(-(((x + h) / 2.0) ** 2)) - 2 * g + np.exp(-(((x - h) / 2.0) ** 2))) / h**2
    worst = max(worst, float(np.max(np.abs(fd - gxx))))
    if worst > 1e-4:
        raise RuntimeError(f"解析导数和差分不一致: {worst}")
    return worst


def grids():
    x0, x1 = DEFAULTS["domain_x"]
    t0, t1 = DEFAULTS["domain_t"]
    x_all = np.linspace(x0, x1, DEFAULTS["n_x_pde"])
    t_all = np.linspace(t0, t1, DEFAULTS["n_t"])
    xx, tt = np.meshgrid(x_all[1:-1], t_all[1:], indexing="xy")
    pde = (xx.ravel(), tt.ravel())
    t_bc = np.concatenate([t_all, t_all])
    x_bc = np.concatenate([np.full_like(t_all, x0), np.full_like(t_all, x1)])
    x_ic = np.linspace(x0, x1, DEFAULTS["n_x_ic"])
    return pde, (x_bc, t_bc), (x_ic, np.zeros_like(x_ic))


def exact_u(x, t):
    return tanh_wave(x, t, K, 0.0)[0]


def pack():
    pde, bc, ic = grids()
    return {
        "pde": eval_library(*pde),
        "bc": eval_library(*bc),
        "ic": eval_library(*ic),
        "u_bc": exact_u(*bc),
        "u_ic": exact_u(*ic),
        "counts": {"pde": int(pde[0].size), "bc": int(bc[0].size), "ic": int(ic[0].size)},
    }


def column_scales(data):
    phi, dt, dxx = data["pde"]
    blocks = [dt - dxx - phi, data["bc"][0], data["ic"][0]]
    scales = [np.linalg.norm(block, axis=0) / np.sqrt(block.shape[0]) for block in blocks]
    return np.maximum(np.sqrt(np.sum(np.vstack(scales) ** 2, axis=0)), 1e-15)


def apply_scale(data, scale):
    out = {key: tuple(arr / scale for arr in data[key]) for key in ("pde", "bc", "ic")}
    out["u_bc"] = data["u_bc"]
    out["u_ic"] = data["u_ic"]
    out["counts"] = data["counts"]
    out["scale"] = scale
    return out


def residual_and_jacobian(data, xi):
    phi, dt, dxx = data["pde"]
    u = phi @ xi
    r_pde = dt @ xi - dxx @ xi - u + u**3
    j_pde = dt - dxx - phi + 3.0 * (u**2)[:, None] * phi

    def scale(r, j):
        s = np.sqrt(r.size)
        return r / s, j / s

    blocks = [
        scale(r_pde, j_pde),
        scale(data["bc"][0] @ xi - data["u_bc"], data["bc"][0]),
        scale(data["ic"][0] @ xi - data["u_ic"], data["ic"][0]),
    ]
    return np.concatenate([b[0] for b in blocks]), np.vstack([b[1] for b in blocks])


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
            new = soft(float(a[:, j] @ r), lam * w[j]) / col2[j]
            max_delta = max(max_delta, abs(new - beta[j]))
            r -= a[:, j] * new
            beta[j] = new
        if max_delta < DEFAULTS["cd_tol"]:
            break
    return beta


def gauss_newton(data, xi0, update):
    xi = xi0.copy()
    r, _ = residual_and_jacobian(data, xi)
    best = float(np.linalg.norm(r))
    for _ in range(DEFAULTS["gn_max_iter"]):
        r, j = residual_and_jacobian(data, xi)
        proposed = update(j, j @ xi - r, xi)
        step = proposed - xi
        accepted = None
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
    deriv_err = check_derivatives()
    names = library_spec()
    raw = pack()
    scale = column_scales(raw)
    data = apply_scale(raw, scale)
    p = len(names)

    xi_true = np.zeros(p)
    xi_true[0] = scale[0]
    r0, _ = residual_and_jacobian(data, np.zeros(p))
    base = float(np.linalg.norm(r0))
    exact_rel = float(np.linalg.norm(residual_and_jacobian(data, xi_true)[0]) / base)
    if exact_rel > 1e-8:
        raise RuntimeError(f"真解残差不够小: {exact_rel}")

    def ridge_update(j, y, xi):
        gram = j.T @ j + DEFAULTS["ridge_alpha"] * np.eye(p)
        return np.linalg.solve(gram, j.T @ y)

    beta_ridge = gauss_newton(data, np.zeros(p), ridge_update)
    w = 1.0 / (np.abs(beta_ridge) + DEFAULTS["adaptive_eps"]) ** DEFAULTS["adaptive_gamma"]

    r_at0, j_at0 = residual_and_jacobian(data, np.zeros(p))
    lam_max = float(np.max(np.abs(j_at0.T @ (-r_at0)) / w))
    lams = lam_max * np.geomspace(1.0, 1e-6, 40)

    rows = []
    xi = np.zeros(p)
    for lam in lams:
        def l1_update(j, y, current, lam=lam, start=xi):
            return weighted_lasso(j, y, float(lam), w, beta0=start)

        xi = gauss_newton(data, xi, l1_update)
        resid = float(np.linalg.norm(residual_and_jacobian(data, xi)[0]) / base)
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
    if len(support) == 0:
        xi_refit = np.zeros(p)
        refit_resid = 1.0
    else:
        mask = np.zeros(p, dtype=bool)
        mask[support] = True

        def refit_update(j, y, current):
            beta = np.zeros(p)
            coef, _, _, _ = np.linalg.lstsq(j[:, mask], y, rcond=None)
            beta[mask] = coef
            return beta

        xi_refit = gauss_newton(data, xi, refit_update)
        xi_refit[~mask] = 0.0
        refit_resid = float(np.linalg.norm(residual_and_jacobian(data, xi_refit)[0]) / base)

    phys_l1 = xi / scale
    phys_refit = xi_refit / scale
    passed = list(support) == [0]
    coef_back = float(phys_refit[0]) if passed else None
    stable = [r[0] for r in rows if list(r[2]) == [0]]

    report = {
        "defaults": DEFAULTS,
        "rows": raw["counts"],
        "library": names,
        "derivative_check_max_abs": deriv_err,
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
        "coefficient_back_to_1": bool(passed and coef_back is not None and abs(coef_back - 1.0) < 1e-6),
        "n_lambda_true_support": len(stable),
        "xi_adaptive_l1_physical": phys_l1.tolist(),
        "xi_refit_physical": phys_refit.tolist(),
        "beta_ridge_physical": (beta_ridge / scale).tolist(),
        "started_from_zero_only": True,
    }
    out = Path(__file__).with_name("chafee_infante_result.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("derivative check", f"{deriv_err:.3e}")
    print("exact relative residual", f"{exact_rel:.3e}")
    print("rows", raw["counts"], "p", p)
    print("rule:", rule)
    print(f"lambda_max={lam_max:.6e}  lambda={lam:.6e}")
    print(f"L1 relative residual={resid:.3e}")
    print(f"refit relative residual={refit_resid:.3e}")
    print("support:", [names[i] for i in support])
    print("passed_support:", passed)
    print("refit coef:", coef_back)
    print(f"lambdas with only the true term: {len(stable)} / {len(rows)}")
    ridge_phys = beta_ridge / scale
    picked = set(int(s) for s in support)
    print("--- physical (L1 | refit | ridge) ---")
    for i, name in enumerate(names):
        mark = "*" if i in picked else " "
        print(f"{mark} {phys_l1[i]:+.6e}  {phys_refit[i]:+.6e}  {ridge_phys[i]:+.6e}  {name}")


if __name__ == "__main__":
    main()
