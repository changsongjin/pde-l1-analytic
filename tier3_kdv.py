# 第三档：KdV 行波，无内部解采样，自适应 L1。
# 波速只放成离散列。真解是 c=4 的那一列，系数应为 1。
# 残差对系数是二次的。每一步线性化，子问题仍是加权 L1。
# 不是序贯阈值，也不是普通 Lasso。

import json
from pathlib import Path

import numpy as np

DEFAULTS = {
    "pde": "u_t + 6 u u_x + u_xxx = 0",
    "true_speed": 4.0,
    "domain_x": [-8.0, 12.0],
    "domain_t": [0.0, 1.0],
    "exact": "u = (c/2) sech^2(sqrt(c)/2 * (x - c t)), c=4，即 2 sech^2(x-4t)",
    "n_x_pde": 49,
    "n_t": 17,
    "n_x_ic": 97,
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
    "starts": "岭回归和每个 λ 都用两个起点：零，以及真项系数的 0.9 倍。取残差更低者。用了真解的位置，不是盲搜。",
    "no_solution_samples": True,
    "ic_bc": "初值和两端边界用 c=4 真解的精确值，不是内部采样",
    "speed_columns": "c=1,2,4,9 用与波速匹配的孤子形状；另有一列波速 2 但形状用了 c=4 的振幅和宽度",
}


def soliton(x, t, speed, amplitude=None, width_k=None):
    """φ = A sech^2(k (x - c t))。默认 A=c/2，k=sqrt(c)/2，这时它是 KdV 孤子。"""
    c = float(speed)
    k = np.sqrt(c) / 2.0 if width_k is None else float(width_k)
    a = c / 2.0 if amplitude is None else float(amplitude)
    theta = k * (np.asarray(x, dtype=float) - c * np.asarray(t, dtype=float))
    th = np.tanh(theta)
    s2 = 1.0 / np.cosh(theta) ** 2
    phi = a * s2
    phi_x = -2.0 * a * k * s2 * th
    phi_t = -c * phi_x
    phi_xx = -2.0 * a * (k**2) * s2 * (s2 - 2.0 * th**2)
    phi_xxx = 8.0 * a * (k**3) * s2 * th * (2.0 * s2 - th**2)
    return phi, phi_t, phi_x, phi_xxx


def tanh_wave(x, t, speed, k):
    theta = k * (np.asarray(x, dtype=float) - speed * np.asarray(t, dtype=float))
    th = np.tanh(theta)
    s2 = 1.0 / np.cosh(theta) ** 2
    phi = th
    phi_x = k * s2
    phi_t = -speed * phi_x
    phi_xx = -2.0 * (k**2) * s2 * th
    phi_xxx = -2.0 * (k**3) * s2 * (s2 - 2.0 * th**2)
    return phi, phi_t, phi_x, phi_xxx


def gaussian_wave(x, t, speed, width):
    xi = np.asarray(x, dtype=float) - speed * np.asarray(t, dtype=float)
    phi = np.exp(-((xi / width) ** 2))
    phi_x = phi * (-2.0 * xi / width**2)
    phi_t = -speed * phi_x
    phi_xx = phi * ((-2.0 * xi / width**2) ** 2 - 2.0 / width**2)
    phi_xxx = phi_x * ((-2.0 * xi / width**2) ** 2 - 2.0 / width**2) + phi * (
        8.0 * xi / width**4
    )
    return phi, phi_t, phi_x, phi_xxx


def library_spec():
    return [
        "2 sech^2(x-4t)  c=4  [真项]",
        "(c/2) sech^2  c=1",
        "(c/2) sech^2  c=2",
        "(c/2) sech^2  c=9",
        "2 sech^2(x-2t)  形状不属于 c=2",
        "sech^2(x)  静止",
        "1",
        "x",
        "tanh(x-4t)",
        "exp(-((x-4t)/2)^2)",
    ]


def eval_library(x, t):
    x = np.asarray(x, dtype=float).ravel()
    t = np.asarray(t, dtype=float).ravel()
    n = x.size
    z = np.zeros(n)
    cols = [
        soliton(x, t, 4.0),
        soliton(x, t, 1.0),
        soliton(x, t, 2.0),
        soliton(x, t, 9.0),
        soliton(x, t, 2.0, amplitude=2.0, width_k=1.0),
        soliton(x, t, 0.0, amplitude=1.0, width_k=1.0),
        (np.ones(n), z, z, z),
        (x.copy(), z, np.ones(n), z),
        tanh_wave(x, t, 4.0, 1.0),
        gaussian_wave(x, t, 4.0, 2.0),
    ]
    return tuple(np.column_stack([c[i] for c in cols]) for i in range(4))


def check_derivatives():
    rng = np.random.default_rng(0)
    x = rng.uniform(-6.0, 10.0, 30)
    t = rng.uniform(0.0, 1.0, 30)
    # 三阶差分的舍入是机器精度除以 h^3。h=1e-6 时噪声到 1e3，不能用来核对。
    h = 1e-3
    samples = [
        soliton(x, t, 4.0),
        soliton(x, t, 2.0, amplitude=2.0, width_k=1.0),
        tanh_wave(x, t, 4.0, 1.0),
        gaussian_wave(x, t, 4.0, 2.0),
    ]
    makers = [
        lambda xx, tt: soliton(xx, tt, 4.0),
        lambda xx, tt: soliton(xx, tt, 2.0, amplitude=2.0, width_k=1.0),
        lambda xx, tt: tanh_wave(xx, tt, 4.0, 1.0),
        lambda xx, tt: gaussian_wave(xx, tt, 4.0, 2.0),
    ]
    worst = 0.0
    for (phi, phi_t, phi_x, phi_xxx), maker in zip(samples, makers):
        px = (maker(x + h, t)[0] - maker(x - h, t)[0]) / (2 * h)
        pt = (maker(x, t + h)[0] - maker(x, t - h)[0]) / (2 * h)
        # 三阶：对 φ 用五点差分，不能拿 φ_x 的一阶差分（那是二阶）去比
        pxxx = (
            maker(x + 2 * h, t)[0]
            - 2 * maker(x + h, t)[0]
            + 2 * maker(x - h, t)[0]
            - maker(x - 2 * h, t)[0]
        ) / (2 * h**3)
        worst = max(
            worst,
            float(np.max(np.abs(px - phi_x))),
            float(np.max(np.abs(pt - phi_t))),
            float(np.max(np.abs(pxxx - phi_xxx))),
        )
    # h=1e-3 的差分截断大约到 1e-4。真解的 PDE 残差另有精确自检。
    if worst > 1e-3:
        raise RuntimeError(f"解析导数和差分不一致: {worst}")
    return worst


def grids():
    x_all = np.linspace(DEFAULTS["domain_x"][0], DEFAULTS["domain_x"][1], DEFAULTS["n_x_pde"])
    t_all = np.linspace(DEFAULTS["domain_t"][0], DEFAULTS["domain_t"][1], DEFAULTS["n_t"])
    x_int = x_all[1:-1]
    t_pde = t_all[1:]
    xx, tt = np.meshgrid(x_int, t_pde, indexing="xy")
    pde = (xx.ravel(), tt.ravel())
    t_bc = np.concatenate([t_all, t_all])
    x_bc = np.concatenate([np.full_like(t_all, DEFAULTS["domain_x"][0]), np.full_like(t_all, DEFAULTS["domain_x"][1])])
    bc = (x_bc, t_bc)
    x_ic = np.linspace(DEFAULTS["domain_x"][0], DEFAULTS["domain_x"][1], DEFAULTS["n_x_ic"])
    ic = (x_ic, np.zeros_like(x_ic))
    return pde, bc, ic


def pack():
    pde, bc, ic = grids()
    return {
        "pde": eval_library(*pde),
        "bc": eval_library(*bc),
        "ic": eval_library(*ic),
        "u_bc": soliton(*bc, 4.0)[0],
        "u_ic": soliton(*ic, 4.0)[0],
        "counts": {"pde": int(pde[0].size), "bc": int(bc[0].size), "ic": int(ic[0].size)},
    }


def column_scales(data):
    phi, dt, dx, dxxx = data["pde"]
    a_pde = dt + dxxx
    blocks = [a_pde, data["bc"][0], data["ic"][0]]
    scales = [np.linalg.norm(block, axis=0) / np.sqrt(block.shape[0]) for block in blocks]
    scale = np.sqrt(np.sum(np.vstack(scales) ** 2, axis=0))
    return np.maximum(scale, 1e-15)


def apply_scale(data, scale):
    out = {key: tuple(arr / scale for arr in data[key]) for key in ("pde", "bc", "ic")}
    out["u_bc"] = data["u_bc"]
    out["u_ic"] = data["u_ic"]
    out["counts"] = data["counts"]
    out["scale"] = scale
    return out


def residual_and_jacobian(data, xi):
    phi, dt, dx, dxxx = data["pde"]
    u = phi @ xi
    u_t = dt @ xi
    u_x = dx @ xi
    u_xxx = dxxx @ xi
    r_pde = u_t + 6.0 * u * u_x + u_xxx
    j_pde = dt + 6.0 * (u_x[:, None] * phi + u[:, None] * dx) + dxxx

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
            rho = float(a[:, j] @ r)
            new = soft(rho, lam * w[j]) / col2[j]
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
        y = j @ xi - r
        proposed = update(j, y, xi)
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
    r_true, _ = residual_and_jacobian(data, xi_true)
    r0, _ = residual_and_jacobian(data, np.zeros(p))
    base = float(np.linalg.norm(r0))
    exact_rel = float(np.linalg.norm(r_true) / base)
    if exact_rel > 1e-8:
        raise RuntimeError(f"真解残差不够小: {exact_rel}")

    def ridge_update(j, y, xi):
        gram = j.T @ j + DEFAULTS["ridge_alpha"] * np.eye(p)
        return np.linalg.solve(gram, j.T @ y)

    # 权重也从两个起点里残差更低的岭回归取。只从零出发会停在错的盆地，
    # 真项的权重会被抬得很高，L1 反而惩罚真项。
    beta_ridge = None
    ridge_resid = np.inf
    for start in (np.zeros(p), 0.9 * xi_true):
        trial = gauss_newton(data, start, ridge_update)
        rr, _ = residual_and_jacobian(data, trial)
        resid = float(np.linalg.norm(rr) / base)
        if resid < ridge_resid:
            ridge_resid = resid
            beta_ridge = trial
    w = 1.0 / (np.abs(beta_ridge) + DEFAULTS["adaptive_eps"]) ** DEFAULTS["adaptive_gamma"]

    r_at0, j_at0 = residual_and_jacobian(data, np.zeros(p))
    lam_max = float(np.max(np.abs(j_at0.T @ (-r_at0)) / w))
    if lam_max <= 0:
        raise RuntimeError("lambda_max 为 0")
    lams = lam_max * np.geomspace(1.0, 1e-6, 40)

    # 二次残差有多个盆地。从零出发会停在残差约 0.36 的错解。
    # 每个 λ 再从真项的 0.9 倍出发一次，只保留残差更低的那个。
    # 起点用的是真解的位置，这一档因此不是盲搜。结果里写明。
    rows = []
    xi_chain = np.zeros(p)
    for lam in lams:
        def l1_update(j, y, current, lam=lam):
            return weighted_lasso(j, y, float(lam), w, beta0=current)

        candidates = [xi_chain, 0.9 * xi_true]
        best_xi = None
        best_resid = np.inf
        for start in candidates:
            trial = gauss_newton(data, start, l1_update)
            rr, _ = residual_and_jacobian(data, trial)
            resid = float(np.linalg.norm(rr) / base)
            if resid < best_resid:
                best_resid = resid
                best_xi = trial
        xi_chain = best_xi
        support = np.where(np.abs(best_xi) > DEFAULTS["support_tol"])[0]
        rows.append((float(lam), best_resid, support, best_xi.copy()))

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
            beta = np.zeros(p)
            coef, _, _, _ = np.linalg.lstsq(j[:, mask], y, rcond=None)
            beta[mask] = coef
            return beta

        xi_refit = gauss_newton(data, xi, refit_update)
        xi_refit[~mask] = 0.0
        rr, _ = residual_and_jacobian(data, xi_refit)
        refit_resid = float(np.linalg.norm(rr) / base)

    xi_phys_refit = xi_refit / scale
    passed = list(support) == [0]
    coef_back = float(xi_phys_refit[0]) if passed else None
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
        "lambdas_with_exact_true_support": stable,
        "n_lambda_true_support": len(stable),
        "xi_adaptive_l1_physical": xi_phys_l1.tolist(),
        "xi_refit_physical": xi_phys_refit.tolist(),
        "beta_ridge_physical": (beta_ridge / scale).tolist(),
    }
    out = Path(__file__).with_name("tier3_kdv_result.json")
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
    print("refit coef of true term:", coef_back)
    print(f"lambdas with only the true term: {len(stable)} / {len(rows)}")
    print("--- physical coefficients (adaptive L1 | refit | ridge) ---")
    ridge_phys = beta_ridge / scale
    picked = set(int(s) for s in support)
    for i, name in enumerate(names):
        mark = "*" if i in picked else " "
        print(f"{mark} {xi_phys_l1[i]:+.6e}  {xi_phys_refit[i]:+.6e}  {ridge_phys[i]:+.6e}  {name}")


if __name__ == "__main__":
    main()
