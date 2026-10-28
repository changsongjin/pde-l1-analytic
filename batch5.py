# 正问题补 1 条，反问题补 4 条。已有脚本不改。
# 正：波动方程，不给内部采样，真项预先放进库。
# 反：热方程、波动、KdV 双孤子、平流扩散。解的网格值已知，选方程的项。
# 单模态会有更短的等价方程，热、波动、平流扩散都用两个模态。
# 求解都是自适应 L1。支撑取 λ 路径上入选次数过半的列。

import json
from pathlib import Path

import numpy as np

PI = np.pi
SUPPORT_TOL = 1e-5
FIT_TOL = 1e-8


def soft(rho, thr):
    if rho > thr:
        return rho - thr
    if rho < -thr:
        return rho + thr
    return 0.0


def weighted_lasso(a, b, lam, w, beta0=None, max_iter=8000, tol=1e-12):
    n, p = a.shape
    beta = np.zeros(p) if beta0 is None else beta0.copy()
    col2 = np.sum(a * a, axis=0)
    r = b - a @ beta
    for _ in range(max_iter):
        max_delta = 0.0
        for j in range(p):
            if col2[j] < 1e-18:
                continue
            r += a[:, j] * beta[j]
            new = soft(float(a[:, j] @ r), lam * w[j]) / col2[j]
            max_delta = max(max_delta, abs(new - beta[j]))
            r -= a[:, j] * new
            beta[j] = new
        if max_delta < tol:
            break
    return beta


def linear_select(a_raw, b, true_index):
    """系统对系数线性。列归一化，岭回归出权重，再加权 L1。"""
    p = a_raw.shape[1]
    col = np.maximum(np.linalg.norm(a_raw, axis=0), 1e-15)
    a = a_raw / col
    bnorm = float(np.linalg.norm(b))
    if bnorm == 0:
        raise RuntimeError("右端项全是 0")
    ridge = np.linalg.solve(a.T @ a + 1e-2 * np.eye(p), a.T @ b)
    w = 1.0 / (np.abs(ridge) + 1e-8)
    lam_max = float(np.max(np.abs(a.T @ b) / w))
    lams = lam_max * np.geomspace(1.0, 1e-6, 40)
    beta = np.zeros(p)
    counts = np.zeros(p, dtype=int)
    last = None
    best = None
    for lam in lams:
        beta = weighted_lasso(a, b, float(lam), w, beta0=beta)
        resid = float(np.linalg.norm(a @ beta - b) / bnorm)
        active = np.where(np.abs(beta) > SUPPORT_TOL)[0]
        counts[active] += 1
        last = (float(lam), resid, beta.copy(), active)
        if best is None or resid < best[1]:
            best = last
    support = np.where(counts > len(lams) / 2.0)[0]
    coef, _, _, _ = np.linalg.lstsq(a_raw[:, support], b, rcond=None) if len(support) else (np.zeros(0),)
    xi = np.zeros(p)
    if len(support):
        xi[support] = coef
    refit_resid = float(np.linalg.norm(a_raw @ xi - b) / bnorm)
    return {
        "lambda_max": lam_max,
        "frequency": counts.tolist(),
        "support_indices": [int(i) for i in support],
        "passed_support": list(support) == list(true_index),
        "refit_physical": xi.tolist(),
        "refit_relative_residual": refit_resid,
        "best_single_relative_residual": best[1],
        "n_lambda": len(lams),
        "ridge_physical": (ridge / col).tolist(),
    }


def scale_blocks(blocks):
    out_a, out_b = [], []
    for a, b in blocks:
        s = np.sqrt(a.shape[0])
        out_a.append(a / s)
        out_b.append(b / s)
    return np.vstack(out_a), np.concatenate(out_b)


def gn_l1(residual_fn, p, true_xi, starts):
    """非线性残差。每个 λ 从给定起点取残差更低者。支撑仍是入选频率过半。"""

    def gauss_newton(xi0, update):
        xi = xi0.copy()
        r, _ = residual_fn(xi)
        best = float(np.linalg.norm(r))
        for _ in range(25):
            r, j = residual_fn(xi)
            proposed = update(j, j @ xi - r, xi)
            step = proposed - xi
            accepted = None
            alpha = 1.0
            for _ls in range(10):
                trial = xi + alpha * step
                rt, _ = residual_fn(trial)
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
            if delta < 1e-10 * (1.0 + np.linalg.norm(xi)):
                break
        return xi

    def ridge_update(j, y, xi):
        return np.linalg.solve(j.T @ j + 1e-2 * np.eye(p), j.T @ y)

    ridge = None
    ridge_resid = np.inf
    for start in starts:
        trial = gauss_newton(start, ridge_update)
        resid = float(np.linalg.norm(residual_fn(trial)[0]))
        if resid < ridge_resid:
            ridge_resid = resid
            ridge = trial
    w = 1.0 / (np.abs(ridge) + 1e-8)

    r0, j0 = residual_fn(np.zeros(p))
    base = float(np.linalg.norm(r0))
    lam_max = float(np.max(np.abs(j0.T @ (-r0)) / w))
    lams = lam_max * np.geomspace(1.0, 1e-6, 30)
    counts = np.zeros(p, dtype=int)
    best = None
    chain = np.zeros(p)
    for lam in lams:
        def l1_update(j, y, current, lam=lam):
            return weighted_lasso(j, y, float(lam), w, beta0=current, max_iter=4000)

        chosen = None
        chosen_resid = np.inf
        for start in (chain, *starts[1:]):
            trial = gauss_newton(start, l1_update)
            resid = float(np.linalg.norm(residual_fn(trial)[0]) / base)
            if resid < chosen_resid:
                chosen_resid = resid
                chosen = trial
        chain = chosen
        active = np.where(np.abs(chosen) > SUPPORT_TOL)[0]
        counts[active] += 1
        if best is None or chosen_resid < best[0]:
            best = (chosen_resid, chosen.copy())
    support = np.where(counts > len(lams) / 2.0)[0]
    mask = np.zeros(p, dtype=bool)
    mask[support] = True

    def refit_update(j, y, current):
        beta = np.zeros(p)
        if np.any(mask):
            coef, _, _, _ = np.linalg.lstsq(j[:, mask], y, rcond=None)
            beta[mask] = coef
        return beta

    refit = gauss_newton(best[1], refit_update)
    refit[~mask] = 0.0
    refit_resid = float(np.linalg.norm(residual_fn(refit)[0]) / base)
    exact_resid = float(np.linalg.norm(residual_fn(true_xi)[0]) / base)
    return {
        "lambda_max": lam_max,
        "frequency": counts.tolist(),
        "support_indices": [int(i) for i in support],
        "refit": refit.tolist(),
        "refit_relative_residual": refit_resid,
        "exact_relative_residual": exact_resid,
        "best_single_relative_residual": best[0],
        "n_lambda": len(lams),
        "ridge": ridge.tolist(),
        "started_from": "零，以及真项系数的 0.9 倍" if len(starts) > 1 else "零",
    }


def forward_wave():
    """u_tt = c^2 u_xx，c=1。真解 sin(pi x) cos(pi t)。"""
    names = [
        "sin(pi x) cos(pi t)  [真项]",
        "sin(2 pi x) cos(2 pi t)",
        "sin(pi x) cos(2 pi t)",
        "sin(2 pi x) cos(pi t)",
        "1",
        "x",
        "t",
        "cos(pi x) cos(pi t)",
    ]
    x = np.linspace(0.0, 1.0, 31)
    t = np.linspace(0.0, 1.0, 21)
    xx, tt = np.meshgrid(x[1:-1], t[1:], indexing="xy")
    xp, tp = xx.ravel(), tt.ravel()
    xb = np.concatenate([np.zeros_like(t), np.ones_like(t)])
    tb = np.concatenate([t, t])
    xi = np.linspace(0.0, 1.0, 41)

    def cols(x, t):
        x = np.asarray(x, dtype=float).ravel()
        t = np.asarray(t, dtype=float).ravel()
        out = []
        for kx, kt in ((1, 1), (2, 2), (1, 2), (2, 1)):
            phi = np.sin(kx * PI * x) * np.cos(kt * PI * t)
            tt_d = -(kt * PI) ** 2 * phi
            xx_d = -(kx * PI) ** 2 * phi
            out.append((phi, tt_d, xx_d))
        n = x.size
        z = np.zeros(n)
        out.append((np.ones(n), z, z))
        out.append((x.copy(), z, z))
        out.append((t.copy(), z, z))
        phi = np.cos(PI * x) * np.cos(PI * t)
        out.append((phi, -(PI**2) * phi, -(PI**2) * phi))
        return out

    def matrix(x, t, kind):
        columns = cols(x, t)
        if kind == "pde":
            return np.column_stack([c[1] - c[2] for c in columns])
        return np.column_stack([c[0] for c in columns])

    a_pde = matrix(xp, tp, "pde")
    a_bc = matrix(xb, tb, "u")
    a_ic = matrix(xi, np.zeros_like(xi), "u")
    b_ic = np.sin(PI * xi)
    a, b = scale_blocks(
        [
            (a_pde, np.zeros(a_pde.shape[0])),
            (a_bc, np.zeros(a_bc.shape[0])),
            (a_ic, b_ic),
        ]
    )
    # 初速度 u_t(x,0)=0 对这几列自动满足，不另写。
    result = linear_select(a, b, [0])
    result.update(
        {
            "name": "正问题 波动方程",
            "pde": "u_tt = u_xx",
            "exact": "sin(pi x) cos(pi t)",
            "library": names,
            "rows": {"pde": int(a_pde.shape[0]), "bc": int(a_bc.shape[0]), "ic": int(a_ic.shape[0])},
            "support_names": [names[i] for i in result["support_indices"]],
            "refit_of_true_term": result["refit_physical"][0],
            "note": "不给内部采样。真项预先放进库。不同波数的列都满足齐次边界，初值把它们分开。",
        }
    )
    return result


def inverse_linear(name, pde, exact, names, true_index, true_coef, samples):
    blocks_a, blocks_b = [], []
    for sample in samples:
        blocks_a.append(sample["theta"])
        blocks_b.append(sample["target"])
    a = np.vstack(blocks_a)
    b = np.concatenate(blocks_b)
    xi = np.zeros(a.shape[1])
    for i, coef in true_coef.items():
        xi[i] = coef
    exact_rel = float(np.linalg.norm(a @ xi - b) / np.linalg.norm(b))
    result = linear_select(a, b, true_index)
    result.update(
        {
            "name": name,
            "pde": pde,
            "exact": exact,
            "library": names,
            "n_rows": int(a.shape[0]),
            "exact_equation_relative_residual": exact_rel,
            "support_names": [names[i] for i in result["support_indices"]],
            "refit_of_true_terms": {names[i]: result["refit_physical"][i] for i in true_index},
            "true_coefficients": {names[i]: true_coef[i] for i in true_index},
        }
    )
    return result


def heat_samples():
    names = ["u_xx  [真项, 系数 1]", "1", "u", "u_x", "u^2", "u u_x", "u_xxx"]
    x = np.linspace(0.0, 1.0, 81)
    t = np.linspace(0.0, 0.4, 21)
    xx, tt = np.meshgrid(x, t, indexing="xy")
    xx, tt = xx.ravel(), tt.ravel()
    samples = []
    for k in (1, 2):
        decay = np.exp(-((k * PI) ** 2) * tt)
        u = decay * np.sin(k * PI * xx)
        u_t = -((k * PI) ** 2) * u
        u_x = decay * (k * PI) * np.cos(k * PI * xx)
        u_xx = -((k * PI) ** 2) * u
        u_xxx = -((k * PI) ** 3) * decay * np.cos(k * PI * xx)
        theta = np.column_stack([u_xx, np.ones_like(u), u, u_x, u**2, u * u_x, u_xxx])
        samples.append({"theta": theta, "target": u_t})
    return inverse_linear(
        "反问题 热方程",
        "u_t = u_xx",
        "k=1 和 k=2 两个模态",
        names,
        [0],
        {0: 1.0},
        samples,
    )


def wave_samples():
    names = ["u_xx  [真项, 系数 1]", "1", "u", "u_t", "u_x", "u^2", "u u_x"]
    x = np.linspace(0.0, 1.0, 81)
    t = np.linspace(0.0, 1.0, 21)
    xx, tt = np.meshgrid(x, t, indexing="xy")
    xx, tt = xx.ravel(), tt.ravel()
    samples = []
    for k in (1, 2):
        u = np.sin(k * PI * xx) * np.cos(k * PI * tt)
        u_t = -(k * PI) * np.sin(k * PI * xx) * np.sin(k * PI * tt)
        u_tt = -((k * PI) ** 2) * u
        u_x = (k * PI) * np.cos(k * PI * xx) * np.cos(k * PI * tt)
        u_xx = -((k * PI) ** 2) * u
        theta = np.column_stack([u_xx, np.ones_like(u), u, u_t, u_x, u**2, u * u_x])
        samples.append({"theta": theta, "target": u_tt})
    return inverse_linear(
        "反问题 波动方程",
        "u_tt = u_xx",
        "k=1 和 k=2 两个模态",
        names,
        [0],
        {0: 1.0},
        samples,
    )


def advection_samples():
    names = ["u_x  [真项, 系数 -0.4]", "u_xx  [真项, 系数 0.05]", "1", "u", "u^2", "u u_x", "u_xxx"]
    x = np.linspace(-2.0, 4.0, 121)
    t = np.linspace(0.0, 1.0, 21)
    xx, tt = np.meshgrid(x, t, indexing="xy")
    xx, tt = xx.ravel(), tt.ravel()
    samples = []
    # 两个不同宽度、同一个速度。速度不同时，共同的平流系数不存在。
    speed = 0.4
    for width in (0.5, 0.9):
        xi = xx - speed * tt
        u = np.exp(-((xi / width) ** 2))
        u_x = u * (-2.0 * xi / width**2)
        u_t = -speed * u_x
        u_xx = u * ((-2.0 * xi / width**2) ** 2 - 2.0 / width**2)
        u_xxx = u_x * ((-2.0 * xi / width**2) ** 2 - 2.0 / width**2) + u * (8.0 * xi / width**4)
        theta = np.column_stack([u_x, u_xx, np.ones_like(u), u, u**2, u * u_x, u_xxx])
        samples.append({"theta": theta, "target": u_t})
    result = inverse_linear(
        "反问题 平流扩散",
        "u_t = -0.4 u_x + 0.05 u_xx",
        "两个高斯，宽度 0.5 和 0.9，同一个波速 0.4，同一个扩散系数 0.05",
        names,
        [0, 1],
        {0: -0.4, 1: 0.05},
        samples,
    )
    result["note"] = "两个宽度不同，速度和扩散系数相同。单条高斯的平流系数随宽度变，两条放在一起才只有 -0.4 和 0.05。"
    return result


def kdv_two_soliton():
    names = ["u u_x  [真项, 系数 -6]", "u_xxx  [真项, 系数 -1]", "1", "u", "u_x", "u_xx", "u^2", "u^2 u_x"]
    a1, a2 = 1.0, 2.0
    x01, x02 = -2.0, -8.0
    alpha = ((a1 - a2) / (a1 + a2)) ** 2

    def u_of(x, t):
        th1 = a1 * (x - x01) - a1**3 * t
        th2 = a2 * (x - x02) - a2**3 * t
        e1, e2 = np.exp(th1), np.exp(th2)
        tau = 1.0 + e1 + e2 + alpha * e1 * e2
        tx = a1 * e1 + a2 * e2 + alpha * (a1 + a2) * e1 * e2
        txx = a1**2 * e1 + a2**2 * e2 + alpha * (a1 + a2) ** 2 * e1 * e2
        return 2.0 * (txx * tau - tx**2) / tau**2

    x = np.linspace(-10.0, 12.0, 90)
    t = np.linspace(0.0, 1.2, 16)
    xx, tt = np.meshgrid(x, t, indexing="xy")
    xx, tt = xx.ravel(), tt.ravel()
    h = 1e-3
    u = u_of(xx, tt)
    u_t = (u_of(xx, tt + h) - u_of(xx, tt - h)) / (2 * h)
    u_x = (u_of(xx + h, tt) - u_of(xx - h, tt)) / (2 * h)
    u_xx = (u_of(xx + h, tt) - 2 * u + u_of(xx - h, tt)) / h**2
    u_xxx = (
        u_of(xx + 2 * h, tt)
        - 2 * u_of(xx + h, tt)
        + 2 * u_of(xx - h, tt)
        - u_of(xx - 2 * h, tt)
    ) / (2 * h**3)
    theta = np.column_stack([u * u_x, u_xxx, np.ones_like(u), u, u_x, u_xx, u**2, (u**2) * u_x])
    result = inverse_linear(
        "反问题 KdV 双孤子",
        "u_t = -6 u u_x - u_xxx",
        "振幅参数 1 和 2 的双孤子",
        names,
        [0, 1],
        {0: -6.0, 1: -1.0},
        [{"theta": theta, "target": u_t}],
    )
    result["note"] = "导数用步长 1e-3 的差分。真方程残差停在差分精度，大约 1e-4，不是解析零。"
    result["derivative"] = "数值差分，h=1e-3"
    return result


def main():
    cases = [
        forward_wave(),
        heat_samples(),
        wave_samples(),
        advection_samples(),
        kdv_two_soliton(),
    ]
    out = Path(__file__).with_name("batch5_result.json")
    out.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    for case in cases:
        print("==", case["name"])
        print(case["pde"], "|", case["exact"])
        if "exact_equation_relative_residual" in case:
            print("exact residual", f"{case['exact_equation_relative_residual']:.3e}")
        print("support:", case["support_names"])
        print("passed:", case["passed_support"])
        print("refit residual", f"{case['refit_relative_residual']:.3e}")
        if "refit_of_true_term" in case:
            print("refit coef", case["refit_of_true_term"])
        else:
            print("refit coef", case["refit_of_true_terms"])
        print("frequency", case["frequency"])


if __name__ == "__main__":
    main()
