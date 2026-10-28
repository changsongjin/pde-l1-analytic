# 反问题：粘性 Burgers。已知解的网格值，未知方程的项。
# 真方程 u_t = -u u_x + nu u_xx，nu=0.1。
# 一条行波还精确满足 u_t + c u_x = 0，比真方程更稀疏。
# 所以主实验叠两条不同波速的精确解。单条只作对照，写进结果。
# 导数用解析式，不是数值差分。求解是自适应 L1，不是序贯阈值。

import json
from pathlib import Path

import numpy as np

NU = 0.1
DEFAULTS = {
    "pde": "u_t = -u u_x + nu u_xx",
    "nu": NU,
    "speeds": [0.5, 1.0],
    "front_x0": 0.0,
    "domain_x": [-1.0, 2.0],
    "domain_t": [0.0, 1.0],
    "n_x": 121,
    "n_t": 41,
    "exact": "u = c (1 - tanh(c/(2 nu) (x - x0 - c t)))",
    "why_two_waves": "单条行波精确满足 u_t + c u_x = 0，只有一项，短于真方程。两条不同波速不再有共同的平流系数。",
    "derivatives": "解析式，不用数值差分",
    "column_normalize": True,
    "ridge_alpha": 1e-2,
    "adaptive_gamma": 1.0,
    "adaptive_eps": 1e-8,
    "lambda_path": "50 个，从 lambda_max 几何降到 lambda_max * 1e-6",
    "fit_tol": 1e-8,
    "support_tol": 1e-5,
    "stability": "支撑改为入选频率过半。分母是全部 50 个 λ，不是残差达标的那些。阈值不放宽。",
    "cd_max_iter": 8000,
    "cd_tol": 1e-12,
    "solver": "自适应 L1：岭回归得权重，再加权 L1。系统对系数是线性的。不是序贯阈值。",
    "plain_l1": "权重全为 1 的普通 L1 只作对照，不参与通过与否",
}


def wave(x, t, speed):
    k = speed / (2.0 * NU)
    xi = np.asarray(x, dtype=float) - DEFAULTS["front_x0"] - speed * np.asarray(t, dtype=float)
    th = np.tanh(k * xi)
    s2 = 1.0 / np.cosh(k * xi) ** 2
    u = speed * (1.0 - th)
    u_x = -speed * k * s2
    u_t = -speed * u_x
    u_xx = 2.0 * speed * (k**2) * s2 * th
    u_xxx = 2.0 * speed * (k**3) * s2 * (s2 - 2.0 * th**2)
    return u, u_t, u_x, u_xx, u_xxx


def library_spec():
    return [
        "u u_x  [真项, 系数 -1]",
        "u_xx  [真项, 系数 0.1]",
        "1",
        "u",
        "u^2",
        "u^3",
        "u_x",
        "u_xxx",
        "u^2 u_x",
        "u u_xx",
    ]


TRUE_INDEX = [0, 1]
TRUE_COEF = {0: -1.0, 1: NU}


def design(speeds):
    x = np.linspace(DEFAULTS["domain_x"][0], DEFAULTS["domain_x"][1], DEFAULTS["n_x"])
    t = np.linspace(DEFAULTS["domain_t"][0], DEFAULTS["domain_t"][1], DEFAULTS["n_t"])
    xx, tt = np.meshgrid(x, t, indexing="xy")
    blocks_a = []
    blocks_b = []
    for speed in speeds:
        u, u_t, u_x, u_xx, u_xxx = wave(xx.ravel(), tt.ravel(), speed)
        cols = [
            u * u_x,
            u_xx,
            np.ones_like(u),
            u,
            u**2,
            u**3,
            u_x,
            u_xxx,
            (u**2) * u_x,
            u * u_xx,
        ]
        blocks_a.append(np.column_stack(cols))
        blocks_b.append(u_t)
    return np.vstack(blocks_a), np.concatenate(blocks_b)


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


def check_derivatives():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1.0, 2.0, 20)
    t = rng.uniform(0.0, 1.0, 20)
    h = 1e-4
    worst = 0.0
    for speed in (0.5, 1.0):
        u, u_t, u_x, u_xx, u_xxx = wave(x, t, speed)
        ut = (wave(x, t + h, speed)[0] - wave(x, t - h, speed)[0]) / (2 * h)
        ux = (wave(x + h, t, speed)[0] - wave(x - h, t, speed)[0]) / (2 * h)
        uxx = (wave(x + h, t, speed)[0] - 2 * u + wave(x - h, t, speed)[0]) / h**2
        uxxx = (
            wave(x + 2 * h, t, speed)[0]
            - 2 * wave(x + h, t, speed)[0]
            + 2 * wave(x - h, t, speed)[0]
            - wave(x - 2 * h, t, speed)[0]
        ) / (2 * h**3)
        worst = max(
            worst,
            float(np.max(np.abs(ut - u_t))),
            float(np.max(np.abs(ux - u_x))),
            float(np.max(np.abs(uxx - u_xx))),
            float(np.max(np.abs(uxxx - u_xxx))),
        )
    # h=1e-4 的三阶差分截断大约到 1e-4。一阶、二阶远小于此。
    if worst > 1e-3:
        raise RuntimeError(f"解析导数和差分不一致: {worst}")
    return worst


def run_case(speeds):
    names = library_spec()
    a_raw, b = design(speeds)
    p = a_raw.shape[1]
    col_scale = np.maximum(np.linalg.norm(a_raw, axis=0), 1e-15)
    a = a_raw / col_scale
    bnorm = float(np.linalg.norm(b))

    xi_true = np.zeros(p)
    for i, coef in TRUE_COEF.items():
        xi_true[i] = coef
    exact_rel = float(np.linalg.norm(a_raw @ xi_true - b) / bnorm)

    gram = a.T @ a + DEFAULTS["ridge_alpha"] * np.eye(p)
    beta_ridge = np.linalg.solve(gram, a.T @ b)

    def path(w):
        lam_max = float(np.max(np.abs(a.T @ b) / w))
        lams = lam_max * np.geomspace(1.0, 1e-6, 50)
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
            rule = "相对残差 <= 1e-8 的 λ 里，支撑最短；一样短则取更大的 λ"
        else:
            chosen = min(rows, key=lambda r: (r[1], -r[0]))
            rule = "没有任何 λ 达到 1e-8，退而取相对残差最小的 λ"
        return lam_max, rows, chosen, rule

    w_adaptive = 1.0 / (np.abs(beta_ridge) + DEFAULTS["adaptive_eps"]) ** DEFAULTS["adaptive_gamma"]
    lam_max, rows, chosen, rule = path(w_adaptive)
    _, rows_plain, chosen_plain, rule_plain = path(np.ones(p))

    def stable_support(path_rows):
        counts = np.zeros(p, dtype=int)
        for row in path_rows:
            counts[row[2]] += 1
        keep = np.where(counts > len(path_rows) / 2.0)[0]
        return counts, keep

    def refit_on(support, beta):
        xi_l1 = beta / col_scale
        if len(support) == 0:
            xi_refit = np.zeros(p)
            refit_resid = 1.0
        else:
            coef, _, _, _ = np.linalg.lstsq(a_raw[:, support], b, rcond=None)
            xi_refit = np.zeros(p)
            xi_refit[support] = coef
            refit_resid = float(np.linalg.norm(a_raw @ xi_refit - b) / bnorm)
        return xi_l1, xi_refit, refit_resid

    counts, support = stable_support(rows)
    counts_plain, support_plain = stable_support(rows_plain)
    lam, resid, single_support, beta = chosen
    xi_l1, xi_refit, refit_resid = refit_on(support, beta)
    _, xi_plain_refit, _ = refit_on(support_plain, chosen_plain[3])
    stable = [r[0] for r in rows if list(r[2]) == TRUE_INDEX]
    passed = list(support) == TRUE_INDEX
    refit_true = {names[i]: float(xi_refit[i]) for i in TRUE_INDEX}

    return {
        "speeds": list(speeds),
        "n_rows": int(a_raw.shape[0]),
        "exact_equation_relative_residual": exact_rel,
        "selection_rule": "全部 50 个 λ 上，|系数| > 1e-5 的次数过半才进入支撑。单个 λ 的最短支撑只作记录。",
        "single_lambda_rule": rule,
        "single_lambda_support_names": [names[i] for i in single_support],
        "lambda": lam,
        "lambda_max": lam_max,
        "relative_residual_l1": resid,
        "relative_residual_refit": refit_resid,
        "support_indices": [int(i) for i in support],
        "support_names": [names[i] for i in support],
        "passed_support": passed,
        "refit_coefficients_physical": xi_refit.tolist(),
        "l1_coefficients_physical": xi_l1.tolist(),
        "refit_of_true_terms": refit_true,
        "coefficient_back": bool(
            passed
            and abs(xi_refit[0] + 1.0) < 1e-6
            and abs(xi_refit[1] - NU) < 1e-6
        ),
        "n_lambda_true_support": len(stable),
        "selection_frequency": {names[i]: int(counts[i]) for i in range(p)},
        "plain_l1_frequency": {names[i]: int(counts_plain[i]) for i in range(p)},
        "plain_l1_rule": rule_plain,
        "plain_l1_support_names": [names[i] for i in support_plain],
        "plain_l1_passed_support": list(support_plain) == TRUE_INDEX,
        "plain_l1_refit_physical": xi_plain_refit.tolist(),
        "beta_ridge_physical": (beta_ridge / col_scale).tolist(),
        "names": names,
    }


def main():
    deriv_err = check_derivatives()
    two = run_case(DEFAULTS["speeds"])
    one = run_case([DEFAULTS["speeds"][0]])
    report = {
        "defaults": DEFAULTS,
        "library": library_spec(),
        "derivative_check_max_abs": deriv_err,
        "two_waves": two,
        "one_wave_contrast": one,
    }
    out = Path(__file__).with_name("inverse_burgers_result.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    def show(title, case):
        print("==", title)
        print("rows", case["n_rows"], "exact residual", f"{case['exact_equation_relative_residual']:.3e}")
        print("rule:", case["selection_rule"])
        print("single lambda support:", case["single_lambda_support_names"])
        print(f"lambda_max={case['lambda_max']:.6e} lambda={case['lambda']:.6e}")
        print(f"L1 residual={case['relative_residual_l1']:.3e} refit={case['relative_residual_refit']:.3e}")
        print("support:", case["support_names"])
        print("passed:", case["passed_support"], "coef back:", case["coefficient_back"])
        print(f"lambdas with true support: {case['n_lambda_true_support']} / 50")
        print("frequency:", case["selection_frequency"])
        print("plain L1 support:", case["plain_l1_support_names"], "passed:", case["plain_l1_passed_support"])
        print("--- physical (adaptive L1 | refit) ---")
        for i, name in enumerate(case["names"]):
            mark = "*" if i in case["support_indices"] else " "
            print(f"{mark} {case['l1_coefficients_physical'][i]:+.6e}  {case['refit_coefficients_physical'][i]:+.6e}  {name}")

    print("derivative check", f"{deriv_err:.3e}")
    show("two waves", two)
    show("one wave", one)


if __name__ == "__main__":
    main()
