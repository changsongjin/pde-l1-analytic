# 只重做两条反问题：粘性 Burgers 两条波、KdV 双孤子。
# inverse_burgers.py 和 redo_two.py 不改，数据和导数从那边调用。
# λ 路径仍是 40 个，从 lambda_max 几何降到 1e-6 倍。阈值 1e-5 只用来计数。
# 支撑：相对残差 <= 1e-8 的 λ 里，取 L1 目标最小的那个；
# 归一化系数绝对值 < 1e-3 的列丢掉。

import json
from pathlib import Path

import numpy as np

import inverse_burgers
import redo_two

RESID_GATE = 1e-8
DROP_TOL = 1e-3
COUNT_TOL = 1e-5
N_LAMBDA = 40


def select(a_raw, b, names):
    p = a_raw.shape[1]
    col = np.maximum(np.linalg.norm(a_raw, axis=0), 1e-15)
    a = a_raw / col
    bnorm = float(np.linalg.norm(b))
    ridge = np.linalg.solve(a.T @ a + 1e-2 * np.eye(p), a.T @ b)
    w = 1.0 / (np.abs(ridge) + 1e-8)
    lam_max = float(np.max(np.abs(a.T @ b) / w))
    lams = lam_max * np.geomspace(1.0, 1e-6, N_LAMBDA)
    beta = np.zeros(p)
    counts = np.zeros(p, dtype=int)
    rows = []
    for lam in lams:
        beta = inverse_burgers.weighted_lasso(a, b, float(lam), w, beta0=beta)
        resid_vec = a @ beta - b
        resid = float(np.linalg.norm(resid_vec) / bnorm)
        counts[np.abs(beta) > COUNT_TOL] += 1
        # L1 目标是加权 Lasso 的目标：残差平方加 λ 乘加权 L1。
        objective = float(np.dot(resid_vec, resid_vec) + lam * np.sum(w * np.abs(beta)))
        rows.append(
            {
                "lambda": float(lam),
                "relative_residual": resid,
                "objective": objective,
                "beta": beta.copy(),
            }
        )
    eligible = [row for row in rows if row["relative_residual"] <= RESID_GATE]
    out = {
        "library": names,
        "n_rows": int(a_raw.shape[0]),
        "lambda_max": lam_max,
        "n_lambda": N_LAMBDA,
        "n_lambda_residual_leq_1e-8": len(eligible),
        "best_relative_residual_on_path": min(row["relative_residual"] for row in rows),
        "frequency_at_1e-5": {names[i]: int(counts[i]) for i in range(p)},
        "rule": "相对残差 <= 1e-8 的 λ 里取 L1 目标最小；归一化 |系数| < 1e-3 的列丢掉。1e-5 只计数。",
    }
    if not eligible:
        out["stopped"] = True
        out["passed_support"] = False
        out["reason"] = "没有任何 λ 的相对残差 <= 1e-8，不选支撑"
        return out
    chosen = min(eligible, key=lambda row: (row["objective"], -row["lambda"]))
    beta = chosen["beta"]
    counted = np.where(np.abs(beta) > COUNT_TOL)[0]
    dropped = np.where((np.abs(beta) > COUNT_TOL) & (np.abs(beta) < DROP_TOL))[0]
    support = np.where(np.abs(beta) >= DROP_TOL)[0]
    xi = np.zeros(p)
    if len(support):
        coef, _, _, _ = np.linalg.lstsq(a_raw[:, support], b, rcond=None)
        xi[support] = coef
    out.update(
        {
            "stopped": False,
            "lambda": chosen["lambda"],
            "relative_residual_l1": chosen["relative_residual"],
            "l1_objective": chosen["objective"],
            "normalized_coefficients": beta.tolist(),
            "support_indices": [int(i) for i in support],
            "support_names": [names[i] for i in support],
            "dropped_by_1e-3": [
                {"name": names[i], "normalized": float(beta[i])} for i in dropped
            ],
            "counted_at_1e-5": [names[i] for i in counted],
            "refit_physical": xi.tolist(),
            "refit_relative_residual": float(np.linalg.norm(a_raw @ xi - b) / bnorm),
        }
    )
    return out


def burgers(speeds):
    names = inverse_burgers.library_spec()
    a_raw, b = inverse_burgers.design(speeds)
    xi = np.zeros(a_raw.shape[1])
    xi[0] = -1.0
    xi[1] = inverse_burgers.NU
    exact = float(np.linalg.norm(a_raw @ xi - b) / np.linalg.norm(b))
    out = select(a_raw, b, names)
    out["name"] = "反问题 粘性 Burgers"
    out["speeds"] = list(speeds)
    out["exact_equation_relative_residual"] = exact
    out["true_support"] = [names[0], names[1]]
    if not out["stopped"]:
        out["passed_support"] = out["support_indices"] == [0, 1]
        out["refit_of_true_terms"] = {
            names[0]: out["refit_physical"][0],
            names[1]: out["refit_physical"][1],
        }
    return out


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
    u, u_t, u_x, u_xx, u_xxx = redo_two.two_soliton_fields(xx.ravel(), tt.ravel())
    residual = u_t + 6.0 * u * u_x + u_xxx
    exact = float(np.linalg.norm(residual) / np.linalg.norm(u_t))
    theta = np.column_stack(
        [u * u_x, u_xxx, np.ones_like(u), u, u_x, u_xx, u**2, (u**2) * u_x]
    )
    out = select(theta, u_t, names)
    out["name"] = "反问题 KdV 双孤子"
    out["exact_equation_relative_residual"] = exact
    out["true_support"] = [names[0], names[1]]
    out["derivative"] = "沿用 redo_two.py 的解析导数"
    if exact > 1e-8:
        out["stopped"] = True
        out["passed_support"] = False
        out["reason"] = "真方程残差大于 1e-8，不报支撑"
        return out
    if not out["stopped"]:
        out["passed_support"] = out["support_indices"] == [0, 1]
        out["refit_of_true_terms"] = {
            names[0]: out["refit_physical"][0],
            names[1]: out["refit_physical"][1],
        }
    return out


def main():
    cases = [burgers([0.5, 1.0]), burgers([0.5]), kdv()]
    path = Path(__file__).with_name("optimize_ux_result.json")
    path.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    for case in cases:
        label = case["name"]
        if "speeds" in case:
            label += " " + str(case["speeds"])
        print("==", label)
        print("exact", f"{case['exact_equation_relative_residual']:.3e}")
        print("best L1 residual", f"{case['best_relative_residual_on_path']:.3e}")
        print("eligible", case["n_lambda_residual_leq_1e-8"])
        print("stopped", case["stopped"], "passed", case["passed_support"])
        if case["stopped"]:
            print(case["reason"])
        else:
            print("support", case["support_names"])
            print("dropped", case["dropped_by_1e-3"])
            print("refit", case.get("refit_of_true_terms"))
        print("frequency", case["frequency_at_1e-5"])


if __name__ == "__main__":
    main()
