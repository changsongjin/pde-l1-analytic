# 最大库，换成精确 L1 路径和迭代重加权。maxlib_first.py 不改。
# 库按 maxlib_spec.md：正问题 43 列、反问题 43 列，一列不删。
# 第 0 轮权重全是 1。之后 w = 1/(|β|+ε)，β 是上一轮 λ = 1e-3 λ_max 的精确 L1 解。
# 支撑在最终权重的路径上统计：40 个 λ，|归一化系数| > 1e-5 的次数过半。
# 通过只认支撑。

import json
from pathlib import Path

import numpy as np

import inverse_burgers
import maxlib_first as M
import noise_inverse_biglib as B
from lasso_exact import kkt_violation, lars_lasso_path, lasso_at

RATIO = 1e-3
EPS = 1e-8
SUPPORT_TOL = 1e-5
N_LAMBDA = 40
MAX_ITER = 20


def irl1(a, b):
    p = a.shape[1]
    w = np.ones(p)
    history = []
    beta = np.zeros(p)
    for it in range(MAX_ITER):
        knots = lars_lasso_path(a, b, w, lam_min_ratio=RATIO * 0.1)
        beta = lasso_at(knots, knots[0][0] * RATIO)
        active = [int(i) for i in np.where(np.abs(beta) > SUPPORT_TOL)[0]]
        violation = kkt_violation(a, b, beta, knots[0][0] * RATIO, w)
        history.append({"iter": it, "support": active, "kkt": violation})
        new_w = 1.0 / (np.abs(beta) + EPS)
        if it and active == history[-2]["support"]:
            w = new_w
            break
        w = new_w
    knots = lars_lasso_path(a, b, w, lam_min_ratio=1e-6)
    lam_max = knots[0][0]
    counts = np.zeros(p, dtype=int)
    best = None
    for lam in lam_max * np.geomspace(1.0, 1e-6, N_LAMBDA):
        beta_lam = lasso_at(knots, float(lam))
        resid = float(np.linalg.norm(a @ beta_lam - b) / np.linalg.norm(b))
        counts[np.abs(beta_lam) > SUPPORT_TOL] += 1
        if best is None or resid < best[0]:
            best = (resid, beta_lam.copy(), float(lam))
    support = np.where(counts > N_LAMBDA / 2.0)[0]
    return {
        "iterations": history,
        "n_iterations": len(history),
        "frequency_counts": counts,
        "support": support,
        "best_relative_residual": best[0],
        "best_beta": best[1],
        "best_lambda": best[2],
        "final_kkt": history[-1]["kkt"],
    }


def normalize(a):
    col = np.maximum(np.linalg.norm(a, axis=0), 1e-15)
    return a / col, col


def forward_case():
    names, _ = M.forward_names()
    # forward_names 仍按旧删除名单返回第二项。这里 43 列全部保留。
    x = np.linspace(-8.0, 12.0, 81)
    t = np.linspace(0.0, 1.0, 21)
    theta = x[1:-1, None] * 0.0  # 占位，下面用网格函数
    xx, tt = np.meshgrid(x[1:-1], t[1:], indexing="xy")
    grid = xx.ravel() - 4.0 * tt.ravel()
    s2 = 1.0 / np.cosh(grid) ** 2
    th = np.tanh(grid)
    u = 2.0 * s2
    u_theta = -4.0 * s2 * th
    # u = 2 sech²θ。差分核对过：∂θ³u = 16 sech²θ tanhθ (2 sech²θ − tanh²θ)。
    u_theta3 = 16.0 * s2 * th * (2.0 * s2 - th**2)
    h = 1e-3
    fd = (
        2.0 / np.cosh(grid + 2 * h) ** 2
        - 2.0 * 2.0 / np.cosh(grid + h) ** 2
        + 2.0 * 2.0 / np.cosh(grid - h) ** 2
        - 2.0 / np.cosh(grid - 2 * h) ** 2
    ) / (2 * h**3)
    u_x = u_theta
    u_t = -4.0 * u_theta
    u_xxx = u_theta3
    exact = u_t + 6.0 * u * u_x + u_xxx
    exact_rel = float(np.linalg.norm(exact) / np.linalg.norm(u_t))
    fd_err = float(np.max(np.abs(fd - u_xxx)))
    if exact_rel > 1e-8:
        raise RuntimeError(f"真解解析残差过大: {exact_rel}")

    blocks = []
    for xq, tq, target in (
        (np.concatenate([np.full_like(t, -8.0), np.full_like(t, 12.0)]),
         np.concatenate([t, t]),
         None),
        (x, np.zeros_like(x), None),
    ):
        thq = xq - 4.0 * tq
        mat = np.column_stack(M.forward_columns(thq))
        tgt = 2.0 / np.cosh(thq) ** 2
        scale = np.sqrt(mat.shape[0])
        blocks.append((mat / scale, tgt / scale))
    a = np.vstack([b[0] for b in blocks])
    rhs = np.concatenate([b[1] for b in blocks])
    an, col = normalize(a)
    picked = irl1(an, rhs)
    support = [names[i] for i in picked["support"]]
    acceptable = [{"sech^2 θ"}, {"1", "tanh^2 θ"}]
    return {
        "name": "正问题 KdV 孤子，43 列，迭代重加权精确 L1",
        "library_size": len(names),
        "exact_pde_relative_residual": exact_rel,
        "analytic_third_derivative_check": fd_err,
        "iterations": picked["n_iterations"],
        "final_kkt": picked["final_kkt"],
        "best_relative_residual": picked["best_relative_residual"],
        "frequency": {names[i]: int(picked["frequency_counts"][i]) for i in range(len(names)) if picked["frequency_counts"][i]},
        "support_names": support,
        "passed_support": set(support) in acceptable,
        "note": "43 列都不删。选择只用初值和两端边界，对系数线性。PDE 残差用解析导数核对真解。",
    }


def inverse_columns():
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
    return cols


def make_inverse_data(case, cols, power):
    """和 noise_inverse_biglib.make_data 相同，只把测试函数的次数换成 power。"""
    import numpy.polynomial.polynomial as Poly

    x = np.linspace(*case["x"])
    t = np.linspace(*case["t"])
    dx, dt = x[1] - x[0], t[1] - t[0]
    tt, xx = np.meshgrid(t, x, indexing="ij")
    hx, ht = case["half"]

    def bump(n_half, h, max_d):
        s = np.linspace(-1.0, 1.0, 2 * n_half + 1)
        a = n_half * h
        poly = Poly.polypow([1.0, 0.0, -1.0], power)
        out, q = [], poly
        for d in range(max_d + 1):
            out.append(Poly.polyval(s, q) / a**d)
            q = Poly.polyder(q)
        return out

    px, pt = bump(hx, dx, 4), bump(ht, dt, 2)
    blocks_a, blocks_b = [], []
    for f in case["traj"]:
        u = f(xx.ravel(), tt.ravel()).reshape(xx.shape)

        def integ(field, kx, kt, px=px, pt=pt):
            from scipy.signal import correlate

            ker = np.outer(pt[kt], px[kx])
            return (correlate(field, ker, mode="valid", method="fft") * dx * dt)[::ht, ::hx].ravel()

        a = []
        for k, fn in cols:
            if fn == "1":
                field = np.ones_like(u)
            elif fn == "x u":
                field = xx * u
            elif fn == "x u^2":
                field = xx * u**2
            else:
                field = B.FUN[fn](u)
            a.append((-1) ** k * integ(field, k, 0))
        b = -integ(u, 0, 1)
        a = np.column_stack(a)
        s = np.linalg.norm(b)
        blocks_a.append(a / s)
        blocks_b.append(b / s)
    return np.vstack(blocks_a), np.concatenate(blocks_b)


def inverse_case():
    cols = inverse_columns()
    names = [B.name(k, f) for k, f in cols]
    case = {
        "pde": "u_t = -u u_x + 0.1 u_xx",
        "traj": [
            lambda x, t: 0.5 * (1.0 - np.tanh(2.5 * (x - 0.3 - 0.5 * t))),
            lambda x, t: 1.0 * (1.0 - np.tanh(5.0 * (x + 0.4 - t))),
        ],
        "x": (-1.0, 2.0, 641),
        "t": (0.0, 1.0, 161),
        "half": (24, 16),
        "target": "u_t",
        "true": {"(u^2)_x": -0.5, "u_xx": 0.1},
    }
    # 次数 6 的核在 641×161 上残差是 7.2e-6。次数 10 降到 3.6e-9，次数 14 降到 7.0e-11。
    test_power = 14
    theta, target = make_inverse_data(case, cols, test_power)
    xi = np.zeros(len(names))
    for key, coef in case["true"].items():
        xi[names.index(key)] = coef
    exact_rel = float(np.linalg.norm(theta @ xi - target) / np.linalg.norm(target))
    out = {
        "name": "反问题 Burgers 两条轨迹，43 列，迭代重加权精确 L1",
        "library_size": len(names),
        "grid": {"x": case["x"], "t": case["t"], "half": case["half"], "test_power": test_power},
        "n_rows": int(theta.shape[0]),
        "exact_equation_relative_residual": exact_rel,
    }
    if exact_rel > 1e-8:
        out["stopped"] = True
        out["passed_support"] = False
        out["reason"] = "加密网格后真方程残差仍大于 1e-8，不报支撑"
        return out
    an, _ = normalize(theta)
    picked = irl1(an, target)
    support = [names[i] for i in picked["support"]]
    out.update(
        {
            "stopped": False,
            "iterations": picked["n_iterations"],
            "final_kkt": picked["final_kkt"],
            "best_relative_residual": picked["best_relative_residual"],
            "frequency": {names[i]: int(picked["frequency_counts"][i]) for i in range(len(names)) if picked["frequency_counts"][i]},
            "support_names": support,
            "passed_support": support == ["(u^2)_x", "u_xx"],
        }
    )
    return out


def main():
    report = {"forward": forward_case(), "inverse": inverse_case()}
    path = Path(__file__).with_name("maxlib_irl1_result.json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for key in ("forward", "inverse"):
        case = report[key]
        print("==", case["name"])
        print("columns", case["library_size"])
        print("exact", f"{case.get('exact_pde_relative_residual', case.get('exact_equation_relative_residual')):.3e}")
        print("stopped", case.get("stopped", False), "passed", case.get("passed_support"))
        print("support", case.get("support_names"))
        print("frequency", case.get("frequency"))
        if case.get("reason"):
            print(case["reason"])


if __name__ == "__main__":
    main()
