# 正问题：通用候选库 + 连续波参数。回答「库里本来就放了答案」的质疑。旧脚本不改。
#
# 与 tier2 / tier3 / chafee_infante / kdv_forward_blind 的区别：
#   1. 库不再含真解那一列。三个方程共用同一个库：行波变量 θ = k (x - x0 - c t) 上的
#      1, tanh, tanh², tanh³, tanh⁴（tanh 展开法的通用基），外加 sech、exp(-θ²)、θ 三列干扰。
#      真解只能由几列拼出来：KdV 2sech² = 2 - 2tanh²；Burgers c(1 - tanh)；Chafee–Infante tanh。
#   2. 波参数 (k, c, x0) 是连续未知数，不预先离散成列。搜索范围按通用量级给：
#      k ∈ [0.1, 10]（对数均匀，k>0 去掉 θ→-θ 的对称），c ∈ [-8, 8]，x0 ∈ 计算域。
#   3. 第一步 A：多起点 Levenberg–Marquardt，未知数 = 全部 8 个系数 + (log k, c, x0)，不加正则。
#      每个起点的系数初值只用初边值（已知数据）做线性最小二乘。取残差最小的 p*。
#      门槛：相对残差 ≤ 1e-8 才认为库里有精确解；否则按规则停、不报支撑。
#   4. 第二步 B：p* 固定，系数上用 kdv_forward_blind.py 的 IRL1 + λ 路径定支撑（精确 LARS，阈值 1e-5 不放宽）。
#   5. 第三步：只在支撑上联合重拟合系数和 (k, c, x0)，报告与真值的误差。重拟合不决定支撑。
#   6. 反例（应当被门槛拦下）：KdV 删掉 tanh² 列；KdV 双孤子（不是单一行波，库表示不了）。
#   不提供解的内部采样：损失只有配点 PDE 残差 + 初边值残差（+ L1）。真解只用于报告，不进入求解。

import json
import platform
from pathlib import Path

import numpy as np
from numpy.polynomial import polynomial as P

import kdv_forward_blind as B

GATE = 1e-8
N_STARTS = 40
SEED = 0
SEED_CHECK = list(range(5))
K_RANGE = (0.1, 10.0)
C_RANGE = (-8.0, 8.0)
LM_MAX_ITER = 300
SWEEP_RATIOS = [1e-2, 1e-3, 1e-4]

NAMES = ["1", "tanh θ", "tanh² θ", "tanh³ θ", "tanh⁴ θ", "sech θ", "exp(-θ²)", "θ"]


# ---------- 库：g(θ) 及对 θ 的一到三阶导 ----------

def _tanh_poly(n):
    """tanh^n 写成 T 的多项式系数，求导算子 D p(T) = p'(T)(1 - T²)。"""
    coef = np.zeros(n + 1)
    coef[n] = 1.0
    out = [coef]
    for _ in range(3):
        d = P.polymul(P.polyder(out[-1]), [1.0, 0.0, -1.0])
        out.append(d)
    return out


TANH_POLYS = [_tanh_poly(n) for n in range(5)]


def basis(theta, cols):
    th = np.tanh(theta)
    s = 1.0 / np.cosh(theta)
    e = np.exp(-theta**2)
    full = []
    for polys in TANH_POLYS:
        full.append([P.polyval(th, q) for q in polys])
    full.append([s, -s * th, s * (2 * th**2 - 1), s * th * (5 - 6 * th**2)])
    full.append([e, -2 * theta * e, (4 * theta**2 - 2) * e, (-8 * theta**3 + 12 * theta) * e])
    one, zero = np.ones_like(theta), np.zeros_like(theta)
    full.append([theta.copy(), one, zero, zero])
    return [np.column_stack([full[j][d] for j in cols]) for d in range(4)]


def check_basis():
    th = np.linspace(-3, 3, 41)
    h = 1e-4
    cols = list(range(len(NAMES)))
    g = basis(th, cols)
    worst = 0.0
    for d in range(3):
        fd = (basis(th + h, cols)[d] - basis(th - h, cols)[d]) / (2 * h)
        worst = max(worst, float(np.max(np.abs(fd - g[d + 1]))))
    if worst > 1e-6:
        raise RuntimeError(f"库导数与差分不一致: {worst}")
    return worst


# ---------- 方程 ----------

def kdv_pde(u, ut, ux, uxx, uxxx):
    r = ut + 6 * u * ux + uxxx
    return r, {"u": 6 * ux, "ut": 1.0, "ux": 6 * u, "uxx": 0.0, "uxxx": 1.0}


NU = 0.1


def burgers_pde(u, ut, ux, uxx, uxxx):
    r = ut + u * ux - NU * uxx
    return r, {"u": ux, "ut": 1.0, "ux": u, "uxx": -NU, "uxxx": 0.0}


def ci_pde(u, ut, ux, uxx, uxxx):
    r = ut - uxx - u + u**3
    return r, {"u": -1.0 + 3 * u**2, "ut": 1.0, "ux": 0.0, "uxx": -1.0, "uxxx": 0.0}


def kdv_soliton(x, t):
    return 2.0 / np.cosh(x - 4 * t) ** 2


def burgers_front(x, t):
    return 0.5 * (1 - np.tanh(2.5 * (x - 0.3 - 0.5 * t)))


def ci_kink(x, t):
    return np.tanh(x / np.sqrt(2.0)) + 0 * t


# KdV 双孤子：u = 2 (ln F)_xx，F = 1 + e^{η1} + e^{η2} + A e^{η1+η2}，η_i = k_i x - k_i³ t + δ_i
K1, K2, D1, D2 = 2.0, 1.0, 12.0, 0.0
A12 = ((K1 - K2) / (K1 + K2)) ** 2


def kdv_two(x, t):
    e1 = np.exp(K1 * x - K1**3 * t + D1)
    e2 = np.exp(K2 * x - K2**3 * t + D2)
    e12 = A12 * e1 * e2
    f = 1 + e1 + e2 + e12
    fx = K1 * e1 + K2 * e2 + (K1 + K2) * e12
    fxx = K1**2 * e1 + K2**2 * e2 + (K1 + K2) ** 2 * e12
    return 2 * (f * fxx - fx**2) / f**2


def check_two_soliton():
    rng = np.random.default_rng(1)
    x = rng.uniform(-10, 15, 30)
    t = rng.uniform(0, 3, 30)
    h = 1e-3
    u = kdv_two(x, t)
    ut = (kdv_two(x, t + h) - kdv_two(x, t - h)) / (2 * h)
    ux = (kdv_two(x + h, t) - kdv_two(x - h, t)) / (2 * h)
    uxxx = (kdv_two(x + 2 * h, t) - 2 * kdv_two(x + h, t) + 2 * kdv_two(x - h, t) - kdv_two(x - 2 * h, t)) / (2 * h**3)
    return float(np.max(np.abs(ut + 6 * u * ux + uxxx)))


CASES = {
    "kdv": {"pde_text": "u_t + 6 u u_x + u_xxx = 0", "pde": kdv_pde, "exact": kdv_soliton,
            "x": (-8.0, 12.0), "t": (0.0, 1.0), "grid": (49, 17, 97),
            "true_p": {"k": 1.0, "c": 4.0, "x0": 0.0}, "true_xi": {0: 2.0, 2: -2.0}, "kind": "positive"},
    "burgers": {"pde_text": "u_t + u u_x = 0.1 u_xx", "pde": burgers_pde, "exact": burgers_front,
                "x": (0.0, 1.0), "t": (0.0, 0.8), "grid": (41, 21, 81),
                "true_p": {"k": 2.5, "c": 0.5, "x0": 0.3}, "true_xi": {0: 0.5, 1: -0.5}, "kind": "positive"},
    "chafee_infante": {"pde_text": "u_t = u_xx + u - u^3", "pde": ci_pde, "exact": ci_kink,
                       "x": (-6.0, 6.0), "t": (0.0, 1.0), "grid": (41, 11, 81),
                       "true_p": {"k": 1 / np.sqrt(2.0), "c": 0.0, "x0": 0.0}, "true_xi": {1: 1.0}, "kind": "positive"},
    "kdv_no_tanh2": {"pde_text": "u_t + 6 u u_x + u_xxx = 0（库删掉 tanh²）", "pde": kdv_pde, "exact": kdv_soliton,
                     "x": (-8.0, 12.0), "t": (0.0, 1.0), "grid": (49, 17, 97), "drop": [2], "kind": "negative"},
    "kdv_two_soliton": {"pde_text": "u_t + 6 u u_x + u_xxx = 0（双孤子初边值）", "pde": kdv_pde, "exact": kdv_two,
                        "x": (-15.0, 20.0), "t": (0.0, 3.0), "grid": (71, 25, 141), "kind": "negative"},
}


# ---------- 模型：u = Σ ξ_j g_j(k (x - x0 - c t)) ----------

class Model:
    def __init__(self, case):
        self.case = case
        self.cols = [j for j in range(len(NAMES)) if j not in case.get("drop", [])]
        self.names = [NAMES[j] for j in self.cols]
        self.m = len(self.cols)
        nx, nt, nic = case["grid"]
        x0, x1 = case["x"]
        t0, t1 = case["t"]
        xa = np.linspace(x0, x1, nx)
        ta = np.linspace(t0, t1, nt)
        xx, tt = np.meshgrid(xa[1:-1], ta[1:], indexing="xy")
        self.pde_pts = (xx.ravel(), tt.ravel())
        self.bc_pts = (np.concatenate([np.full(nt, x0), np.full(nt, x1)]), np.concatenate([ta, ta]))
        xi_ic = np.linspace(x0, x1, nic)
        self.ic_pts = (xi_ic, np.zeros(nic))
        self.u_bc = case["exact"](*self.bc_pts)
        self.u_ic = case["exact"](*self.ic_pts)
        self.counts = {"pde": int(xx.size), "bc": 2 * nt, "ic": nic}

    def _g(self, pts, p, deriv=True):
        k, c, x0 = np.exp(p[0]), p[1], p[2]
        g = basis(k * (pts[0] - x0 - c * pts[1]), self.cols)
        return g, k, c

    def residual(self, xi, p):
        g, k, c = self._g(self.pde_pts, p)
        u, ux, uxx, uxxx = g[0] @ xi, k * (g[1] @ xi), k**2 * (g[2] @ xi), k**3 * (g[3] @ xi)
        ut = -c * ux
        r_pde, dr = self.case["pde"](u, ut, ux, uxx, uxxx)
        # 对 ξ 的雅可比是解析的
        coef = lambda key: np.broadcast_to(dr[key], u.shape)[:, None]
        j_pde = (coef("u") * g[0] + (coef("ux") - c * coef("ut")) * k * g[1]
                 + coef("uxx") * k**2 * g[2] + coef("uxxx") * k**3 * g[3])
        g_bc = self._g(self.bc_pts, p)[0][0]
        g_ic = self._g(self.ic_pts, p)[0][0]
        blocks = [(r_pde, j_pde), (g_bc @ xi - self.u_bc, g_bc), (g_ic @ xi - self.u_ic, g_ic)]
        r = np.concatenate([b[0] / np.sqrt(b[0].size) for b in blocks])
        j = np.vstack([b[1] / np.sqrt(b[0].size) for b in blocks])
        return r, j

    def base(self):
        return float(np.linalg.norm(self.residual(np.zeros(self.m), np.zeros(3))[0]))

    def full_residual(self, z):
        """z = [ξ (m 个), log k, c, x0]。对 p 的雅可比用中心差分。"""
        xi, p = z[: self.m], z[self.m:]
        r, jx = self.residual(xi, p)
        jp = np.zeros((r.size, 3))
        for i in range(3):
            h = 1e-6 * (1 + abs(p[i]))
            e = np.zeros(3)
            e[i] = h
            jp[:, i] = (self.residual(xi, p + e)[0] - self.residual(xi, p - e)[0]) / (2 * h)
        return r, np.hstack([jx, jp])

    def ic_bc_init(self, p):
        """系数初值：只用初边值（已知数据）做线性最小二乘。"""
        g_bc = self._g(self.bc_pts, p)[0][0]
        g_ic = self._g(self.ic_pts, p)[0][0]
        a = np.vstack([g_bc, g_ic])
        b = np.concatenate([self.u_bc, self.u_ic])
        return np.linalg.lstsq(a, b, rcond=None)[0]


def lm(fun, z0, mask=None, max_iter=LM_MAX_ITER):
    """Levenberg–Marquardt，Marquardt 对角缩放。mask 为 False 的分量固定为 0。"""
    z = z0.copy()
    free = np.ones(z.size, dtype=bool) if mask is None else mask
    z[~free] = 0.0
    r, j = fun(z)
    f = float(r @ r)
    f0 = f
    mu = 1e-3
    for _ in range(max_iter):
        jf = j[:, free]
        h = jf.T @ jf
        g = jf.T @ r
        d = np.diag(h).copy() + 1e-300
        accepted = False
        for _try in range(30):
            try:
                step = np.linalg.solve(h + mu * np.diag(d), -g)
            except np.linalg.LinAlgError:
                mu *= 10
                continue
            zn = z.copy()
            zn[free] += step
            rn, jn = fun(zn)
            fn = float(rn @ rn)
            if np.isfinite(fn) and fn < f:
                z, r, j, f, accepted = zn, rn, jn, fn, True
                mu = max(mu / 3, 1e-12)
                break
            mu *= 4
        if not accepted or f < 1e-32 * f0:
            break
        if float(np.linalg.norm(step)) < 1e-14 * (1 + float(np.linalg.norm(z))):
            break
    return z, f


def stage_a(model, seed):
    rng = np.random.default_rng(seed)
    base = model.base()
    fits = []
    for _ in range(N_STARTS):
        p = np.array([rng.uniform(np.log(K_RANGE[0]), np.log(K_RANGE[1])),
                      rng.uniform(*C_RANGE), rng.uniform(*model.case["x"])])
        z0 = np.concatenate([model.ic_bc_init(p), p])
        z, f = lm(model.full_residual, z0)
        fits.append((float(np.sqrt(f)) / base, z))
    fits.sort(key=lambda t: t[0])
    return fits


class FixedP:
    """p 固定后的系数问题，接口与 kdv_forward_blind.Problem 一致（归一化列）。"""

    def __init__(self, model, p):
        self.model, self.pv = model, p
        self.names = model.names
        self.p = model.m
        _, j0 = model.residual(np.zeros(self.p), p)
        self.scale = np.maximum(np.linalg.norm(j0, axis=0), 1e-15)
        self.r0, self.j0 = self.residual(np.zeros(self.p))
        self.base = float(np.linalg.norm(self.r0))

    def residual(self, x):
        r, j = self.model.residual(x / self.scale, self.pv)
        return r, j / self.scale

    def rel(self, x):
        return float(np.linalg.norm(self.residual(x)[0]) / self.base)

    def objective(self, x, lam, w):
        r = self.residual(x)[0]
        return float(0.5 * r @ r + lam * np.sum(w * np.abs(x)))

    def lam_max(self, w):
        return float(np.max(np.abs(self.j0.T @ (-self.r0)) / w))


def stage_b(model, p_star, xi_star, ratio):
    pb = FixedP(model, p_star)
    x_star = xi_star * pb.scale
    w, hist = B.irl1_weights(pb, x_star, ratio)
    counts, support, rows, _ = B.path_support(pb, x_star, w)
    return pb, w, hist, counts, support, rows


def refit(model, z_star, support):
    mask = np.zeros(model.m + 3, dtype=bool)
    mask[support] = True
    mask[model.m:] = True
    z, f = lm(model.full_residual, z_star, mask=mask)
    return z, float(np.sqrt(f)) / model.base()


def p_dict(p):
    return {"k": float(np.exp(p[0])), "c": float(p[1]), "x0": float(p[2])}


def run_case(name, case):
    model = Model(case)
    base = model.base()
    out = {"pde": case["pde_text"], "kind": case["kind"], "library": model.names, "rows": model.counts,
           "domain_x": list(case["x"]), "domain_t": list(case["t"])}
    if "true_p" in case:
        tp = case["true_p"]
        xi_t = np.zeros(model.m)
        for j, v in case["true_xi"].items():
            xi_t[model.cols.index(j)] = v
        p_t = np.array([np.log(tp["k"]), tp["c"], tp["x0"]])
        out["exact_solution_relative_residual"] = float(np.linalg.norm(model.residual(xi_t, p_t)[0]) / base)
        out["true_support"] = [model.names[i] for i in np.where(xi_t != 0)[0]]
        out["true_params"] = tp

    fits = stage_a(model, SEED)
    best_rel, z_star = fits[0]
    out["stage_a"] = {"n_starts": N_STARTS, "seed": SEED, "best_relative_residual": best_rel,
                      "n_below_gate": int(sum(f[0] <= GATE for f in fits)),
                      "relative_residuals_sorted": [f[0] for f in fits],
                      "best_params": p_dict(z_star[model.m:]),
                      "best_coefficients": dict(zip(model.names, map(float, z_star[: model.m])))}
    print(f"== {name}  [{case['kind']}]")
    if "exact_solution_relative_residual" in out:
        print(f"  exact residual {out['exact_solution_relative_residual']:.2e}")
    print(f"  stage A best {best_rel:.2e}  hits {out['stage_a']['n_below_gate']}/{N_STARTS}  p* {out['stage_a']['best_params']}")

    if best_rel > GATE:
        out["verdict"] = "门槛未过：库里没有精确解，按规则停、不报支撑"
        out["passed"] = case["kind"] == "negative"
        print("  gate NOT reached ->", "PASSED (正确拦下)" if out["passed"] else "FAILED")
        return out

    p_star, xi_star = z_star[model.m:], z_star[: model.m]
    pb, w, hist, counts, support, rows = stage_b(model, p_star, xi_star, B.IRL1_RATIO)
    z_r, rel_r = refit(model, z_star, support)
    out["stage_b"] = {"irl1_ratio": B.IRL1_RATIO, "irl1_history": hist,
                      "frequency": {model.names[i]: int(counts[i]) for i in range(model.m)},
                      "support_names": [model.names[i] for i in support], "path": rows}
    out["refit"] = {"coefficients": {model.names[i]: float(z_r[i]) for i in support},
                    "params": p_dict(z_r[model.m:]), "relative_residual": rel_r}
    print("  IRL1:", [h["support_names"] for h in hist])
    print("  frequency:", out["stage_b"]["frequency"])
    print("  support:", out["stage_b"]["support_names"])
    print("  refit:", out["refit"]["coefficients"], out["refit"]["params"], f"res {rel_r:.2e}")

    if case["kind"] == "negative":
        out["passed"] = False
        out["verdict"] = "反例却过了门槛：需要检查"
        return out

    passed = out["stage_b"]["support_names"] == out["true_support"]
    out["passed"] = bool(passed)
    tp = case["true_p"]
    out["param_abs_error"] = {key: abs(out["refit"]["params"][key] - tp[key]) for key in tp}
    print("  param abs error:", {k: f"{v:.1e}" for k, v in out["param_abs_error"].items()}, "PASSED" if passed else "FAILED")

    seeds = []
    for sd in SEED_CHECK:
        f2 = stage_a(model, sd)
        rel2, z2 = f2[0]
        ok = rel2 <= GATE
        if ok:
            *_, s2, _ = stage_b(model, z2[model.m:], z2[: model.m], B.IRL1_RATIO)
            names2 = [model.names[i] for i in s2]
            ok = names2 == out["true_support"]
        else:
            names2 = None
        seeds.append({"seed": sd, "best_relative_residual": rel2, "n_below_gate": int(sum(f[0] <= GATE for f in f2)),
                      "support_names": names2, "passed": bool(ok)})
    out["seed_check"] = {"n_passed": sum(s["passed"] for s in seeds), "n": len(seeds), "runs": seeds}
    sweep = []
    for r in SWEEP_RATIOS:
        *_, s3, _ = stage_b(model, p_star, xi_star, r)
        names3 = [model.names[i] for i in s3]
        sweep.append({"ratio": r, "support_names": names3, "passed": names3 == out["true_support"]})
    out["irl1_ratio_check"] = sweep
    print(f"  seeds {out['seed_check']['n_passed']}/{len(seeds)}  ratios", [(s["ratio"], s["passed"]) for s in sweep])
    return out


def main():
    report = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "library": NAMES,
        "wave_variable": "θ = k (x - x0 - c t)，(k, c, x0) 连续未知",
        "search_ranges": {"k": list(K_RANGE), "c": list(C_RANGE), "x0": "计算域"},
        "gate": GATE,
        "basis_derivative_check": check_basis(),
        "two_soliton_fd_check": check_two_soliton(),
        "rule": "A：多起点 LM（无正则）定 p*，相对残差 ≤ 1e-8 才继续；B：p* 固定，IRL1 + 40 个 λ 过半定支撑，阈值 1e-5；"
                "重拟合只在支撑上联合拟合系数和 p，不决定支撑。",
        "cases": {},
    }
    print("python", report["python"], "numpy", report["numpy"])
    print(f"basis check {report['basis_derivative_check']:.1e}  two-soliton FD check {report['two_soliton_fd_check']:.1e}")
    for name, case in CASES.items():
        report["cases"][name] = run_case(name, case)
    report["summary"] = {k: v["passed"] for k, v in report["cases"].items()}
    print("summary:", report["summary"])
    out = Path(__file__).with_name("forward_generic_result.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
