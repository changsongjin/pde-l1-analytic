# 正问题：大候选函数库（导师建议“把能找到的函数都放进去”）。旧脚本不改，只复用它们的 LM、IRL1、λ 路径。
#
# 与 forward_generic(_v2).py 相同的部分：
#   行波变量 θ = k (x − x0 − c t)，(k, c, x0) 连续未知；损失 = 配点 PDE 残差 + 边界 + 初值（不用解的内部采样）；
#   第一步 A：全库多起点 LM（40 个起点，最好的再延长迭代 3000 步），相对残差 ≤ 1e-8 才认为库里有精确解，否则拒绝；
#   第二步 B：p* 固定，IRL1 的 λ 路径（精确 LARS，阈值 1e-5 不放宽）给出候选支撑；
#   第三步：只在选中的支撑上联合重拟合系数和 (k, c, x0)。
# 不同的部分：
#   1. 库从 8 列扩到 27 列：双曲（tanh、sech 的各次幂和乘积）、logistic、高斯、三角、有理、多项式、arctan。
#      库里有精确恒等式（如 tanh² = 1 − sech²），同一个解有多种写法。所以：
#   2. 判定标准改为“拼出的解等于真解”（测试网格相对 L2 误差 ≤ 1e-6），并记录用了几项、是否达到已知最简写法。
#   3. 从 λ 路径上的候选支撑里选“残差过门槛的支撑中项数最少的”（同项数取残差小的）。L1 只提供候选，
#      门槛只判断“精确与否”，不另加项。
#   4. 方程从 3 个扩到 11 个，真解全部用 sympy 代回方程核对过（残差到机器精度）。
#   5. 对照：(a) 旧 8 列库；(b) 在同一个 p* 上比较产生候选支撑的方法：普通 L1、IRL1、序贯阈值（STRidge 式）、
#      遗传算法、穷举（≤ 3 项），统计重拟合次数和耗时。
# 库函数的导数：用 T = tanh θ、S = sech θ、H = tanh(θ/2) 作为中间变量求导（dT = 1 − T²，dS = −S T，dH = (1 − H²)/2），
# 大 |θ| 时不会出现 inf/inf。

import itertools
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import sympy as sp

GATE = 1e-8
N_STARTS = 40
POLISH_ITER = 3000
N_POLISH = 5
K_RANGE = (0.1, 10.0)
C_RANGE = (-8.0, 8.0)
SEEDS = [0, 1, 2, 3, 4]
WORKERS = 8
SOLUTION_TOL = 1e-6
GA = {"pop": 30, "gens": 40, "p_cross": 0.9, "elite": 2, "tour": 3}
EXHAUSTIVE_MAX = 3
SCALES = [1.0, 2.0, 0.5]
K_B = 5     # 用前几个起点的 p 走 L1 路径
N_SUP = 8   # 每条路径最多试几个支撑（按项数从少到多）

# ---------- 候选函数库 ----------
th, T, S, H = sp.symbols("th T S H", real=True)
LIB = [
    ("1", sp.Integer(1)),
    ("tanh θ", T), ("tanh² θ", T**2), ("tanh³ θ", T**3), ("tanh⁴ θ", T**4),
    ("sech θ", S), ("sech² θ", S**2), ("sech³ θ", S**3), ("sech⁴ θ", S**4),
    ("tanh θ sech θ", T * S), ("tanh θ sech² θ", T * S**2), ("tanh θ sech³ θ", T * S**3),
    ("1/(1+e^θ)", (1 - H) / 2), ("1/(1+e^θ)²", ((1 - H) / 2) ** 2),
    ("e^(−θ²)", sp.exp(-th**2)), ("θ e^(−θ²)", th * sp.exp(-th**2)),
    ("sin θ", sp.sin(th)), ("cos θ", sp.cos(th)), ("sin² θ", sp.sin(th) ** 2), ("sin θ cos θ", sp.sin(th) * sp.cos(th)),
    ("1/(1+θ²)", 1 / (1 + th**2)), ("θ/(1+θ²)", th / (1 + th**2)), ("1/(1+θ²)²", 1 / (1 + th**2) ** 2),
    ("θ", th), ("θ²", th**2),
    ("arctan θ", sp.atan(th)), ("arctan e^θ", sp.pi / 4 + sp.atan(H)),  # arctan e^θ = π/4 + arctan tanh(θ/2)
]
LIB_NAMES = [n for n, _ in LIB]
# 旧 8 列库（forward_generic.py）：1, tanh..tanh⁴, sech, e^(−θ²), θ
SMALL = [LIB_NAMES.index(n) for n in ["1", "tanh θ", "tanh² θ", "tanh³ θ", "tanh⁴ θ", "sech θ", "e^(−θ²)", "θ"]]


def dtheta(e):
    return sp.diff(e, th) + sp.diff(e, T) * (1 - T**2) + sp.diff(e, S) * (-S * T) + sp.diff(e, H) * (1 - H**2) / 2


DMAX = 5  # 四阶方程对 p 的解析雅可比要用到五阶导
_EXPR = []
for _, e in LIB:
    ds = [e]
    for _ in range(DMAX):
        ds.append(sp.expand(dtheta(ds[-1])))
    _EXPR.append(ds)
_DERIVS = [[sp.lambdify((th, T, S, H), d, "numpy") for d in ds] for ds in _EXPR]
_BATCH = {}  # (列组, 阶) → 一次算出全部列的函数（列多时省掉逐列调用的开销）


def lib_eval(theta, cols, dmax):
    """返回 [G_0, ..., G_dmax]，G_d[:, i] = cols[i] 号函数对 θ 的 d 阶导。"""
    with np.errstate(over="ignore", invalid="ignore"):
        tv, sv, hv = np.tanh(theta), 1.0 / np.cosh(theta), np.tanh(theta / 2)
    out = []
    key = tuple(cols)
    for d in range(dmax + 1):
        g = np.empty((theta.size, len(cols)))
        if len(cols) > 6:
            if (key, d) not in _BATCH:
                _BATCH[(key, d)] = sp.lambdify((th, T, S, H), [_EXPR[j][d] for j in cols], "numpy", cse=True)
            for i, v in enumerate(_BATCH[(key, d)](theta, tv, sv, hv)):
                g[:, i] = v
        else:
            for i, j in enumerate(cols):
                g[:, i] = _DERIVS[j][d](theta, tv, sv, hv)
        out.append(g)
    return out


def check_library():
    """解析导数与差分对比；返回最大误差。"""
    x = np.linspace(-3, 3, 31)
    h = 1e-5
    cols = list(range(len(LIB)))
    g = lib_eval(x, cols, DMAX)
    worst = 0.0
    for d in range(DMAX):
        fd = (lib_eval(x + h, cols, d + 1)[d] - lib_eval(x - h, cols, d + 1)[d]) / (2 * h)
        worst = max(worst, float(np.max(np.abs(fd - g[d + 1]) / (1 + np.abs(g[d + 1])))))
    big = lib_eval(np.array([-800.0, 800.0]), cols, DMAX)
    if worst > 1e-5 or not all(np.all(np.isfinite(b)) for b in big[:2]):
        raise RuntimeError(f"库导数检查失败 {worst}")
    return worst


# ---------- 方程（物理变量写，行波代换自动完成） ----------
u_, ux, uxx, uxxx, uxxxx, ut, utt, uxt, uxxt = sp.symbols("u ux uxx uxxx uxxxx ut utt uxt uxxt")
D = sp.symbols("D0:5")
kk, cc = sp.symbols("k c")
TW = {u_: D[0], ux: kk * D[1], uxx: kk**2 * D[2], uxxx: kk**3 * D[3], uxxxx: kk**4 * D[4],
      ut: -cc * kk * D[1], utt: cc**2 * kk**2 * D[2], uxt: -cc * kk**2 * D[2], uxxt: -cc * kk**3 * D[3]}
x_, t_ = sp.symbols("x t", real=True)
r209 = sp.sqrt(209)
KS_A1, KS_A3, KS_K = -135 * r209 / 361, 165 * r209 / 361, r209 / 38
SG_G = 1 / sp.sqrt(sp.Rational(3, 4))

CASES = {
    "kdv": dict(pde="u_t + 6 u u_x + u_xxx = 0", F=ut + 6 * u_ * ux + uxxx,
                exact=2 / sp.cosh(x_ - 4 * t_) ** 2, x=(-8.0, 12.0), t=(0.0, 1.0), minimal=1, kind="positive"),
    "mkdv": dict(pde="u_t + 6 u² u_x + u_xxx = 0", F=ut + 6 * u_**2 * ux + uxxx,
                 exact=1 / sp.cosh(x_ - t_), x=(-10.0, 10.0), t=(0.0, 2.0), minimal=1, kind="positive"),
    "burgers": dict(pde="u_t + u u_x = 0.1 u_xx", F=ut + u_ * ux - sp.Rational(1, 10) * uxx,
                    exact=(1 - sp.tanh(sp.Rational(5, 2) * (x_ - sp.Rational(3, 10) - t_ / 2))) / 2,
                    x=(0.0, 1.0), t=(0.0, 0.8), minimal=1, kind="positive"),
    "fisher": dict(pde="u_t = u_xx + u (1 − u)", F=ut - uxx - u_ * (1 - u_),
                   exact=1 / (1 + sp.exp((x_ - 5 * t_ / sp.sqrt(6)) / sp.sqrt(6))) ** 2,
                   x=(-15.0, 25.0), t=(0.0, 4.0), minimal=1, kind="positive"),
    "chafee_infante": dict(pde="u_t = u_xx + u − u³", F=ut - uxx - u_ + u_**3,
                           exact=sp.tanh(x_ / sp.sqrt(2)) + 0 * t_, x=(-6.0, 6.0), t=(0.0, 1.0), minimal=1, kind="positive"),
    "bbm": dict(pde="u_t + u_x + u u_x − u_xxt = 0", F=ut + ux + u_ * ux - uxxt,
                exact=3 / sp.cosh(sp.sqrt(2) / 4 * (x_ - 2 * t_)) ** 2, x=(-20.0, 30.0), t=(0.0, 5.0), minimal=1, kind="positive"),
    "boussinesq": dict(pde="u_tt − u_xx − 3 (u²)_xx − u_xxxx = 0", F=utt - uxx - 3 * (2 * ux**2 + 2 * u_ * uxx) - uxxxx,
                       exact=sp.Rational(3, 2) / sp.cosh(sp.sqrt(3) / 2 * (x_ - 2 * t_)) ** 2,
                       x=(-10.0, 14.0), t=(0.0, 2.0), minimal=1, kind="positive"),
    "kdv_burgers": dict(pde="u_t + u u_x − u_xx + 0.1 u_xxx = 0", F=ut + u_ * ux - uxx + sp.Rational(1, 10) * uxxx,
                        exact=sp.Rational(17, 10) - sp.Rational(12, 5) * sp.tanh(x_ - t_ / 2) - sp.Rational(6, 5) * sp.tanh(x_ - t_ / 2) ** 2,
                        x=(-8.0, 10.0), t=(0.0, 4.0), minimal=3, kind="positive"),
    "kuramoto_sivashinsky": dict(pde="u_t + u u_x + u_xx + u_xxxx = 0", F=ut + u_ * ux + uxx + uxxxx,
                                 exact=1 + KS_A1 * sp.tanh(KS_K * (x_ - t_)) + KS_A3 * sp.tanh(KS_K * (x_ - t_)) ** 3,
                                 x=(-20.0, 25.0), t=(0.0, 4.0), minimal=3, kind="positive"),
    "klein_gordon": dict(pde="u_tt − u_xx + u = 0", F=utt - uxx + u_, exact=sp.sin(x_ - sp.sqrt(2) * t_),
                         x=(0.0, float(4 * np.pi)), t=(0.0, 2.0), minimal=1, kind="positive"),
    "sine_gordon": dict(pde="u_tt − u_xx + sin u = 0", F=utt - uxx + sp.sin(u_),
                        exact=4 * sp.atan(sp.exp(SG_G * (x_ - t_ / 2))), x=(-8.0, 8.0), t=(0.0, 3.0), minimal=1, kind="positive"),
    # 反例：门槛应当拒绝
    "kdv_two_soliton": dict(pde="KdV，双孤子初边值", F=ut + 6 * u_ * ux + uxxx, exact="kdv_two",
                            x=(-15.0, 20.0), t=(0.0, 3.0), grid=(71, 25, 141), kind="negative"),
    "burgers_cole_hopf": dict(pde="Burgers，Cole–Hopf 三激波合并", F=ut + u_ * ux - sp.Rational(1, 10) * uxx, exact="cole_hopf",
                              x=(-1.0, 3.0), t=(0.0, 1.0), kind="negative"),
    "sine_gordon_no_arctan": dict(pde="sine-Gordon，库删掉 arctan e^θ", F=utt - uxx + sp.sin(u_),
                                  exact=4 * sp.atan(sp.exp(SG_G * (x_ - t_ / 2))), x=(-8.0, 8.0), t=(0.0, 3.0),
                                  drop=["arctan e^θ"], kind="negative"),
}


def pde_order(F):
    xo = max([{ux: 1, uxx: 2, uxxx: 3, uxxxx: 4, uxt: 2, uxxt: 3}.get(s, 0) for s in F.free_symbols] + [0])
    to = max([{ut: 1, utt: 2, uxt: 1, uxxt: 1}.get(s, 0) for s in F.free_symbols] + [0])
    return xo, to


def exact_funcs(case):
    """返回 u(x,t)、u_t(x,t)、u_x(x,t) 的数值函数。"""
    e = case["exact"]
    if e == "kdv_two":
        import forward_generic as G
        f = G.kdv_two
        h = 1e-5
        return f, lambda x, t: (f(x, t + h) - f(x, t - h)) / (2 * h), lambda x, t: (f(x + h, t) - f(x - h, t)) / (2 * h)
    if e == "cole_hopf":
        import inverse_irl1 as I
        f = lambda x, t: I.cole_hopf(np.asarray(x, float), np.asarray(t, float))[0]
        h = 1e-6
        return f, lambda x, t: (f(x, t + h) - f(x, t - h)) / (2 * h), lambda x, t: (f(x + h, t) - f(x - h, t)) / (2 * h)
    lam = lambda ex: (lambda x, t, _f=sp.lambdify((x_, t_), ex, "numpy"): np.broadcast_to(_f(x, t), np.shape(x)).astype(float))
    return lam(e), lam(sp.diff(e, t_)), lam(sp.diff(e, x_))


# ---------- 模型 ----------

class Model:
    def __init__(self, name, lib_cols):
        case = CASES[name]
        self.name, self.case = name, case
        drop = [LIB_NAMES.index(n) for n in case.get("drop", [])]
        self.cols = [j for j in lib_cols if j not in drop]
        self.names = [LIB_NAMES[j] for j in self.cols]
        self.m = len(self.cols)
        F = case["F"]
        self.xo, self.to = pde_order(F)
        self.dmax = self.xo
        Ftw = sp.expand(F.subs(TW))
        self.F = sp.lambdify((D, kk, cc), Ftw, "numpy")
        self.dF = [sp.lambdify((D, kk, cc), sp.diff(Ftw, D[n]), "numpy") for n in range(self.dmax + 1)]
        self.dFk = sp.lambdify((D, kk, cc), sp.diff(Ftw, kk), "numpy")
        self.dFc = sp.lambdify((D, kk, cc), sp.diff(Ftw, cc), "numpy")
        nx, nt, nic = case.get("grid", (49, 17, 97))
        x0, x1 = case["x"]
        t0, t1 = case["t"]
        xa, ta = np.linspace(x0, x1, nx), np.linspace(t0, t1, nt)
        xx, tt = np.meshgrid(xa[1:-1], ta[1:], indexing="xy")
        self.pde_pts = (xx.ravel(), tt.ravel())
        self.bc_pts = (np.concatenate([np.full(nt, x0), np.full(nt, x1)]), np.concatenate([ta, ta]))
        self.ic_pts = (np.linspace(x0, x1, nic), np.zeros(nic))
        fu, fut, fux = exact_funcs(case)
        self.u_bc, self.u_ic = fu(*self.bc_pts), fu(*self.ic_pts)
        self.ut_ic = fut(*self.ic_pts) if self.to >= 2 else None   # 时间二阶：初值再给 u_t
        self.ux_bc = fux(*self.bc_pts) if self.xo >= 4 else None   # 空间四阶：边界再给 u_x
        xt, tt2 = np.meshgrid(np.linspace(x0, x1, 101), np.linspace(t0, t1, 31), indexing="xy")
        self.test_pts = (xt.ravel(), tt2.ravel())
        self.u_test = fu(*self.test_pts)
        self._base = None

    def theta(self, pts, p):
        k, c, x0 = np.exp(p[0]), p[1], p[2]
        return k * (pts[0] - x0 - c * pts[1]), k, c

    def residual(self, xi, p, sub=None, jac_p=False):
        """sub = 参与的库列位置（默认全部），xi 与 sub 对应。返回 r 与对 xi 的雅可比；
        jac_p=True 时再拼上对 p = (log k, c, x0) 的解析雅可比（θ 对 p 的导数：θ、−k t、−k）。"""
        sub = list(range(self.m)) if sub is None else list(sub)
        cols = [self.cols[i] for i in sub]
        extra = 1 if jac_p else 0
        th_p, k, c = self.theta(self.pde_pts, p)
        g = lib_eval(th_p, cols, self.dmax + extra)
        Dv = [gd @ xi for gd in g[: self.dmax + 1]] + [np.zeros(th_p.size)] * (4 - self.dmax)
        dFn = [np.broadcast_to(self.dF[n](Dv, k, c), th_p.shape) for n in range(self.dmax + 1)]
        r_pde = np.broadcast_to(self.F(Dv, k, c), th_p.shape)
        j_pde = sum(dFn[n][:, None] * g[n] for n in range(self.dmax + 1))
        th_bc = self.theta(self.bc_pts, p)[0]
        th_ic = self.theta(self.ic_pts, p)[0]
        g_bc = lib_eval(th_bc, cols, 1 + extra)
        g_ic = lib_eval(th_ic, cols, 1 + extra)
        blocks = [(r_pde, j_pde), (g_bc[0] @ xi - self.u_bc, g_bc[0]), (g_ic[0] @ xi - self.u_ic, g_ic[0])]
        if jac_p:
            dth = lambda thv, pts: np.column_stack([thv, -k * pts[1], np.full(thv.size, -k)])
            dpde = sum(dFn[n] * (g[n + 1] @ xi) for n in range(self.dmax + 1))[:, None] * dth(th_p, self.pde_pts)
            dpde[:, 0] += k * np.broadcast_to(self.dFk(Dv, k, c), th_p.shape)
            dpde[:, 1] += np.broadcast_to(self.dFc(Dv, k, c), th_p.shape)
            jps = [dpde, (g_bc[1] @ xi)[:, None] * dth(th_bc, self.bc_pts), (g_ic[1] @ xi)[:, None] * dth(th_ic, self.ic_pts)]
        if self.ut_ic is not None:
            a = -c * k * g_ic[1]
            blocks.append((a @ xi - self.ut_ic, a))
            if jac_p:
                q = (-c * k * (g_ic[2] @ xi))[:, None] * dth(th_ic, self.ic_pts)
                q[:, 0] += a @ xi
                q[:, 1] += -k * (g_ic[1] @ xi)
                jps.append(q)
        if self.ux_bc is not None:
            a = k * g_bc[1]
            blocks.append((a @ xi - self.ux_bc, a))
            if jac_p:
                q = (k * (g_bc[2] @ xi))[:, None] * dth(th_bc, self.bc_pts)
                q[:, 0] += a @ xi
                jps.append(q)
        r = np.concatenate([b[0] / np.sqrt(b[0].size) for b in blocks])
        j = np.vstack([b[1] / np.sqrt(b[0].size) for b in blocks])
        if jac_p:
            j = np.hstack([j, np.vstack([q / np.sqrt(b[0].size) for q, b in zip(jps, blocks)])])
        return r, j

    def base(self):
        if self._base is None:
            self._base = float(np.linalg.norm(self.residual(np.zeros(0), np.zeros(3), sub=[])[0]))
        return self._base

    def full_residual(self, z, sub=None):
        sub = list(range(self.m)) if sub is None else sub
        n = len(sub)
        return self.residual(z[:n], z[n:], sub, jac_p=True)

    def check_jac_p(self, z):
        """解析雅可比与中心差分对比（只在自检时用）。"""
        r, j = self.full_residual(z)
        n = self.m
        worst = 0.0
        for i in range(3):
            h = 1e-6 * (1 + abs(z[n + i]))
            e = np.zeros(z.size)
            e[n + i] = h
            fd = (self.residual(z[:n], (z + e)[n:])[0] - self.residual(z[:n], (z - e)[n:])[0]) / (2 * h)
            worst = max(worst, float(np.max(np.abs(fd - j[:, n + i])) / (1 + np.max(np.abs(fd)))))
        return worst

    def ic_bc_init(self, p, sub=None):
        sub = list(range(self.m)) if sub is None else sub
        cols = [self.cols[i] for i in sub]
        a = np.vstack([lib_eval(self.theta(self.bc_pts, p)[0], cols, 0)[0], lib_eval(self.theta(self.ic_pts, p)[0], cols, 0)[0]])
        return np.linalg.lstsq(a, np.concatenate([self.u_bc, self.u_ic]), rcond=None)[0]

    def solution_error(self, xi, p, sub):
        cols = [self.cols[i] for i in sub]
        u = lib_eval(self.theta(self.test_pts, p)[0], cols, 0)[0] @ xi
        return float(np.linalg.norm(u - self.u_test) / np.linalg.norm(self.u_test))


def p_dict(p):
    return {"k": float(np.exp(p[0])), "c": float(p[1]), "x0": float(p[2])}


# ---------- 第一步 A ----------

def stage_a(model, seed):
    import forward_generic as G
    rng = np.random.default_rng(seed)
    base = model.base()
    fits = []
    for _ in range(N_STARTS):
        p = np.array([rng.uniform(np.log(K_RANGE[0]), np.log(K_RANGE[1])), rng.uniform(*C_RANGE), rng.uniform(*model.case["x"])])
        try:
            z, f = G.lm(model.full_residual, np.concatenate([model.ic_bc_init(p), p]))
        except (np.linalg.LinAlgError, ValueError):
            continue
        fits.append((float(np.sqrt(f)) / base, z) if np.isfinite(f) else (np.inf, z))
    fits.sort(key=lambda q: q[0])
    # 库里有精确恒等式，JᵀJ 奇异，LM 在解附近收敛很慢。前 N_POLISH 个起点再 LM 延长迭代，
    # 然后用最小范数最小二乘做 Gauss–Newton（对奇异 J 稳定，近解处二次收敛）。门槛不变。
    for i in range(min(N_POLISH, len(fits))):
        z, f = G.lm(model.full_residual, fits[i][1], max_iter=POLISH_ITER // N_POLISH)
        z, f = gn_polish(model.full_residual, z)
        fits[i] = (float(np.sqrt(f)) / base, z)
    fits.sort(key=lambda q: q[0])
    return fits


def gn_polish(fun, z, iters=100):
    r, j = fun(z)
    f = float(r @ r)
    for _ in range(iters):
        if not (np.all(np.isfinite(j)) and np.all(np.isfinite(r))):
            break
        try:
            step = np.linalg.lstsq(j, -r, rcond=1e-13)[0]
        except np.linalg.LinAlgError:
            break
        a = 1.0
        while a > 1e-6:
            zn = z + a * step
            rn, jn = fun(zn)
            fn = float(rn @ rn)
            if np.isfinite(fn) and fn < f:
                break
            a *= 0.5
        else:
            break
        if f - fn < 1e-14 * f:
            z, r, j, f = zn, rn, jn, fn
            break
        z, r, j, f = zn, rn, jn, fn
    return z, f


# ---------- 固定 p* 后的支撑评估 ----------

class SupportEval:
    """p 固定，只在支撑 S 上用 LM 拟合系数，返回相对残差。带缓存，统计重拟合次数。"""

    def __init__(self, model, p_star):
        self.model, self.p = model, p_star
        self.cache, self.calls = {}, 0

    def fit(self, S):
        import forward_generic as G
        S = tuple(sorted(int(i) for i in S))
        if S in self.cache:
            return self.cache[S]
        self.calls += 1
        if not S:
            out = (1.0, np.zeros(0))
        else:
            fun = lambda z: self.model.residual(z, self.p, S)
            z, f = G.lm(fun, self.model.ic_bc_init(self.p, S), max_iter=200)
            out = (float(np.sqrt(f)) / self.model.base(), z)
        self.cache[S] = out
        return out

    def pick(self, supports):
        """过门槛的支撑中项数最少的；同项数取残差最小。都不过门槛时返回残差最小的并标记。"""
        sc = [(S, self.fit(S)[0]) for S in {tuple(sorted(s)) for s in supports}]
        ok = [q for q in sc if q[1] <= GATE]
        if ok:
            return min(ok, key=lambda q: (len(q[0]), q[1]))[0], True
        return min(sc, key=lambda q: q[1])[0], False


def l1_path_supports(model, p_star, xi_star, weighted):
    """p 固定后，对配点残差在 ξ = 0 处的线性化做精确加权 LARS 路径，收集路径上的支撑。

    不用全库最小二乘解当起点：库里有精确恒等式（tanh² = 1 − sech² 等），全库解不唯一，
    从某一个特解出发的 Gauss–Newton L1 会停在与真解等价但含无关列的写法上，路径里没有真支撑
    （KdV–Burgers、Kuramoto–Sivashinsky 因此被拒，见 _diag_kb_ks.py）。
    线性化 L1 的解只由数据和 λ 决定。权重仍来自迭代重加权：第 0 轮普通 L1，之后 w = 1/(|β|+ε)，
    β 是上一轮在 λ = 1e-3 λ_max 处的解。阈值 1e-5 不放宽，作用在归一化系数上。"""
    import forward_generic as G
    import kdv_forward_blind as B
    from lasso_exact import lars_lasso_path, lasso_at
    pb = G.FixedP(model, p_star)
    a, b = pb.j0, -pb.r0
    w = np.ones(pb.p)
    knots = None
    for _ in range(B.IRL1_MAX_ITER if weighted else 1):
        knots = lars_lasso_path(a, b, w, lam_min_ratio=B.LAMBDA_LO)
        if not weighted:
            break
        beta = lasso_at(knots, knots[0][0] * B.IRL1_RATIO)
        w = 1.0 / (np.abs(beta) + B.IRL1_EPS)
    sups = []
    for lam, beta in knots:
        if lam < knots[0][0] * B.LAMBDA_LO:
            break
        on = np.where(np.abs(beta) * pb.scale > B.SUPPORT_TOL)[0]
        if on.size:
            sups.append(tuple(int(i) for i in on))
    return sups


def stridge_supports(ev, scale, n_tol=25):
    sups = []
    m = ev.model.m
    for tol in np.geomspace(1e-5, 0.5, n_tol):
        Sc = list(range(m))
        for _ in range(10):
            res, z = ev.fit(Sc)
            xn = np.abs(z * scale[Sc])
            keep = [Sc[i] for i in range(len(Sc)) if xn[i] >= tol * xn.max()]
            if keep == Sc or not keep:
                break
            Sc = keep
        sups.append(tuple(Sc))
    return sups


def ga_supports(ev, rng):
    m = ev.model.m
    def fit(ind):
        Sc = tuple(np.where(ind)[0])
        res = ev.fit(Sc)[0]
        return len(Sc) + (0.0 if res <= GATE else 1.0 + np.log10(res / GATE))
    pop = rng.random((GA["pop"], m)) < 0.15
    scores = np.array([fit(i) for i in pop])
    for _ in range(GA["gens"]):
        order = np.argsort(scores)
        new = [pop[i].copy() for i in order[: GA["elite"]]]
        while len(new) < GA["pop"]:
            def pick():
                c = rng.choice(GA["pop"], GA["tour"], replace=False)
                return pop[c[np.argmin(scores[c])]]
            a, b = pick(), pick()
            child = np.where(rng.random(m) < 0.5, a, b) if rng.random() < GA["p_cross"] else a.copy()
            child ^= rng.random(m) < 1.0 / m
            new.append(child)
        pop = np.array(new)
        scores = np.array([fit(i) for i in pop])
    return [tuple(np.where(i)[0]) for i in pop]


def exhaustive_supports(ev):
    m = ev.model.m
    for size in range(1, EXHAUSTIVE_MAX + 1):
        sups = list(itertools.combinations(range(m), size))
        found = [s for s in sups if ev.fit(s)[0] <= GATE]
        if found:
            return found
    return sups


# ---------- 一次运行 ----------

def finish(model, z_star, S):
    """在支撑 S 上联合重拟合系数和 p，报告解的误差。"""
    import forward_generic as G
    S = list(S)
    n = len(S)
    z0 = np.concatenate([z_star[: model.m][S], z_star[model.m:]])
    fun = lambda q: model.full_residual(q, S)
    z, f = G.lm(fun, z0)
    z, f = gn_polish(fun, z)
    xi, p = z[:n], z[n:]
    err = model.solution_error(xi, p, S)
    return {"support": [model.names[i] for i in S], "n_terms": n,
            "coefficients": {model.names[i]: float(v) for i, v in zip(S, xi)}, "params": p_dict(p),
            "relative_residual": float(np.sqrt(f)) / model.base(), "solution_rel_error": err,
            "correct": bool(err <= SOLUTION_TOL)}


def job(args):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    name, lib, seed, mode = args
    np.seterr(over="ignore", invalid="ignore", divide="ignore")
    cols = list(range(len(LIB))) if lib == "big" else SMALL
    model = Model(name, cols)
    t0 = time.perf_counter()
    fits = stage_a(model, seed)
    rel_a, z_star = fits[0]
    out = {"case": name, "kind": model.case["kind"], "library": lib, "library_size": model.m, "seed": seed, "mode": mode,
           "stage_a_relative_residual": rel_a, "stage_a_params": p_dict(z_star[model.m:]),
           "time_stage_a_s": time.perf_counter() - t0}
    # 大库下全库拟合只用来“定位”波：27 个函数能在一片 (k, x0) 上把解近似到 1e-5～1e-7，精确点是一个窄坑，
    # 全库 LM 会沿平谷漂走（从真值附近出发也会）。所以：在前 K_B 个起点 × 3 个尺度的 p 上走 L1 路径，
    # 路径上的支撑按项数从少到多，在支撑上联合重拟合 (ξ_S, k, c, x0)；残差 ≤ 1e-8 的认定为精确解。
    # 门槛（1e-8）不变，只是改在稀疏模型上检验；没有任何支撑过门槛就拒绝。支撑仍只由 L1 路径给出。
    # 尺度：库里有半尺度函数（1/(1+e^θ) = (1 − tanh(θ/2))/2 等），同一个解在 k、2k、k/2 下都有精确写法。
    import forward_generic as G
    t1 = time.perf_counter()
    tries, seen, best = [], set(), None
    for i in range(min(K_B, len(fits))):
        if best is not None and best["n_terms"] == 1:  # 已有一项的精确解，不可能更简
            break
        for s in SCALES:
            if best is not None and best["n_terms"] == 1:
                break
            if s == 1.0:
                z_s = fits[i][1]
            else:
                p = fits[i][1][model.m:].copy()
                p[0] += np.log(s)
                z_s = G.lm(model.full_residual, np.concatenate([model.ic_bc_init(p), p]), max_iter=500)[0]
            try:
                sups = sorted(set(l1_path_supports(model, z_s[model.m:], z_s[: model.m], weighted=True)), key=len)
            except (np.linalg.LinAlgError, ValueError):
                continue
            for S in sups[:N_SUP]:
                if best is not None and len(S) > best["n_terms"]:
                    break
                key = (S, round(float(z_s[model.m]), 1), round(float(z_s[model.m + 1]), 1))
                if key in seen:
                    continue
                seen.add(key)
                try:
                    fin = finish(model, z_s, S)
                except (np.linalg.LinAlgError, ValueError):  # 参数跑到数值溢出区：这个支撑记为失败
                    continue
                fin.update({"start_rank": i, "scale": s})
                tries.append(fin)
                if fin["relative_residual"] <= GATE and (best is None or (fin["n_terms"], fin["relative_residual"]) < (best["n_terms"], best["relative_residual"])):
                    best = fin
    out.update({"n_sparse_refits": len(tries), "time_select_s": time.perf_counter() - t1,
                "best_sparse_residual": min(t["relative_residual"] for t in tries) if tries else None})
    if best is None:
        out.update({"rejected": True, "time_s": time.perf_counter() - t0})
        return out
    out.update({"rejected": False, **best})
    if mode == "compare":
        # 在已认定的精确解的波参数 p 上（全库系数先在 p 固定时拟合），比较各种产生候选支撑的方法
        p_star = np.array([np.log(best["params"]["k"]), best["params"]["c"], best["params"]["x0"]])
        xi_star = G.lm(lambda q: model.residual(q, p_star), model.ic_bc_init(p_star), max_iter=500)[0]
        z_star = np.concatenate([xi_star, p_star])
        scale = G.FixedP(model, p_star).scale
        comp = {}
        for meth in ["L1", "IRL1", "STRidge", "GA", "Exhaustive"]:
            ev = SupportEval(model, p_star)
            t1 = time.perf_counter()
            if meth in ("L1", "IRL1"):
                sups = l1_path_supports(model, p_star, xi_star, weighted=meth == "IRL1")
            elif meth == "STRidge":
                sups = stridge_supports(ev, scale)
            elif meth == "GA":
                sups = ga_supports(ev, np.random.default_rng(seed))
            else:
                sups = exhaustive_supports(ev)
            S, ok = ev.pick(sups)
            dt = time.perf_counter() - t1
            fin = finish(model, z_star, S)
            comp[meth] = {"support": fin["support"], "n_terms": fin["n_terms"], "admissible": ok, "correct": fin["correct"],
                          "solution_rel_error": fin["solution_rel_error"], "refits": ev.calls, "time_s": dt}
        out["comparison"] = comp
    out["time_s"] = time.perf_counter() - t0
    return out


def main():
    smoke = "--smoke" in sys.argv  # 自检：每个情形只跑大库、种子 0
    only = [a for a in sys.argv[1:] if not a.startswith("--")]  # 可选：只跑指定情形
    names = [n for n in CASES if not only or n in only]
    if smoke:
        jobs = [(n, "big", 0, "main") for n in names]
    else:
        jobs = [(n, lib, sd, "main") for n in names for lib in ("big", "small") for sd in SEEDS]
        jobs += [(n, "big", 0, "compare") for n in names if CASES[n]["kind"] == "positive"]
    report = {"python": platform.python_version(), "numpy": np.__version__, "sympy": sp.__version__,
              "library": LIB_NAMES, "small_library": [LIB_NAMES[j] for j in SMALL],
              "library_derivative_check": check_library(), "gate": GATE, "n_starts": N_STARTS, "polish_iter": POLISH_ITER,
              "search": {"k": K_RANGE, "c": C_RANGE, "x0": "计算域"}, "solution_tol": SOLUTION_TOL,
              "selection": "p 固定后对配点残差线性化做精确 LARS（迭代重加权，阈值 1e-5），路径上的支撑中残差过门槛（1e-8）且项数最少者",
              "cases": {n: {k: (str(v) if k in ("F", "exact") else v) for k, v in CASES[n].items()} for n in names},
              "runs": []}
    t0 = time.time()
    with ProcessPoolExecutor(WORKERS) as ex:
        for r in ex.map(job, jobs):
            report["runs"].append(r)
            tag = "REJECT" if r["rejected"] else ("OK" if r.get("correct") else ("cmp" if r["mode"] == "compare" else "BAD"))
            print(f"{r['case']:22s} {r['library']:5s} seed={r['seed']} {r['mode']:7s} {tag:6s} resA={r['stage_a_relative_residual']:.1e} "
                  f"sparse={r.get('best_sparse_residual') or 0:.1e} support={r.get('support')} err={r.get('solution_rel_error')} {r['time_s']:.0f}s", flush=True)
    report["wall_time_s"] = time.time() - t0
    out = Path(__file__).with_name("forward_biglib_smoke.json" if smoke else
                                   ("forward_biglib_result.json" if not only else "forward_biglib_partial.json"))
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
