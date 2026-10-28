# 精确的加权 L1（Lasso）同伦路径，只用 NumPy。
# 目标：0.5 * ||b - A β||² + λ Σ w_j |β_j|。
# 做法：列除以 w 变成普通 Lasso，用 LARS-Lasso 同伦求整条分段线性路径。
# 每一个折点都是精确解（到浮点舍入），不依赖坐标下降的迭代次数。
# 旧脚本不改，新脚本从这里调用。

import numpy as np


def lars_lasso_path(a, b, w=None, lam_min_ratio=0.0, max_steps=500, tol=1e-14):
    """返回 [(λ, β)] 折点列表，β 是对原始列 a（未除 w）的系数。"""
    n, p = a.shape
    w = np.ones(p) if w is None else np.asarray(w, dtype=float)
    x = a / w
    beta = np.zeros(p)
    corr = x.T @ b
    lam = float(np.max(np.abs(corr)))
    lam_stop = lam * lam_min_ratio
    active = []
    knots = [(lam, beta.copy())]
    j0 = int(np.argmax(np.abs(corr)))
    active.append(j0)
    for _ in range(max_steps):
        if lam <= lam_stop + tol * knots[0][0] or not active:
            break
        xa = x[:, active]
        signs = np.sign(corr[active])
        gram = xa.T @ xa
        try:
            d_act = np.linalg.solve(gram, signs)
        except np.linalg.LinAlgError:
            d_act = np.linalg.lstsq(gram, signs, rcond=None)[0]
        direction = np.zeros(p)
        direction[active] = d_act
        a_dir = x.T @ (xa @ d_act)
        # λ 减小 γ 时：β_A += γ d，相关系数 c_j 变 c_j - γ a_dir_j
        gamma = lam - lam_stop
        event = ("end", None)
        for j in range(p):
            if j in active:
                continue
            for s in (1.0, -1.0):
                denom = s - a_dir[j]
                if abs(denom) < 1e-15:
                    continue
                g = (lam * s - corr[j]) / denom
                if tol < g < gamma:
                    gamma = g
                    event = ("add", j)
        for k, j in enumerate(active):
            if abs(d_act[k]) < 1e-300:
                continue
            g = -beta[j] / d_act[k]
            if tol < g < gamma:
                gamma = g
                event = ("drop", j)
        beta = beta + gamma * direction
        corr = corr - gamma * a_dir
        lam = lam - gamma
        if event[0] == "drop":
            beta[event[1]] = 0.0
            active.remove(event[1])
        elif event[0] == "add":
            active.append(event[1])
        knots.append((lam, (beta / w).copy()))
        if event[0] == "end":
            break
    return knots


def lasso_at(knots, lam):
    """在折点之间线性插值，得到给定 λ 的精确解。"""
    lams = [k[0] for k in knots]
    if lam >= lams[0]:
        return knots[0][1].copy()
    for i in range(1, len(knots)):
        if lam >= lams[i]:
            l0, b0 = knots[i - 1]
            l1, b1 = knots[i]
            if l0 == l1:
                return b1.copy()
            s = (l0 - lam) / (l0 - l1)
            return (1 - s) * b0 + s * b1
    return knots[-1][1].copy()


def kkt_violation(a, b, beta, lam, w):
    """最优性检查：非零项 |a_j^T r - λ w_j sign| ，零项 max(0, |a_j^T r| - λ w_j)，再除以 λ。"""
    r = b - a @ beta
    g = a.T @ r
    v = np.zeros_like(g)
    nz = np.abs(beta) > 0
    v[nz] = np.abs(g[nz] - lam * w[nz] * np.sign(beta[nz]))
    v[~nz] = np.maximum(0.0, np.abs(g[~nz]) - lam * w[~nz])
    return float(np.max(v) / max(lam, 1e-300))
