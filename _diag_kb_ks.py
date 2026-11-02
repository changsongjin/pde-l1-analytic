# 临时诊断：KdV–Burgers 与 Kuramoto–Sivashinsky 在 27 列库上为什么被拒。
# 不改已有脚本，只 import forward_biglib。结果打印，不写结果 JSON。

import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

import forward_biglib as F


def true_param(name):
    """由真解读出 (k, c, x0) 和真项在库里的名字。只用于诊断，不进搜索。"""
    if name == "kdv_burgers":
        return 1.0, 0.5, 0.0, ["1", "tanh θ", "tanh² θ"]
    if name == "kuramoto_sivashinsky":
        return float(np.sqrt(209) / 38), 1.0, 0.0, ["1", "tanh θ", "tanh³ θ"]
    raise ValueError(name)


def report(name, seed=0):
    k, c, x0, true_names = true_param(name)
    model = F.Model(name, list(range(len(F.LIB))))
    p = np.array([np.log(k), c, x0])
    idx = [model.names.index(n) for n in true_names]
    xi0 = model.ic_bc_init(p, idx)
    print(f"\n===== {name}  真参数 k={k:.6g} c={c} x0={x0}")
    print("  真项最小二乘初值", dict(zip(true_names, np.round(xi0, 6))))
    r, _ = model.residual(xi0, p, idx)
    print(f"  真支撑残差（最小二乘初值） {np.linalg.norm(r) / model.base():.3e}")
    fin = F.finish(model, np.concatenate([np.zeros(model.m), p]), idx)
    print(f"  真支撑联合重拟合 残差 {fin['relative_residual']:.3e} 解误差 {fin['solution_rel_error']:.3e}")
    print("    系数", {k_: round(v, 6) for k_, v in fin["coefficients"].items()})
    print("    参数", {k_: round(v, 6) for k_, v in fin["params"].items()})

    # 真值附近的全库系数，看 L1 路径是否含真支撑
    import forward_generic as G
    xi_full = G.lm(lambda q: model.residual(q, p), model.ic_bc_init(p), max_iter=500)[0]
    rfull, _ = model.residual(xi_full, p)
    print(f"  全库在真 p 上拟合残差 {np.linalg.norm(rfull) / model.base():.3e}")
    big = np.argsort(-np.abs(xi_full))[:8]
    print("  全库系数最大的 8 项", [(model.names[i], round(float(xi_full[i]), 4)) for i in big])
    # 用搜索自己找到的 p（前 5 个起点），不告诉它真参数
    fits = F.stage_a(model, seed)
    print(f"  多起点最好残差 {fits[0][0]:.3e}  参数 { {kk: round(vv, 5) for kk, vv in F.p_dict(fits[0][1][model.m:]).items()} }")
    for i in range(min(3, len(fits))):
        p_i = fits[i][1][model.m:]
        sups = F.l1_path_supports(model, p_i, fits[i][1][: model.m], weighted=True)
        uniq = []
        for s in sups:
            if s not in uniq:
                uniq.append(s)
        print(f"  起点 {i}（残差 {fits[i][0]:.2e}，k={np.exp(p_i[0]):.4f} c={p_i[1]:.4f}）路径 {len(uniq)} 个支撑，最短几个：")
        for s in sorted(uniq, key=len)[:4]:
            rel = model.residual(model.ic_bc_init(p_i, s), p_i, s)[0]
            print("     ", len(s), [model.names[j] for j in s], f"初值残差 {np.linalg.norm(rel) / model.base():.2e}")
    return
    for weighted, label in ((True, "IRL1"), (False, "L1")):
        sups = F.l1_path_supports(model, p, xi_full, weighted=weighted)
        uniq = []
        for s in sups:
            if s not in uniq:
                uniq.append(s)
        print(f"  {label} 路径不同支撑 {len(uniq)} 个：")
        for s in sorted(uniq, key=len):
            print("   ", len(s), [model.names[i] for i in s])
    # 全库系数里 |系数| 超过若干阈值的项，看有没有别的精确写法
    for tol in (1e-2, 1e-3, 1e-6, 1e-8):
        on = [i for i in range(model.m) if abs(xi_full[i]) > tol]
        r2, _ = model.residual(xi_full[on] if False else np.array([xi_full[i] for i in on]), p, on)
        print(f"  |系数|>{tol:.0e} 的 {len(on)} 项，残差 {np.linalg.norm(r2) / model.base():.3e}:",
              [model.names[i] for i in on])


if __name__ == "__main__":
    np.seterr(over="ignore", invalid="ignore", divide="ignore")
    names = sys.argv[1:] or ["kdv_burgers", "kuramoto_sivashinsky"]
    for n in names:
        report(n)
