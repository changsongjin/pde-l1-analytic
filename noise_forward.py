# 正问题加高斯噪声：初值和边界数据带噪声（正问题本来就不用解的内部采样）。旧脚本不改。
#
# 流程与 forward_generic_v2.py 相同（通用 tanh 库、连续波参数、多起点 LM + 延长迭代、IRL1），改动三处：
#   1. 噪声：u_ic、u_bc 各加 σ·std(u_ic)·N(0,1)。
#   2. 门槛：精确解也拟合不了噪声，残差到不了 1e-8。改为差异原理（Morozov）：
#      门槛 = 3 × 噪声本身贡献的相对残差 = 3·√2·σ·std(u_ic)/‖数据‖（两块初边值各贡献 σ·std）。假定 σ 已知。
#   3. 定支撑：带噪声时 λ 很小的一端会把干扰项也拟合进来，所以和反问题用同一个判据：
#      最终权重 λ 路径上出现过的每个支撑都与 (k,c,x0) 联合重拟合，取
#      L(S) = sqrt(RSS_S − RSS_全库)/‖数据‖ + |S|/p 最小者。论文原来的“过半”规则一并记录，只作对照。
# 反例（KdV 删 tanh²、KdV 双孤子）一并加噪声，看门槛还能不能拦下。
# 并行：每个（情形, σ, 种子）一个进程。

import json
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

NOISE = [0.001, 0.01, 0.05, 0.1]
SEEDS = [0, 1, 2, 3, 4]
WORKERS = 6


def job(args):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    name, sigma, seed = args
    import forward_generic as G
    import forward_generic_v2  # noqa: F401  导入即把第一步换成带延长迭代的版本
    import kdv_forward_blind as B

    case = G.CASES[name]
    model = G.Model(case)
    rng = np.random.default_rng(100 * seed + 17)
    s = float(np.std(model.u_ic))
    model.u_bc = model.u_bc + sigma * s * rng.standard_normal(model.u_bc.shape)
    model.u_ic = model.u_ic + sigma * s * rng.standard_normal(model.u_ic.shape)
    base = model.base()
    gate = max(G.GATE, 3.0 * np.sqrt(2.0) * sigma * s / base)

    t0 = time.perf_counter()
    fits = G.stage_a(model, seed)
    rel_a, z_star = fits[0]
    out = {"case": name, "kind": case["kind"], "sigma": sigma, "seed": seed, "gate": gate,
           "stage_a_relative_residual": rel_a, "stage_a_params": G.p_dict(z_star[model.m:]),
           "n_starts_below_gate": int(sum(f[0] <= gate for f in fits))}
    if rel_a > gate:
        out.update({"rejected": True, "support": None, "time_s": time.perf_counter() - t0})
        return out

    p_star, xi_star = z_star[model.m:], z_star[: model.m]
    pb = G.FixedP(model, p_star)
    x_star = xi_star * pb.scale
    w, _ = B.irl1_weights(pb, x_star, B.IRL1_RATIO)
    # λ 路径：与 kdv_forward_blind.path_support 相同的热启动链，同时记下每个 λ 的支撑
    lam_max = pb.lam_max(w)
    chain = np.zeros(pb.p)
    counts = np.zeros(pb.p, dtype=int)
    supports = set()
    for lam in lam_max * np.geomspace(1.0, B.LAMBDA_LO, B.N_LAMBDA):
        cands = [B.gn_l1(pb, chain, float(lam), w), B.gn_l1(pb, x_star, float(lam), w)]
        chain = cands[int(np.argmin([pb.objective(c, float(lam), w) for c in cands]))]
        on = np.abs(chain) > B.SUPPORT_TOL
        counts[on] += 1
        if on.any():
            supports.add(tuple(int(i) for i in np.where(on)[0]))
    rule_support = [int(i) for i in np.where(counts > B.N_LAMBDA / 2.0)[0]]

    rss_full = (rel_a * base) ** 2
    scored = []
    for S in supports:
        z_r, rel_r = G.refit(model, z_star, list(S))
        loss = np.sqrt(max((rel_r * base) ** 2 - rss_full, 0.0)) / base + len(S) / model.m
        scored.append((loss, S, z_r, rel_r))
    loss, S, z_r, rel_r = min(scored, key=lambda t: t[0])
    out.update({
        "rejected": False,
        "support": [model.names[i] for i in S],
        "support_rule_majority": [model.names[i] for i in rule_support],
        "criterion": loss,
        "refit_relative_residual": rel_r,
        "refit_coefficients": {model.names[i]: float(z_r[i]) for i in S},
        "refit_params": G.p_dict(z_r[model.m:]),
        "n_candidate_supports": len(supports),
        "time_s": time.perf_counter() - t0,
    })
    if case["kind"] == "positive":
        true_names = sorted(model.names[model.cols.index(j)] for j in case["true_xi"])
        out["exact"] = sorted(out["support"]) == true_names
        out["exact_rule_majority"] = sorted(out["support_rule_majority"]) == true_names
        tp = case["true_p"]
        out["param_abs_error"] = {k: abs(out["refit_params"][k] - tp[k]) for k in tp}
        if out["exact"]:
            tx = {model.names[model.cols.index(j)]: v for j, v in case["true_xi"].items()}
            out["coef_rel_err"] = float(np.sqrt(sum((out["refit_coefficients"][k] - v) ** 2 for k, v in tx.items())
                                                / sum(v**2 for v in tx.values())))
    return out


def main():
    import forward_generic as G
    jobs = [(name, s, sd) for name in G.CASES for s in NOISE for sd in SEEDS]
    results = []
    t0 = time.time()
    with ProcessPoolExecutor(WORKERS) as ex:
        for r in ex.map(job, jobs):
            results.append(r)
            tag = "rejected" if r["rejected"] else (("OK " if r.get("exact") else "BAD") if r["kind"] == "positive" else "ACCEPTED")
            print(f"{r['case']:16s} σ={r['sigma']:<5} seed={r['seed']} {tag} res={r['stage_a_relative_residual']:.2e} "
                  f"gate={r['gate']:.2e} support={r.get('support')} {r['time_s']:.0f}s", flush=True)
    summary = {}
    for name in G.CASES:
        for s in NOISE:
            rs = [r for r in results if r["case"] == name and r["sigma"] == s]
            if G.CASES[name]["kind"] == "positive":
                ok = [r for r in rs if r.get("exact")]
                summary[f"{name}|{s}"] = {
                    "exact": f"{len(ok)}/{len(rs)}",
                    "exact_rule_majority": f"{sum(bool(r.get('exact_rule_majority')) for r in rs)}/{len(rs)}",
                    "rejected": sum(r["rejected"] for r in rs),
                    "coef_rel_err_median": float(np.median([r["coef_rel_err"] for r in ok])) if ok else None,
                    "param_abs_error_median": {k: float(np.median([r["param_abs_error"][k] for r in ok])) for k in ("k", "c", "x0")} if ok else None,
                }
            else:
                summary[f"{name}|{s}"] = {"rejected": f"{sum(r['rejected'] for r in rs)}/{len(rs)}",
                                          "accepted_supports": [r["support"] for r in rs if not r["rejected"]]}
    report = {"python": platform.python_version(), "numpy": np.__version__, "noise": NOISE, "seeds": SEEDS,
              "gate": "3·√2·σ·std(u_ic)/‖数据‖（差异原理，假定 σ 已知），不低于 1e-8",
              "selection": "λ 路径上的支撑各自与 (k,c,x0) 联合重拟合，取 sqrt(RSS_S − RSS_全库)/‖数据‖ + |S|/p 最小者",
              "summary": summary, "runs": results, "wall_time_s": time.time() - t0}
    Path(__file__).with_name("noise_forward_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
