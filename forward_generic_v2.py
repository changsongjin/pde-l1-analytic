# 通用库正问题 v2。forward_generic.py 不改，只在第一步 A 后加一步「延长迭代」。
#
# 诊断（见 HANDOFF.md）：Burgers 种子 3、4 不是错盆地。tanh 幂次库能近似吸收平移，
# 残差沿 x0 几乎是平的，LM 300 步走不到门槛 1e-8。
# 改动只有一处：多起点 LM 之后，只对残差最小的那个结果从原地继续 LM，最多 POLISH_ITER 步。
# 不换起点、不加信息；门槛 1e-8、IRL1、λ 路径、阈值 1e-5、重拟合全部与 v1 相同。

import json
import platform
from pathlib import Path

import numpy as np

import forward_generic as G

POLISH_ITER = 3000


def stage_a_polished(model, seed):
    fits = G.stage_a_v1(model, seed)
    rel0, z0 = fits[0]
    z, f = G.lm(model.full_residual, z0, max_iter=POLISH_ITER)
    rel = float(np.sqrt(f)) / model.base()
    fits[0] = (rel, z)
    fits.sort(key=lambda t: t[0])
    POLISH_LOG.append({"seed": seed, "pde": model.case["pde_text"], "before": rel0, "after": rel})
    return fits


POLISH_LOG = []
# run_case 内部按模块名调用 stage_a，这里替换成带延长迭代的版本
G.stage_a_v1 = G.stage_a
G.stage_a = stage_a_polished


def main():
    report = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "library": G.NAMES,
        "wave_variable": "θ = k (x - x0 - c t)，(k, c, x0) 连续未知",
        "search_ranges": {"k": list(G.K_RANGE), "c": list(G.C_RANGE), "x0": "计算域"},
        "gate": G.GATE,
        "polish_iter": POLISH_ITER,
        "basis_derivative_check": G.check_basis(),
        "two_soliton_fd_check": G.check_two_soliton(),
        "rule": "与 forward_generic.py 相同，只在 A 之后对残差最小的结果继续 LM 最多 3000 步。",
        "cases": {},
    }
    print("python", report["python"], "numpy", report["numpy"], "polish", POLISH_ITER)
    for name, case in G.CASES.items():
        report["cases"][name] = G.run_case(name, case)
    report["polish_log"] = POLISH_LOG
    report["summary"] = {k: v["passed"] for k, v in report["cases"].items()}
    print("summary:", report["summary"])
    out = Path(__file__).with_name("forward_generic_v2_result.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
