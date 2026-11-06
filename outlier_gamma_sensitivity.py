"""Small gamma sensitivity scan for the joint sparse-error L1 model."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np

import noise_inverse as N
from outlier_l1_compare import run_one


GAMMAS = [0.5, 1.0, 2.0, 5.0, 10.0]
CASES = ["heat", "burgers", "kdv"]
FRACTIONS = [0.0, 0.05]
AMPLITUDE = 10.0
SEEDS = [0]
LAYER = "L13"


def main() -> None:
    t0 = time.perf_counter()
    rows = []
    for gamma in GAMMAS:
        for cname in CASES:
            for fraction in FRACTIONS:
                for seed in SEEDS:
                    result = run_one(N.CASES[cname], LAYER, fraction, AMPLITUDE, seed, gamma)
                    rows.append({
                        "gamma": gamma,
                        "case": cname,
                        "fraction": fraction,
                        "seed": seed,
                        "joint_L1": {
                            "beta": result["joint_L1"]["beta_metrics"],
                            "error": result["joint_L1"]["error_metrics"],
                        },
                        "joint_IRL1": {
                            "beta": result["joint_IRL1"]["beta_metrics"],
                            "error": result["joint_IRL1"]["error_metrics"],
                        },
                    })
                    print(gamma, cname, fraction, flush=True)
    report = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "method": "gamma sensitivity for joint weighted L1 and joint IRL1",
        "layer": LAYER,
        "gammas": GAMMAS,
        "fractions": FRACTIONS,
        "amplitude_rms": AMPLITUDE,
        "seeds": SEEDS,
        "rows": rows,
        "wall_time_s": time.perf_counter() - t0,
    }
    out = Path(__file__).with_name("outlier_gamma_sensitivity_result.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
