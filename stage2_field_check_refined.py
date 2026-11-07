"""Refine the KdV diagnostic stencil; preserve the original failed check.

No data generator, PDE coefficient, acceptance tolerance, or regression input
is changed. The first check's complete 36-row evidence is retained by hash.
"""

import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np

import stage2_data as D
import stage2_field_check as C
from stage0_evaluation import strict_json_write


ROOT = Path(__file__).resolve().parent


def main():
    started = time.perf_counter()
    previous_path = ROOT / "stage2_field_check_result.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    for name, recorded in previous["source_hashes"].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != recorded:
            raise RuntimeError("original field-check source changed; rerun independently before reuse: " + name)
    result = copy.deepcopy(previous)
    spec = D.CASES["kdv"]
    hx0, ht0 = C.STEPS["kdv"]
    rng = np.random.default_rng(62000 + list(D.CASES).index("kdv"))
    x = rng.uniform(spec["x"][0]+3*hx0, spec["x"][1]-3*hx0, 128)
    t = rng.uniform(spec["t"][0]+3*ht0, min(spec["t"][1]-3*ht0, .8*spec["t"][1]), 128)
    for row in result["field_checks"]:
        row["source"] = "first check, unchanged" if row["case"] != "kdv" else "first check plus two refined KdV steps"
        if row["case"] != "kdv":
            continue
        p = D.trajectory_parameters("kdv", row["trajectory_id"])
        row["levels"].extend(C.pde_difference("kdv", p, x, t, hx0*q, ht0*q) for q in (.125, .0625))
        coarse, fine = row["levels"][0]["relative_residual"], row["levels"][-1]["relative_residual"]
        row["fine_over_coarse_error_ratio"] = fine / max(coarse, 1e-300)
        row["convergence_or_numerical_floor"] = fine <= C.CONVERGENCE_FLOOR or fine <= coarse/4
        row["passed"] = fine <= C.FINE_RELATIVE_TOL and row["convergence_or_numerical_floor"]
        row["successive_error_ratios"] = [a["relative_residual"]/max(b["relative_residual"], 1e-300)
                                          for a, b in zip(row["levels"][:-1], row["levels"][1:])]
    failures = [{"case": r["case"], "trajectory_id": r["trajectory_id"],
                 "finest_relative_residual": r["levels"][-1]["relative_residual"]}
                for r in result["field_checks"] if not r["passed"]]
    failures.extend(f for f in previous["failures"] if "trajectory_id" not in f)
    result.update({"component": "refined independent field correctness diagnostics",
                   "all_checks_passed": not failures, "failures": failures,
                   "previous_result": {"file": previous_path.name,
                                       "sha256": hashlib.sha256(previous_path.read_bytes()).hexdigest(),
                                       "original_failed_checks": previous["failures"]},
                   "refinement": {"case": "kdv", "trajectory_ids": list(range(6)),
                                  "additional_step_factors": [.125, .0625],
                                  "unchanged_acceptance_tolerance": C.FINE_RELATIVE_TOL,
                                  "reason": "Three original KdV checks exceeded tolerance while converging at the expected third-derivative stencil truncation order.",
                                  "generator_modified": False},
                   "elapsed_seconds": time.perf_counter()-started})
    result["source_hashes"][Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["finite_difference_protocol"]["levels_by_case"] = {c: list(C.LEVELS) + ([.125, .0625] if c == "kdv" else []) for c in D.CASES}
    strict_json_write(ROOT/"stage2_field_check_refined_result.json", result)
    print(json.dumps({"all_checks_passed": not failures, "checks": len(result["field_checks"]),
                      "failures": failures,
                      "max_finest_relative_residual": max(r["levels"][-1]["relative_residual"] for r in result["field_checks"])}))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
