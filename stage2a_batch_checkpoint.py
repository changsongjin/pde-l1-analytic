"""Save the exact resumption state of the first <=2-hour development batch."""

import hashlib
import json
from pathlib import Path

from stage0_evaluation import strict_json_write

ROOT = Path(__file__).resolve().parent


def main():
    groups = [("stage2a_heat_pilot_result.json",),
              ("stage2a_heat_reconciled_result.json", "stage2a_heat_noise_pilot_result.json"),
              ("stage2a_other_cases_reconciled_result.json", "stage2a_other_cases_pilot_result.json")]
    sources, methods, unfinished = [], [], []
    for options in groups:
        path = next((ROOT/name for name in options if (ROOT/name).exists()), None)
        if path is None:
            raise RuntimeError("no saved campaign for " + repr(options))
        report = json.loads(path.read_text(encoding="utf-8"))
        sources.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "execution_complete": report["execution_complete"]})
        for row in report["rows"]:
            for layer, spec in row["layers"].items():
                for method, result in spec["methods"].items():
                    key = {"case": row["case"], "sigma": row["sigma"], "seed": row["seed"],
                           "layer": layer, "method": method}
                    numerical = result["path"]["all_solutions_certified"]
                    record = {**key, "numerically_complete": numerical,
                              "certified_lambda_count": sum(p["status"] == "certified" for p in result["path"]["points"]),
                              "requested_lambda_count": len(result["path"]["points"]),
                              "source_file": path.name}
                    methods.append(record)
                    if not numerical:
                        unfinished.append(record)
    complete_grid = len(methods) == 72 and len({tuple(m[k] for k in ("case","sigma","seed","layer","method"))
                                               for m in methods}) == 72
    report = {"stage": "2a", "max_seconds_per_batch_user_constraint": 7200,
              "sources": sources, "method_runs": methods, "expected_runs": 72,
              "recorded_runs": len(methods), "numerically_complete_runs": sum(m["numerically_complete"] for m in methods),
              "unfinished_numerical_runs": unfinished,
              "all_checks_passed": complete_grid and all(s["execution_complete"] for s in sources) and not unfinished,
              "test_data_generated_or_used": False, "new_statistical_L1_variant_tested": False,
              "next_batch_order": [
                  "Resolve any remaining uncertified lambda points with preserved original objectives and certificates.",
                  "Stage 2b: separate missing-path-structure from validation-score failure; test one targeted L1 change at a time.",
                  "Prioritize dynamical/rollout validation and noise propagation where smaller weak residual selects unstable dynamics.",
                  "Keep original L1/IRL1; freeze a justified modification before independent development replication and final test seeds.",
                  "Only then enter stage 3 paired multi-seed comparisons and later publication evidence review."
              ],
              "pending_scope": ["independent development replications", "native comparison pipelines",
                                "full held-out PDE rollout", "54 nonpolynomial supplementary terms",
                                "formal >=20 paired noise seeds", "publication novelty and evidence", "later raw-field outliers"],
              "interpretation": "Development checkpoint; complete numerical paths do not imply correct supports or publication readiness."}
    strict_json_write(ROOT/"stage2a_batch_checkpoint_result.json", report)
    print(json.dumps({"recorded": len(methods), "numerically_complete": report["numerically_complete_runs"],
                      "all_checks_passed": report["all_checks_passed"]}))


if __name__ == "__main__":
    main()
