"""Final integrity check for this execution checkpoint; no experiment reruns."""

import hashlib
import json
from pathlib import Path

from stage0_evaluation import strict_json_write

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    baseline = json.loads((ROOT/"stage0_gate_result.json").read_text(encoding="utf-8"))
    changed = [name for name, expected in baseline["historical_script_sha256"].items()
               if not (ROOT/name).is_file() or digest(ROOT/name) != expected]
    errors, scripts, results = [], {}, {}
    for path in sorted(ROOT.glob("stage*.py")):
        try:
            compile(path.read_text(encoding="utf-8-sig"), str(path), "exec")
            scripts[path.name] = digest(path)
        except SyntaxError as exc:
            errors.append({"file": path.name, "error": str(exc)})
    for path in sorted(ROOT.glob("stage*_result.json")):
        if path.name == "stage_execution_audit_result.json":
            continue
        try:
            def reject(value):
                raise ValueError("nonfinite token "+value)
            data = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
            results[path.name] = {"sha256": digest(path), "size_bytes": path.stat().st_size,
                                  "reported_all_checks_passed": data.get("all_checks_passed"),
                                  "stage": data.get("stage")}
        except (ValueError, UnicodeError) as exc:
            errors.append({"file": path.name, "error": str(exc)})
    report = {"scope": "source preservation, syntax and strict serialization; not scientific acceptance",
              "historical_scripts_unchanged": not changed, "changed_historical_scripts": changed,
              "new_scripts": scripts, "result_manifests": results, "errors": errors,
              "all_checks_passed": not changed and not errors,
              "preserved_negative_evidence": [
                  "stage2_field_check_result.json: initial KdV finite-difference truncation above accuracy target; refined report resolves it",
                  "stage1_weak_reduction_result.json: formal/convergence checks passed but coarse high-order quadrature accuracy inadequate; v2 adds absolute accuracy checks",
                  "stage0_l1_solver_result.json: failed legacy low-lambda solution and nonconverged independent CD attempts retained"
              ],
              "numerical_tail_status": "stage2a_batch_checkpoint_result.json is authoritative; its expected false gate records the remaining two uncertified lambda points",
              "publication_status": "not approved; later method freezing, independent multi-seed testing and contribution review still required"}
    strict_json_write(ROOT/"stage_execution_audit_result.json", report)
    print(json.dumps({"all_checks_passed": report["all_checks_passed"],
                      "scripts": len(scripts), "results": len(results), "errors": errors}))
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
