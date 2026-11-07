"""Second-batch preservation and serialization audit; no scientific reruns."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import stage3_data_v2 as DATA
import stage3_campaign as CAMPAIGN

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "stage_execution_audit_v2_result.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    result = json.loads(path.read_text(encoding="utf-8-sig"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    pending = [result]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Nonfinite parsed numeric value")
    return result


def main():
    if OUT.exists():
        raise FileExistsError("Preserve delivered audit; use a new version.")
    previous = read(ROOT / "stage_execution_audit_result.json")
    baseline = read(ROOT / "stage0_gate_result.json")
    snapshot = read(ROOT / "stage2b_protocol_result.json")
    checks, errors, scripts, results = {}, [], {}, {}
    for label, mapping in [
        ("historical_scripts", baseline["historical_script_sha256"]),
        ("first_batch_scripts", previous["new_scripts"]),
        ("second_batch_start_scripts", snapshot["previous_script_sha256"]),
        ("first_batch_results", {n: r["sha256"] for n, r in previous["result_manifests"].items()})]:
        changed = [n for n, h in mapping.items() if not (ROOT / n).is_file() or digest(ROOT / n) != h]
        checks[label + "_unchanged"] = not changed
        errors.extend({"file": n, "error": label + " changed"} for n in changed)
    for path in sorted(ROOT.glob("stage*.py")):
        try:
            compile(path.read_text(encoding="utf-8-sig"), str(path), "exec")
            scripts[path.name] = digest(path)
        except Exception as exc:
            errors.append({"file": path.name, "error": str(exc)})
    for path in sorted(ROOT.glob("stage*.json")):
        if path == OUT:
            continue
        try:
            report = read(path)
            results[path.name] = {"sha256": digest(path), "size_bytes": path.stat().st_size,
                "reported_all_checks_passed": report.get("all_checks_passed")}
        except Exception as exc:
            errors.append({"file": path.name, "error": str(exc)})
    protocol = DATA.load_frozen_protocol()
    CAMPAIGN.protocol_check(protocol)
    checks["frozen_protocol_source_hashes_current"] = True
    checks["no_formal_observations_generated"] = not list((ROOT / "stage3_data").rglob("*.npz"))
    checks["formal_campaign_not_started"] = not (ROOT / "stage3_campaign_result.json").exists()
    now = datetime.now(timezone.utc)
    checks["saved_before_second_batch_deadline"] = now < datetime(2026, 11, 6, 12, 48, tzinfo=timezone.utc)
    checks["syntax_and_finite_serialization"] = not errors
    report = {"stage": "second_batch_integrity_checkpoint", "completed_utc": now.isoformat(),
        "batch_start_utc_approximate": "2026-11-06T10:49:00+00:00",
        "conservative_batch_deadline_utc": "2026-11-06T12:48:00+00:00",
        "all_checks_passed": all(checks.values()), "checks": checks, "errors": errors,
        "new_scripts": scripts, "result_manifests": results,
        "scope": "Preservation, syntax, finite JSON and frozen source provenance; not publication acceptance or formal outcomes.",
        "numerical_status": {"stage2a_complete_paths": 72, "stage2b_complete_paths": 36,
                             "stage2c_complete_paths": 36, "formal_complete_paths": 0},
        "formal_protocol_sha256": digest(ROOT / "stage3_protocol_result.json"),
        "next_action": "Start an explicitly bounded stage3_campaign batch with the frozen protocol. No frozen source or method changes based on formal outcomes.",
        "known_limits": ["No observed improvement from the relative-epsilon IRL1 ablation.",
            "All true PDEs are polynomial; extra functional families currently test distraction resistance.",
            "Complete end-to-end timing is unavailable after some interruptions; no runtime superiority claim.",
            "No formal multi-seed result, general nonlinear rollout or publication-ready claim."]}
    with OUT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"all_checks_passed": report["all_checks_passed"], "scripts": len(scripts),
                      "results": len(results), "errors": errors}))
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
