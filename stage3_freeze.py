"""Write-once formal protocol and readiness record; never generate observations."""
from __future__ import annotations

import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import stage3_campaign as CAMPAIGN
import stage3_data as DATA
import stage3_data_v2 as BINDING

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "stage3_freeze_result.json"
PROTOCOL = ROOT / "stage3_protocol_result.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"),
        parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))


def source_closure(names):
    pending, found = list(names), set()
    while pending:
        name = pending.pop()
        if name in found:
            continue
        path = ROOT / name
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        found.add(name)
        for node in ast.walk(tree):
            modules = ([a.name for a in node.names] if isinstance(node, ast.Import)
                       else [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for module in modules:
                local = module.split(".")[0] + ".py"
                if (ROOT / local).is_file() and local not in found:
                    pending.append(local)
    return {name: digest(ROOT / name) for name in sorted(found)}


def write_new(path, data):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def main():
    if PROTOCOL.exists() or OUTPUT.exists():
        raise FileExistsError("Freeze is write-once; preserve the existing protocol and report.")
    if list((ROOT / "stage3_data").rglob("*.npz")) or (ROOT / "stage3_campaign_result.json").exists():
        raise ValueError("Formal observations/outcomes already exist before first freeze.")
    draft = read(ROOT / "stage3_protocol_draft.json")
    checks = {}
    evidence = {}
    gates = ["stage2a_batch_checkpoint_v2_result.json", "stage2b_batch_checkpoint_result.json",
             "stage1_nonpolynomial_result.json", "stage1_nonpolynomial_curved_check_result.json",
             "stage1_extended_evaluation_result.json", "stage2c_review_v2_result.json",
             "stage3_data_preflight_result.json", "stage3_data_v2_preflight_result.json",
             "stage3_campaign_preflight_result.json"]
    for name in gates:
        result = read(ROOT / name)
        checks[name] = result["all_checks_passed"] is True
        evidence[name] = digest(ROOT / name)
        for source, expected in result.get("source_sha256", {}).items():
            checks[f"{name}:source:{source}"] = digest(ROOT / source) == expected
    for name, expected in draft["source_review_sha256"].items():
        checks["draft_evidence:" + name] = digest(ROOT / name) == expected
    # Revalidate current aliases, instead of claiming the older v1 preflight used this draft.
    protocol = deepcopy(draft)
    protocol.update(stage="3_systematic_noise_frozen_protocol", status="frozen", is_frozen=True,
        execution_authorized=True, frozen_utc=datetime.now(timezone.utc).isoformat(),
        purpose="User-authorized staged formal noise comparison with immutable statistical choices.",
        author_role="root reviewed final contracts and froze before any formal observations",
        draft_sha256=digest(ROOT / "stage3_protocol_draft.json"))
    seeds = set(draft["source_sha256_required"]) | {
        "stage3_data_v2.py", "stage3_campaign_preflight.py", "stage3_freeze.py",
        "stage2b_numerical_retry.py", "stage2c_numerical_retry.py"}
    protocol["source_sha256"] = source_closure(seeds)
    protocol["source_sha256_required"] = sorted(protocol["source_sha256"])
    protocol["source_sha256_state"] = "Frozen exact source bytes; transitive local imports included."
    protocol["readiness"] = {
        "maximum_library_pilot": "36/36 complete, 864 final lambda points; independent review passed",
        "data_and_runner_preflights": "Passed; current aliases rechecked in stage3_freeze_result.json",
        "formal_observations_before_freeze": False,
        "evidence_sha256": evidence,
        "complete_pilot_sha256": digest(ROOT / "stage2c_reconcile_complete_result.json"),
        "v1_preflight_draft_note": "The v1 report retains its earlier draft hash. Its unchanged source and all current protocol aliases are checked here; the campaign preflight binds the current draft."
    }
    protocol["execution_and_cost"].update(
        forecast="Maximum-library pilot: 1040.057 solver seconds for 36 paths including retries, approximately 28.9 seconds/path. This heterogeneous development average is not a formal runtime forecast; estimate batch throughput from actual checkpoints.",
        timing_completeness="Saved solver/continuation costs and matrix-build records remain attributable. Raw generation/load timings are explicitly this-call; an interruption before dataset finalization may lose earlier setup timing. Fine-grained stage timings need not add to end-to-end time.",
        ranking_limit="No end-to-end speed superiority claim from these records. Structure/noise is the primary comparison; missing setup costs are unavailable, never treated as zero.")
    protocol["report_only_code_policy"] = "Later report/plot code must implement these already frozen statistical definitions, bind protocol and input hashes, and receive independent verification. It cannot modify data, supports, models, selection or release conditions."
    DATA._protocol_content_checks(protocol)
    CAMPAIGN.protocol_check(protocol)
    checks["current_protocol_both_interfaces"] = True
    tasks = CAMPAIGN.requested_datasets(protocol, protocol["cases"], protocol["noise_levels"], protocol["noise_seeds"])
    checks["606_unique_datasets_3636_methods"] = len(tasks) == len(set(tasks)) == 606 and len(tasks) * 6 == 3636
    old = read(ROOT / "stage_execution_audit_result.json")
    baseline = read(ROOT / "stage0_gate_result.json")["historical_script_sha256"]
    snapshot = read(ROOT / "stage2b_protocol_result.json")["previous_script_sha256"]
    for label, mapping in [("historical", baseline), ("previous_batch", old["new_scripts"]), ("second_batch_snapshot", snapshot)]:
        checks[label + "_scripts_unchanged"] = all(digest(ROOT / n) == h for n, h in mapping.items())
    checks["previous_batch_results_unchanged"] = all(digest(ROOT / n) == r["sha256"] for n, r in old["result_manifests"].items())
    versions = {"python": platform.python_version(), "python_executable": sys.executable}
    for package in ("numpy", "scipy", "scikit-learn", "clarabel", "threadpoolctl", "sympy"):
        if package == "clarabel":
            local = {d.metadata["Name"].lower(): d.version for d in
                     importlib.metadata.distributions(path=[str(ROOT / ".research_deps")])}
            versions[package] = local[package]
        else:
            versions[package] = importlib.metadata.version(package)
    protocol["runtime_versions_at_freeze"] = versions
    if not all(checks.values()):
        raise ValueError("Readiness failures: " + repr([n for n, ok in checks.items() if not ok]))
    write_new(PROTOCOL, protocol)
    loaded = BINDING.load_frozen_protocol()
    CAMPAIGN.protocol_check(loaded)
    checks["written_protocol_loads_with_all_source_hashes"] = loaded == protocol
    report = {"stage": "3_formal_freeze_readiness", "all_checks_passed": all(checks.values()),
        "checks": checks, "protocol_sha256": digest(PROTOCOL), "source_sha256": protocol["source_sha256"],
        "runtime_versions": versions, "formal_or_test_data_generated": False,
        "formal_campaign_started": False, "draft_preserved": True,
        "scope": "Configuration/source freeze and inherited numerical/data-contract gates; no formal statistical results.",
        "timing_limit": protocol["execution_and_cost"]["timing_completeness"]}
    write_new(OUTPUT, report)
    print(json.dumps({"all_checks_passed": report["all_checks_passed"], "checks": len(checks),
        "frozen_sources": len(protocol["source_sha256"]), "protocol_sha256": digest(PROTOCOL)}))


if __name__ == "__main__":
    main()
