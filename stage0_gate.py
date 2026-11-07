"""Aggregate numerical/evaluation gates without modifying historical experiments."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE_COMMIT = "4632b4d0f90af6eb5be436d0c646807f55c350d8"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", BASE_COMMIT], cwd=ROOT
    ).decode("utf-8").splitlines()
    originals = [p for p in paths if p.endswith(".py")]
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", BASE_COMMIT, "--", "*.py"], cwd=ROOT
    ).decode("utf-8").splitlines()
    reports = {}
    for name in ("stage0_l1_solver", "stage0_evaluation", "stage0_data_bundle"):
        p = ROOT / (name + "_result.json")
        reports[name] = json.loads(p.read_text(encoding="utf-8"),
                                  parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    checks = {"historical_scripts_unchanged": not changed}
    checks.update({name: bool(report.get("all_checks_passed", False))
                   for name, report in reports.items()})
    report = {
        "stage": 0, "date": "2026-11-06", "base_commit": BASE_COMMIT,
        "python_executable": __import__("sys").executable,
        "python": platform.python_version(),
        "packages": {p: importlib.metadata.version(p)
                     for p in ("numpy", "scipy", "scikit-learn", "sympy")},
        "historical_script_sha256": {p: sha256(ROOT / p) for p in originals},
        "historical_script_changes": changed,
        "component_result_sha256": {name: sha256(ROOT / (name + "_result.json"))
                                     for name in reports},
        "checks": checks, "all_checks_passed": all(checks.values()),
        "scope": "implementation gates; historical benchmark rankings are not rehabilitated",
        "remaining_formal_requirements": [
            "All new comparisons must load identical saved raw observations and declared trajectory splits.",
            "Native DLGA/SGA reruns require verified multi-trajectory input; legacy first-trajectory runs stay excluded.",
            "Every future solve must carry its own certificate; this audit is not a universal solver guarantee.",
            "Final noise curves, independent tests, scientific novelty and publication evidence remain pending."
        ],
    }
    out = ROOT / "stage0_gate_result.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"all_checks_passed": report["all_checks_passed"], "checks": checks}))
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
