"""Bounded resumption of unfinished development L1 paths, preserving old files."""

import hashlib
import json
import time
from pathlib import Path

from threadpoolctl import threadpool_limits

from stage0_evaluation import array_hash, normalize_training, strict_json_write
from stage2_identifiability import save_matrices, stack_split
from stage2a_solver_retry import retry_l1_path

ROOT = Path(__file__).resolve().parent


def main():
    original = ROOT / "stage2a_other_cases_pilot_result.json"
    source = json.loads(original.read_text(encoding="utf-8"))
    if not source["execution_complete"]:
        raise RuntimeError("original campaign is still running")
    jobs = [(row, layer, spec, method, record)
            for row in source["rows"] for layer, spec in row["layers"].items()
            for method, record in spec["methods"].items()
            if method == "L1" and not record["path"]["all_solutions_certified"]]
    report = {"scope": "numerical resumption only; no statistical method change",
              "source_file": original.name, "source_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
              "max_batch_seconds": 7200, "per_path_retry_seconds": 90, "rows": [],
              "execution_complete": False, "all_checks_passed": False,
              "pending": [{"case": r["case"], "sigma": r["sigma"], "layer": l, "method": m}
                          for r, l, _, m, _ in jobs]}
    output = ROOT / "stage2a_retry_other_cases_result.json"
    start = time.perf_counter()
    with threadpool_limits(limits=1):
        for row, layer, spec, method, record in jobs:
            if time.perf_counter()-start+95 > 7200:
                break
            matrices, targets, _ = save_matrices(row["case"], row["sigma"], row["seed"])
            a, b = stack_split(matrices, targets, range(4))
            a, b, _ = normalize_training(a[:, spec["indices_in_maximum_library"]], b)
            hashes = {"A": array_hash(a), "b": array_hash(b)}
            if hashes != spec["solver_input_hashes"]:
                raise ValueError("retry input differs from original")
            progress = ROOT / "stage2a_progress" / "retries" / (row["case"]+"_L1_result.json")
            progress.parent.mkdir(parents=True, exist_ok=True)
            path = retry_l1_path(a, b, record["path"], budget_s=90., progress_path=progress)
            report["rows"].append({"case": row["case"], "sigma": row["sigma"], "seed": row["seed"],
                                   "layer": layer, "method": method, "input_hashes": hashes,
                                   "original_path_seconds": record["path"]["elapsed_s"], "path": path})
            report["pending"].pop(0)
            report["elapsed_s"] = time.perf_counter()-start
            strict_json_write(output, report)
            print(row["case"], row["sigma"], path["status"],
                  "certified_points", sum(p["status"] == "certified" for p in path["points"]),
                  "retry_seconds", path["retry_elapsed_s"], flush=True)
    report["execution_complete"] = not report["pending"]
    report["all_checks_passed"] = report["execution_complete"] and all(
        row["path"]["all_solutions_certified"] for row in report["rows"])
    report["elapsed_s"] = time.perf_counter()-start
    strict_json_write(output, report)


if __name__ == "__main__":
    main()
