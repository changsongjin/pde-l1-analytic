"""Independent read-only stage-2 artifact review and support-order deduplication.

Does not rebuild fields/matrices or rerun any solver. The original stage-2
report stays unchanged. Passing permits a development L1 baseline, not a paper
claim, exact recovery guarantee, or global sparse-identifiability conclusion.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from stage0_data_bundle import load_bundle
from stage0_evaluation import array_hash, canonical_term, strict_json_write

ROOT = Path(__file__).resolve().parent
EXPECTED_CASES = {"heat", "wave", "advection_diffusion", "burgers", "kdv", "logheat"}


def strict_load(path):
    def reject(token):
        raise ValueError("non-finite JSON token: " + token)
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def review_row(case, row):
    meta = row["metadata"]
    sigma, seed = row["sigma"], row["seed"]
    tag = f"{case}_sigma_{repr(float(sigma)).replace('.', 'p')}_seed_{seed}"
    matrix_file = ROOT / "stage2_matrices" / (tag + ".npz")
    raw_file = ROOT / "stage2_data" / (tag + ".npz")
    matrix_meta = strict_load(matrix_file.with_suffix(".json"))
    raw_manifest = strict_load(raw_file.with_suffix(".json"))
    if matrix_meta != meta:
        raise ValueError("saved matrix metadata and stage-2 report differ")
    if hashlib.sha256(raw_file.read_bytes()).hexdigest() != meta["raw_bundle_sha256"]:
        raise ValueError("raw input file hash differs from matrix source")
    for name, digest in meta["source_hashes"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError("changed matrix source: " + name)
    x, t, fields, restored = load_bundle(raw_file)
    if restored != raw_manifest or len(fields) != 6 or raw_manifest["trajectory_count"] != 6:
        raise ValueError("raw bundle does not contain all six development trajectories")
    raw_rows, matrix_rows = raw_manifest["trajectories"], meta["trajectories"]
    expected_splits = ["train"]*4 + ["validation"]*2
    if ([r["split"] for r in raw_rows] != expected_splits
            or [r["split"] for r in matrix_rows] != expected_splits
            or len({r["trajectory_id"] for r in raw_rows}) != 6
            or [r["trajectory"] for r in matrix_rows] != list(range(6))
            or meta["test_data_used"] is not False):
        raise ValueError("development split is inconsistent or test data was used")
    columns = 1159 if case == "wave" else 1158
    with np.load(matrix_file, allow_pickle=False) as bundle:
        if set(bundle.files) != {f"{prefix}_{i}" for i in range(6) for prefix in ("a", "b")}:
            raise ValueError("matrix bundle missing/extra trajectory arrays")
        for i, entry in enumerate(matrix_rows):
            a, b = bundle[f"a_{i}"], bundle[f"b_{i}"]
            if (a.shape != (entry["rows"], columns) or b.shape != (entry["rows"],)
                    or not np.isfinite(a).all() or not np.isfinite(b).all()
                    or array_hash(a) != entry["matrix_hash"]
                    or array_hash(b) != entry["target_hash"]
                    or array_hash(fields[i]) != entry["raw_hash"]
                    or fields[i].shape != (len(t), len(x))):
                raise ValueError("matrix/raw-array shape, finite-value, or hash failure")
    if len({entry["raw_hash"] for entry in matrix_rows}) != 6:
        raise ValueError("duplicate raw observations across development trajectories")
    geometry_checks = {}
    for name in ("geometry_full_train", "geometry_small_train", "geometry_single_train_small"):
        geometry = row[name]
        spectrum = np.asarray(geometry["singular_values"])
        geometry_checks[name] = bool(np.isfinite(spectrum).all() and np.all(spectrum >= 0)
                                    and len(spectrum) == min(geometry["shape"]))
    if not all(geometry_checks.values()):
        raise ValueError("invalid geometry singular spectrum")
    competitors = row["bounded_competitors"]
    if competitors["not_global_oracle"] is not True:
        raise ValueError("bounded competition falsely presented as global")
    canonical = [canonical_term(n) for n in meta["names"]]
    true_indices = tuple(canonical.index(canonical_term(n)) for n in row["true_support"])
    supports = [()] + [(i,) for i in range(columns)]
    supports += list(itertools.combinations(meta["small_indices"], 2))
    supports.append(true_indices)
    original_count = len(set(supports))
    corrected_count = len({tuple(sorted(s)) for s in supports})
    if original_count != competitors["candidates"]:
        raise ValueError("recorded competitor count does not match declared bounded search")
    seen, unique, removed = set(), [], []
    for rank, candidate in enumerate(competitors["top_validation"], 1):
        support = tuple(sorted(candidate["support_indices"]))
        if support in seen:
            removed.append({"original_rank": rank, "support": candidate["support"],
                            "support_set_indices": list(support)})
        else:
            seen.add(support)
            unique.append({**candidate, "original_rank": rank,
                           "support_set_indices": list(support)})
    return {"case": case, "sigma": sigma, "seed": seed, "all_checks_passed": True,
            "matrix_target_pairs_verified": 6, "raw_arrays_verified": 8,
            "source_files_verified": sorted(meta["source_hashes"]),
            "raw_bundle_sha256": meta["raw_bundle_sha256"],
            "split": {"train": [0, 1, 2, 3], "validation": [4, 5], "test_used": False},
            "geometry_checks": geometry_checks,
            "known_coefficients_validation_residual": row["known_coefficients_validation_residual"],
            "corrected_candidate_counts": {"original_order_sensitive": original_count,
                "unique_support_sets": corrected_count, "duplicates_removed": original_count-corrected_count},
            "top_validation_unique_by_support_set": unique,
            "top_validation_duplicates_removed": removed,
            "deduplicated_top_scope": "deduplicated saved top ten only; no unsaved candidate was recomputed"}


def main():
    source = ROOT / "stage2_identifiability_result.json"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    original = strict_load(source)
    checks = {"six_expected_cases": set(original["cases"]) == EXPECTED_CASES,
              "stage2_execution_complete": original["all_checks_passed"] is True,
              "test_data_not_used": original["test_data_used"] is False}
    rows, errors = [], []
    for case, records in original["cases"].items():
        checks[case+"_noise_grid"] = len(records) == 2 and {r["sigma"] for r in records} == {0.0, 0.01}
        for row in records:
            try:
                rows.append(review_row(case, row))
            except (ValueError, KeyError, OSError, AssertionError) as exc:
                errors.append({"case": case, "sigma": row.get("sigma"), "error": str(exc)})
    pairs = sum(row["matrix_target_pairs_verified"] for row in rows)
    checks["all_72_matrix_target_pairs_verified"] = pairs == 72 and not errors
    checks["original_report_unchanged"] = hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    passed = all(checks.values())
    report = {"stage": 2, "component": "independent artifact and evaluation review",
        "all_checks_passed": passed, "checks": checks, "errors": errors, "rows": rows,
        "matrix_target_pairs_verified": pairs,
        "original_stage2_result_sha256": source_hash,
        "allowed_next_step": "stage2a development L1/IRL1 baseline" if passed else None,
        "pass_meaning": "artifacts and declared diagnostics are consistent enough to run a development baseline; not publication acceptance or exact support recovery",
        "support_order_issue": "Some original true-support tuples were appended in a different order from the same enumerated pair. Counts and saved top candidates are deduplicated here; original report unchanged.",
        "scope_limits": ["No field, matrix, or solver was rerun.",
            "Competition is all full-library singletons plus J18 pairs and true-support diagnostics, not a global oracle.",
            "Rank/small singular values do not alone prove sparse non-identifiability.",
            "Validation errors are weak equation residuals, not learned-PDE trajectory rollout errors.",
            "One development noise seed and two noise levels do not establish an algorithm robustness ranking.",
            "No requirement that L1 recover the true structure in every noisy or large-library case."]}
    strict_json_write(ROOT / "stage2_review_result.json", report)
    print(json.dumps({"all_checks_passed": passed, "matrix_target_pairs_verified": pairs,
                      "errors": errors, "allowed_next_step": report["allowed_next_step"]}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
