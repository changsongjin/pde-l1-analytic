"""Read-only matrix/protocol/result audit for the full declared candidate set.

No field generation, preprocessing, or L1 optimizer is called. A preflight can
be saved before the pilot; later reviews use new output files and audit the
saved method records, including honestly incomplete paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

import stage1_extended_evaluation as E
from stage0_data_bundle import load_bundle
from stage0_evaluation import SUPPORT_TOL, array_hash, normalize_training, strict_json_write
from stage0_l1_solver import certify_l1
from stage1_grammar import nonpolynomial_recipes
from stage2_data import CASES, bundle_path
from stage2_identifiability import stack_split
from stage2a_l1_paths import DEFAULT_RATIOS

ROOT = Path(__file__).resolve().parent


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON: "+value)))


def check_hashes(mapping):
    for name, expected in mapping.items():
        if digest(ROOT/name) != expected: raise ValueError("source hash mismatch: "+name)


def read_matrices(case, sigma):
    """Load existing caches directly; missing inputs fail instead of generating."""
    old_path = ROOT/"stage2_matrices"/f"{case}_sigma_{repr(float(sigma)).replace('.', 'p')}_seed_0.npz"
    extra_path = ROOT/"stage2c_matrices"/f"{case}_sigma_{float(sigma)}_seed_0_supplement54.npz"
    old, meta = read(old_path.with_suffix(".json")), read(extra_path.with_suffix(".json"))
    check_hashes(old["source_hashes"]); check_hashes(meta["source_hashes"])
    if (meta["polynomial_source_metadata_sha256"] != digest(old_path.with_suffix(".json"))
            or meta["old_polynomial_metadata"] != old or meta["npz_sha256"] != digest(extra_path)):
        raise ValueError("saved polynomial/supplement provenance changed")
    x, t, observations, raw_meta = load_bundle(bundle_path(case, sigma, 0))
    recipes = nonpolynomial_recipes()
    nold = 1159 if case == "wave" else 1158
    if (len(meta["trajectories"]) != 6 or len(observations) != 6 or meta["case"] != case
            or meta["sigma"] != sigma or meta["seed"] != 0 or meta["test_data_used"]
            or old["raw_bundle_sha256"] != meta["raw_bundle_sha256"] or meta["raw_bundle_sha256"] != raw_meta["npz_sha256"]
            or meta["names"] != old["names"]+[r["name"] for r in recipes]
            or meta["supplement_recipes"] != recipes or meta["n_total_columns"] != nold+54
            or meta["n_polynomial_and_nuisance_columns"] != nold or meta["n_supplement_columns"] != 54):
        raise ValueError("catalog/order/raw observation/split mismatch")
    names = meta["names"]
    costs = [E.term_complexity(n) for n in names]
    canonical = [E.canonical_term(n) for n in names]
    if costs[:nold] != old["complexities"] or len(set(canonical)) != len(names):
        raise ValueError("old costs changed or canonical duplicate introduced")
    matrices, targets = [], []
    with np.load(old_path, allow_pickle=False) as old_npz, np.load(extra_path, allow_pickle=False) as extra_npz:
        if set(extra_npz.files) != {f"additional_{i}" for i in range(6)}:
            raise ValueError("supplement array keys changed")
        for i in range(6):
            poly, target, extra = old_npz[f"a_{i}"].copy(), old_npz[f"b_{i}"].copy(), extra_npz[f"additional_{i}"].copy()
            entry, old_entry = meta["trajectories"][i], old["trajectories"][i]
            local = entry["supplement_metadata"]
            hx, ht = CASES[case]["half"]
            expected_x, expected_t = list(range(15+hx, len(x)-15-hx, hx)), list(range(ht, len(t)-ht, ht))
            combined = np.column_stack([poly, extra])
            observed_hash = array_hash(observations[i])
            if (entry["trajectory"] != i or entry["split"] != ("train" if i < 4 else "validation")
                    or entry["raw_hash"] != observed_hash or old_entry["raw_hash"] != observed_hash
                    or local["input_observation_hash"] != observed_hash or local["recipes"] != recipes
                    or local["names"] != [r["name"] for r in recipes]
                    or local["center_x_indices"] != expected_x or local["center_t_indices"] != expected_t
                    or local["target_order"] != (2 if case == "wave" else 1)
                    or extra.shape != (len(expected_x)*len(expected_t), 54)
                    or poly.shape != (len(target), nold)
                    or array_hash(poly) != old_entry["matrix_hash"] or array_hash(poly) != entry["polynomial_hash"]
                    or array_hash(target) != old_entry["target_hash"] or array_hash(target) != entry["target_hash"]
                    or local["target_hash"] != array_hash(target) or array_hash(extra) != entry["supplement_hash"]
                    or array_hash(combined) != entry["combined_hash"]
                    or not np.array_equal(combined[:, :nold], poly)):
                raise ValueError(f"matrix/target/raw input mismatch: {case}/{sigma}/{i}")
            matrices.append(combined); targets.append(target)
    return matrices, targets, meta, costs, canonical


def preflight():
    source = ROOT/"stage2c_extended_matrices_result.json"
    matrix_report = read(source)
    check_hashes(matrix_report["source_sha256"])
    if not matrix_report["all_checks_passed"]: raise ValueError("upstream matrix gate failed")
    expected = {(c, s, 0) for c in CASES for s in (0., .01, .05)}
    if len(matrix_report["rows"]) != 18 or {(r["case"], r["sigma"], r["seed"]) for r in matrix_report["rows"]} != expected:
        raise ValueError("not exactly the18 declared development datasets")
    rows = []
    for r in matrix_report["rows"]:
        matrices, targets, meta, costs, keys = read_matrices(r["case"], r["sigma"])
        if (meta["npz_sha256"] != r["supplement_npz_sha256"] or meta["raw_bundle_sha256"] != r["raw_bundle_sha256"]
                or r["columns"] != len(costs)):
            raise ValueError("matrix campaign row mismatch")
        rows.append({"case":r["case"], "sigma":r["sigma"], "trajectories":6, "columns":len(costs),
            "all_array_hashes_reverified":True, "polynomial_block_and_target_bitwise_preserved":True,
            "raw_observations_and_centers_verified":True, "exact54_order_verified":True,
            "all_polynomial_costs_unchanged":True, "canonical_keys_unique":True})
    return {"all_checks_passed":True, "matrix_report_sha256":digest(source), "rows":rows,
            "verified_trajectories":sum(r["trajectories"] for r in rows)}


def review_method(record, method, a, b, av, bv, an, bn, names, costs, dataset_id):
    path, evaluation = record["path"], record["evaluation"]
    if path["shape"] != list(an.shape) or path["ratios"] != DEFAULT_RATIOS.tolist() or len(path["points"]) != 24:
        raise ValueError("matrix shape or full24 lambda grid changed")
    if path["support_tolerance"] != SUPPORT_TOL: raise ValueError("support threshold changed")
    weights = np.ones(an.shape[1]); outer_count = 0
    for outer in path.get("outer", []):
        if outer["status"] != "certified": continue
        lam = .001*float(np.max(np.abs(an.T@bn)/weights))
        if not np.isclose(lam, outer["lambda"], rtol=2e-13, atol=0): raise ValueError("outer lambda changed")
        beta = np.asarray(outer["beta"])
        if not certify_l1(an, bn, beta, outer["lambda"], weights)["certified"]:
            raise ValueError("claimed outer certificate not reproducible")
        if outer["support"] != np.flatnonzero(np.abs(beta)>SUPPORT_TOL).tolist(): raise ValueError("outer support changed")
        weights = 1./(np.abs(beta)+1e-8); outer_count += 1
    if path.get("outer") and outer_count == len(path["outer"]) and not np.array_equal(weights, np.asarray(path["weights"])):
        raise ValueError("originalIRL1 weight update changed")
    weights = np.asarray(path["weights"])
    if method == "L1" and not np.array_equal(weights, np.ones_like(weights)): raise ValueError("ordinary L1 weighted")
    lmax = float(np.max(np.abs(an.T@bn)/weights)); certified_count = 0; supports = {}
    for index, (point, ratio) in enumerate(zip(path["points"], DEFAULT_RATIOS)):
        if point["ratio"] != ratio: raise ValueError("lambda order changed")
        if point["status"] != "certified": continue
        if not np.isclose(point["lambda"], ratio*lmax, rtol=2e-13, atol=0): raise ValueError("final lambda changed")
        beta = np.asarray(point["beta"])
        if not certify_l1(an, bn, beta, point["lambda"], weights)["certified"]:
            raise ValueError("claimed final certificate not reproducible")
        support = np.flatnonzero(np.abs(beta)>SUPPORT_TOL).tolist()
        if point["support"] != support: raise ValueError("L1 support changed")
        supports[index] = support; certified_count += 1
    complete = certified_count == 24 and outer_count == len(path.get("outer", []))
    if bool(path["all_solutions_certified"]) != complete: raise ValueError("path numerical completion flag mismatch")
    candidates = evaluation["candidates"]
    by_id = {c["candidate_id"]:c for c in candidates}
    if len(by_id) != len(candidates): raise ValueError("duplicate candidate identifiers")
    represented = {}
    for mapping in evaluation["point_to_candidate"]:
        index = mapping["point_index"]
        if index not in supports or by_id[mapping["candidate_id"]]["support_indices"] != supports[index]:
            raise ValueError("support-to-candidate mapping differs from L1")
        represented[index] = True
    if set(represented) != set(supports): raise ValueError("certified point omitted from selection")
    for c in candidates:
        support, coefficient = c["support_indices"], np.asarray(c["coefficients"])
        error = float(np.linalg.norm(av[:,support]@coefficient-bv)/np.linalg.norm(bv))
        train_error = float(np.linalg.norm(a[:,support]@coefficient-b)/np.linalg.norm(b))
        if (not np.isclose(error, c["validation_error"], rtol=1e-11, atol=1e-12)
                or not np.isclose(train_error, c["train_equation_relative_error"], rtol=1e-11, atol=1e-12)
                or c["complexity"] != sum(costs[i] for i in support) or c["support"] != [names[i] for i in support]
                or c["dataset_id"] != dataset_id or c["ls_readout_changes_support"]
                or c["support_source"] != "normalized_L1_beta_only"):
            raise ValueError("candidate residual/cost/support/data mismatch")
    selected = min(candidates, key=lambda c:(c["validation_error"]+.005*c["complexity"], c["complexity"],
                     c["validation_error"], tuple(c["support_indices"]))) if candidates else None
    selected_id = selected["candidate_id"] if selected is not None else None
    if (evaluation["primary_selected_candidate_id"] != (selected_id if complete else None)
            or evaluation["selection_eligible"] != complete
            or evaluation["exploratory_best_available_candidate_id"] != selected_id
            or evaluation["selection_rule"]["complexity_coefficient"] != .005):
        raise ValueError("fixed selector or partial policy changed")
    return {"method":method, "requested_final_points":24, "certified_final_points":certified_count,
            "certified_outer_points":outer_count, "numerically_complete":complete,
            "selection_and_partial_policy_reproduced":True, "costs_and_L1_support_reverified":True,
            "primary_selected_candidate_id":evaluation["primary_selected_candidate_id"]}


def scan_pilot(path):
    initial_hash = digest(path); report = read(path)
    check_hashes(report["source_sha256"]); check_hashes(report["gate_sources"])
    if (report["methods"] != ["L1", "IRL1_original"] or report["lambda_ratios"] != DEFAULT_RATIOS.tolist()
            or report["alpha_frozen"] != .005 or report["support_tolerance"] != SUPPORT_TOL or report["test_data_used"]):
        raise ValueError("frozen pilot statistical protocol changed")
    rows = []
    for r in report["rows"]:
        matrices, targets, meta, costs, canonical = read_matrices(r["case"], r["sigma"])
        a,b = stack_split(matrices, targets, range(4)); av,bv = stack_split(matrices, targets, range(4,6))
        an,bn,scales = normalize_training(a,b)  # Same contiguous layout as this pilot, not historical subset indexing.
        inputs = {"train_matrix":array_hash(a), "train_target":array_hash(b),
                  "validation_matrix":array_hash(av), "validation_target":array_hash(bv)}
        lib = r["layers"]["J1212"]
        if (inputs != r["input_hashes"] or r["library_names"] != meta["names"] or r["library_complexities"] != costs
                or r["canonical_library_names"] != canonical or scales != lib["normalization"]
                or lib["solver_input_hashes"] != {"A":array_hash(an),"b":array_hash(bn)}
                or r["splits"]["train_trajectories"] != [0,1,2,3] or r["splits"]["validation_trajectories"] != [4,5]
                or r["truth_used_for_selection"] or r["test_data_used"]):
            raise ValueError("saved pilot input/normalization/split mismatch")
        for method, record in lib["methods"].items():
            if method not in ("L1", "IRL1_original"): raise ValueError("undeclared method")
            source_path = ROOT/record["path_source_file"]
            if digest(source_path) != record["path_source_sha256"]: raise ValueError("saved path source changed")
            rows.append({"case":r["case"], "sigma":r["sigma"], **review_method(record,method,a,b,av,bv,an,bn,
                         meta["names"],costs,r["dataset_id"])})
    unchanged = digest(path) == initial_hash
    return {"source_file":str(path), "source_sha256":initial_hash, "source_unchanged_during_review":unchanged,
            "rows":rows, "recorded_methods":len(rows), "campaign_execution_complete":report["execution_complete"],
            "all_recorded_results_consistent":unchanged,
            "all36_paths_numerically_complete":unchanged and len(rows)==36 and all(r["numerically_complete"] for r in rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT/"stage2c_review_result.json")
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("preserve existing review; select a new versioned output")
    start = time.perf_counter()
    with threadpool_limits(limits=1):
        matrices = preflight()
        pilot = scan_pilot(args.pilot) if args.pilot else None
    result = {"stage":"2c_independent_read_only_review", "all_checks_passed":matrices["all_checks_passed"]
              and (pilot is None or pilot["all_recorded_results_consistent"]), "matrix_preflight":matrices,
              "pilot_review":pilot, "campaign_completion_gate":bool(pilot and pilot["all36_paths_numerically_complete"]),
              "source_script_sha256":digest(Path(__file__)), "new_L1_solver_calls":0, "new_data_generated":False,
              "test_data_used":False, "elapsed_s":time.perf_counter()-start,
              "completion_meaning":"Audit consistency is distinct from36-path numerical completion or support correctness.",
              "limits":["Candidate-family expansion changes the design, not the L1/originalIRL1 algorithms.",
                  "Validation residuals are recomputed from saved fixed-support LS coefficients; LS never changes support.",
                  "Partial paths remain without a primary model; successful audit is not noise-robustness success."]}
    strict_json_write(args.output,result)
    print({"all_checks_passed":result["all_checks_passed"], "campaign_completion_gate":result["campaign_completion_gate"],
           "elapsed_s":result["elapsed_s"]})


if __name__ == "__main__": main()
