"""Reconcile a certified numerical retry into a separate frozen-rule baseline.

Original baseline/path/retry files are read only. This does not solve L1 again,
change the lambda grid, select a new alpha, or retune other methods. Time spent
on the original timeout, polish attempt, retry, and reconciliation is retained.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import stage2a_baseline as B
from stage0_evaluation import (SUPPORT_TOL, array_hash, normalize_training,
                               pareto_front, physical_coefficients, strict_json_write)
from stage0_l1_solver import certify_l1
from stage2_identifiability import save_matrices, stack_split

ROOT = Path(__file__).resolve().parent


def _load(path):
    def reject(token):
        raise ValueError("non-finite JSON token " + token)
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reconcile_baseline(source_path, retry_path, output_path, *, case, sigma, layer,
                       method="L1", seed=0, polish_path=None):
    """Reuse for another dataset only with its corresponding verified retry.

    The retry schema must contain input_hashes and a complete original-grid
    path. Its source_result points to the preceding polish result, whose
    original_reference identifies the first attempted path and its timing.
    """
    start = time.perf_counter()
    source_path, retry_path, output_path = map(Path, (source_path,retry_path,output_path))
    original, retry = _load(source_path), _load(retry_path)
    polish_path = Path(polish_path) if polish_path else retry_path.parent/retry["source_result"]["file"]
    polish = _load(polish_path)
    original_path_file = ROOT/polish["original_reference"]["file"]
    sources = [source_path, retry_path, polish_path, original_path_file]
    if output_path.resolve() in {p.resolve() for p in sources}:
        raise ValueError("reconciliation output must be a new file, not any source")
    hashes = {str(p.resolve()): _hash(p) for p in sources}
    if (_hash(polish_path) != retry["source_result"]["sha256"]
            or _hash(original_path_file) != polish["original_reference"]["sha256"]):
        raise ValueError("retry provenance hashes do not match source files")
    if (original["alpha_frozen"] != B.FROZEN_ALPHA
            or original["alpha_sensitivity"] != B.SENSITIVITY_ALPHAS):
        raise ValueError("current evaluator does not match the frozen selection rule")
    matches = [i for i,r in enumerate(original["rows"]) if (r["case"],r["sigma"],r["seed"])==(case,sigma,seed)]
    if len(matches) != 1:
        raise ValueError("requested baseline dataset is missing or ambiguous")
    row_index = matches[0]
    report = copy.deepcopy(original)
    row = report["rows"][row_index]
    layer_record = row["layers"][layer]
    old_method = copy.deepcopy(layer_record["methods"][method])
    old_path = old_method["path"]
    path = copy.deepcopy(retry["path"])
    if old_path != _load(original_path_file):
        # Baseline adds physical-coefficient readouts after saving worker paths.
        without_readouts = copy.deepcopy(old_path)
        for point in without_readouts["points"]:
            point.pop("physical_L1_coefficients", None)
        if without_readouts != _load(original_path_file):
            raise ValueError("retry original path does not belong to this baseline method")
    if (not retry["all_checks_passed"] or not path["all_solutions_certified"]
            or path["status"] != "complete" or path["method"] != method
            or path["ratios"] != old_path["ratios"] or path["weights"] != old_path["weights"]
            or path["support_tolerance"] != SUPPORT_TOL):
        raise ValueError("retry changed method/grid/weights/support convention or is incomplete")
    cache = ROOT/"stage2_matrices"/f"{case}_sigma_{repr(float(sigma)).replace('.', 'p')}_seed_{seed}.npz"
    if not cache.exists():
        raise ValueError("cached matrices required: reconciliation may not regenerate data")
    matrices, targets, metadata = save_matrices(case,sigma,seed)
    a,b = stack_split(matrices,targets,range(4))
    av,bv = stack_split(matrices,targets,range(4,6))
    indices = layer_record["indices_in_maximum_library"]
    at,va = a[:,indices], av[:,indices]
    an,bn,scales = normalize_training(at,b)
    expected_hashes = {"A":array_hash(an),"b":array_hash(bn)}
    if (expected_hashes != retry["input_hashes"] or expected_hashes != polish["input_hashes"]
            or expected_hashes != layer_record["solver_input_hashes"]
            or array_hash(av) != row["input_hashes"]["validation_matrix"]
            or array_hash(bv) != row["input_hashes"]["validation_target"]
            or scales != layer_record["normalization"]):
        raise ValueError("retry training input or validation/scaling contract changed")
    weights = np.asarray(path["weights"])
    lambda_max = float(np.max(np.abs(an.T@bn)/weights))
    old_tolerances = next(p["certificate"]["tolerances"] for p in old_path["points"]
                          if p["status"]=="certified")
    if len(path["points"]) != len(path["ratios"]):
        raise ValueError("incomplete retry lambda list")
    recertified = []
    for point, ratio in zip(path["points"],old_path["ratios"]):
        beta = np.asarray(point["beta"])
        if (point["ratio"] != ratio or point["status"] != "certified"
                or not np.isclose(point["lambda"],ratio*lambda_max,rtol=1e-14,atol=0)
                or point["support"] != np.flatnonzero(np.abs(beta)>SUPPORT_TOL).tolist()
                or point["certificate"]["tolerances"] != old_tolerances):
            raise ValueError("retry point changed lambda/support/certification rules")
        cert = certify_l1(an,bn,beta,point["lambda"],weights)
        if not cert["certified"] or cert["tolerances"] != old_tolerances:
            raise ValueError("retry point failed independent original-tolerance recertification")
        recertified.append({"ratio":ratio,"certificate":cert})
        point["physical_L1_coefficients"] = physical_coefficients(beta,scales).tolist()
    validation_blocks = [(i,matrices[i][:,indices],targets[i]) for i in (4,5)]
    eval_start = time.perf_counter()
    evaluation = B._candidate_records(path,at,b,va,bv,layer_record["names"],layer_record["complexities"],
                                      row["dataset_id"],validation_blocks)
    evaluation_s = time.perf_counter()-eval_start
    initial_s = float(old_path["elapsed_s"])
    polish_s = float(polish["path"]["elapsed_s"])
    retry_s = float(path["retry_elapsed_s"])
    historical_solver_s = initial_s+polish_s+retry_s
    timing = {"original_attempt_status":old_path["status"],"original_attempt_budget_s":old_path["budget_s"],
              "original_attempt_elapsed_s":initial_s,"polish_attempt_elapsed_s":polish_s,
              "retry_only_elapsed_s":retry_s,"additional_recovery_solver_elapsed_s":polish_s+retry_s,
              "total_historical_solver_elapsed_s":historical_solver_s,
              "interpretation":"retry-only duration is not a full-path runtime; original timeout remains part of the historical cost"}
    new_method = {**old_method,"path":path,"evaluation":evaluation,
        "time_s":{**old_method["time_s"],"solver_with_worker_startup":historical_solver_s,
                  "original_fixed_support_validation":old_method["time_s"]["fixed_support_validation"],
                  "fixed_support_validation":evaluation_s},
        "numerical_reconciliation":{"timing":timing,"recertified_points":recertified,
            "original_primary_selected_candidate_id":old_method["evaluation"]["primary_selected_candidate_id"],
            "old_timeout_not_reclassified":True,"same_statistical_method":True,
            "selection_rule_unchanged":True,"no_solver_rerun_in_reconciliation":True},
        "timing_note":"Total solver timing includes original timeout plus polish plus retry; field-generation cost is still unavailable."}
    layer_record["methods"][method] = new_method
    # Only the numerical fields used by Pareto are copied; existing truth labels
    # from the source are not inputs to this recomputation.
    common = []
    for lname,lib in row["layers"].items():
        for mname,m in lib["methods"].items():
            if not m["path"]["all_solutions_certified"]:
                continue
            for candidate in m["evaluation"]["candidates"]:
                common.append({k:candidate[k] for k in ("candidate_id","support","complexity","validation_error","dataset_id")}
                              | {"layer":lname,"method":mname})
    row["common_validation_pareto"]={"dataset_id":row["dataset_id"],"partial_methods_excluded":True,
        "points":[{k:c[k] for k in ("layer","method","candidate_id","support","complexity","validation_error")}
                  for c in pareto_front(common)]}
    # True support/coefficients are first accessed after fixed-rule selection.
    B._add_truth_metrics(evaluation,row["truth"])
    row["all_methods_numerically_complete"] = all(m["path"]["all_solutions_certified"]
        for lib in row["layers"].values() for m in lib["methods"].values())
    elapsed_s = time.perf_counter()-start
    row["time_s"]["original_dataset_total"] = row["time_s"]["dataset_total"]
    row["time_s"]["numerical_recovery_solver_additional"] = polish_s+retry_s
    row["time_s"]["reconciliation_this_run"] = elapsed_s
    row["time_s"]["dataset_total"] += polish_s+retry_s+elapsed_s
    report["all_checks_passed"] = all(r["all_methods_numerically_complete"] for r in report["rows"])
    report["execution_complete"] = True
    report["reconciliation"]={"source_baseline":str(source_path),"retry_result":str(retry_path),
        "polish_result":str(polish_path),"source_hashes":hashes,
        "selector":{"case":case,"sigma":sigma,"seed":seed,"layer":layer,"method":method},
        "timing":timing,"source_script_sha256":_hash(Path(__file__)),
        "completion_meaning":"Numerical completeness after additional attempts; original budget failure preserved, not statistical superiority."}
    unaffected = all(report["rows"][i] == original["rows"][i] for i in range(len(original["rows"])) if i != row_index)
    other_methods_unchanged = all(m == original["rows"][row_index]["layers"][lname]["methods"][mname]
        for lname,lib in row["layers"].items() for mname,m in lib["methods"].items()
        if (lname,mname) != (layer,method))
    if not unaffected or not other_methods_unchanged or any(_hash(p)!=digest for p,digest in hashes.items()):
        raise ValueError("unrelated rows/methods or source files changed")
    strict_json_write(output_path,report)
    selected = evaluation["primary_selected_candidate_id"]
    candidate = next(c for c in evaluation["candidates"] if c["candidate_id"]==selected)
    summary={"all_checks_passed":True,"output":str(output_path),"recertified_points":len(recertified),
        "unaffected_rows_unchanged":unaffected,"other_methods_unchanged":other_methods_unchanged,
        "source_files_unchanged":True,"fixed_rule":evaluation["selection_rule"],
        "selected_support":candidate["support"],"coefficients":candidate["coefficients"],
        "validation_error":candidate["validation_error"],"timing":timing,
        "baseline_all_methods_numerically_complete":report["all_checks_passed"],
        "truth_not_used_for_reselection":True,"new_solver_work_performed":False}
    return summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",type=Path,default=ROOT/"stage2a_heat_noise_pilot_result.json")
    ap.add_argument("--retry",type=Path,default=ROOT/"stage2a_solver_retry_result.json")
    ap.add_argument("--output",type=Path,default=ROOT/"stage2a_heat_reconciled_result.json")
    ap.add_argument("--case",default="heat")
    ap.add_argument("--sigma",type=float,default=.05)
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--layer",default="J1158")
    ap.add_argument("--method",default="L1")
    args=ap.parse_args()
    with threadpool_limits(limits=1):
        result=reconcile_baseline(args.source,args.retry,args.output,case=args.case,sigma=args.sigma,
                                  seed=args.seed,layer=args.layer,method=args.method)
    strict_json_write(ROOT/"stage2a_reconcile_result.json",result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=="__main__":
    main()
