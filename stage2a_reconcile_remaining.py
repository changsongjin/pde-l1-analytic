"""Merge generic numerical continuations without rerunning successful methods.

The original baseline is copied once. Only complete recovered paths are scored
again under the original fixed validation rule. Partial recovered paths retain
their original ineligible evaluation and updated numerical progress. Final
outputs are write-once: use a new output name for a later continuation.
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
from stage0_evaluation import (SUPPORT_TOL,array_hash,normalize_training,pareto_front,
                               physical_coefficients,strict_json_write)
from stage0_l1_solver import certify_l1
from stage2_identifiability import save_matrices,stack_split

ROOT=Path(__file__).resolve().parent


def _load(path):
    def reject(value):
        raise ValueError("non-finite JSON: "+value)
    return json.loads(Path(path).read_text(encoding="utf-8"),parse_constant=reject)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _identity(row):
    return row["case"],row["sigma"],row["seed"],row["layer"],row["method"]


def _validate_path(path,original,an,bn):
    if (path["method"]!=original["method"] or path["shape"]!=list(an.shape)
            or path["ratios"]!=original["ratios"] or path["weights"]!=original["weights"]
            or path["support_tolerance"]!=SUPPORT_TOL or len(path["points"])!=len(original["points"])):
        raise ValueError("continuation changed objective/grid/weights/shape/support convention")
    weights=np.asarray(path["weights"])
    lambda_max=float(np.max(np.abs(an.T@bn)/weights))
    tolerances=next(p["certificate"]["tolerances"] for p in original["points"] if p["status"]=="certified")
    certificates=[]
    inherited=0
    for point,old,ratio in zip(path["points"],original["points"],original["ratios"]):
        if point["ratio"]!=ratio or point["lambda"] is None or not np.isclose(point["lambda"],lambda_max*ratio,rtol=1e-14,atol=0):
            raise ValueError("requested lambda is missing or changed")
        if old["status"]=="certified":
            if point!=old:
                raise ValueError("previously certified point was modified by a generic continuation")
            inherited+=1
        if point["status"]!="certified":
            certificate = (certify_l1(an,bn,np.asarray(point["beta"]),point["lambda"],weights)
                           if point.get("beta") is not None else None)
            certificates.append({"ratio":ratio,"status":point["status"],"certified":False,
                                 "independent_certificate":certificate,
                                 "no_automatic_promotion_of_incomplete_point":True})
            continue
        beta=np.asarray(point["beta"])
        if (point["support"]!=np.flatnonzero(np.abs(beta)>SUPPORT_TOL).tolist()
                or point["certificate"]["tolerances"]!=tolerances):
            raise ValueError("support or original certificate tolerance changed")
        certificate=certify_l1(an,bn,beta,point["lambda"],weights)
        if not certificate["certified"] or certificate["tolerances"]!=tolerances:
            raise ValueError("independent certification failed at original tolerances")
        certificates.append({"ratio":ratio,"status":"certified","certificate":certificate,"certified":True})
    complete=all(c["certified"] for c in certificates)
    if bool(path["all_solutions_certified"])!=complete or (complete and path["status"]!="complete"):
        raise ValueError("path completion flags disagree with point certifications")
    return complete,certificates,inherited


def _common_frontier(row):
    candidates=[]
    for layer,lib in row["layers"].items():
        for method,result in lib["methods"].items():
            if not result["path"]["all_solutions_certified"]:
                continue
            for candidate in result["evaluation"]["candidates"]:
                candidates.append({key:candidate[key] for key in
                    ("candidate_id","support","complexity","validation_error","dataset_id")}
                    |{"layer":layer,"method":method})
    return {"dataset_id":row["dataset_id"],"partial_methods_excluded":True,
        "points":[{key:candidate[key] for key in
            ("layer","method","candidate_id","support","complexity","validation_error")}
            for candidate in pareto_front(candidates)]}


def _verify_report_sources(continuation_path,report,baseline_path,known_sources):
    """Every continuation must bind the original baseline and prior attempts."""
    if not report.get("execution_complete"):
        raise ValueError("continuation attempts are still running")
    if report.get("test_data_used",False) or not report.get("no_true_support_used",True):
        raise ValueError("continuation violates development-only/no-truth solver contract")
    if report.get("source_file")==baseline_path.name:
        if report.get("source_sha256")!=_hash(baseline_path):
            raise ValueError("original baseline provenance mismatch")
    elif report.get("source_file") in {p.name for p in known_sources}:
        previous=next(p for p in known_sources if p.name==report["source_file"])
        if report.get("source_sha256")!=_hash(previous):
            raise ValueError("preceding continuation provenance mismatch")
    else:
        raise ValueError("continuation must name original baseline or verified previous continuation")
    verified=[]
    # Known v2/v3 predecessor manifests use paired *_file/*_sha256 fields.
    for key,value in report.items():
        if key.endswith("_file") and key!="source_file" and isinstance(value,str):
            hash_key=key[:-5]+"_sha256"
            if hash_key in report:
                path=continuation_path.parent/value
                if _hash(path)!=report[hash_key]:
                    raise ValueError("continuation predecessor hash mismatch: "+value)
                verified.append({"file":str(path),"sha256":report[hash_key]})
    return verified


def reconcile_remaining(source_path,continuation_paths,output_path):
    start=time.perf_counter()
    source_path,output_path=Path(source_path),Path(output_path)
    continuation_paths=list(map(Path,continuation_paths))
    if output_path.exists():
        raise ValueError("final output already exists; choose a new version rather than overwrite")
    original=_load(source_path)
    if original["alpha_frozen"]!=B.FROZEN_ALPHA or original["alpha_sensitivity"]!=B.SENSITIVITY_ALPHAS:
        raise ValueError("selection constants differ from original baseline")
    sources={str(source_path.resolve()):_hash(source_path)}
    latest,history={},{}
    known=[]
    predecessor_audits=[]
    for path in continuation_paths:
        report=_load(path)
        predecessor_audits.extend(_verify_report_sources(path,report,source_path,known))
        sources[str(path.resolve())]=_hash(path)
        for row in report["rows"]:
            identity=_identity(row)
            latest[identity]=row
            history.setdefault(identity,[]).append({"file":str(path),"sha256":_hash(path),
                "metadata_without_path":{key:value for key,value in row.items() if key!="path"}})
        known.append(path)
    output=copy.deepcopy(original)
    summaries,changed_methods,changed_rows,truth_pending=[],set(),set(),[]
    for identity,continuation in latest.items():
        case,sigma,seed,layer,method=identity
        indices=[i for i,row in enumerate(output["rows"]) if (row["case"],row["sigma"],row["seed"])==(case,sigma,seed)]
        if len(indices)!=1:
            raise ValueError("continuation dataset missing or ambiguous in original baseline")
        row_index=indices[0]
        row=output["rows"][row_index]
        library=row["layers"][layer]
        old=original["rows"][row_index]["layers"][layer]["methods"][method]
        if old["path"]["all_solutions_certified"]:
            raise ValueError("successful original method must not be replaced")
        cache=ROOT/"stage2_matrices"/f"{case}_sigma_{repr(float(sigma)).replace('.', 'p')}_seed_{seed}.npz"
        if not cache.exists():
            raise ValueError("cached matrices required; no matrix generation in reconciliation")
        matrices,targets,meta=save_matrices(case,sigma,seed)
        a,b=stack_split(matrices,targets,range(4))
        av,bv=stack_split(matrices,targets,range(4,6))
        subset=library["indices_in_maximum_library"]
        at,va=a[:,subset],av[:,subset]
        an,bn,scales=normalize_training(at,b)
        hashes={"A":array_hash(an),"b":array_hash(bn)}
        if (hashes!=continuation["input_hashes"] or hashes!=library["solver_input_hashes"]
                or scales!=library["normalization"]
                or array_hash(av)!=row["input_hashes"]["validation_matrix"]
                or array_hash(bv)!=row["input_hashes"]["validation_target"]
                or meta["raw_bundle_sha256"]!=row["raw_bundle_sha256"]):
            raise ValueError("training/validation/scaling/raw-input contract mismatch")
        path=copy.deepcopy(continuation["path"])
        complete,certificates,inherited=_validate_path(path,old["path"],an,bn)
        if complete:
            begin=time.perf_counter()
            evaluation=B._candidate_records(path,at,b,va,bv,library["names"],library["complexities"],
                row["dataset_id"],[(i,matrices[i][:,subset],targets[i]) for i in (4,5)])
            evaluation_s=time.perf_counter()-begin
            truth_pending.append((row_index,evaluation))
        else:
            evaluation=copy.deepcopy(old["evaluation"])
            evaluation_s=0.
            if evaluation["primary_selected_candidate_id"] is not None or evaluation["selection_eligible"]:
                raise ValueError("partial original method unexpectedly has a primary model")
        for point in path["points"]:
            if point.get("beta") is not None and "physical_L1_coefficients" not in point:
                point["physical_L1_coefficients"]=physical_coefficients(point["beta"],scales).tolist()
        timing=_timing_costs(continuation,history[identity],old)
        replacement={**copy.deepcopy(old),"path":path,"evaluation":evaluation,
            "time_s":{**old["time_s"],"solver_with_worker_startup":timing["historical_solver_cost_total_s"],
                "original_fixed_support_validation":old["time_s"]["fixed_support_validation"],
                "fixed_support_validation":evaluation_s if complete else old["time_s"]["fixed_support_validation"]},
            "numerical_reconciliation":{"complete":complete,"continuation_sources":history[identity],
                "independent_certificates":certificates,"original_certified_points_unchanged":inherited,
                "timing":timing,"fixed_alpha_unchanged":True,"same_validation_data":True,
                "partial_evaluation_policy":"Original ineligible evaluation preserved; newly certified partial points are not rescored.",
                "no_solver_rerun_in_reconciliation":True},
            "timing_note":"Historical original timeout, interface failures, and every numerical retry remain in solver cost; this is not the runtime of one fresh complete path."}
        library["methods"][method]=replacement
        changed_methods.add((row_index,layer,method));changed_rows.add(row_index)
        row["time_s"]["original_dataset_total"]=original["rows"][row_index]["time_s"]["dataset_total"]
        row["time_s"]["numerical_recovery_solver_additional"]=timing["historical_solver_cost_total_s"]-old["path"]["elapsed_s"]
        row["time_s"]["dataset_total"]=row["time_s"]["original_dataset_total"]+row["time_s"]["numerical_recovery_solver_additional"]+evaluation_s
        summaries.append({"case":case,"sigma":sigma,"seed":seed,"layer":layer,"method":method,
            "numerically_complete":complete,"certified_points":sum(c["certified"] for c in certificates),
            "requested_points":len(certificates),"primary_selected_candidate_id":evaluation["primary_selected_candidate_id"],
            "selected_support":next((c["support"] for c in evaluation["candidates"]
                if c["candidate_id"]==evaluation["primary_selected_candidate_id"]),None),"timing":timing})
    for index in changed_rows:
        row=output["rows"][index]
        row["common_validation_pareto"]=_common_frontier(row)
        row["all_methods_numerically_complete"]=all(result["path"]["all_solutions_certified"]
            for lib in row["layers"].values() for result in lib["methods"].values())
    # Truth read only after all complete-path selections and common frontiers.
    for row_index,evaluation in truth_pending:
        B._add_truth_metrics(evaluation,output["rows"][row_index]["truth"])
    unchanged_rows=all(row==original["rows"][i] for i,row in enumerate(output["rows"]) if i not in changed_rows)
    unchanged_methods=all(result==original["rows"][i]["layers"][layer]["methods"][method]
        for i,row in enumerate(output["rows"]) for layer,lib in row["layers"].items()
        for method,result in lib["methods"].items() if (i,layer,method) not in changed_methods)
    source_unchanged=all(_hash(path)==digest for path,digest in sources.items())
    if not all((unchanged_rows,unchanged_methods,source_unchanged)):
        raise ValueError("unrelated data/methods or source files changed")
    output["all_checks_passed"]=all(row["all_methods_numerically_complete"] for row in output["rows"])
    output["execution_complete"]=True
    output["numerical_reconciliation"]={"sources":sources,"predecessor_file_checks":predecessor_audits,
        "rows":summaries,"elapsed_s":time.perf_counter()-start,"source_script_sha256":_hash(Path(__file__)),
        "integrity_checks_passed":True,"unaffected_rows_unchanged":unchanged_rows,
        "successful_and_unaffected_methods_unchanged":unchanged_methods,
        "original_source_files_unchanged":source_unchanged,
        "completion_meaning":"Complete paths may be scored; partial paths remain ineligible. Historical timeouts are numerical outcomes, not statistical recovery failures."}
    strict_json_write(output_path,output)
    return {"all_checks_passed":True,"reconciliation_integrity_passed":True,
        "all_baseline_paths_numerically_complete":output["all_checks_passed"],"output":str(output_path),
        "rows":summaries,"source_files_unchanged":source_unchanged,
        "successful_methods_unchanged":unchanged_methods,"unaffected_rows_unchanged":unchanged_rows,
        "elapsed_s":time.perf_counter()-start,"no_solver_rerun":True,
        "pass_meaning":"Reconciliation integrity, not universal numerical completion or paper readiness."}


def _timing_costs(latest,history,old):
    # Generic v2/v3 share original and historical cumulative costs. Preserve
    # every source row's detailed costs alongside these non-overlapping totals.
    original=float(latest["original_path_seconds"])
    interface=float(latest["previous_interface_failure_seconds"])
    # v3 preserves the v2 historical total and adds its own duration separately;
    # path.elapsed_s includes both. Never lose the v3 effort when reconciling.
    historical=float(latest["historical_solver_cost_total_s"])+float(latest.get("v3_retry_seconds",0.))
    if not np.isclose(original,old["path"]["elapsed_s"],rtol=0,atol=1e-8):
        raise ValueError("historical original timeout cost changed")
    attempts=[]
    previous=original+interface
    for entry in history:
        row=entry["metadata_without_path"]
        base=float(row["historical_solver_cost_total_s"])
        extra=float(row.get("v3_retry_seconds",0.))
        cumulative=base+extra
        increment=cumulative-previous
        if increment < -1e-8:
            raise ValueError("retry historical costs decreased")
        attempts.append({"source":entry["file"],"additional_attempt_elapsed_s":max(0.,increment),
                         "source_reported_v2_historical_total_s":base,
                         "source_reported_v3_retry_seconds":extra,
                         "resolved_cumulative_historical_total_s":cumulative})
        previous=cumulative
    if abs(previous-historical)>1e-8:
        raise ValueError("historical retry cost mismatch")
    if abs(float(latest["path"]["elapsed_s"])-historical)>1e-8:
        raise ValueError("path elapsed time disagrees with complete historical cost")
    return {"original_attempt_status":old["path"]["status"],"original_path_seconds":original,
        "previous_interface_failure_seconds":interface,"continuation_attempts":attempts,
        "historical_solver_cost_total_s":historical,
        "total_new_numerical_work_s":historical-original,
        "interpretation":"Sum of all historical attempts; not one fresh-path timing."}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",type=Path,default=ROOT/"stage2a_other_cases_pilot_result.json")
    ap.add_argument("--continuations",nargs="+",type=Path,default=[ROOT/"stage2a_retry_remaining_v2_result.json",ROOT/"stage2a_retry_remaining_v3_result.json"])
    ap.add_argument("--output",type=Path,default=ROOT/"stage2a_other_cases_reconciled_result.json")
    ap.add_argument("--review-output",type=Path,default=ROOT/"stage2a_reconcile_remaining_result.json")
    args=ap.parse_args()
    if args.review_output.exists():
        raise ValueError("review output exists; choose a new name")
    with threadpool_limits(limits=1):
        result=reconcile_remaining(args.source,args.continuations,args.output)
    strict_json_write(args.review_output,result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=="__main__":
    main()
