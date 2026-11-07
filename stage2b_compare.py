"""Frozen single-factor IRL1 development comparison; resumable, one worker.

Reuse completed historical controls after checking identical solver/validation
inputs. New-cohort controls and the candidate each get the same 120 s initial
wall budget. Numerical failures remain explicit and are not statistical losses.
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
import stage2b_data as NEW
from stage0_evaluation import (SUPPORT_TOL,array_hash,normalize_training,physical_coefficients,
                               strict_json_write)
from stage0_l1_solver import certify_l1
from stage2_data import CASES
from stage2_identifiability import save_matrices as old_matrices, stack_split
from stage2a_l1_paths import DEFAULT_RATIOS,solve_l1_path,solve_original_irl1
from stage2a_reconcile_remaining import _common_frontier
from stage2b_smoothed_irl1 import solve_smoothed_irl1


ROOT=Path(__file__).resolve().parent
OUTPUT=ROOT/"stage2b_compare_result.json"
PROGRESS=ROOT/"stage2b_progress"
COHORTS=("development_ids0_5_seed0","independent_development_ids12_17_seed1")
METHODS=("L1","IRL1_original","IRL1_smoothed")
METHOD_SECONDS=120.


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    def reject(token):raise ValueError("nonfinite JSON: "+token)
    return json.loads(Path(path).read_text(encoding="utf-8"),parse_constant=reject)


def require_gates():
    names=["stage2a_batch_checkpoint_v2_result.json","stage2b_protocol_result.json",
           "stage2b_data_result.json","stage2b_smoothed_irl1_result.json"]
    reports={name:load(ROOT/name) for name in names}
    if not all(r["all_checks_passed"] for r in reports.values()):
        raise RuntimeError("required numerical/protocol/data/API gate has not passed")
    checkpoint=reports[names[0]]
    if checkpoint["recorded_runs"]!=72 or checkpoint["numerically_complete_runs"]!=72:
        raise RuntimeError("stage2a full numerical completion required")
    protocol=reports[names[1]]
    if (protocol["selection_alpha"]!=B.FROZEN_ALPHA or protocol["epsilon_relative"]!=.01
            or protocol["epsilon_floor"]!=1e-8 or protocol["per_method_initial_wall_budget_s"]!=METHOD_SECONDS
            or protocol["methods"]!=list(METHODS) or protocol["support_tolerance"]!=SUPPORT_TOL):
        raise ValueError("runner does not match the frozen protocol")
    if reports[names[2]]["protocol_sha256"]!=digest(ROOT/names[1]):
        raise ValueError("new data belongs to a different protocol")
    return checkpoint,protocol,{name:digest(ROOT/name) for name in names}


def control_rows(checkpoint,protocol):
    wanted={(case,float(sigma),0) for case,sigma in protocol["cases_and_noise"]}
    rows={};source_hashes={}
    for item in checkpoint["sources"]:
        path=ROOT/item["file"]
        if digest(path)!=item["sha256"]:raise ValueError("baseline source hash changed")
        source=load(path)
        source_hashes[path.name]=item["sha256"]
        if path.name!="stage2a_reconcile_complete_result.json":continue
        if source["stage"]!="2a_development_baseline":raise ValueError("unexpected historical baseline schema")
        if source["alpha_frozen"]!=B.FROZEN_ALPHA:raise ValueError("baseline selection alpha changed")
        for row in source["rows"]:
            identity=(row["case"],float(row["sigma"]),row["seed"])
            if identity in wanted:
                if identity in rows:raise ValueError("ambiguous duplicated historical cohort")
                rows[identity]=(row,{"file":path.name,"sha256":item["sha256"],
                                    "historical_nominal_budgets_s":source["budgets_s"]})
    if set(rows)!=wanted:raise ValueError("missing historical controls")
    return rows,source_hashes


def validate_path(path,a,b):
    """No automatic promotion: certify every supplied successful point."""
    if path["shape"]!=list(a.shape) or path["support_tolerance"]!=SUPPORT_TOL:
        raise ValueError("path input/support convention mismatch")
    if path["ratios"]!=DEFAULT_RATIOS.tolist() or len(path["points"])!=24:
        raise ValueError("every frozen lambda must be represented")
    weights=np.asarray(path["weights"])
    lmax=float(np.max(np.abs(a.T@b)/weights))
    for point,ratio in zip(path["points"],DEFAULT_RATIOS):
        if point["ratio"]!=ratio:raise ValueError("lambda order changed")
        if point["status"]!="certified":continue
        if not np.isclose(point["lambda"],ratio*lmax,rtol=2e-13,atol=0):
            raise ValueError("lambda objective changed")
        beta=np.asarray(point["beta"])
        if point["support"]!=np.flatnonzero(np.abs(beta)>SUPPORT_TOL).tolist():
            raise ValueError("path support not computed from normalized L1 coefficients")
        if not certify_l1(a,b,beta,point["lambda"],weights)["certified"]:
            raise ValueError("a claimed successful point fails the unchanged certificate")
    complete=(len(path["points"])==24 and all(p["status"]=="certified" for p in path["points"])
              and all(p["status"]=="certified" for p in path.get("outer",[])))
    if bool(path["all_solutions_certified"])!=complete:
        raise ValueError("path completion flag inconsistent with its points")


def interrupted_path(path):
    """Preserve an interrupted checkpoint, never silently launch a fresh solve."""
    path=copy.deepcopy(path)
    done={p["ratio"] for p in path["points"]}
    lmax=path.get("path_metadata",{}).get("lambda_max") if path.get("path_metadata") else None
    for ratio in DEFAULT_RATIOS:
        if float(ratio) not in done:
            path["points"].append({"ratio":float(ratio),"lambda":float(ratio*lmax) if lmax is not None else None,
                "status":"interrupted","beta":None,"support":None,"certificate":None,"attempts":[],
                "error":"previous worker checkpoint incomplete; no automatic rerun"})
    path.update({"status":"interrupted","all_solutions_certified":False,
                 "error":"previous task interrupted; retained for explicit numerical continuation"})
    return path


def obtain_path(solver,a,b,manifest,tag):
    job=PROGRESS/(tag+"_job.json");path_file=PROGRESS/(tag+"_path_result.json")
    if job.exists():
        if load(job)!=manifest:raise ValueError("stored method job does not match current inputs/sources")
    else:strict_json_write(job,manifest)
    if path_file.exists():
        path=load(path_file)
        if path["status"]=="running":
            path=interrupted_path(path)
            preserved=PROGRESS/(tag+"_interrupted_result.json")
            if not preserved.exists():strict_json_write(preserved,path)
        return path,{"origin":"saved_method_checkpoint","file":str(path_file.relative_to(ROOT)),
                     "sha256":digest(path_file),"new_solver_seconds":0.}
    path=solver(a,b,ratios=DEFAULT_RATIOS,budget_s=METHOD_SECONDS,progress_path=path_file)
    return path,{"origin":"new_solver_call","file":str(path_file.relative_to(ROOT)),
                 "sha256":digest(path_file),"new_solver_seconds":path["elapsed_s"]}


def finish_row(row):
    ready=all(set(layer["methods"])==set(METHODS) for layer in row["layers"].values()) and len(row["layers"])==2
    row["all_methods_executed"]=ready
    row["all_methods_numerically_complete"]=ready and all(m["path"]["all_solutions_certified"]
        for layer in row["layers"].values() for m in layer["methods"].values())
    if not ready:return
    row["common_validation_pareto"]=_common_frontier(row)
    # Truth first enters new evaluations only after all fixed selections exist.
    for layer in row["layers"].values():
        for record in layer["methods"].values():
            if record["origin"]!="reused_stage2a_control":
                B._add_truth_metrics(record["evaluation"],CASES[row["case"]]["true"])
    row["truth"]=CASES[row["case"]]["true"]


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--max-seconds",type=float,default=2700.)
    args=parser.parse_args()
    if not 150<=args.max_seconds<=2700:raise ValueError("runner budget must be 150..2700 seconds")
    checkpoint,protocol,gates=require_gates()
    controls,control_hashes=control_rows(checkpoint,protocol)
    sources=["stage2b_compare.py","stage2b_data.py","stage2b_protocol.py","stage2b_smoothed_irl1.py",
             "stage2a_baseline.py","stage2a_l1_paths.py","stage0_l1_solver.py","stage0_evaluation.py",
             "stage2_data.py","stage2_identifiability.py","stage1_grammar.py","stage1_jet_preprocess.py",
             "stage1_weak_reduction.py","stage1_weak_reduction_v2.py","stage2a_reconcile_remaining.py","lasso_exact.py"]
    hashes={name:digest(ROOT/name) for name in sources}
    if OUTPUT.exists():
        report=load(OUTPUT)
        if report["source_sha256"]!=hashes or report["gate_sources"]!=gates or report["control_sources"]!=control_hashes:
            raise ValueError("saved runner sources changed; preserve result and create a new version")
    else:
        report={"stage":"2b_single_factor_development_comparison","source_sha256":hashes,"gate_sources":gates,
                "control_sources":control_hashes,"cases_and_noise":protocol["cases_and_noise"],
                "cohorts":list(COHORTS),"layers":protocol["layers"],"methods":list(METHODS),
                "alpha_frozen":B.FROZEN_ALPHA,"alpha_sensitivity":B.SENSITIVITY_ALPHAS,
                "lambda_ratios":DEFAULT_RATIOS.tolist(),"per_method_initial_wall_budget_s":METHOD_SECONDS,
                "validation_metric":B.METRIC,"rows":[],"batches":[],"execution_complete":False,"all_checks_passed":False,
                "test_data_used":False,"statistical_parameter_tuning_after_outcomes":False,
                "limits":["Selected development cases and one new cohort do not establish a general noise-robustness ranking.",
                          "Primary selection uses unchanged validation weak residual plus .005*complexity; not rollout accuracy.",
                          "Uncertified or unfinished paths have no primary selected model.",
                          "Historical control costs remain separate from new computation; reused controls are not zero-cost methods."]}
    PROGRESS.mkdir(exist_ok=True)
    batch={"index":len(report["batches"]),"wall_limit_s":args.max_seconds,"status":"running","elapsed_s":0.}
    report["batches"].append(batch);start=time.perf_counter()
    def save():
        batch["elapsed_s"]=time.perf_counter()-start
        observed={(r["cohort"],r["case"],r["sigma"],l,m)
                  for r in report["rows"] for l,lib in r["layers"].items() for m in lib["methods"]}
        report["pending_jobs"]=[{"cohort":c,"case":case,"sigma":sigma,"layer":layer,"method":method}
                                for c in COHORTS for case,sigma in protocol["cases_and_noise"]
                                for layer in protocol["layers"] for method in METHODS
                                if (c,case,sigma,layer,method) not in observed]
        report["uncertified_jobs"]=[{"cohort":r["cohort"],"case":r["case"],"sigma":r["sigma"],"layer":l,"method":m,
                                      "path_status":record["path"]["status"]}
                                     for r in report["rows"] for l,lib in r["layers"].items() for m,record in lib["methods"].items()
                                     if not record["path"]["all_solutions_certified"]]
        strict_json_write(OUTPUT,report)
    save();stopped=False
    with threadpool_limits(limits=1):
        for cohort in COHORTS:
            for case,sigma in protocol["cases_and_noise"]:
                seed=0 if cohort==COHORTS[0] else 1
                original,source=controls[(case,float(sigma),0)]
                begin=time.perf_counter()
                matrices,targets,meta=(old_matrices(case,sigma,0) if seed==0 else NEW.save_matrices(case,sigma,1))
                a,b=stack_split(matrices,targets,range(4));av,bv=stack_split(matrices,targets,range(4,6))
                names=tuple(meta["names"]);canonical,costs=B._symbolic_metadata(names)
                if list(costs)!=meta["complexities"]:raise ValueError("grammar/evaluation complexity mismatch")
                input_hashes={"train_matrix":array_hash(a),"train_target":array_hash(b),
                              "validation_matrix":array_hash(av),"validation_target":array_hash(bv)}
                if seed==0 and (input_hashes!=original["input_hashes"] or meta["raw_bundle_sha256"]!=original["raw_bundle_sha256"]):
                    raise ValueError("historical control observations/validation split changed")
                dataset_id=original["dataset_id"] if seed==0 else f"{cohort}:{case}:sigma={sigma!r}:validation={array_hash(bv)}"
                row=next((r for r in report["rows"] if (r["cohort"],r["case"],r["sigma"])==(cohort,case,sigma)),None)
                if row is None:
                    ids=list(range(6)) if seed==0 else list(range(12,18))
                    row={"cohort":cohort,"case":case,"sigma":sigma,"seed":seed,"dataset_id":dataset_id,
                         "raw_bundle_sha256":meta["raw_bundle_sha256"],"matrix_metadata":meta,"input_hashes":input_hashes,
                         "library_names":list(names),"canonical_library_names":list(canonical),"library_complexities":list(costs),
                         "splits":{"train_trajectories":ids[:4],"validation_trajectories":ids[4:],"test":"reserved and not generated"},
                         "layers":{},"time_s":{"matrix_load_and_metadata_this_run":time.perf_counter()-begin,
                            "matrix_build_recorded_source_sum":sum(t["build_seconds"] for t in meta["trajectories"])},
                         "validation_metric":B.METRIC,"test_data_used":False,"truth_used_for_selection":False}
                    report["rows"].append(row)
                elif row["input_hashes"]!=input_hashes:raise ValueError("resumed dataset input changed")
                for layer in protocol["layers"]:
                    indices=meta["small_indices"] if layer=="J18" else list(range(a.shape[1]))
                    at,va=a[:,indices],av[:,indices]
                    an,bn,scales=normalize_training(at,b)
                    solver_hashes={"A":array_hash(an),"b":array_hash(bn)}
                    if layer not in row["layers"]:
                        row["layers"][layer]={"indices_in_maximum_library":indices,"names":[names[i] for i in indices],
                            "n_columns":len(indices),"complexities":[costs[i] for i in indices],"normalization":scales,
                            "solver_input_hashes":solver_hashes,"methods":{}}
                    lib=row["layers"][layer]
                    if lib["solver_input_hashes"]!=solver_hashes or lib["normalization"]!=scales:
                        raise ValueError("resumed normalization differs")
                    for method in METHODS:
                        if method in lib["methods"]:continue
                        if seed==0 and method!="IRL1_smoothed":
                            prior=original["layers"][layer]
                            if prior["solver_input_hashes"]!=solver_hashes or prior["normalization"]!=scales:
                                raise ValueError("historical solver input/scaling differs")
                            record=copy.deepcopy(prior["methods"][method])
                            validate_path(record["path"],an,bn)
                            if not record["path"]["all_solutions_certified"]:raise ValueError("historical control incomplete")
                            record.update({"origin":"reused_stage2a_control","reuse_provenance":source,
                                           "new_solver_seconds_this_comparison":0.})
                            lib["methods"][method]=record;save();continue
                        if time.perf_counter()-start+METHOD_SECONDS+20.>args.max_seconds:
                            stopped=True;break
                        tag=f"{cohort}_{case}_{sigma}_{layer}_{method}".replace(".","p")
                        manifest={"cohort":cohort,"case":case,"sigma":sigma,"seed":seed,"layer":layer,"method":method,
                                  "solver_input_hashes":solver_hashes,"raw_bundle_sha256":meta["raw_bundle_sha256"],
                                  "source_sha256":hashes,"gate_sources":gates,"ratios":DEFAULT_RATIOS.tolist(),
                                  "normalization":scales,"budget_s":METHOD_SECONDS}
                        solver={"L1":solve_l1_path,"IRL1_original":solve_original_irl1,"IRL1_smoothed":solve_smoothed_irl1}[method]
                        path,provenance=obtain_path(solver,an,bn,manifest,tag)
                        validate_path(path,an,bn)
                        evaluate_start=time.perf_counter()
                        val_blocks=[(row["splits"]["validation_trajectories"][i-4],matrices[i][:,indices],targets[i]) for i in (4,5)]
                        evaluation=B._candidate_records(path,at,b,va,bv,lib["names"],lib["complexities"],dataset_id,val_blocks)
                        for point in path["points"]:
                            if point["beta"] is not None:point["physical_L1_coefficients"]=physical_coefficients(point["beta"],scales).tolist()
                        record={"origin":"stage2b_new_method_run","path":path,"evaluation":evaluation,"execution_provenance":provenance,
                                "time_s":{"solver_with_worker_startup":path["elapsed_s"],
                                          "fixed_support_validation":time.perf_counter()-evaluate_start},
                                "new_solver_seconds_this_comparison":provenance["new_solver_seconds"],
                                "full_end_to_end_timing_available":False}
                        lib["methods"][method]=record
                        strict_json_write(PROGRESS/(tag+"_evaluation_result.json"),record)
                        save()
                        print(cohort,case,layer,method,path["status"],round(path["elapsed_s"],2),flush=True)
                    if stopped:break
                finish_row(row);save()
                if stopped:break
            if stopped:break
    batch["status"]="budget_checkpoint" if stopped else "all_initial_attempts_finished"
    report["execution_complete"]=not stopped and all(r.get("all_methods_executed",False) for r in report["rows"]) and len(report["rows"])==6
    report["all_checks_passed"]=report["execution_complete"] and all(r["all_methods_numerically_complete"] for r in report["rows"])
    report["completion_meaning"]="All initial method tasks attempted; numerical completion and support correctness are separate."
    save()
    print(json.dumps({"execution_complete":report["execution_complete"],"all_checks_passed":report["all_checks_passed"],
                      "pending_jobs":len(report["pending_jobs"]),"uncertified_jobs":report["uncertified_jobs"],
                      "batch_elapsed_s":batch["elapsed_s"]},allow_nan=False))


if __name__=="__main__":main()
