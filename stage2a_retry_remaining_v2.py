"""Resume uncertified ordinary-L1 points using verified lambda reconstruction.

Each point has a 90-second subprocess wall budget; the batch is <=2 hours.
Existing certified points and all historical artifacts remain unchanged.
Re-running this new script resumes its own saved incomplete points only.
"""
from __future__ import annotations

import copy
import hashlib
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from stage0_evaluation import array_hash, normalize_training, strict_json_write
from stage0_l1_solver import SUPPORT_TOL, _inputs, certify_l1
from stage2_identifiability import save_matrices, stack_split
from stage2a_clarabel_probe import solve_and_certify


ROOT=Path(__file__).resolve().parent
OUTPUT=ROOT/"stage2a_retry_remaining_v2_result.json"
POINT_BUDGET=90.
BATCH_BUDGET=7200.


def _worker(conn,a,b,lam,weights):
    try:
        with threadpool_limits(limits=1):
            solution=solve_and_certify(a,b,lam,weights,lambda kind,value:conn.send((kind,value)),
                                       solver_time_limit=60.)
        conn.send(("solution",solution))
    except Exception as exc:
        conn.send(("worker_error",{"type":type(exc).__name__,"message":str(exc)}))
    finally:
        conn.close()


def solve_point(a,b,lam,weights,attempt,save):
    """Save stage progress while enforcing the entire worker's wall limit."""
    context=mp.get_context("spawn")
    receiving,sending=context.Pipe(duplex=False)
    process=context.Process(target=_worker,args=(sending,a,b,lam,weights))
    start=time.perf_counter()
    process.start();sending.close()
    solution=None
    attempt.update({"status":"running","stages":[],"wall_budget_s":POINT_BUDGET})
    save()
    try:
        while True:
            remaining=POINT_BUDGET-(time.perf_counter()-start)
            if remaining<=0:
                attempt["status"]="budget_exhausted";break
            if receiving.poll(min(.2,remaining)):
                try:kind,value=receiving.recv()
                except EOFError:
                    attempt["status"]="worker_closed";break
                if kind=="solver_stage":attempt["stages"].append(value)
                elif kind=="solution":
                    solution=value
                    attempt["status"]="certified" if value["certificate"]["certified"] else "uncertified"
                    break
                elif kind=="worker_error":
                    attempt["status"]="worker_error";attempt["error"]=value;break
                attempt["wall_seconds"]=time.perf_counter()-start
                save()
            elif not process.is_alive():
                attempt["status"]="worker_exited";break
    finally:
        if process.is_alive():process.terminate()
        process.join(timeout=5.);receiving.close()
    attempt["wall_seconds"]=time.perf_counter()-start
    if solution is not None:
        certificate=certify_l1(a,b,np.asarray(solution["beta"]),lam,weights)
        if certificate["certified"]!=solution["certificate"]["certified"]:
            raise RuntimeError("parent and worker certificates disagree")
        attempt["certificate"]=certificate
        solution["certificate"]=certificate
    save()
    return solution


def validate_and_restore(a,b,path,expected_hashes):
    hashes={"A":array_hash(a),"b":array_hash(b)}
    if hashes!=expected_hashes:
        raise ValueError("retry solver inputs differ from the original")
    a,b,weights=_inputs(a,b,1.,path["weights"])
    recorded=float(path["path_metadata"]["lambda_max"])
    recalculated=float(np.max(np.abs(a.T@b)/weights))
    tolerance=64*np.finfo(float).eps*max(1.,abs(recorded),abs(recalculated))
    if not np.isfinite(recorded) or recorded<=0 or abs(recorded-recalculated)>tolerance:
        raise ValueError("recorded lambda_max does not match the same A,b,weights")
    if path["shape"]!=list(a.shape) or path["support_tolerance"]!=SUPPORT_TOL:
        raise ValueError("path shape or fixed support convention mismatch")
    if len(path["points"])!=len(path["ratios"]):
        raise ValueError("original path does not represent every requested ratio")
    restored=[]
    for index,(point,ratio) in enumerate(zip(path["points"],path["ratios"])):
        if point["ratio"]!=ratio or not np.isfinite(ratio) or not 0<ratio<=1:
            raise ValueError("original ratio metadata mismatch")
        target=float(recorded*ratio)
        if point["lambda"] is None:
            if point["status"]=="certified":
                raise ValueError("certified point must already contain lambda")
            point["lambda"]=target
            restored.append({"index":index,"ratio":ratio,"lambda":target})
        elif abs(float(point["lambda"])-target)>64*np.finfo(float).eps*max(abs(target),1e-300):
            raise ValueError("saved lambda disagrees with recorded lambda_max times ratio")
        if point["status"]=="certified":
            beta=np.asarray(point["beta"])
            certificate=certify_l1(a,b,beta,target,weights)
            if not certificate["certified"] or point["support"]!=np.flatnonzero(np.abs(beta)>SUPPORT_TOL).tolist():
                raise ValueError("inherited point certificate/support invalid")
    return {"input_hashes":hashes,"lambda_max_recorded":recorded,"lambda_max_recalculated":recalculated,
            "lambda_max_absolute_difference":abs(recorded-recalculated),"verification_tolerance":tolerance,
            "restored_pending_lambdas":restored},weights


def _job_key(case,sigma,seed,layer):
    return f"{case}:{sigma!r}:{seed}:{layer}:L1"


def main():
    original_file=ROOT/"stage2a_other_cases_pilot_result.json"
    failed_file=ROOT/"stage2a_retry_other_cases_result.json"
    original=json.loads(original_file.read_text(encoding="utf-8"))
    failed=json.loads(failed_file.read_text(encoding="utf-8"))
    if not original["execution_complete"]:
        raise RuntimeError("original campaign is not complete")
    source_hash=hashlib.sha256(original_file.read_bytes()).hexdigest()
    jobs=[(row,layer,spec,spec["methods"]["L1"])
          for row in original["rows"] for layer,spec in row["layers"].items()
          if not spec["methods"]["L1"]["path"]["all_solutions_certified"]]
    failed_lookup={_job_key(r["case"],r["sigma"],r["seed"],r["layer"]):r for r in failed["rows"]}
    if OUTPUT.exists():
        report=json.loads(OUTPUT.read_text(encoding="utf-8"))
        if report["source_sha256"]!=source_hash:
            raise RuntimeError("source changed since this continuation; refusing to reuse its points")
    else:
        report={"scope":"numerical continuation only; no L1 objective/threshold/selection change",
                "source_file":original_file.name,"source_sha256":source_hash,
                "previous_interface_failure_file":failed_file.name,
                "previous_interface_failure_sha256":hashlib.sha256(failed_file.read_bytes()).hexdigest(),
                "lambda_restoration_rule":"recorded lambda_max * original ratio, checked against current A,b,weights",
                "max_batch_seconds":BATCH_BUDGET,"per_point_wall_seconds":POINT_BUDGET,
                "source_script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "clarabel_api_sha256":hashlib.sha256((ROOT/"stage2a_clarabel_probe.py").read_bytes()).hexdigest(),
                "dependency_manifest_sha256":hashlib.sha256((ROOT/"requirements-research.txt").read_bytes()).hexdigest(),
                "rows":[],"batches":[],"execution_complete":False,"all_checks_passed":False,
                "test_data_used":False,"no_true_support_used":True,
                "cost_note":"Historical timeout, interface-failure retries and current continuation costs are separate; no timeout is a statistical recovery failure."}
    batch={"index":len(report["batches"]),"budget_s":BATCH_BUDGET,"status":"running","elapsed_s":0.}
    report["batches"].append(batch)
    start=time.perf_counter()
    def save():
        batch["elapsed_s"]=time.perf_counter()-start
        report["pending"]=[{"job":r["job_key"],"ratio":p["ratio"],"lambda":p["lambda"],"status":p["status"]}
                           for r in report["rows"] for p in r["path"]["points"] if p["status"]!="certified"]
        strict_json_write(OUTPUT,report)
    stop=False
    with threadpool_limits(limits=1):
        for dataset,layer,spec,old_method in jobs:
            key=_job_key(dataset["case"],dataset["sigma"],dataset["seed"],layer)
            row=next((r for r in report["rows"] if r["job_key"]==key),None)
            if row is None:
                prior=failed_lookup[key]
                row={"job_key":key,"case":dataset["case"],"sigma":dataset["sigma"],"seed":dataset["seed"],
                     "layer":layer,"method":"L1","path":copy.deepcopy(old_method["path"]),"point_attempts":[],
                     "original_certified_point_indices":[i for i,p in enumerate(old_method["path"]["points"]) if p["status"]=="certified"],
                     "original_path_seconds":old_method["path"]["elapsed_s"],
                     "previous_interface_failure_seconds":prior["path"]["retry_elapsed_s"],
                     "previous_interface_failure":prior["path"].get("error")}
                report["rows"].append(row)
            matrices,targets,_=save_matrices(dataset["case"],dataset["sigma"],dataset["seed"])
            a,b=stack_split(matrices,targets,range(4))
            a,b,_=normalize_training(a[:,spec["indices_in_maximum_library"]],b)
            verification,weights=validate_and_restore(a,b,row["path"],spec["solver_input_hashes"])
            row.setdefault("input_verification_history",[]).append(verification)
            row["input_hashes"]=verification["input_hashes"]
            save()
            for index,point in enumerate(row["path"]["points"]):
                if point["status"]=="certified":continue
                if time.perf_counter()-start+POINT_BUDGET+10.>BATCH_BUDGET:
                    stop=True;break
                attempt={"point_index":index,"ratio":point["ratio"],"lambda":point["lambda"],"batch_index":batch["index"],
                         "prior_status":point["status"],"status":"pending","wall_seconds":0.}
                row["point_attempts"].append(attempt)
                solution=solve_point(a,b,float(point["lambda"]),weights,attempt,save)
                replacement={"ratio":point["ratio"],"lambda":point["lambda"],"status":attempt["status"],
                             "beta":None,"support":None,"certificate":None,"attempts":[],
                             "elapsed_s":attempt["wall_seconds"],"error":attempt.get("error")}
                if solution is not None:
                    replacement.update({k:solution[k] for k in ("beta","support","certificate")})
                    replacement["attempts"]=[{"solver":"Clarabel_residual_QP_and_same_objective_polish",
                                               "external_solver":solution["external_solver"],"stages":solution["stages"],
                                               "raw_to_final_prediction_difference":solution["raw_to_final_prediction_difference"],
                                               "raw_to_final_objective_change":solution["raw_to_final_objective_change"]}]
                row["path"]["points"][index]=replacement
                row["new_retry_total_wall_s"]=sum(a["wall_seconds"] for a in row["point_attempts"])
                row["historical_solver_cost_total_s"]=(row["original_path_seconds"]+row["previous_interface_failure_seconds"]
                                                        +row["new_retry_total_wall_s"])
                completed=all(p["status"]=="certified" for p in row["path"]["points"])
                row["path"].update({"status":"complete" if completed else "partial_failure",
                                     "all_solutions_certified":completed,"elapsed_s":row["historical_solver_cost_total_s"],
                                     "continuation_wall_s":row["new_retry_total_wall_s"],
                                     "implementation":"inherited_certified_LARS_points_plus_verified_Clarabel_continuation"})
                save()
                print(dataset["case"],point["ratio"],attempt["status"],round(attempt["wall_seconds"],2),flush=True)
            # Equality check protects inherited candidates, including all coefficient values.
            assert all(row["path"]["points"][i]==old_method["path"]["points"][i]
                       for i in row["original_certified_point_indices"])
            if stop:break
    report["execution_complete"]=not stop and len(report["rows"])==len(jobs)
    report["all_checks_passed"]=report["execution_complete"] and all(r["path"]["all_solutions_certified"] for r in report["rows"])
    batch["status"]="batch_budget_checkpoint" if stop else "attempts_finished"
    save()
    print(json.dumps({"execution_complete":report["execution_complete"],"all_checks_passed":report["all_checks_passed"],
                      "pending_points":len(report["pending"]),"batch_elapsed_s":batch["elapsed_s"]},allow_nan=False))


if __name__=="__main__":main()
