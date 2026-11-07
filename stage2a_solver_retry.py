"""Resume only uncertified L1 points; keep verified previous path points intact.

The saved normalized matrix hashes and all inherited certificates are checked.
Warm QP starts at the nearest certified larger-lambda neighbor, not a known
incorrect LARS orthant. Every lambda and all numerical tolerances are retained.
"""

import copy
import hashlib
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import stage2a_solver_polish as P
from stage0_evaluation import array_hash, normalize_training, strict_json_write
from stage0_l1_solver import SUPPORT_TOL, certify_l1
from stage2_identifiability import stack_split
from stage2a_l1_paths import _save_progress


ROOT=Path(__file__).resolve().parent


def _worker(conn,a,b,source):
    points=copy.deepcopy(source["points"])
    weights=np.asarray(source["weights"])
    real_certificate=P.certify_l1
    current_point=None
    history=[]
    def audited_certificate(*args,**kwargs):
        certificate=real_certificate(*args,**kwargs)
        history.append({k:certificate[k] for k in ["certified","active_count_exact_nonzero",
                        "kkt_max_relative_to_penalty","dual_gap_relative","primal_objective"]})
        if len(history)%20==0:
            conn.send(("progress",{"point_index":current_point,"certificate_calls":len(history),
                                   "latest":history[-1],"beta":np.asarray(args[2]).tolist()}))
        return certificate
    P.certify_l1=audited_certificate
    try:
        with threadpool_limits(limits=1):
            for index,point in enumerate(points):
                if point["status"]=="certified":
                    continue
                current_point=index
                history=[]
                neighbors=[p for p in points[:index] if p["status"]=="certified"]
                if not neighbors:
                    raise RuntimeError("no certified larger-lambda warm initialization exists")
                neighbor=neighbors[-1]
                initial=np.asarray(neighbor["beta"])
                lam=float(point["lambda"])
                before=real_certificate(a,b,initial,lam,weights)
                start=time.perf_counter()
                conn.send(("start_point",{"index":index,"lambda":lam,"ratio":point["ratio"],
                                          "warm_neighbor_ratio":neighbor["ratio"],"initial_certificate":before}))
                try:
                    beta,details=P.polish_fixed_l1(a,b,lam,weights,initial,max_iterations=500)
                    after=real_certificate(a,b,beta,lam,weights)
                    if after["primal_objective"]>before["primal_objective"]+2*before["roundoff_allowance"]:
                        raise RuntimeError("fixed-lambda objective increased beyond roundoff")
                    replacement={"ratio":point["ratio"],"lambda":lam,"status":"certified" if after["certified"] else "uncertified",
                                 "beta":beta.tolist(),"support":np.flatnonzero(np.abs(beta)>SUPPORT_TOL).tolist(),
                                 "certificate":after,"elapsed_s":time.perf_counter()-start,
                                 "attempts":[{"solver":"warm_QP_from_certified_neighbor","neighbor_ratio":neighbor["ratio"],
                                              **details,"certificate":after}],"error":None,
                                 "polish_effect":{"initial_certificate_at_target_lambda":before,
                                                  "objective_change":after["primal_objective"]-before["primal_objective"],
                                                  "prediction_difference":float(np.linalg.norm(a@(beta-initial))),
                                                  "coefficient_difference":float(np.linalg.norm(beta-initial))},
                                 "certificate_history":history}
                    points[index]=replacement
                    conn.send(("point",{"index":index,"value":replacement}))
                except (RuntimeError,np.linalg.LinAlgError) as exc:
                    conn.send(("failure",{"index":index,"error":str(exc),"elapsed_s":time.perf_counter()-start,
                                          "certificate_history":history}))
            conn.send(("done",True))
    except Exception as exc:
        conn.send(("fatal",{"type":type(exc).__name__,"message":str(exc)}))
    finally:
        conn.close()


def retry_l1_path(a,b,source,budget_s=90.,progress_path=None):
    if source["method"]!="L1":
        raise ValueError("this retry preserves ordinary L1 only")
    source=copy.deepcopy(source)
    weights=np.asarray(source["weights"])
    if source["shape"]!=list(a.shape) or source["support_tolerance"]!=SUPPORT_TOL:
        raise ValueError("retry shape/support convention mismatch")
    inherited=[]
    for index,point in enumerate(source["points"]):
        if point["status"]=="certified":
            certificate=certify_l1(a,b,np.asarray(point["beta"]),point["lambda"],weights)
            if not certificate["certified"]:
                raise ValueError("inherited point no longer certifies")
            inherited.append(index)
    report={**source,"status":"retry_running","all_solutions_certified":False,
            "implementation":"certified_neighbor_warm_QP_retry","retry_budget_s":float(budget_s),
            "inherited_certified_point_indices":inherited,"retry_failures":[],"retry_progress":None}
    _save_progress(progress_path,report)
    context=mp.get_context("spawn")
    receiver,sender=context.Pipe(duplex=False)
    process=context.Process(target=_worker,args=(sender,a,b,source))
    start=time.perf_counter();process.start();sender.close()
    terminal="worker_error"
    try:
        while True:
            remaining=budget_s-(time.perf_counter()-start)
            if remaining<=0:
                terminal="budget_exhausted";break
            if receiver.poll(min(.2,remaining)):
                try:kind,value=receiver.recv()
                except EOFError:break
                if kind=="point":report["points"][value["index"]]=value["value"]
                elif kind=="failure":report["retry_failures"].append(value)
                elif kind in ("start_point","progress"):report["retry_progress"]=value
                elif kind=="fatal":report["error"]=value;break
                elif kind=="done":
                    terminal="complete" if all(p["status"]=="certified" for p in report["points"]) else "partial_failure"
                    break
                report["retry_elapsed_s"]=time.perf_counter()-start
                _save_progress(progress_path,report)
            elif not process.is_alive():break
    finally:
        if process.is_alive():process.terminate()
        process.join(timeout=5.);receiver.close()
    report.update({"status":terminal,"all_solutions_certified":terminal=="complete",
                   "retry_elapsed_s":time.perf_counter()-start,
                   "elapsed_s":source["elapsed_s"]+time.perf_counter()-start,
                   "timing_note":"elapsed_s sums previous attempted path and this retry; retry_elapsed_s isolates resumption"})
    _save_progress(progress_path,report)
    return report


def main():
    input_path=ROOT/"stage2a_solver_polish_result.json"
    source=json.loads(input_path.read_text(encoding="utf-8"))
    with np.load(ROOT/"stage2_matrices/heat_sigma_0p05_seed_0.npz",allow_pickle=False) as data:
        a,b=stack_split([data[f"a_{i}"] for i in range(4)],[data[f"b_{i}"] for i in range(4)],range(4))
    a,b,_=normalize_training(a[:,list(range(a.shape[1]))],b)
    assert array_hash(a)==source["input_hashes"]["A"] and array_hash(b)==source["input_hashes"]["b"]
    path=retry_l1_path(a,b,source["path"],90.,ROOT/"stage2a_solver_retry_progress_result.json")
    result={"scope":"resume only failed ordinary-L1 point; original path/data/tolerances preserved",
            "source_result":{"file":input_path.name,"sha256":hashlib.sha256(input_path.read_bytes()).hexdigest()},
            "input_hashes":source["input_hashes"],"path":path,"all_checks_passed":path["all_solutions_certified"],
            "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "test_data_used":False,"no_statistical_noise_robustness_claim":True}
    strict_json_write(ROOT/"stage2a_solver_retry_result.json",result)
    print(json.dumps({"all_checks_passed":result["all_checks_passed"],"status":path["status"],
                      "inherited_points":len(path["inherited_certified_point_indices"]),
                      "retry_elapsed_s":path["retry_elapsed_s"],"last_certificate":path["points"][-1]["certificate"]},allow_nan=False))


if __name__=="__main__":main()
