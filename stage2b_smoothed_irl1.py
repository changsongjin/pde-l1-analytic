"""Single-factor IRL1 candidate: relative epsilon, frozen before development.

Only epsilon changes: max(1e-8, .01*max(abs(previous normalized L1 beta))).
Original outer lambda ratio, iteration limit, support stop, final path, solver,
certificate and 1e-5 support convention are retained. This is not a new-method
novelty claim and does not promise improved support or validation error.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from stage0_evaluation import array_hash, strict_json_write
from stage0_l1_solver import SUPPORT_TOL, _inputs, certify_l1, _coordinate_descent
from stage2a_l1_paths import (DEFAULT_RATIOS, IRL1_RATIO, IRL1_MAX_ITER,
                             _path_points, _save_progress, solve_original_irl1)


ROOT=Path(__file__).resolve().parent
EPSILON_FLOOR=1e-8
RELATIVE_EPSILON=0.01
MAX_TASK_SECONDS=120.
FROZEN_SELECTION_ALPHA=0.005


def smoothed_weights(beta):
    """Use only the previous normalized L1 solution; no truth or LS readout."""
    beta=np.asarray(beta,dtype=float)
    if beta.ndim!=1 or not beta.size or not np.isfinite(beta).all():
        raise ValueError("need a finite nonempty normalized coefficient vector")
    largest=float(np.max(np.abs(beta)))
    epsilon=max(EPSILON_FLOOR,RELATIVE_EPSILON*largest)
    weights=1./(np.abs(beta)+epsilon)
    if not np.isfinite(weights).all() or not np.all(weights>0):
        raise ValueError("weight update must remain finite and strictly positive")
    ratio=float(weights.max()/weights.min())
    if ratio>1.+1./RELATIVE_EPSILON+1e-12:
        raise AssertionError("relative smoothing weight-ratio bound violated")
    return weights,{"epsilon":epsilon,"maximum_absolute_normalized_beta":largest,
                    "weights_min":float(weights.min()),"weights_max":float(weights.max()),
                    "weight_ratio":ratio,"weight_ratio_bound":101.,
                    "coefficient_source":"previous_certified_normalized_L1_solution",
                    "input_beta_sha256":array_hash(beta)}


def _worker(conn,a,b,ratios):
    def send(kind,value):conn.send((kind,value))
    try:
        with threadpool_limits(limits=1):
            weights=np.ones(a.shape[1])
            previous_support=None
            for iteration in range(IRL1_MAX_ITER):
                start=time.perf_counter()
                messages=[]
                outer=list(_path_points(a,b,weights,[IRL1_RATIO],lambda kind,value:messages.append((kind,value))))[0]
                outer.update({"iteration":iteration,"weights_used":weights.tolist(),
                              "path_metadata":next(value for kind,value in messages if kind=="path_metadata"),
                              "fixed_weight_solve_wall_s":time.perf_counter()-start})
                if outer["status"]!="certified":
                    send("outer",outer)
                    send("outer_stop","uncertified_outer_fixed_weight_solution")
                    send("partial","outer solution failed unchanged full-column certificate; final weights not frozen")
                    return
                weights,update=smoothed_weights(outer["beta"])
                outer["weight_update"]=update
                send("outer",outer)
                send("weights",weights.tolist())
                support=tuple(outer["support"])
                if previous_support==support:
                    send("outer_stop","support_unchanged_original_rule")
                    break
                previous_support=support
            else:
                send("outer_stop","maximum_8_iterations_original_rule")
            # The final weights include the last update even when support stops,
            # matching the historical original-IRL1 control exactly.
            for _ in _path_points(a,b,weights,ratios,send):pass
        send("done",True)
    except Exception as exc:
        send("fatal",{"type":type(exc).__name__,"message":str(exc)})
    finally:conn.close()


def solve_smoothed_irl1(a,b,ratios=None,budget_s=120.,progress_path=None):
    """Return baseline-compatible path records, with a <=120 s whole-task cap.

    This API does not choose models. Downstream selection must retain the
    previously frozen alpha=.005 and use validation data only.
    """
    a,b,_=_inputs(a,b,1.,None)
    ratios=np.asarray(DEFAULT_RATIOS if ratios is None else ratios,dtype=float)
    if ratios.shape!=DEFAULT_RATIOS.shape or not np.array_equal(ratios,DEFAULT_RATIOS):
        raise ValueError("this single-factor experiment fixes all 24 original lambda ratios")
    if not np.isfinite(budget_s) or not 0<budget_s<=MAX_TASK_SECONDS:
        raise ValueError("each task must have a positive wall budget no greater than 120 seconds")
    report={"method":"IRL1_relative_epsilon_0p01","shape":list(a.shape),"ratios":ratios.tolist(),
            "weights":np.ones(a.shape[1]).tolist(),"support_tolerance":SUPPORT_TOL,
            "budget_s":float(budget_s),"status":"running","points":[],"outer":[],"outer_stop":None,
            "path_metadata":None,"all_solutions_certified":False,
            "input_hashes":{"A":array_hash(a),"b":array_hash(b)},
            "single_factor_change":{"old_epsilon":EPSILON_FLOOR,"relative_epsilon":RELATIVE_EPSILON,
                                    "new_rule":"epsilon_i=max(1e-8,0.01*max(abs(beta_i))); weights=1/(abs(beta_i)+epsilon_i)",
                                    "frozen_before_new_method_results":True},
            "unchanged_settings":{"outer_relative_lambda":IRL1_RATIO,"max_outer_iterations":IRL1_MAX_ITER,
                                  "outer_stop":"support unchanged after weight update",
                                  "final_lambda_points":24,"final_lambda_min_ratio":1e-6,
                                  "selection_alpha":FROZEN_SELECTION_ALPHA,"support_threshold":SUPPORT_TOL,
                                  "fixed_weight_solver":"stage2a_l1_paths._path_points unchanged"},
            "no_true_support_used":True,"no_ls_weight_initialization":True,"no_column_pruning":True,
            "controls_retained":["L1","IRL1_original"],"test_data_used":False,
            "limitations":["Relative smoothing is a candidate; no superiority or novelty is assumed.",
                           "Unchanged support is the historical stop rule, not convergence of weights or a nonconvex global optimum.",
                           "Any unfinished or uncertified requested point makes the path ineligible for primary selection."]}
    context=mp.get_context("spawn");receiver,sender=context.Pipe(duplex=False)
    process=context.Process(target=_worker,args=(sender,a,b,ratios.tolist()))
    start=time.perf_counter();process.start();sender.close()
    _save_progress(progress_path,report)
    terminal="worker_error"
    try:
        while True:
            remaining=budget_s-(time.perf_counter()-start)
            if remaining<=0:terminal="budget_exhausted";break
            if receiver.poll(min(.2,remaining)):
                try:kind,value=receiver.recv()
                except EOFError:
                    report["error"]="worker closed without completion";break
                if kind=="outer":report["outer"].append(value)
                elif kind=="point":report["points"].append(value)
                elif kind in ("weights","path_metadata","outer_stop"):report[kind]=value
                elif kind=="done":
                    terminal="complete" if all(p["status"]=="certified" for p in report["points"]) else "partial_failure"
                    break
                elif kind=="partial":terminal="partial_failure";report["error"]=value;break
                elif kind=="fatal":report["error"]=value;break
                report["elapsed_s"]=time.perf_counter()-start
                _save_progress(progress_path,report)
            elif not process.is_alive():
                report["error"]="worker exited before all requested points";break
    finally:
        if process.is_alive():process.terminate()
        process.join(timeout=5.);receiver.close()
    completed={p["ratio"] for p in report["points"]}
    final_max=report["path_metadata"]["lambda_max"] if report["path_metadata"] else None
    for ratio in ratios:
        if float(ratio) not in completed:
            report["points"].append({"ratio":float(ratio),"lambda":float(ratio*final_max) if final_max is not None else None,
                                      "status":terminal,"beta":None,"support":None,"certificate":None,"attempts":[],
                                      "error":"requested final-weight point unfinished; no primary model selection"})
    report.update({"status":terminal,"elapsed_s":time.perf_counter()-start,
                   "all_solutions_certified":terminal=="complete" and len(report["points"])==24
                   and all(p["status"]=="certified" for p in report["outer"]),
                   "epsilon_trajectory":[p["weight_update"]["epsilon"] for p in report["outer"] if "weight_update" in p]})
    _save_progress(progress_path,report)
    return report


def _independent_checks(a,b,path):
    checks=[]
    weighted_points=[(f"outer_{p['iteration']}",p,np.asarray(p["weights_used"])) for p in path["outer"]]
    weighted_points.extend((f"final_{i}",p,np.asarray(path["weights"])) for i,p in enumerate(path["points"]))
    for label,point,weights in weighted_points:
        if point["status"]!="certified":
            checks.append({"label":label,"passed":False,"reason":"original path point not certified"});continue
        independent,details=_coordinate_descent(a,b,point["lambda"],weights,max_iter=100000)
        other=certify_l1(a,b,independent,point["lambda"],weights)
        current=point["certificate"]
        delta=float(np.linalg.norm(a@(np.asarray(point["beta"])-independent)))
        bound=float(np.sqrt(2*max(0.,current["dual_gap"]+other["dual_gap"])+1e-12*float(b@b)))
        difference=abs(current["primal_objective"]-other["primal_objective"])
        objective_bound=abs(current["dual_gap"])+abs(other["dual_gap"])+current["roundoff_allowance"]+other["roundoff_allowance"]
        checks.append({"label":label,"lambda":point["lambda"],"independent_certificate":other,
                       "iterations":details["iterations"],"warnings":details["warnings"],
                       "prediction_difference":delta,"prediction_gap_bound":bound,
                       "objective_absolute_difference":difference,"objective_gap_bound":objective_bound,
                       "passed":bool(other["certified"] and delta<=bound and difference<=objective_bound)})
    return checks


def main():
    started=time.perf_counter()
    protected=["stage2a_l1_paths.py","stage0_l1_solver.py","stage2a_baseline.py"]
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in protected}
    unit=[]
    for beta in [np.zeros(4),np.array([1.,0.,-.01,1e-12]),np.array([1e-9,-1e-11,0.])]:
        weights,meta=smoothed_weights(beta)
        assert meta["epsilon"]==max(EPSILON_FLOOR,RELATIVE_EPSILON*max(abs(beta)))
        assert np.all(weights>0) and meta["weight_ratio"]<=101.+1e-12
        unit.append(meta)
    base=np.array([1.,.1,0.]);w1,_=smoothed_weights(base);w2,_=smoothed_weights(7*base)
    assert np.allclose(w2,w1/7,rtol=1e-14,atol=0.)
    rows=[]
    rng=np.random.default_rng(6201)
    with threadpool_limits(limits=1):
        for label,correlation in [("synthetic_generic",False),("synthetic_correlated",True)]:
            a=rng.normal(size=(70,12))
            if correlation:a[:,1]=a[:,0]+.12*rng.normal(size=70)
            a/=np.linalg.norm(a,axis=0)
            coefficient=np.zeros(12);coefficient[[0,5,8]]=[1.,-.4,.2]
            b=a@coefficient+.003*rng.normal(size=70)
            path=solve_smoothed_irl1(a,b,budget_s=35.)
            checks=_independent_checks(a,b,path)
            original=solve_original_irl1(a,b,budget_s=15.) if not correlation else None
            same_initial=True
            if original is not None:
                same_initial=bool(original["all_solutions_certified"] and path["outer"]
                                  and original["outer"][0]["lambda"]==path["outer"][0]["lambda"]
                                  and np.array_equal(original["outer"][0]["beta"],path["outer"][0]["beta"]))
            rows.append({"label":label,"path":path,"independent_fixed_weight_checks":checks,
                         "original_IRL1_synthetic_control":original,"same_unweighted_first_outer_solve":same_initial,
                         "passed":path["all_solutions_certified"] and all(c["passed"] for c in checks) and same_initial})
            print(label,rows[-1]["passed"],"outer",len(path["outer"]),"seconds",round(path["elapsed_s"],2),flush=True)
    budget=solve_smoothed_irl1(np.eye(2),np.ones(2),budget_s=.001)
    budget_ok=budget["status"]=="budget_exhausted" and len(budget["points"])==24 and not budget["all_solutions_certified"]
    unchanged=hashes=={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in protected}
    result={"stage":"2b_API_and_synthetic_verification_only","all_checks_passed":all(r["passed"] for r in rows) and budget_ok and unchanged,
            "formula_unit_checks":unit,"rows":rows,"budget_guard":{"passed":budget_ok,"result":budget},
            "protected_sources_unchanged":unchanged,"source_hashes":hashes,
            "variant_source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "frozen_relative_epsilon":RELATIVE_EPSILON,"frozen_floor":EPSILON_FLOOR,
            "unchanged_selection_alpha":FROZEN_SELECTION_ALPHA,"elapsed_s":time.perf_counter()-started,
            "development_PDE_comparison_started":False,"test_data_used":False,
            "limits":["Synthetic numerical checks do not establish PDE robustness or an improved method.",
                      "Real development comparisons wait for the root agent's numerical-stage gate."]}
    strict_json_write(ROOT/"stage2b_smoothed_irl1_result.json",result)
    print(json.dumps({"all_checks_passed":result["all_checks_passed"],"elapsed_s":result["elapsed_s"],
                      "development_PDE_comparison_started":False},allow_nan=False))


if __name__=="__main__":main()
