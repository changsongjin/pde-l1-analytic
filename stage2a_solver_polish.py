"""Warm fixed-lambda convex polishing of LARS; retain every original path point.

Same L1 objective, 1e-5 reporting threshold, and stage-0 KKT/duality criteria.
This is a numerical performance repair, not a new statistical estimator.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from lasso_exact import lars_lasso_path, lasso_at
from stage0_evaluation import array_hash, normalize_training, strict_json_write
from stage0_l1_solver import (SUPPORT_TOL, _inputs, certify_l1,
                             _coordinate_descent)
from stage2a_l1_paths import DEFAULT_RATIOS, _save_progress
from stage2_identifiability import stack_split


ROOT = Path(__file__).resolve().parent


def polish_fixed_l1(a, b, lam, weights, initial, max_iterations=100):
    """Newton correction on the current orthant, exact first-bound steps, KKT scan.

    Initialization uses every actual nonzero, independently of SUPPORT_TOL.
    A rank-deficient active matrix is explicitly rejected, never ridge-corrected.
    """
    a, b, weights = _inputs(a, b, lam, weights)
    beta = np.asarray(initial, dtype=float).copy()
    if beta.shape != weights.shape or not np.isfinite(beta).all():
        raise ValueError("invalid initial coefficients")
    before = certify_l1(a, b, beta, lam, weights)
    events = []
    svd_seconds = 0.
    for iteration in range(max_iterations):
        certificate = certify_l1(a, b, beta, lam, weights)
        if certificate["certified"]:
            return beta, {"iterations": iteration, "events": events, "svd_seconds": svd_seconds,
                          "initial_certificate": before, "final_certificate": certificate}
        active = np.flatnonzero(beta != 0.)
        if active.size:
            signs = np.sign(beta[active])
            signed = a[:, active] * signs
            start = time.perf_counter()
            u, singular, vt = np.linalg.svd(signed, full_matrices=False)
            svd_seconds += time.perf_counter()-start
            cutoff = singular[0]*np.finfo(float).eps*max(signed.shape)
            if len(singular) != len(active) or singular[-1] <= cutoff:
                raise RuntimeError("warm active matrix is numerically singular; no column pruning or ridge used")
            current = beta[active]*signs
            candidate = current.copy()
            for _ in range(3):
                gradient = signed.T@(b-signed@candidate) - lam*weights[active]
                candidate += vt.T@((vt@gradient)/singular**2)
            if np.any(candidate <= 0.):
                moving_out = np.flatnonzero(candidate <= 0.)
                denominators = current[moving_out]-candidate[moving_out]
                if np.any(denominators <= 0.):
                    raise RuntimeError("invalid first-bound step; no coefficient threshold applied")
                fractions = current[moving_out]/denominators
                alpha = float(np.min(fractions))
                hit = moving_out[fractions == alpha]
                feasible = current + alpha*(candidate-current)
                feasible[hit] = 0.
                if np.any(feasible < 0.):
                    raise RuntimeError("roundoff violated feasible orthant at a bound step")
                beta[active] = feasible*signs
                events.append({"iteration": iteration, "event": "bound_step", "fraction": alpha,
                               "removed_exact_zero": active[hit].tolist(), "active_count": len(active)})
                continue
            beta[active] = candidate*signs
            certificate = certify_l1(a, b, beta, lam, weights)
            events.append({"iteration": iteration, "event": "orthant_newton", "active_count": len(active),
                           "active_condition": float(singular[0]/singular[-1]),
                           "certificate": certificate})
            if certificate["certified"]:
                return beta, {"iterations": iteration+1, "events": events, "svd_seconds": svd_seconds,
                              "initial_certificate": before, "final_certificate": certificate}
        corr = a.T@(b-a@beta)
        violation = np.abs(corr)-lam*weights
        violation[beta != 0.] = -np.inf
        entering = int(np.argmax(violation))
        if not np.isfinite(violation[entering]) or violation[entering] <= 0.:
            raise RuntimeError("full certificate failed despite no entering KKT violation; tolerances unchanged")
        # A one-coordinate exact convex minimization makes the new variable
        # feasible and nonzero, after which it joins the next orthant solve.
        norm = float(a[:, entering]@a[:, entering])
        if norm <= 0.:
            raise RuntimeError("zero column cannot have positive KKT violation")
        beta[entering] = np.sign(corr[entering])*violation[entering]/norm
        events.append({"iteration": iteration, "event": "enter", "index": entering,
                       "full_library_violation": float(violation[entering])})
    raise RuntimeError("warm fixed-lambda QP iteration budget exhausted")


def _worker(conn, a, b, weights, ratios):
    def send(kind, value):
        conn.send((kind, value))
    try:
        with threadpool_limits(limits=1):
            start = time.perf_counter()
            lambda_max = float(np.max(np.abs(a.T@b)/weights))
            limit = max(500, 4*min(a.shape))
            knots = lars_lasso_path(a,b,weights,lam_min_ratio=min(ratios),max_steps=limit)
            send("path_metadata", {"lambda_max":lambda_max,"knots":len(knots),"max_steps":limit,
                                   "last_lambda":float(knots[-1][0]),"path_build_s":time.perf_counter()-start})
            for ratio in ratios:
                point_start = time.perf_counter()
                lam = float(ratio*lambda_max)
                initial = lasso_at(knots,lam)
                beta = initial.copy()
                original = certify_l1(a,b,beta,lam,weights)
                certificate = original
                attempts = [{"solver":"shared_LARS_path","certificate":original}]
                if not certificate["certified"]:
                    polish_start = time.perf_counter()
                    try:
                        beta, detail = polish_fixed_l1(a,b,lam,weights,initial)
                        certificate = certify_l1(a,b,beta,lam,weights)
                        attempts.append({"solver":"warm_orthant_SVD_newton","elapsed_s":time.perf_counter()-polish_start,
                                         **detail,"certificate":certificate})
                    except (RuntimeError,np.linalg.LinAlgError) as exc:
                        attempts.append({"solver":"warm_orthant_SVD_newton","elapsed_s":time.perf_counter()-polish_start,
                                         "error":str(exc)})
                        beta = initial.copy()
                    if not certificate["certified"]:
                        cd_start = time.perf_counter()
                        beta, detail = _coordinate_descent(a,b,lam,weights,initial=beta,max_iter=2000)
                        certificate = certify_l1(a,b,beta,lam,weights)
                        attempts.append({"solver":"official_CD_warm_polish_2000","elapsed_s":time.perf_counter()-cd_start,
                                         **detail,"certificate":certificate})
                objective_change = certificate["primal_objective"]-original["primal_objective"]
                allowance = original["roundoff_allowance"]+certificate["roundoff_allowance"]
                if certificate["certified"] and objective_change > allowance:
                    raise RuntimeError("polish increased the fixed-lambda objective beyond roundoff")
                send("point", {"ratio":float(ratio),"lambda":lam,
                               "status":"certified" if certificate["certified"] else "uncertified",
                               "beta":beta.tolist(),"support":np.flatnonzero(np.abs(beta)>SUPPORT_TOL).tolist(),
                               "certificate":certificate,"attempts":attempts,
                               "polish_effect":{"objective_change":objective_change,
                                                "prediction_difference":float(np.linalg.norm(a@(beta-initial))),
                                                "coefficient_difference":float(np.linalg.norm(beta-initial)),
                                                "initial_exact_nonzeros":int(np.count_nonzero(initial)),
                                                "final_exact_nonzeros":int(np.count_nonzero(beta))},
                               "elapsed_s":time.perf_counter()-point_start,"error":None})
            send("done",True)
    except Exception as exc:
        send("fatal",{"type":type(exc).__name__,"message":str(exc)})
    finally:
        conn.close()


def solve_l1_path(a,b,ratios=None,weights=None,budget_s=120.,progress_path=None):
    """Drop-in L1 path API; full-column certificate and original lambda grid."""
    a,b,weights = _inputs(a,b,1.,weights)
    ratios = np.asarray(DEFAULT_RATIOS if ratios is None else ratios,dtype=float)
    if (ratios.ndim!=1 or not ratios.size or not np.isfinite(ratios).all()
            or not np.all((ratios>0)&(ratios<=1)) or not np.all(np.diff(ratios)<0)):
        raise ValueError("ratios must be positive, decreasing and no greater than one")
    if not np.isfinite(budget_s) or budget_s<=0:
        raise ValueError("positive finite wall budget required")
    lambda_max = float(np.max(np.abs(a.T@b)/weights))
    if lambda_max<=0:
        raise ValueError("zero lambda_max")
    report = {"method":"L1","implementation":"LARS_plus_warm_fixed_lambda_polish",
              "shape":list(a.shape),"ratios":ratios.tolist(),"weights":weights.tolist(),
              "support_tolerance":SUPPORT_TOL,"budget_s":float(budget_s),"status":"running",
              "points":[],"outer":[],"outer_stop":None,"path_metadata":None,
              "no_truth_used":True,"no_column_pruning":True,"tolerances_unchanged":True}
    context=mp.get_context("spawn")
    receiving,sending=context.Pipe(duplex=False)
    process=context.Process(target=_worker,args=(sending,a,b,weights,ratios.tolist()))
    start=time.perf_counter()
    process.start(); sending.close()
    terminal="worker_error"
    try:
        while True:
            remaining=budget_s-(time.perf_counter()-start)
            if remaining<=0:
                terminal="budget_exhausted"; break
            if receiving.poll(min(.2,remaining)):
                try:
                    kind,value=receiving.recv()
                except EOFError:
                    report["error"]="worker pipe closed without completion"; break
                if kind=="point": report["points"].append(value)
                elif kind=="path_metadata": report["path_metadata"]=value
                elif kind=="done":
                    terminal="complete" if all(p["status"]=="certified" for p in report["points"]) else "partial_failure"
                    break
                elif kind=="fatal": report["error"]=value; break
                report["elapsed_s"]=time.perf_counter()-start
                _save_progress(progress_path,report)
            elif not process.is_alive():
                report["error"]="worker exited before every lambda completed"; break
    finally:
        if process.is_alive(): process.terminate()
        process.join(timeout=5.)
        receiving.close()
    completed={p["ratio"] for p in report["points"]}
    for ratio in ratios:
        if float(ratio) not in completed:
            report["points"].append({"ratio":float(ratio),"lambda":float(ratio*lambda_max),"status":terminal,
                                      "beta":None,"support":None,"certificate":None,"attempts":[],
                                      "error":"requested lambda unfinished; no model-ranking eligibility"})
    report.update({"status":terminal,"elapsed_s":time.perf_counter()-start,
                   "all_solutions_certified":terminal=="complete" and len(report["points"])==len(ratios)})
    _save_progress(progress_path,report)
    return report


def main():
    matrix_path=ROOT/"stage2_matrices/heat_sigma_0p05_seed_0.npz"
    original_path=ROOT/"stage2a_progress/stage2a_heat_noise_pilot_result/heat_0p05_0_J1158_L1_path_result.json"
    original=json.loads(original_path.read_text(encoding="utf-8"))
    with np.load(matrix_path,allow_pickle=False) as data:
        a,b=stack_split([data[f"a_{i}"] for i in range(4)],
                        [data[f"b_{i}"] for i in range(4)],range(4))
    # Match the baseline's column-indexed memory layout as well as values;
    # column-norm summation can otherwise differ by last-bit roundoff.
    a,b,_=normalize_training(a[:,list(range(a.shape[1]))],b)
    baseline=json.loads((ROOT/"stage2a_heat_noise_pilot_result.json").read_text(encoding="utf-8"))
    reference=next(row for row in baseline["rows"] if row["sigma"]==.05)["layers"]["J1158"]["solver_input_hashes"]
    assert array_hash(a)==reference["A"] and array_hash(b)==reference["b"]
    path=solve_l1_path(a,b,budget_s=90.,progress_path=ROOT/"stage2a_solver_polish_progress_result.json")
    polished=[p for p in path["points"] if len(p["attempts"])>1]
    report={"scope":"numerical performance repair probe; same heat5% J1158 inputs, no model selection",
            "original_reference":{"file":str(original_path.relative_to(ROOT)),
                                  "sha256":hashlib.sha256(original_path.read_bytes()).hexdigest(),
                                  "status":original["status"],"elapsed_s":original["elapsed_s"],
                                  "path_metadata":original["path_metadata"]},
            "input_hashes":reference,"path":path,"all_checks_passed":path["all_solutions_certified"],
            "polished_ratios":[p["ratio"] for p in polished],
            "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "interpretation":"timeout is a computational outcome, not statistical noise intolerance",
            "limitations":["One development matrix; timing is not a formal multi-case benchmark.",
                           "Warm orthant SVD rejects active rank deficiency instead of changing the objective.",
                           "No validation score, true support, or statistical variant is tuned here."]}
    strict_json_write(ROOT/"stage2a_solver_polish_result.json",report)
    print(json.dumps({"all_checks_passed":report["all_checks_passed"],"status":path["status"],
                      "elapsed_s":path["elapsed_s"],"polished_ratios":report["polished_ratios"],
                      "polish_seconds":[p["elapsed_s"] for p in polished]},allow_nan=False))


if __name__=="__main__":
    main()
