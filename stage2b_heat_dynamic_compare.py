"""Frozen stage-2a heat models: full-spectrum development diagnostics only.

No L1 solve, coefficient refit, support deletion, or model reselection. The new
cohort is used only at matched 1% noise. Unavailable nonlinear/odd-derivative
rollouts remain unsupported, not failures or silently excluded competitors.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from stage0_data_bundle import load_bundle
from stage0_evaluation import array_hash,canonical_term,strict_json_write
import stage2b_heat_full_spectrum as D
from stage2b_protocol import PROTOCOL

ROOT=Path(__file__).resolve().parent


def _load(path):
    def reject(value):raise ValueError("non-finite JSON "+value)
    return json.loads(Path(path).read_text(encoding="utf-8"),parse_constant=reject)


def _hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def review_new_data():
    gate=_load(ROOT/"stage2b_data_result.json")
    protocol=_load(ROOT/"stage2b_protocol_result.json")
    if not gate["all_checks_passed"] or not protocol["all_checks_passed"]:
        raise ValueError("new-cohort data/protocol gate is not ready")
    if gate["protocol_sha256"]!=_hash(ROOT/"stage2b_protocol_result.json"):
        raise ValueError("data used a different frozen protocol")
    for name,digest in gate["source_hashes"].items():
        if _hash(ROOT/name)!=digest:raise ValueError("new data source changed: "+name)
    ids=PROTOCOL["independent_development_replication"]["trajectory_ids"]
    if ids!=list(range(12,18)) or set(ids)&set(range(6,12)):
        raise ValueError("development trajectory identities overlap reserved final test")
    rows=[]
    for meta in gate["rows"]:
        case,sigma=meta["case"],meta["sigma"]
        tag=f"{case}_sigma_{sigma}_seed_1_ids12_17"
        x,t,observations,raw=load_bundle(ROOT/"stage2b_data"/(tag+".npz"))
        stored=_load(ROOT/"stage2b_matrices"/(tag+".json"))
        matrix_path=ROOT/"stage2b_matrices"/(tag+".npz")
        if (stored!=meta or _hash(matrix_path)!=meta["npz_sha256"]
                or raw["npz_sha256"]!=meta["raw_bundle_sha256"]
                or raw["seed"]!=1 or raw["test_data_generated_or_used"]
                or [r["trajectory_id"] for r in raw["trajectories"]]!=ids
                or [r["split"] for r in raw["trajectories"]]!=["train"]*4+["validation"]*2):
            raise ValueError("new cohort manifest/split/hash mismatch")
        if len({array_hash(u) for u in observations})!=6:
            raise ValueError("duplicate raw trajectory across train/validation")
        with np.load(matrix_path,allow_pickle=False) as data:
            for i,m in enumerate(meta["trajectories"]):
                a,b=data[f"a_{i}"],data[f"b_{i}"]
                if (a.shape!=(m["rows"],1158) or b.shape!=(m["rows"],)
                        or array_hash(a)!=m["matrix_hash"] or array_hash(b)!=m["target_hash"]
                        or array_hash(observations[i])!=m["raw_hash"]
                        or m["trajectory"]!=ids[i] or m["split"]!=raw["trajectories"][i]["split"]):
                    raise ValueError("new matrix/target/raw contract mismatch")
        rows.append({"case":case,"sigma":sigma,"noise_seed":1,"trajectory_ids":ids,
                     "matrix_target_pairs_verified":6,"raw_bundle_sha256":raw["npz_sha256"]})
    if len(rows)!=3 or not all(x["old_ids_0_5_exact_match"] for x in gate["parameter_distribution_equivalence"]):
        raise ValueError("data cohort or generator-equivalence audit incomplete")
    if len(gate["field_checks"])!=18 or not all(r["passed"] for r in gate["field_checks"]):
        raise ValueError("independent PDE field checks incomplete")
    return {"all_checks_passed":True,"rows":rows,"pairs_verified":18,
        "distribution_audit":"same parameter families and RNG draw order as stage2; original IDs0..5 exactly reproduce old parameters",
        "split_audit":"new train12..15/validation16..17, seed1; disjoint from original0..5/seed0 and reserved final-test6..11/seeds>=1000",
        "reserved_test_generated_or_used":False}


def _cohort(sigma,new=False):
    if new:
        if sigma!=.01:raise ValueError("new-cohort comparison is frozen to matched1% noise")
        tag="heat_sigma_0.01_seed_1_ids12_17"
        raw_path=ROOT/"stage2b_data"/(tag+".npz")
        matrix_path=ROOT/"stage2b_matrices"/(tag+".npz")
        ids=[16,17];cohort="new_development_ids12_17_seed1"
    else:
        tag=f"heat_sigma_{repr(float(sigma)).replace('.', 'p')}_seed_0"
        raw_path=ROOT/"stage2_data"/(tag+".npz")
        matrix_path=ROOT/"stage2_matrices"/(tag+".npz")
        ids=[4,5];cohort="original_development_ids0_5_seed0"
    x,t,fields,raw=load_bundle(raw_path)
    metadata=_load(matrix_path.with_suffix(".json"))
    if raw["npz_sha256"]!=metadata["raw_bundle_sha256"]:raise ValueError("matrix/raw cohort mismatch")
    with np.load(matrix_path,allow_pickle=False) as data:
        matrices=[data[f"a_{i}"].copy() for i in (4,5)]
        targets=[data[f"b_{i}"].copy() for i in (4,5)]
    for a,b,m in zip(matrices,targets,metadata["trajectories"][4:]):
        if array_hash(a)!=m["matrix_hash"] or array_hash(b)!=m["target_hash"]:
            raise ValueError("validation matrix/target hash mismatch")
    observations=fields[4:6]
    initial={scheme:[D.prepare_initialization(x,u[0],scheme=scheme,boundary_realization=D.BOUNDARY)
                     for u in observations] for scheme in D.SCHEMES}
    return {"x":x,"t":t,"observations":observations,"matrices":matrices,"targets":targets,
            "metadata":metadata,"raw":raw,"initial":initial,"sigma":sigma,"ids":ids,
            "cohort":cohort,"new":new}


def _fixed_weak_residual(candidate,cohort):
    names=[canonical_term(name) for name in cohort["metadata"]["names"]]
    selected=[names.index(canonical_term(name)) for name in candidate["support"]]
    coefficient=np.asarray(candidate["coefficients"])
    errors=[float(np.linalg.norm(a[:,selected]@coefficient-b)/np.linalg.norm(b))
            for a,b in zip(cohort["matrices"],cohort["targets"])]
    return {"trajectory_relative_errors":errors,"rms_relative_error":float(np.sqrt(np.mean(np.square(errors)))),
            "coefficient_refit":False,"support_unchanged":list(candidate["support"])}


def _aggregate_dynamic(reports):
    if any(r["status"]!="supported_full_resolved_spectral_forecast" for r in reports):
        return {"status":"unsupported","relative_error":None,"log_relative_error":None,
                "not_counted_as_algorithm_failure":True}
    logs=[]
    for report in reports:
        metric=report["error_metric"]
        if metric["status"]=="zero_residual":logs.append(-np.inf)
        elif metric["log_relative_error"] is not None:logs.append(metric["log_relative_error"])
        else:return {"status":"undefined_relative_error","relative_error":None,"log_relative_error":None}
    value=float(.5*(logsumexp(2*np.asarray(logs))-math.log(len(logs))))
    if np.isneginf(value):return {"status":"zero_residual","relative_error":0.,"log_relative_error":None}
    return {"status":"finite" if value<=D.LOG_MAX else "above_float64_range_use_log",
            "relative_error":float(np.exp(value)) if value<=D.LOG_MAX else None,"log_relative_error":value,
            "aggregation":"RMS of per-trajectory relative errors; equal trajectory weight"}


def main():
    output=ROOT/"stage2b_heat_dynamic_compare_result.json"
    if output.exists():raise FileExistsError("preserve completed dynamic diagnostic; new candidates need a new version")
    source_path=ROOT/"stage2a_heat_reconciled_result.json"
    source=_load(source_path)
    full_gate=_load(ROOT/"stage2b_heat_full_spectrum_result.json")
    if not full_gate["all_checks_passed"]:raise ValueError("full-spectrum numerical gate failed")
    if full_gate["source_sha256"]!=_hash(ROOT/"stage2b_heat_full_spectrum.py"):
        raise ValueError("full-spectrum evaluator differs from checked version")
    source_hash=_hash(source_path)
    with threadpool_limits(limits=1):
        data_review=review_new_data()
        cohorts={(sigma,False):_cohort(sigma) for sigma in (.01,.05)}
        cohorts[(.01,True)]=_cohort(.01,new=True)
        rows=[]
        for dataset in source["rows"]:
            if dataset["case"]!="heat" or dataset["sigma"] not in (.01,.05):continue
            sigma=dataset["sigma"]
            for layer,library in dataset["layers"].items():
                for method,result in library["methods"].items():
                    evaluation=result["evaluation"]
                    selected=evaluation["primary_selected_candidate_id"]
                    if selected is None or not result["path"]["all_solutions_certified"]:
                        rows.append({"sigma":sigma,"layer":layer,"method":method,"status":"numerically_incomplete_no_primary"})
                        continue
                    candidate=next(c for c in evaluation["candidates"] if c["candidate_id"]==selected)
                    coefficients=dict(zip(candidate["support"],candidate["coefficients"]))
                    for new in ([False,True] if sigma==.01 else [False]):
                        cohort=cohorts[(sigma,new)]
                        weak=_fixed_weak_residual(candidate,cohort)
                        for scheme in D.SCHEMES:
                            forecasts=[]
                            for uid,u,initial in zip(cohort["ids"],cohort["observations"],cohort["initial"][scheme]):
                                forecast,_=D.evaluate_operator(coefficients,cohort["x"],cohort["t"],u,initial,
                                    boundary_realization=D.BOUNDARY,return_prediction=False)
                                forecasts.append({"trajectory_id":uid,**forecast})
                            rows.append({"case":"heat","training_sigma":sigma,"validation_sigma":cohort["sigma"],
                                "training_noise_seed":dataset["seed"],"validation_noise_seed":1 if new else 0,
                                "layer":layer,"method":method,"initialization_scheme":scheme,
                                "source_primary_candidate_id":selected,"source_primary_support":candidate["support"],
                                "frozen_physical_coefficients":coefficients,"coefficient_source":"original training-only fixed-support LS readout",
                                "source_primary_weak_validation_error":candidate["validation_error"],
                                "source_primary_weak_validation_ids":[4,5],"source_selection_rule":evaluation["selection_rule"],
                                "dynamic_validation_cohort":cohort["cohort"],"dynamic_validation_ids":cohort["ids"],
                                "cohort_previously_used_for_this_models_selection":not new,
                                "matched_noise_level":True,"raw_bundle_sha256":cohort["raw"]["npz_sha256"],
                                "fixed_model_weak_error_on_this_cohort":weak,
                                "dynamic_error_equal_trajectory_rms":_aggregate_dynamic(forecasts),
                                "per_trajectory":forecasts,"no_refit_no_reselection":True,
                                "support_not_deleted_by_LS":True,"test_data_used":False})
    checks={"independent_new_data_review":data_review["all_checks_passed"],
            "all_24_declared_model_cohort_initialization_records_present":len(rows)==24,
            "all_noise_comparisons_matched":all(r.get("training_sigma")==r.get("validation_sigma") for r in rows),
            "source_frozen_models_unchanged":_hash(source_path)==source_hash,
            "new_cohort_only_1percent":all(r["training_sigma"]==.01 for r in rows if r.get("cohort_previously_used_for_this_models_selection") is False),
            "all_supported_forecasts_use255_modes":all(f.get("mode_count")==255 for r in rows for f in r.get("per_trajectory",[]) if f["status"]=="supported_full_resolved_spectral_forecast")}
    report={"stage":"2b_heat_frozen_model_dynamic_diagnostic","all_checks_passed":all(checks.values()),
        "checks":checks,"source_models":source_path.name,"source_sha256":source_hash,"new_data_review":data_review,
        "rows":rows,"method_selection_changed":False,"new_statistical_method_results_used":False,
        "test_data_generated_or_used":False,"clean_reference_used":False,
        "source_script_sha256":_hash(Path(__file__)),
        "evaluator_sha256":_hash(ROOT/"stage2b_heat_full_spectrum.py"),
        "protocol_sha256":_hash(ROOT/"stage2b_protocol_result.json"),
        "coverage_summary":{"records":len(rows),"supported":sum(r.get("dynamic_error_equal_trajectory_rms",{}).get("status")!="unsupported" for r in rows),
            "unsupported":sum(r.get("dynamic_error_equal_trajectory_rms",{}).get("status")=="unsupported" for r in rows)},
        "limits":["Dynamic diagnostics keep the original weak-residual-selected primary model; no stability-based or rollout-based reselection occurs.",
            "Unsupported structures stay visible and are not failures or silently filtered competitors.",
            "Old validation IDs4,5 were used in primary selection; new IDs16,17 provide independent development validation for these original models.",
            "Only1% original models are evaluated on the new1% cohort; no cross-noise result is mixed into matched comparisons.",
            "Raw and frozen-smoothed t0 initialization use the same common scheme for all methods, retaining all255 resolved modes.",
            "High-order propagation uses the explicit Dirichlet-Laplacian polynomial boundary realization, not unspecified general PDE boundary conditions.",
            "Full finite-grid dynamics and continuum stability are separate; positive finite-band growth is not automatically high-frequency ill-posedness.",
            "This is development evidence, not a formal multi-seed robustness ranking or final-test result."]}
    strict_json_write(output,report)
    print({"all_checks_passed":report["all_checks_passed"],"records":len(rows),"coverage":report["coverage_summary"]})
    if not report["all_checks_passed"]:raise SystemExit(1)


if __name__=="__main__":main()
