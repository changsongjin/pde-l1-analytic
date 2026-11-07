"""Post-selection full-spectrum diagnostics for the finished stage-2b heat run.

Native validation for 12 frozen models, plus original-cohort models evaluated
on new-cohort validation. Both initializations retain every resolved mode.
No fitting, model selection, or handling of unsupported models as losses.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from threadpoolctl import threadpool_limits

from stage0_evaluation import strict_json_write
import stage2b_heat_full_spectrum as D
import stage2b_heat_dynamic_compare as V1

ROOT=Path(__file__).resolve().parent
OLD="development_ids0_5_seed0"
NEW="independent_development_ids12_17_seed1"
METHODS={"L1","IRL1_original","IRL1_smoothed"}


def run(source_path,output_path):
    source_path,output_path=Path(source_path),Path(output_path)
    if output_path.exists():raise FileExistsError("preserve completed diagnostic; choose a new version")
    source=V1._load(source_path)
    if not source.get("execution_complete"):
        raise RuntimeError("research task is still running; dynamic comparison not started")
    if source.get("test_data_used",False):raise ValueError("development-only source required")
    source_hash=V1._hash(source_path)
    frozen=V1._load(ROOT/"stage2b_protocol_result.json")
    if source["alpha_frozen"]!=frozen["selection_alpha"]:
        raise ValueError("primary score differs from frozen protocol")
    full_gate=V1._load(ROOT/"stage2b_heat_full_spectrum_result.json")
    if not full_gate["all_checks_passed"] or full_gate["source_sha256"]!=V1._hash(ROOT/"stage2b_heat_full_spectrum.py"):
        raise ValueError("full-spectrum evaluator changed or numerical gate not passed")
    datasets=[r for r in source["rows"] if r["case"]=="heat" and r["sigma"]==.01]
    if len(datasets)!=2 or {r["cohort"] for r in datasets}!={OLD,NEW}:
        raise ValueError("need exactly the original and independent heat1% cohorts")
    data_review=V1.review_new_data()
    cohorts={OLD:V1._cohort(.01,False),NEW:V1._cohort(.01,True)}
    rows=[]
    for dataset in datasets:
        native=cohorts[dataset["cohort"]]
        if dataset["raw_bundle_sha256"]!=native["raw"]["npz_sha256"]:
            raise ValueError("selected model source cohort changed")
        if set(dataset["layers"])!={"J18","J1158"}:
            raise ValueError("source is missing a declared library")
        for layer,library in dataset["layers"].items():
            if set(library["methods"])!=METHODS:raise ValueError("source has missing or extra declared methods")
            for method,model in library["methods"].items():
                evaluation=model["evaluation"]
                selected=evaluation["primary_selected_candidate_id"]
                available=(model["path"]["all_solutions_certified"] and evaluation["selection_eligible"]
                           and selected is not None)
                candidate=(next(c for c in evaluation["candidates"] if c["candidate_id"]==selected)
                           if available else None)
                targets=[dataset["cohort"]]+([NEW] if dataset["cohort"]==OLD else [])
                for target in targets:
                    cohort=cohorts[target]
                    coefficients=(dict(zip(candidate["support"],candidate["coefficients"])) if available else None)
                    weak=V1._fixed_weak_residual(candidate,cohort) if available else None
                    for scheme in D.SCHEMES:
                        row={"case":"heat","training_sigma":.01,"validation_sigma":.01,
                            "source_cohort":dataset["cohort"],"dynamic_validation_cohort":target,
                            "evaluation_relation":"native_validation" if target==dataset["cohort"] else "old_model_external_new_validation",
                            "training_noise_seed":dataset["seed"],"validation_noise_seed":1 if target==NEW else 0,
                            "training_ids":dataset["splits"]["train_trajectories"],
                            "source_primary_validation_ids":dataset["splits"]["validation_trajectories"],
                            "dynamic_validation_ids":cohort["ids"],
                            "cohort_previously_used_for_this_models_selection":target==dataset["cohort"],
                            "layer":layer,"method":method,"initialization_scheme":scheme,
                            "path_status":model["path"]["status"],"source_primary_candidate_id":selected if available else None,
                            "source_model_raw_bundle_sha256":dataset["raw_bundle_sha256"],
                            "dynamic_raw_bundle_sha256":cohort["raw"]["npz_sha256"],
                            "no_refit_no_reselection":True,"test_data_used":False}
                        if not available:
                            row.update({"status":"numerically_unavailable","primary_model_used":False,
                                "exploratory_candidate_substituted":False,"dynamic_error_equal_trajectory_rms":None,
                                "not_counted_as_statistical_failure":True})
                            rows.append(row)
                            continue
                        forecasts=[]
                        for uid,u,initial in zip(cohort["ids"],cohort["observations"],cohort["initial"][scheme]):
                            forecast,_=D.evaluate_operator(coefficients,cohort["x"],cohort["t"],u,initial,
                                boundary_realization=D.BOUNDARY,return_prediction=False)
                            forecasts.append({"trajectory_id":uid,**forecast})
                        aggregate=V1._aggregate_dynamic(forecasts)
                        row.update({"status":"unsupported" if aggregate["status"]=="unsupported" else "diagnosed",
                            "primary_model_used":True,"source_primary_support":candidate["support"],
                            "frozen_physical_coefficients":coefficients,
                            "source_primary_weak_validation_error":candidate["validation_error"],
                            "source_selection_rule":evaluation["selection_rule"],
                            "fixed_model_weak_error_on_dynamic_cohort":weak,
                            "support_not_deleted_by_LS":True,"per_trajectory":forecasts,
                            "dynamic_error_equal_trajectory_rms":aggregate,
                            "existing_postselection_support_diagnostic":candidate.get("truth_evaluation_after_selection"),
                            "truth_diagnostic_role":"copied after original selection; not used for initialization or dynamics"})
                        rows.append(row)
    checks={"data_review_passed":data_review["all_checks_passed"],"all36_diagnostic_slots_present":len(rows)==36,
        "native_records24_external_records12":sum(r["evaluation_relation"]=="native_validation" for r in rows)==24
            and sum(r["evaluation_relation"]=="old_model_external_new_validation" for r in rows)==12,
        "noise_levels_matched":all(r["training_sigma"]==r["validation_sigma"]==.01 for r in rows),
        "all_supported_forecasts_keep255_modes":all(f["mode_count"]==255 and f["modes_dropped"]==0
            for r in rows for f in r.get("per_trajectory",[]) if f["status"]=="supported_full_resolved_spectral_forecast"),
        "source_results_unchanged":V1._hash(source_path)==source_hash}
    coverage={key:sum(r["status"]==key for r in rows) for key in ("diagnosed","unsupported","numerically_unavailable")}
    report={"stage":"2b_heat_full_spectrum_all_frozen_methods","all_checks_passed":all(checks.values()),
        "pass_meaning":"diagnostic integrity/coverage, not all models stable or accurate", "checks":checks,
        "source_file":source_path.name,"source_sha256":source_hash,"source_execution_complete":True,
        "source_all_numerically_complete":source.get("all_checks_passed"),"coverage":coverage,"rows":rows,
        "source_components_sha256":{name:V1._hash(ROOT/name) for name in [
            "stage2b_heat_dynamic_compare_v2.py","stage2b_heat_dynamic_compare.py","stage2b_heat_full_spectrum.py",
            "stage2b_protocol_result.json","stage2b_data_result.json"]},
        "new_data_review":data_review,"test_data_generated_or_used":False,
        "coefficient_or_support_or_primary_selection_changed":False,"clean_reference_used":False,
        "limits":["The raw and frozen-smoothed initializations are common to every method and use t0 only.",
            "No failed numerical path is replaced by an exploratory candidate; its dynamic slot remains unavailable.",
            "Unsupported structures remain visible and are not failed algorithms or silently deleted candidates.",
            "Native validation was used for primary weak-residual selection; only old-model forecasts on new IDs16,17 are independent of those original selections.",
            "All models retain their original training-only LS readout and L1-selected support.",
            "Full255-mode spectral propagation relies on a Dirichlet-Laplacian polynomial boundary realization.",
            "Development evidence on one new cohort does not establish formal noise robustness or final-test accuracy."]}
    strict_json_write(output_path,report)
    return {"all_checks_passed":report["all_checks_passed"],"coverage":coverage,"records":len(rows),
            "source":source_path.name,"source_sha256":source_hash}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,default=ROOT/"stage2b_compare_result.json")
    parser.add_argument("--output",type=Path,default=ROOT/"stage2b_heat_dynamic_compare_v2_result.json")
    args=parser.parse_args()
    with threadpool_limits(limits=1):result=run(args.source,args.output)
    print(result)
    if not result["all_checks_passed"]:raise SystemExit(1)


if __name__=="__main__":main()
