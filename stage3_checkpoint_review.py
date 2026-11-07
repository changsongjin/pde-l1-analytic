"""Read-only audit of a stopped formal campaign; no observations or fits selected.

Rechecks fixed numerical certificates and independently reconstructs selection
scores and held-out residuals. New timestamped reports preserve prior audits.
The same frozen certificate primitives are used; this is not another solver.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

import stage1_extended_evaluation as E
import stage3_campaign as C
import stage3_data as OLD
import stage3_data_v2 as DATA
from stage0_data_bundle import load_bundle
from stage0_evaluation import array_hash, normalize_training
from stage2_identifiability import stack_split
from stage2_data import CASES

ROOT = Path(__file__).resolve().parent


def require(ok, message):
    if not ok:
        raise ValueError(message)


def close(actual, expected, message):
    require(np.allclose(actual, expected, rtol=2e-11, atol=2e-25), message)


def matrices(directory, role):
    path = directory / (role + "_matrices.npz")
    meta = C.read(path.with_suffix(".json"))
    require(C.digest(path) == meta["npz_sha256"], "matrix file hash")
    with np.load(path, allow_pickle=False) as data:
        aa = [data[f"a_{i}"].copy() for i in range(6)]
        bb = [data[f"b_{i}"].copy() for i in range(6)]
    for a, b, record in zip(aa, bb, meta["trajectories"]):
        require(array_hash(a) == record["matrix_hash"] and array_hash(b) == record["target_hash"], "matrix array hashes")
    return aa, bb, meta


def method_review(ref, aa, bb, meta, protocol):
    record = C.read(C.check_reference(ref))
    require(record['truth_used'] is False and record['test_data_used'] is False, 'selection data roles')
    job = C.read(C.check_reference(record["job"]))
    indices = meta["library_indices"][record["layer"]]
    blocks = [a[:, indices] for a in aa]
    a, b = stack_split(blocks, bb, range(4))
    av, bv = stack_split(blocks, bb, [4, 5])
    an, bn, scales = normalize_training(a, b)
    require(record["normalization"] == scales == job["normalization"], "normalization")
    require(record["solver_input_hashes"] == {"A": array_hash(an), "b": array_hash(bn)}, "solver hashes")
    path = C.read(C.check_reference(record["path_source"]))
    certificates = C.validate_campaign_path(path, an, bn, record["method"])
    require(record["numerically_complete"] == path["all_solutions_certified"], "completion")
    ev = record["evaluation"]
    names = [meta["names"][i] for i in indices]
    by_id = {c["candidate_id"]: c for c in ev["candidates"]}
    for mapping in ev["point_to_candidate"]:
        point = path["points"][mapping["point_index"]]
        support = np.flatnonzero(np.abs(point["beta"]) > protocol["support_threshold"]).tolist()
        candidate = by_id[mapping["candidate_id"]]
        require(candidate["support_indices"] == support, "L1 support mapping")
        require(candidate["support"] == [names[i] for i in support], "support labels")
    # Compute every saved candidate's readout using training observations only.
    # These verification coefficients are never written into campaign artifacts.
    for candidate in by_id.values():
        support = candidate["support_indices"]
        cols = a[:, support]
        norms = np.linalg.norm(cols, axis=0)
        norms[norms == 0] = 1.
        beta = np.linalg.lstsq(cols / norms, b, rcond=None)[0]
        coeff = beta / norms
        close(candidate["coefficients"], coeff, "training readout")
        error = np.linalg.norm(av[:, support] @ coeff - bv) / np.linalg.norm(bv)
        close(candidate["validation_error"], error, "validation error")
        require(candidate["complexity"] == sum(E.term_complexity(n) for n in candidate["support"]), "complexity")
    chosen = min(by_id.values(), key=lambda c: (c["validation_error"] + protocol["selection_alpha"] * c["complexity"],
        c["complexity"], c["validation_error"], tuple(c["support_indices"]))) if by_id else None
    require(ev["primary_selected_candidate_id"] == (chosen["candidate_id"] if record["numerically_complete"] else None), "fixed primary selection")
    return record, len(certificates), sum(p["status"] == "certified" for p in path["points"])


def main():
    started = time.perf_counter()
    protocol = DATA.load_frozen_protocol()
    index_ref = C.reference(C.INDEX)
    index = C.read(C.INDEX)
    require(index["batches"][-1]["status"] != "running", "Stop campaign before this numerical audit")
    completed, pending, selected_rows, reports = 0, 0, [], []
    final_points, all_points, complete_methods = 0, 0, 0
    with threadpool_limits(limits=1):
        for tag, row in index["datasets"].items():
            directory = C.DIRECTORY / tag
            methods = row.get("methods", {})
            if not methods:
                reports.append({"dataset": row["dataset"], "status": row.get("status"), "checked_methods": 0})
                continue
            aa, bb, meta = matrices(directory, "fit")
            identity = row["dataset"]
            fit_path = OLD._bundle_path(identity["case"], identity["sigma"], identity["noise_seed"], "fit")
            fit = load_bundle(fit_path)
            require(fit[3]["npz_sha256"] == meta["raw_bundle_sha256"], "fit raw matrix binding")
            for u, trajectory in zip(fit[2], meta["trajectories"]):
                require(array_hash(u) == trajectory["raw_hash"], "fit trajectory matrix binding")
            for value in methods.values():
                record, count, finals = method_review(value["result"], aa, bb, meta, protocol)
                require(record["numerically_complete"] == value["numerically_complete"], "index completion flag")
                all_points += count
                final_points += finals
                complete_methods += int(record["numerically_complete"])
            detail = {"dataset": identity, "status": row.get("status"), "checked_methods": len(methods)}
            if row.get("status") != "complete":
                pending += 1
                test_file = OLD._bundle_path(identity["case"], identity["sigma"], identity["noise_seed"], "test")
                if test_file.exists():
                    # A process deadline may interrupt evaluation after valid release.
                    selection = C.read(directory / "selection_manifest.json")
                    DATA.validate_selection_bindings(selection, protocol)
                    OLD._selection_headers(selection, identity["case"], identity["sigma"], identity["noise_seed"], protocol, index["protocol"])
                    for model in selection["models"]:
                        OLD._validate_selected_model(model, protocol)
                    detail["sealed_before_interrupted_test_evaluation"] = True
                reports.append(detail)
                continue
            result = C.read(C.check_reference(row["result"]))
            for ref in result["artifacts"]:
                C.check_reference(ref)
            manifest_path = C.check_reference(result["selection_manifest"])
            selection = C.read(manifest_path)
            OLD._selection_headers(selection, identity["case"], identity["sigma"], identity["noise_seed"], protocol, index["protocol"])
            DATA.validate_selection_bindings(selection, protocol)
            for model in selection["models"]:
                OLD._validate_selected_model(model, protocol)
            test_path = OLD._bundle_path(identity["case"], identity["sigma"], identity["noise_seed"], "test")
            test = load_bundle(test_path)
            require(test[3]["selection_manifest"] == {"file":str(manifest_path.relative_to(ROOT)), "sha256":C.digest(manifest_path)}, "test release manifest")
            ta, tb, tm = matrices(directory, "test")
            require(tm["raw_bundle_sha256"] == test[3]["npz_sha256"], "test raw binding")
            for u, trajectory in zip(test[2], tm["trajectories"]):
                require(array_hash(u) == trajectory["raw_hash"], "test trajectory matrix binding")
            evaluation = C.read(C.check_reference(result["test_evaluation"]))
            require(evaluation["no_test_refit_or_selection"] is True, "test no refit flag")
            for model, saved in zip(selection["models"], evaluation["rows"]):
                require(saved["selected_model_sha256"] == DATA.selected_model_digest(model), "model hash at evaluation")
                require(saved["support"] == model["support"] and saved["physical_coefficients"] == model["physical_coefficients"], "frozen support/coefficient")
                js = [tm["names"].index(n) for n in model["support"]]
                coeff = np.array([model["physical_coefficients"][n] for n in model["support"]])
                errors = [float(np.linalg.norm(a[:, js] @ coeff - b) / np.linalg.norm(b)) for a, b in zip(ta, tb)]
                close(errors, [r["relative_error"] for r in saved["heldout_weak"]["trajectory_errors"]], "test residuals")
                close(math.hypot(*errors) / math.sqrt(6), saved["heldout_weak"]["rms_relative_error"], "test RMS")
                truth = E.support_metrics(model["support"], CASES[identity["case"]]["true"])
                require(all(saved["truth_evaluation_after_selection"][k] == v for k, v in truth.items()), "truth support metrics")
                selected_rows.append({**identity, "library":model["layer"], "method":model["method"],
                    "support":model["support"], "truth":truth,
                    "test_weak_error":saved["heldout_weak"]["rms_relative_error"]})
            completed += 1
            reports.append(detail)
            print(tag, "reviewed", flush=True)
    require(C.digest(C.INDEX) == index_ref["sha256"], "index changed during stopped audit")
    summary = []
    for case in protocol["cases"]:
        for sigma in protocol["noise_levels"]:
            seeds = {r["noise_seed"] for r in selected_rows if r["case"] == case and r["sigma"] == sigma}
            if seeds:
                summary.append({"case":case, "sigma":sigma, "completed_six_method_seeds":sorted(seeds),
                    "planned_seeds":1 if sigma == 0 else 20,
                    "formal_noise_interval_eligible":sigma != 0 and seeds == set(protocol["noise_seeds"])})
    now = datetime.now(timezone.utc)
    report = {"stage":"3_read_only_checkpoint_review", "all_checks_passed":True,
        "created_utc":now.isoformat(), "index":index_ref, "source":C.reference(Path(__file__)),
        "protocol":index["protocol"], "checked_datasets":reports, "completed_datasets":completed,
        "pending_datasets_with_methods":pending, "complete_method_paths":complete_methods,
        "certified_final_points":final_points, "certified_points_including_outer":all_points,
        "selected_rows":selected_rows, "completion_by_case_noise":summary,
        "formal_grid_complete":index["execution_complete"],
        "no_new_observations_or_solver_calls":True, "no_saved_campaign_artifact_modified":True,
        "elapsed_s":time.perf_counter()-started,
        "scope":"Certificates recomputed with frozen primitives; scores/readouts and test residuals recomputed from saved matrices. Dynamics, source formula accuracy and full field-to-matrix reconstruction are covered by prior development checks, not revalidated here.",
        "inference_limit":"Only complete 20-seed paired sets qualify for formal noise intervals. No algorithm ranking from an incomplete grid; clean cases are deterministic single runs."}
    path = ROOT / ("stage3_checkpoint_review_" + now.strftime("%Y%m%dT%H%M%SZ") + "_result.json")
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"file":path.name, "all_checks_passed":True, "completed_datasets":completed,
        "complete_method_paths":complete_methods, "certified_final_points":final_points}))


if __name__ == "__main__":
    main()
