"""Additional dataset/job/matrix/path binding before the v1 held-out release.

The original data gateway stays unchanged. This wrapper recomputes solver input
hashes from the saved maximum fit matrix and the frozen library subsets before
delegating to its existing six-selected-model release checks. No solver or
field generation is involved in the additional binding audit.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile

import numpy as np
from threadpoolctl import threadpool_limits

import stage3_data as OLD
import stage1_extended_evaluation as E
from stage0_evaluation import array_hash, normalize_training, strict_json_write
from stage1_grammar import LAYERS, enumerate_jet, key, nonpolynomial_recipes
from stage2_identifiability import stack_split

ROOT, PROTOCOL_PATH = OLD.ROOT, OLD.PROTOCOL_PATH
selected_model_digest = OLD.selected_model_digest
digest = OLD.digest


def _reference(file, sha256): return {"file":file, "sha256":sha256}


def _read_reference(reference):
    path = OLD._project_file(reference["file"])
    if digest(path) != reference["sha256"]: raise ValueError("selected artifact checksum changed")
    return OLD._read(path), path


def load_frozen_protocol():
    protocol = OLD.load_frozen_protocol()
    if "stage3_data_v2.py" not in protocol["source_sha256"]:
        raise ValueError("the frozen protocol omits the binding gateway v2")
    return protocol


def get_fit_bundle(case, sigma, seed):
    load_frozen_protocol()
    return OLD.get_fit_bundle(case, sigma, seed)


def _catalog(case):
    terms = enumerate_jet()
    small_keys = {key(term) for term in enumerate_jet(*LAYERS["J18"])}
    small = [i for i, term in enumerate(terms) if key(term) in small_keys]
    names = [term["name"] for term in terms]
    if case == "wave":
        small.append(len(names)); names.append("u_t")
    n_poly = len(names)
    names += [r["name"] for r in nonpolynomial_recipes()]
    return names, [E.term_complexity(n) for n in names], {
        "J18":small, "J1158":list(range(n_poly)), "J1212":list(range(len(names)))}


def _load_bound_matrix(selection, protocol):
    first = selection["models"][0]["trainfit_provenance"]
    reference = _reference(first["fit_matrix_metadata_file"], first["fit_matrix_metadata_sha256"])
    for model in selection["models"]:
        p = model["trainfit_provenance"]
        if _reference(p["fit_matrix_metadata_file"], p["fit_matrix_metadata_sha256"]) != reference:
            raise ValueError("the six models do not share one maximum fit-matrix cache")
    meta, metadata_path = _read_reference(reference)
    identity = selection["dataset"]
    names, costs, indices = _catalog(identity["case"])
    expected = {**identity, "role":"fit", "trajectory_ids":list(range(6)),
                "raw_bundle_sha256":selection["fit_bundle_sha256"], "protocol":selection["protocol"],
                "source_sha256":protocol["source_sha256"], "names":names, "complexities":costs,
                "library_indices":indices, "test_data_used":False}
    if any(meta.get(k) != v for k,v in expected.items()):
        raise ValueError("fit-matrix dataset/role/raw/protocol/catalog binding mismatch")
    matrix_path = metadata_path.with_suffix(".npz")
    if digest(matrix_path) != meta["npz_sha256"] or len(meta["trajectories"]) != 6:
        raise ValueError("fit-matrix payload checksum or trajectory count mismatch")
    matrices, targets = [], []
    with np.load(matrix_path, allow_pickle=False) as arrays:
        if set(arrays.files) != {f"{prefix}_{i}" for prefix in ("a", "b") for i in range(6)}:
            raise ValueError("fit matrix must contain exactly six observed trajectory matrices/targets")
        for i, entry in enumerate(meta["trajectories"]):
            a, b = arrays[f"a_{i}"].copy(), arrays[f"b_{i}"].copy()
            if (entry["trajectory_id"] != i or a.shape != (len(b),len(names)) or entry["shape"] != list(a.shape)
                    or array_hash(a) != entry["matrix_hash"] or array_hash(b) != entry["target_hash"]):
                raise ValueError("fit trajectory matrix/target checksum or identity mismatch")
            matrices.append(a); targets.append(b)
    return matrices, targets, meta


def _layer_inputs(matrices, targets, meta, layer):
    subset = meta["library_indices"][layer]
    sliced = [a[:,subset] for a in matrices]
    a,b = stack_split(sliced, targets, range(4)); av,bv = stack_split(sliced, targets, [4,5])
    an,bn,scales = normalize_training(a,b)
    return {"input_hashes":{"train_matrix":array_hash(a), "train_target":array_hash(b),
                             "validation_matrix":array_hash(av), "validation_target":array_hash(bv)},
            "solver_input_hashes":{"A":array_hash(an), "b":array_hash(bn)}, "normalization":scales,
            "names":[meta["names"][i] for i in subset], "complexities":[meta["complexities"][i] for i in subset],
            "solver_shape":list(an.shape)}


def _validate_model_binding(selection, protocol, model, meta, computed):
    p = model["trainfit_provenance"]
    record, _ = _read_reference(_reference(p["method_result_file"], p["method_result_sha256"]))
    job, _ = _read_reference(record["job"])
    expected_identity = {**selection["dataset"], "layer":model["layer"], "method":model["method"]}
    for artifact in (record, job):
        if (any(artifact.get(k) != v for k,v in expected_identity.items())
                or artifact.get("protocol") != selection["protocol"]):
            raise ValueError("selected method/job belongs to another dataset/library/method/protocol")
    if (job.get("source_sha256") != protocol["source_sha256"]
            or job.get("raw_bundle_sha256") != selection["fit_bundle_sha256"]
            or job.get("lambda_ratios") != protocol["lambda_ratios"]
            or job.get("support_threshold") != protocol["support_threshold"]
            or job.get("selection_alpha") != protocol["selection_alpha"]
            or job.get("budget_s") != protocol["initial_path_budget_s"]
            or record.get("truth_used") is not False or record.get("test_data_used") is not False
            or record.get("numerically_complete") is not True):
        raise ValueError("method/job raw input, frozen objective, or fit-only role mismatch")
    for field in ("input_hashes", "solver_input_hashes", "normalization"):
        if record.get(field) != computed[field] or job.get(field) != computed[field]:
            raise ValueError("saved method/job does not use this maximum-matrix subset: "+field)
    if (p.get("input_hashes") != computed["input_hashes"] or p.get("normalization") != computed["normalization"]
            or job.get("names") != computed["names"] or job.get("complexities") != computed["complexities"]
            or p.get("coefficient_fit_split") != "train_ids0_3_only"):
        raise ValueError("sealed training readout provenance differs from its frozen subset")
    path_reference = _reference(p["path_source_file"], p["path_source_sha256"])
    if record.get("path_source") != path_reference:
        raise ValueError("sealed solver path is not the path referenced by the method record")
    path, _ = _read_reference(path_reference)
    if path.get("method") != model["method"] or path.get("shape") != computed["solver_shape"]:
        raise ValueError("path method/shape differs from its job's matrix subset")
    if "input_hashes" in path and path["input_hashes"] != computed["solver_input_hashes"]:
        raise ValueError("path's explicit solver input hashes differ from the bound job")
    evaluation = record["evaluation"]
    selected = next((c for c in evaluation["candidates"] if c["candidate_id"] == p["candidate_id"]), None)
    if selected is None or selected["support"] != [computed["names"][i] for i in selected["support_indices"]]:
        raise ValueError("selected support labels differ from its bound library indices")
    selected_indices = [entry["point_index"] for entry in evaluation["point_to_candidate"]
                        if entry["candidate_id"] == p["candidate_id"]]
    if not selected_indices or p["selected_path_point_indices"] != selected_indices:
        raise ValueError("selected candidate is not linked to the declared solver path points")
    for i in selected_indices:
        point = path["points"][i]
        beta = np.asarray(point["beta"])
        if (beta.shape != (computed["solver_shape"][1],)
                or point["support"] != selected["support_indices"]
                or np.flatnonzero(np.abs(beta)>protocol["support_threshold"]).tolist() != selected["support_indices"]):
            raise ValueError("selected support does not come from the bound L1 path")


def validate_selection_bindings(selection, protocol):
    """Pure provenance audit; it cannot generate fields or unlock test alone."""
    matrices, targets, meta = _load_bound_matrix(selection, protocol)
    inputs = {}
    with threadpool_limits(limits=1):
        for model in selection["models"]:
            layer = model["layer"]
            if layer not in inputs: inputs[layer] = _layer_inputs(matrices, targets, meta, layer)
            _validate_model_binding(selection, protocol, model, meta, inputs[layer])
    return {"all_bindings_verified":True, "models":len(selection["models"]),
            "maximum_matrix_sha256":meta["npz_sha256"], "raw_bundle_sha256":meta["raw_bundle_sha256"]}


def get_test_bundle(case, sigma, seed, selection_manifest_path):
    protocol = load_frozen_protocol()
    case,sigma,seed = OLD._request(case,sigma,seed,protocol)
    path = OLD._project_file(selection_manifest_path)
    before = digest(path)
    selection = OLD._read(path)
    OLD._selection_headers(selection,case,sigma,seed,protocol,{"file":PROTOCOL_PATH.name,"sha256":digest(PROTOCOL_PATH)})
    validate_selection_bindings(selection,protocol)
    if digest(path) != before: raise ValueError("selection changed during binding audit")
    return OLD.get_test_bundle(case,sigma,seed,path)


def main():
    output = ROOT/"stage3_data_v2_preflight_result.json"
    if output.exists(): raise FileExistsError("preserve existing preflight result")
    formal_before = sorted(str(p.relative_to(ROOT)) for p in (ROOT/"stage3_data").glob("**/*.npz"))
    checks, rejection_cases = {}, []
    with tempfile.TemporaryDirectory(prefix="stage3_binding_synthetic_", dir=ROOT) as temporary:
        directory = Path(temporary)
        def write(name, value):
            path = directory/name
            strict_json_write(path,value)
            return {"file":str(path.relative_to(ROOT)).replace("\\","/"),"sha256":digest(path)}
        identity = {"case":"heat","sigma":.01,"noise_seed":1000}  # Labels only; no formal observations.
        protocol = {"source_sha256":{"synthetic_only":"not_frozen"},"lambda_ratios":np.geomspace(1.,1e-6,24).tolist(),
                    "support_threshold":1e-5,"selection_alpha":.005,"initial_path_budget_s":120.}
        protocol_ref = {"file":"synthetic_protocol_only","sha256":"not_a_formal_protocol"}
        names,costs,indices = _catalog("heat")
        matrices = [np.zeros((2,len(names))) for _ in range(6)]
        for i,a in enumerate(matrices): a[:,names.index("u")] = [1.+i,.5+i]
        targets = [np.array([1.+i,2.+i]) for i in range(6)]
        npz = directory/"fit_matrices.npz"
        np.savez_compressed(npz,**{f"a_{i}":a for i,a in enumerate(matrices)},**{f"b_{i}":b for i,b in enumerate(targets)})
        meta = {**identity,"role":"fit","trajectory_ids":list(range(6)),"raw_bundle_sha256":"synthetic_raw_only",
                "protocol":protocol_ref,"source_sha256":protocol["source_sha256"],"names":names,"complexities":costs,
                "library_indices":indices,"test_data_used":False,"npz_sha256":digest(npz),
                "trajectories":[{"trajectory_id":i,"shape":list(a.shape),"matrix_hash":array_hash(a),"target_hash":array_hash(targets[i])}
                                for i,a in enumerate(matrices)]}
        meta_ref = write("fit_matrices.json",meta)
        computed = _layer_inputs(matrices,targets,meta,"J18")
        uid = computed["names"].index("u")
        beta = np.zeros(len(computed["names"])); beta[uid] = .5
        path = {"method":"L1","shape":computed["solver_shape"],"points":[{"beta":beta.tolist(),"support":[uid]}]}
        path_ref = write("path.json",path)
        job = {**identity,"layer":"J18","method":"L1","protocol":protocol_ref,
               "source_sha256":protocol["source_sha256"],"raw_bundle_sha256":"synthetic_raw_only",
               **{k:computed[k] for k in ("input_hashes","solver_input_hashes","normalization","names","complexities")},
               "lambda_ratios":protocol["lambda_ratios"],"support_threshold":1e-5,"selection_alpha":.005,"budget_s":120.}
        job_ref = write("job.json",job)
        record = {**identity,"layer":"J18","method":"L1","protocol":protocol_ref,"job":job_ref,"path_source":path_ref,
                  **{k:computed[k] for k in ("input_hashes","solver_input_hashes","normalization")},
                  "truth_used":False,"test_data_used":False,"numerically_complete":True,
                  "evaluation":{"candidates":[{"candidate_id":0,"support":["u"],"support_indices":[uid]}],
                                "point_to_candidate":[{"point_index":0,"candidate_id":0}]}}
        record_ref = write("method.json",record)
        model = {"layer":"J18","method":"L1","trainfit_provenance":{
                 "method_result_file":record_ref["file"],"method_result_sha256":record_ref["sha256"],
                 "path_source_file":path_ref["file"],"path_source_sha256":path_ref["sha256"],
                 "fit_matrix_metadata_file":meta_ref["file"],"fit_matrix_metadata_sha256":meta_ref["sha256"],
                 "input_hashes":computed["input_hashes"],"normalization":computed["normalization"],
                 "coefficient_fit_split":"train_ids0_3_only","candidate_id":0,"selected_path_point_indices":[0]}}
        selection = {"dataset":identity,"protocol":protocol_ref,"fit_bundle_sha256":"synthetic_raw_only","models":[model]}
        checks["positive_synthetic_binding_only"] = validate_selection_bindings(selection,protocol)["all_bindings_verified"]
        # Each negative fixture has internally valid file hashes; the binding,
        # rather than a deliberately broken checksum, must reject it.
        for label, change in (("wrong_case",{"case":"wave"}),("wrong_sigma",{"sigma":.03}),
                              ("wrong_seed",{"noise_seed":1001}),("wrong_layer",{"layer":"J1158"}),
                              ("wrong_method",{"method":"IRL1_original"})):
            wrong_record = {**record,**change}
            ref = write(label+".json",wrong_record)
            wrong = copy.deepcopy(selection)
            wrong["models"][0]["trainfit_provenance"].update(method_result_file=ref["file"],method_result_sha256=ref["sha256"])
            try: validate_selection_bindings(wrong,protocol)
            except ValueError: rejection_cases.append(label)
        for label, change in (("wrong_matrix_raw",{"raw_bundle_sha256":"other_raw"}),
                              ("wrong_matrix_role",{"role":"test"}),
                              ("wrong_matrix_protocol",{"protocol":{"file":"other","sha256":"other"}}),
                              ("wrong_matrix_source",{"source_sha256":{"other":"other"}})):
            ref = write(label+".json",{**meta,**change})
            wrong = copy.deepcopy(selection)
            wrong["models"][0]["trainfit_provenance"].update(fit_matrix_metadata_file=ref["file"],fit_matrix_metadata_sha256=ref["sha256"])
            try: validate_selection_bindings(wrong,protocol)
            except ValueError: rejection_cases.append(label)
        other_path = write("other_path.json",{**path,"note":"different saved artifact, same support"})
        wrong = copy.deepcopy(selection)
        wrong["models"][0]["trainfit_provenance"].update(path_source_file=other_path["file"],path_source_sha256=other_path["sha256"])
        try: validate_selection_bindings(wrong,protocol)
        except ValueError: rejection_cases.append("wrong_path_with_same_support")
        wrong_job = write("wrong_job.json",{**job,"solver_input_hashes":{"A":"different","b":"different"}})
        wrong_record = write("wrong_job_record.json",{**record,"job":wrong_job})
        wrong = copy.deepcopy(selection)
        wrong["models"][0]["trainfit_provenance"].update(method_result_file=wrong_record["file"],method_result_sha256=wrong_record["sha256"])
        try: validate_selection_bindings(wrong,protocol)
        except ValueError: rejection_cases.append("wrong_job_solver_inputs")
        checks["all11_wrong_binding_fixtures_rejected"] = len(rejection_cases)==11
    checks["v1_data_gateway_unchanged"] = digest(ROOT/"stage3_data.py") == OLD._read(ROOT/"stage3_data_preflight_result.json")["source_sha256"]["stage3_data.py"]
    formal_after = sorted(str(p.relative_to(ROOT)) for p in (ROOT/"stage3_data").glob("**/*.npz"))
    checks["formal_bundle_inventory_unchanged"] = formal_after==formal_before
    sources = ["stage3_data_v2.py","stage3_data.py","stage0_evaluation.py","stage1_extended_evaluation.py",
               "stage1_grammar.py","stage2_identifiability.py"]
    result = {"stage":"3_data_binding_v2_preflight_only","all_checks_passed":all(checks.values()),"checks":checks,
              "source_sha256":{n:digest(ROOT/n) for n in sources},"rejected_binding_fixtures":rejection_cases,
              "new_L1_solver_calls":0,"formal_or_test_observations_generated":False,"test_parameters_instantiated":False,
              "formal_bundles_before":formal_before,"formal_bundles_after":formal_after,
              "scope":"Closes cross-dataset/layer/raw/path provenance binding before delegating to the unchanged v1 strict release gate.",
              "limits":["Synthetic positive fixture tests binding only; it is not a certified model or a valid six-model release.",
                  "No frozen protocol is created and no test-release API is exercised by preflight.",
                  "Numerical certification remains the frozen runner/reviewer responsibility; this layer binds its saved inputs and evidence."]}
    strict_json_write(output,result)
    print({"all_checks_passed":result["all_checks_passed"],"checks":checks,"rejections":rejection_cases})
    if not result["all_checks_passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
