"""Role-separated formal observations behind frozen-protocol/model gates.

Default execution performs only parameter/guard/synthetic serialization checks.
It never generates formal seeds or instantiates held-out trajectory parameters.
The fit API returns IDs0..5 only. IDs6..11 become available only after all six
library-method models for that dataset are selected and sealed together.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from stage0_data_bundle import load_bundle, save_bundle
from stage0_evaluation import strict_json_write
from stage2_data import CASES, field_value, trajectory_parameters

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT/"stage3_protocol_result.json"
FIT_IDS, TEST_IDS = list(range(6)), list(range(6, 12))
REQUIRED_SOURCES = {"stage3_data.py", "stage3_campaign.py", "stage2_data.py", "stage0_data_bundle.py",
                    "stage0_evaluation.py", "inverse_irl1.py", "redo_two.py"}
MODEL_HASH_KEYS = ("layer", "method", "status", "numerically_complete", "primary_selected",
                   "requested_final_points", "certified_final_points", "outer_all_certified",
                   "support", "physical_coefficients", "trainfit_provenance")


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON: "+value)))


def _project_file(value):
    path = (ROOT/Path(value)).resolve()
    if not path.is_relative_to(ROOT.resolve()): raise ValueError("provenance file must remain inside the project")
    return path


def _protocol_content_checks(protocol):
    if (protocol.get("status") != "frozen" or protocol.get("is_frozen") is not True
            or protocol.get("execution_authorized") is not True):
        raise ValueError("formal protocol is not frozen and execution-authorized")
    if (protocol["cases"] != list(CASES) or protocol["noise_seeds"] != list(range(1000, 1020))
            or protocol["libraries"] != ["J18", "J1158", "J1212"]
            or protocol["methods"] != ["L1", "IRL1_original"]
            or protocol["fit_ids"] != FIT_IDS or protocol["train_ids"] != [0, 1, 2, 3]
            or protocol["validation_ids"] != [4, 5] or protocol["test_ids"] != TEST_IDS):
        raise ValueError("formal case order/seed range/library/method/split contract changed")
    split = protocol["noise_and_data_contract"]["split_ids"]
    if (protocol["cases"] != protocol["scientific_scope"]["cases"]
            or protocol["methods"] != protocol["scientific_scope"]["primary_methods"]
            or protocol["noise_levels"] != protocol["planned_grid"]["noise_fractions"]
            or protocol["noise_seeds"] != protocol["planned_grid"]["noise_seeds"]
            or protocol["train_ids"] != split["train"] or protocol["validation_ids"] != split["validation"]
            or protocol["test_ids"] != split["heldout_test"]
            or protocol["selection_alpha"] != protocol["method_and_selection"]["alpha"]
            or protocol["support_threshold"] != protocol["method_and_selection"]["support_threshold"]):
        raise ValueError("top-level and descriptive protocol configurations disagree")
    levels = protocol["noise_levels"]
    if (not levels or len(set(levels)) != len(levels)
            or any(isinstance(s, bool) or not isinstance(s, (int, float)) or not np.isfinite(s) or s < 0 for s in levels)
            or protocol["selection_alpha"] != .005 or protocol["support_threshold"] != 1e-5
            or protocol["lambda_ratios"] != np.geomspace(1., 1e-6, 24).tolist()):
        raise ValueError("frozen noise levels/selector/lambda grid invalid")
    required = REQUIRED_SOURCES | set(protocol.get("source_sha256_required", []))
    if not required.issubset(protocol["source_sha256"]):
        raise ValueError("frozen source mapping omits a required data/campaign dependency")


def load_frozen_protocol():
    """Return the frozen dict; draft/missing/changed-source protocols fail first."""
    if not PROTOCOL_PATH.is_file(): raise RuntimeError("formal frozen protocol does not exist; no data generated")
    before = digest(PROTOCOL_PATH)
    protocol = _read(PROTOCOL_PATH)
    _protocol_content_checks(protocol)
    for name, expected in protocol["source_sha256"].items():
        if digest(_project_file(name)) != expected: raise ValueError("frozen source changed: "+name)
    if digest(PROTOCOL_PATH) != before: raise ValueError("protocol changed while being validated")
    return protocol


def _request(case, sigma, seed, protocol):
    if not isinstance(case, str) or case not in protocol["cases"]: raise ValueError("undeclared case")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) not in protocol["noise_seeds"]): raise ValueError("formal noise seed outside1000..1019")
    if (isinstance(sigma, (bool, np.bool_)) or not isinstance(sigma, (int, float, np.integer, np.floating))
            or not np.isfinite(sigma) or float(sigma) not in protocol["noise_levels"]):
        raise ValueError("undeclared or invalid noise level")
    sigma, seed = float(sigma), int(seed)
    if sigma == 0:
        sigma = 0.
        if seed != 1000: raise ValueError("zero-noise dataset is generated only once under seed1000")
    return case, sigma, seed


def _bundle_path(case, sigma, seed, role):
    if role not in ("fit", "test"): raise ValueError("unknown observation role")
    tag = repr(float(sigma)).replace(".", "p")
    return ROOT/"stage3_data"/role/f"{case}_sigma_{tag}_seed_{seed}_{role}.npz"


def selected_model_digest(model):
    """Checksum the complete declared selected-model payload, excluding itself."""
    payload = {key:model[key] for key in MODEL_HASH_KEYS}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _selection_headers(selection, case, sigma, seed, protocol, protocol_identity):
    expected_pairs = {(layer, method) for layer in protocol["libraries"] for method in protocol["methods"]}
    models = selection.get("models", [])
    if (selection.get("stage") != "stage3_sealed_dataset_selection" or selection.get("sealed") is not True
            or selection.get("dataset") != {"case":case, "sigma":sigma, "noise_seed":seed}
            or selection.get("protocol") != protocol_identity or selection.get("source_sha256") != protocol["source_sha256"]
            or selection.get("train_ids") != [0, 1, 2, 3] or selection.get("validation_ids") != [4, 5]
            or selection.get("no_test_data_used") is not True or len(models) != 6
            or {(m.get("layer"), m.get("method")) for m in models} != expected_pairs):
        raise ValueError("test access needs one sealed matching six-model dataset manifest")
    for model in models:
        if (model.get("status") != "selected" or model.get("numerically_complete") is not True
                or model.get("primary_selected") is not True or model.get("outer_all_certified") is not True
                or model.get("requested_final_points") != 24 or model.get("certified_final_points") != 24):
            raise ValueError("unfinished/unavailable model cannot unlock test observations")


def _validate_selected_model(model, protocol):
    support, coefficient = model["support"], model["physical_coefficients"]
    if (not isinstance(support, list) or any(not isinstance(n, str) or not n for n in support)
            or len(set(support)) != len(support) or not isinstance(coefficient, dict) or set(coefficient) != set(support)
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) for v in coefficient.values())
            or model["selected_model_sha256"] != selected_model_digest(model)):
        raise ValueError("selected support/physical coefficient/checksum mismatch")
    provenance = model["trainfit_provenance"]
    if not isinstance(provenance, dict) or not provenance: raise ValueError("missing training fit provenance")
    for key in ("method_result_file", "path_source_file", "fit_matrix_metadata_file"):
        if key not in provenance or key[:-5]+"_sha256" not in provenance:
            raise ValueError("missing selected-model source file/hash: "+key)
    for key, value in provenance.items():
        if key.endswith("_file"):
            if digest(_project_file(value)) != provenance[key[:-5]+"_sha256"]:
                raise ValueError("selected-model source changed: "+key)
    path = _read(_project_file(provenance["path_source_file"]))
    if (path.get("all_solutions_certified") is not True or path.get("ratios") != protocol["lambda_ratios"]
            or len(path.get("points", [])) != 24 or path.get("method") != model["method"]
            or path.get("support_tolerance") != protocol["support_threshold"]):
        raise ValueError("selected model does not reference a complete certified frozen path")
    for point in path["points"]+path.get("outer", []):
        if point.get("status") != "certified" or point.get("certificate", {}).get("certified") is not True:
            raise ValueError("uncertified saved point prevents test release")
    if model["method"] == "IRL1_original" and not 1 <= len(path.get("outer", [])) <= 8:
        raise ValueError("original IRL1 outer path missing")
    record = _read(_project_file(provenance["method_result_file"]))
    evaluation = record["evaluation"]
    selected_id = evaluation.get("primary_selected_candidate_id")
    if (evaluation.get("selection_eligible") is not True or selected_id is None
            or selected_id != provenance["candidate_id"]): raise ValueError("primary selection not fixed before test")
    selected = [c for c in evaluation["candidates"] if c["candidate_id"] == selected_id]
    if (len(selected) != 1 or selected[0]["support"] != support
            or dict(zip(support, selected[0]["coefficients"])) != coefficient
            or selected[0].get("ls_readout_changes_support") is not False):
        raise ValueError("sealed physical model differs from saved training-only readout")


def _parameter_formula(case, trajectory_id):
    """Same six-family RNG rule. Test IDs are called only after release checks.

    Preflight calls this function exclusively for old IDs0..5 to establish
    equality with the unchanged historical parameter generator.
    """
    if case not in CASES or type(trajectory_id) is not int or not 0 <= trajectory_id <= 11:
        raise ValueError("parameter recipe only covers declared IDs0..11")
    rng = np.random.default_rng(50000+100*list(CASES).index(case)+trajectory_id)
    if case in ("heat", "wave"):
        coeff = rng.uniform(.35, 1., 6)*rng.choice([-1., 1.], 6)/np.sqrt(np.arange(1, 7))
        return {"cos_coefficients":coeff.tolist(),
                "sin_coefficients":rng.uniform(-.5, .5, 6).tolist() if case == "wave" else [0.]*6}
    if case == "logheat":
        coefficients = rng.uniform(-1., 1., 6)
        coefficients *= .55/np.sum(np.abs(coefficients))
        return {"coefficients":coefficients.tolist(), "positive_heat_field_lower_bound":.45}
    if case == "advection_diffusion":
        return {"widths":rng.uniform(.35, 1.05, 3).tolist(), "centers":rng.uniform(-1., 1., 3).tolist(),
                "amplitudes":rng.uniform(.4, 1.2, 3).tolist()}
    if case == "burgers":
        return {"ks":[0., *(-np.sort(rng.uniform(3., 16., 3))).tolist()],
                "x0s":[0., *rng.uniform(.1, 1.2, 3).tolist()], "nu":.1}
    if case == "kdv":
        return {"a1":float(rng.uniform(.7, 1.25)), "a2":float(rng.uniform(1.6, 2.3)),
                "x01":float(rng.uniform(-3., -1.)), "x02":float(rng.uniform(-9., -6.))}
    raise ValueError(case)


def _noisy_observation(clean, sigma, case_index, seed, trajectory_id):
    standard = np.random.default_rng(np.random.SeedSequence([20261106, case_index, seed, trajectory_id])).standard_normal(clean.shape)
    scale = float(np.std(clean))
    return clean+sigma*scale*standard, scale


def _expected(case, sigma, seed, role, protocol, selection_identity=None):
    return {"stage":"3_formal_observations", "case":case, "sigma":sigma, "seed":seed,
            "role":role, "trajectory_ids":FIT_IDS if role == "fit" else TEST_IDS,
            "protocol":{"file":PROTOCOL_PATH.name, "sha256":digest(PROTOCOL_PATH)},
            "source_sha256":protocol["source_sha256"], "target":CASES[case]["target"],
            "selection_manifest":selection_identity, "numpy_version":np.__version__}


def _load_existing(path, expected):
    loaded = load_bundle(path)
    manifest = loaded[3]
    if any(manifest.get(k) != v for k, v in expected.items()) or manifest["trajectory_count"] != 6:
        raise ValueError("formal observation cache contract changed; preserve and version")
    ids = expected["trajectory_ids"]
    splits = ["test" if expected["role"] == "test" else ("train" if tid < 4 else "validation") for tid in ids]
    if ([r["trajectory_id"] for r in manifest["trajectories"]] != ids
            or [r["split"] for r in manifest["trajectories"]] != splits):
        raise ValueError("wrong-role trajectory IDs in cached bundle")
    return loaded


def _load_or_generate(case, sigma, seed, role, protocol, selection_identity=None):
    path = _bundle_path(case, sigma, seed, role)
    expected = _expected(case, sigma, seed, role, protocol, selection_identity)
    if path.exists() or path.with_suffix(".json").exists():
        return _load_existing(path, expected)
    spec = CASES[case]
    x, t = np.linspace(*spec["x"]), np.linspace(*spec["t"])
    xx, tt = np.meshgrid(x, t)
    observations, trajectories = [], []
    for tid in expected["trajectory_ids"]:
        # This is the only held-out parameter-instantiation call site, reached
        # only from get_test_bundle after the complete six-model release gate.
        parameters = trajectory_parameters(case, tid) if role == "fit" else _parameter_formula(case, tid)
        clean = field_value(case, xx, tt, parameters)
        observed, scale = _noisy_observation(clean, sigma, list(CASES).index(case), seed, tid)
        observations.append(observed)
        trajectories.append({"trajectory_id":tid, "split":"test" if role == "test" else ("train" if tid < 4 else "validation"),
                             "parameters":parameters, "clean_std_for_synthetic_noise":scale})
    save_bundle(path, x, t, observations, {**expected, "trajectories":trajectories,
        "noise_model":"additive Gaussian sigma*std(clean trajectory), paired standard noise across sigma",
        "noise_seed_sequence":"[20261106, frozen case index, noise seed, trajectory_id]",
        "normalization":"none; fit normalization from training matrices only",
        "clean_values_saved":False, "analytical_derivatives_returned":False})
    return _load_existing(path, expected)


def get_fit_bundle(case, sigma, seed):
    protocol = load_frozen_protocol()
    case, sigma, seed = _request(case, sigma, seed, protocol)
    return _load_or_generate(case, sigma, seed, "fit", protocol)


def get_test_bundle(case, sigma, seed, selection_manifest_path):
    protocol = load_frozen_protocol()
    case, sigma, seed = _request(case, sigma, seed, protocol)
    selection_path = _project_file(selection_manifest_path)
    selection_hash = digest(selection_path)
    selection = _read(selection_path)
    identity = {"file":PROTOCOL_PATH.name, "sha256":digest(PROTOCOL_PATH)}
    _selection_headers(selection, case, sigma, seed, protocol, identity)
    fit = _load_existing(_bundle_path(case, sigma, seed, "fit"), _expected(case, sigma, seed, "fit", protocol))
    if fit[3]["npz_sha256"] != selection["fit_bundle_sha256"]: raise ValueError("selected models used a different fit bundle")
    for model in selection["models"]: _validate_selected_model(model, protocol)
    if digest(selection_path) != selection_hash: raise ValueError("selection changed during release validation")
    release = {"file":str(selection_path.relative_to(ROOT)), "sha256":selection_hash}
    return _load_or_generate(case, sigma, seed, "test", protocol, release)


def main():
    output = ROOT/"stage3_data_preflight_result.json"
    if output.exists(): raise FileExistsError("preserve prior preflight; create a versioned script/result for changes")
    formal_before = sorted(str(p.relative_to(ROOT)) for p in (ROOT/"stage3_data").glob("**/*.npz"))
    checks = {}
    equal = [{"case":case, "trajectory_id":tid,
              "identical":_parameter_formula(case, tid) == trajectory_parameters(case, tid)}
             for case in CASES for tid in FIT_IDS]
    checks["all36_old_parameter_recipes_exactly_equal"] = all(r["identical"] for r in equal)
    rejected = []
    draft = _read(ROOT/"stage3_protocol_draft.json")
    try: _protocol_content_checks(draft)
    except ValueError: rejected.append("draft_not_frozen")
    request_examples = [("unknown", .01, 1000), ("heat", .01, 0), ("heat", .01, 999),
                        ("heat", .01, 1020), ("heat", .01, True), ("heat", -.01, 1000),
                        ("heat", np.nan, 1000), ("heat", np.inf, 1000), ("heat", 0., 1001)]
    for i, args in enumerate(request_examples):
        try: _request(*args, draft)
        except ValueError: rejected.append("invalid_request_"+str(i))
    checks["draft_and_invalid_requests_rejected_without_generation"] = len(rejected) == len(request_examples)+1
    try: _selection_headers({}, "heat", .01, 1000, draft, {"file":"unused", "sha256":"unused"})
    except ValueError: checks["missing_selection_manifest_rejected"] = True
    else: checks["missing_selection_manifest_rejected"] = False
    identity = {"file":"synthetic_header_only", "sha256":"not_a_protocol_file"}
    models = [{"layer":layer, "method":method, "status":"selected", "numerically_complete":True,
               "primary_selected":True, "requested_final_points":24, "certified_final_points":24,
               "outer_all_certified":True} for layer in draft["libraries"] for method in draft["methods"]]
    header = {"stage":"stage3_sealed_dataset_selection", "sealed":True,
              "dataset":{"case":"heat", "sigma":.01, "noise_seed":1000}, "protocol":identity,
              "source_sha256":draft["source_sha256"], "train_ids":[0,1,2,3], "validation_ids":[4,5],
              "models":models, "no_test_data_used":True}
    _selection_headers(header, "heat", .01, 1000, draft, identity)  # Pure header validation; no release API called.
    selection_rejections = []
    for key, value in [("status", "numerically_unavailable"), ("numerically_complete", False),
                       ("primary_selected", False), ("certified_final_points", 23), ("outer_all_certified", False)]:
        wrong = {**header, "models":[{**models[0], key:value}, *models[1:]]}
        try: _selection_headers(wrong, "heat", .01, 1000, draft, identity)
        except ValueError: selection_rejections.append(key)
    for label, wrong_models in [("missing_identity", models[:-1]), ("duplicate_identity", [models[1], *models[1:]])]:
        try: _selection_headers({**header, "models":wrong_models}, "heat", .01, 1000, draft, identity)
        except ValueError: selection_rejections.append(label)
    checks["six_complete_selected_models_required"] = len(selection_rejections) == 7
    sample_model = {**models[0], "support":["u"], "physical_coefficients":{"u":0.},
                    "trainfit_provenance":{"synthetic_only":True}}
    checks["selected_model_digest_binds_coefficients"] = selected_model_digest(sample_model) != selected_model_digest(
        {**sample_model, "physical_coefficients":{"u":1.}})
    clean = np.arange(12., dtype=float).reshape(3, 4)
    n1, scale1 = _noisy_observation(clean, .01, 0, 17, 0)
    n2, scale2 = _noisy_observation(clean, .05, 0, 17, 0)
    checks["synthetic_only_standard_noise_paired_across_sigma"] = bool(scale1 == scale2 and np.allclose(
        (n1-clean)/.01, (n2-clean)/.05, rtol=1e-12, atol=1e-12))
    with tempfile.TemporaryDirectory(prefix="stage3_synthetic_preflight_", dir=ROOT) as directory:
        path = Path(directory)/"synthetic_only.npz"
        x, t = np.arange(4.), np.arange(3.)
        saved = save_bundle(path, x, t, [clean, n1], {"role":"synthetic_preflight_only", "formal_seed_used":False})
        restored = load_bundle(path)
        checks["small_synthetic_bundle_roundtrip"] = bool(np.array_equal(restored[0], x)
            and np.array_equal(restored[1], t) and np.array_equal(restored[2][0], clean)
            and np.array_equal(restored[2][1], n1) and saved == restored[3])
    formal_after = sorted(str(p.relative_to(ROOT)) for p in (ROOT/"stage3_data").glob("**/*.npz"))
    checks["formal_bundle_inventory_unchanged"] = formal_after == formal_before
    report = {"stage":"3_data_preflight_only", "all_checks_passed":all(checks.values()), "checks":checks,
              "source_sha256":{name:digest(ROOT/name) for name in sorted(REQUIRED_SOURCES-{"stage3_campaign.py"})},
              "draft_protocol_sha256":digest(ROOT/"stage3_protocol_draft.json"), "parameter_equivalence":equal,
              "guard_rejections":rejected, "selection_header_rejections":selection_rejections,
              "formal_or_test_observations_generated":False,
              "test_parameters_instantiated":False, "formal_noise_seeds_sampled":False,
              "synthetic_noise_seed":17, "formal_bundles_before":formal_before, "formal_bundles_after":formal_after,
              "interfaces":{"fit":"get_fit_bundle(case,sigma,seed) -> x,t,list6observations,manifest (IDs0..5)",
                  "test":"get_test_bundle(case,sigma,seed,selection_manifest_path) -> x,t,list6observations,manifest (IDs6..11)",
                  "protocol":"load_frozen_protocol() -> validated protocol dict",
                  "model_digest_keys":list(MODEL_HASH_KEYS)},
              "limits":["Preflight does not freeze the protocol or authorize any formal execution.",
                  "Holdout creation requires all six selected models and their saved complete24-point/outer certification records.",
                  "The gateway verifies saved certification/selection provenance; numerical certification itself belongs to the frozen solver/reviewer.",
                  "Test-family parameter formulas are only checked for equality on old IDs0..5; no IDs6..11 parameters are inspected here.",
                  "No formal field-generation or test-release positive integration test is performed before root freezes and starts the campaign."]}
    strict_json_write(output, report)
    print({"all_checks_passed":report["all_checks_passed"], "checks":checks,
           "formal_or_test_observations_generated":False})
    if not report["all_checks_passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
