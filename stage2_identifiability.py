"""Development multi-trajectory geometry, competing supports and preprocessing.

This is a bounded diagnostic, never a global sparse oracle. Validation errors
are weak equation residuals, NOT learned-PDE time-integration predictions.
"""

import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from stage0_evaluation import array_hash, canonical_term, strict_json_write
from stage1_grammar import LAYERS, enumerate_jet, key
from stage1_jet_preprocess import weak_jet_matrix
from stage2_data import CASES, get_bundle

ROOT = Path(__file__).resolve().parent


def source_hashes():
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ("stage2_data.py", "stage1_grammar.py", "stage1_jet_preprocess.py",
             "stage1_weak_reduction.py", "stage1_weak_reduction_v2.py")}


def build_matrices(case, sigma, seed=0, compare_direct=False):
    from stage1_weak_reduction_v2 import weak_reduced_matrix
    terms = enumerate_jet()
    names = [term["name"] for term in terms]
    costs = [term["complexity"] for term in terms]
    x, t, observations, manifest = get_bundle(case, sigma, seed)
    spec = CASES[case]
    target_order = 2 if spec["target"] == "u_tt" else 1
    small_keys = {key(term) for term in enumerate_jet(*LAYERS["J18"])}
    small_indices = [i for i, term in enumerate(terms) if key(term) in small_keys]
    matrices, targets, metadata = [], [], []
    direct_difference = None
    for i, observed in enumerate(observations):
        start = time.perf_counter()
        a, b, meta = weak_reduced_matrix(x, t, observed, terms, target_order=target_order, half=spec["half"])
        if compare_direct and i == 0:
            direct, direct_b, _ = weak_jet_matrix(x, t, observed, terms, target_order=target_order, half=spec["half"])
            target_difference = float(np.linalg.norm(b-direct_b)/max(np.linalg.norm(b), 1e-300))
            denominator = np.linalg.norm(a, axis=0)
            errors = np.linalg.norm(direct-a, axis=0)/np.maximum(denominator, 1e-300)
            direct_difference = {"trajectory": i, "comparison": "direct_jet_vs_exact_weak_reduction",
                                 "median_relative_column_difference": float(np.median(errors)),
                                 "max_relative_column_difference": float(errors.max()),
                                 "relative_difference_by_column": errors.tolist()}
            direct_difference["target_relative_difference_after_kernel_correction"] = target_difference
        if target_order == 2:
            # Predeclared wave nuisance u_t, outside the spatial jet grammar.
            _, ut, _ = weak_reduced_matrix(x, t, observed, [terms[0]], target_order=1, half=spec["half"])
            a = np.column_stack([a, ut])
        matrices.append(a)
        targets.append(b)
        metadata.append({"trajectory": i, "split": "train" if i < 4 else "validation",
                         "rows": len(b), "build_seconds": time.perf_counter()-start,
                         "raw_hash": meta["input_observation_hash"],
                         "matrix_hash": array_hash(a), "target_hash": array_hash(b)})
    if target_order == 2:
        names.append("u_t")
        costs.append(2)
        small_indices.append(len(names)-1)
    return matrices, targets, {"case": case, "sigma": sigma, "seed": seed,
        "raw_bundle_sha256": manifest["npz_sha256"], "names": names, "complexities": costs,
        "small_indices": small_indices, "trajectories": metadata,
        "direct_jet_comparison": direct_difference, "test_data_used": False,
        "source_hashes": source_hashes(),
        "data_chain": "exact weak integration-by-parts reduction plus observed jets of order <=3",
        "target_order": target_order, "half": list(spec["half"])}


def save_matrices(case, sigma, seed=0, compare_direct=False):
    path = ROOT / "stage2_matrices" / f"{case}_sigma_{repr(float(sigma)).replace('.', 'p')}_seed_{seed}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            matrices = [data[f"a_{i}"].copy() for i in range(6)]
            targets = [data[f"b_{i}"].copy() for i in range(6)]
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if (meta.get("source_hashes") != source_hashes() or meta["case"] != case
                or meta["sigma"] != sigma or meta["seed"] != seed):
            raise ValueError("cached matrix configuration/source mismatch; preserve and version it")
        if meta["raw_bundle_sha256"] != get_bundle(case, sigma, seed)[3]["npz_sha256"]:
            raise ValueError("cached matrix raw-input mismatch")
        for i, (a, b) in enumerate(zip(matrices, targets)):
            assert array_hash(a) == meta["trajectories"][i]["matrix_hash"]
            assert array_hash(b) == meta["trajectories"][i]["target_hash"]
        return matrices, targets, meta
    matrices, targets, meta = build_matrices(case, sigma, seed, compare_direct)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{f"a_{i}": a for i, a in enumerate(matrices)},
                        **{f"b_{i}": b for i, b in enumerate(targets)})
    strict_json_write(path.with_suffix(".json"), meta)
    return matrices, targets, meta


def stack_split(matrices, targets, indices):
    # Each trajectory has equal response energy; each row weight acts on A,b.
    scales = [max(float(np.linalg.norm(targets[i])), 1e-300) for i in indices]
    return (np.vstack([matrices[i]/s for i, s in zip(indices, scales)]),
            np.concatenate([targets[i]/s for i, s in zip(indices, scales)]))


def fit_fixed_support(a, b, av, bv, support):
    support = list(support)
    norms = np.linalg.norm(a[:, support], axis=0)
    norms = np.where(norms == 0, 1., norms)
    beta, _, rank, _ = np.linalg.lstsq(a[:, support]/norms, b, rcond=None)
    coefficient = beta/norms
    return {"support_indices": support, "coefficients": coefficient.tolist(), "rank": int(rank),
            "train_equation_relative_error": float(np.linalg.norm(a[:, support]@coefficient-b)/np.linalg.norm(b)),
            "validation_equation_relative_error": float(np.linalg.norm(av[:, support]@coefficient-bv)/np.linalg.norm(bv))}


def geometry(a):
    norms = np.linalg.norm(a, axis=0)
    normed = a/np.where(norms == 0, 1., norms)
    singular = np.linalg.svd(normed, compute_uv=False)
    machine_tol = np.finfo(float).eps*max(a.shape)*singular[0]
    machine_rank = int(np.sum(singular > machine_tol))
    gram = normed.T@normed
    np.fill_diagonal(gram, 0.)
    return {"shape": list(a.shape), "singular_values": singular.tolist(),
            "rank_machine": machine_rank, "machine_tolerance": float(machine_tol),
            "column_nullity_at_machine_tolerance": a.shape[1]-machine_rank,
            "effective_ranks": {str(tol): int(np.sum(singular > tol*singular[0])) for tol in (1e-3, 1e-6, 1e-9)},
            "full_column_condition": float(singular[0]/singular[-1]) if machine_rank == a.shape[1] else None,
            "condition_status": "full_column_rank" if machine_rank == a.shape[1] else "not_full_column_rank_at_machine_tolerance",
            "max_absolute_column_correlation": float(np.max(np.abs(gram))),
            "interpretation": "matrix rank does not by itself decide sparse identifiability"}


def diagnose(matrices, targets, meta):
    a, b = stack_split(matrices, targets, range(4))
    av, bv = stack_split(matrices, targets, range(4, 6))
    names = meta["names"]
    canonical = [canonical_term(name) for name in names]
    truth = CASES[meta["case"]]["true"]
    true_indices = [canonical.index(canonical_term(name)) for name in truth]
    truth_values = np.array(list(truth.values()))
    known_residual = float(np.linalg.norm(av[:, true_indices]@truth_values-bv)/np.linalg.norm(bv))
    true_fit = fit_fixed_support(a, b, av, bv, true_indices)
    small = meta["small_indices"]
    # Bounded reference: all supports of size <=2 in J18 (+u_t for wave).
    candidates = [()] + [(i,) for i in range(a.shape[1])]
    candidates += list(itertools.combinations(small, 2))
    candidates.append(tuple(true_indices))
    fits = [fit_fixed_support(a, b, av, bv, support) for support in sorted(set(candidates))]
    for fit in fits:
        fit["support"] = [names[i] for i in fit["support_indices"]]
        fit["complexity"] = sum(meta["complexities"][i] for i in fit["support_indices"])
    ranked = sorted(fits, key=lambda f: (f["validation_equation_relative_error"], f["complexity"]))
    return {"true_support": list(truth), "known_coefficients_validation_residual": known_residual,
            "true_support_fit_diagnostic_only": true_fit,
            "geometry_full_train": geometry(a), "geometry_small_train": geometry(a[:, small]),
            "geometry_single_train_small": geometry(matrices[0][:, small]),
            "bounded_competitors": {"scope": "all 1-term supports in full library; all <=2-term supports in J18; known support diagnostic",
                                     "not_global_oracle": True, "candidates": len(fits), "top_validation": ranked[:10]},
            "validation_metric": "weak equation residual; no PDE rollout performed"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    args = parser.parse_args()
    for component in ("stage1_gate", "stage1_weak_reduction_v2", "stage2_field_check_refined"):
        gate = json.loads((ROOT/(component+"_result.json")).read_text(encoding="utf-8"))
        if not gate["all_checks_passed"]:
            raise RuntimeError(component+" has not passed")
    report = {"stage": 2, "cases": {}, "test_data_used": False, "all_checks_passed": False,
              "limits": ["No universal identifiability claim; competitor search is explicitly bounded.",
                         "Validation equation residuals do not replace held-out PDE rollout errors.",
                         "Noise seed 0 is a development diagnostic, not a robustness curve."]}
    out = ROOT / "stage2_identifiability_result.json"
    with threadpool_limits(limits=1):
        for case in args.cases:
            records, clean_matrices = [], None
            for sigma in (0., .01):
                start = time.perf_counter()
                matrices, targets, meta = save_matrices(case, sigma, compare_direct=True)
                row = {"sigma": sigma, "seed": 0, "metadata": meta, **diagnose(matrices, targets, meta)}
                if clean_matrices is None:
                    clean_matrices = matrices
                else:
                    noise = np.linalg.norm(np.vstack(matrices)-np.vstack(clean_matrices), axis=0)
                    clean = np.linalg.norm(np.vstack(clean_matrices), axis=0)
                    ratios = noise/np.maximum(clean, 1e-300)
                    row["matrix_noise_perturbation"] = {"relative_by_column": ratios.tolist(),
                        "median": float(np.median(ratios)), "max": float(ratios.max()),
                        "interpretation": "relative column errors can be large when the clean column is small"}
                row["elapsed_s"] = time.perf_counter()-start
                records.append(row)
                report["cases"][case] = records
                strict_json_write(out, report)
                print(case, sigma, "true-validation", row["known_coefficients_validation_residual"],
                      "seconds", row["elapsed_s"], flush=True)
    report["all_checks_passed"] = True
    report["completion_meaning"] = "data/geometry/competition diagnostics completed; recovery is not asserted"
    strict_json_write(out, report)


if __name__ == "__main__":
    main()
