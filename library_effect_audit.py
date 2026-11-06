"""Audit library size, identifiability, and support selection on one weak data chain.

This is a new audit script. Existing experiment scripts and result JSON files are
left untouched.  The historical 43-term library contains sin/cos columns that
are not a subset of the current 113-term polynomial grammar, so the report keeps
the historical L43 label and also records their union explicitly.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

import gml1_maxlib_compare as G
import maxlib_irl1 as M
import noise_inverse as N
import noise_inverse_biglib as B


SUPPORT_TOL = 1e-5
DEFAULT_NOISE = [0.0, 0.01, 0.03, 0.05]
DEFAULT_SEEDS = 3


def _historical_l43_names() -> list[str]:
    # Translate the historical labels to the current grammar spelling so the
    # same weak-form columns can be selected by name.
    names = []
    for k, f in M.inverse_columns():
        if f == "1":
            term = {"family": "const", "m": 0, "k": 0, "a": 0, "sign": 1}
        elif f.startswith("u^"):
            term = {"family": "pow", "m": int(f[2:]), "k": k, "a": 0, "sign": 1}
        elif f == "u":
            term = {"family": "pow", "m": 1, "k": k, "a": 0, "sign": 1}
        elif f in ("sin u", "cos u"):
            term = {"family": f.split()[0], "m": 1, "k": k, "a": 0, "sign": 1}
        elif f in ("x u", "x u^2"):
            term = {"family": "coord_pow", "m": 1 if f == "x u" else 2, "k": k, "a": 1, "sign": 1}
        else:
            raise ValueError(f)
        names.append(G.term_name(term))
    return names


def library_names(target: str) -> tuple[list[str], dict]:
    """Return names for one frozen library layer."""
    l13 = [B.name(k, f) for k, f in B.library("L13")]
    l43 = _historical_l43_names()
    l113 = [G.term_name(term) for term in G.build_terms(False)]
    if target == "L13":
        names = l13
    elif target == "L43":
        names = l43
    elif target == "L113":
        names = l113
    elif target == "L121_union":
        names = list(dict.fromkeys(l113 + l43))
    else:
        raise ValueError(target)
    return names, {
        "requested": target,
        "size": len(names),
        "l13_size": len(l13),
        "historical_l43_size": len(l43),
        "current_l113_size": len(l113),
        "l43_missing_from_l113": sorted(set(l43) - set(l113)),
        "l113_missing_from_l43": sorted(set(l113) - set(l43)),
        "is_l13_subset_of_l43": set(l13) <= set(l43),
        "is_l43_subset_of_l113": set(l43) <= set(l113),
    }


def _column_subset(a: np.ndarray, all_names: list[str], wanted: list[str]) -> tuple[np.ndarray, list[str], list[str]]:
    pos = {name: i for i, name in enumerate(all_names)}
    missing = [name for name in wanted if name not in pos]
    keep = [name for name in wanted if name in pos]
    return a[:, [pos[name] for name in keep]], keep, missing


def _matrix_metrics(a: np.ndarray, b: np.ndarray, names: list[str], true: dict[str, float]) -> dict:
    norms = np.linalg.norm(a, axis=0)
    valid = norms > 1e-13
    an = a[:, valid] / np.maximum(norms[valid], 1e-300)
    valid_names = [name for name, ok in zip(names, valid) if ok]
    if an.size:
        sv = np.linalg.svd(an, compute_uv=False)
        s0 = float(sv[0])
        rank_machine = int(np.sum(sv > max(an.shape) * np.finfo(float).eps * s0))
        rank_1e6 = int(np.sum(sv / max(s0, 1e-300) > 1e-6))
        rank_1e8 = int(np.sum(sv / max(s0, 1e-300) > 1e-8))
        cond_eff = float(s0 / sv[rank_1e6 - 1]) if rank_1e6 else None
        gram = an.T @ an
        np.fill_diagonal(gram, 0.0)
        pairs = []
        for i, j in zip(*np.triu_indices(len(valid_names), 1)):
            pairs.append((abs(float(gram[i, j])), valid_names[i], valid_names[j]))
        pairs.sort(reverse=True)
    else:
        sv, rank_machine, rank_1e6, rank_1e8, cond_eff, pairs = [], 0, 0, 0, None, []

    true_names = list(true)
    true_present = [name for name in true_names if name in valid_names]
    true_missing = [name for name in true_names if name not in valid_names]
    true_idx = [valid_names.index(name) for name in true_present]
    if true_idx:
        coef = np.linalg.lstsq(an[:, true_idx], b, rcond=None)[0]
        fit = an[:, true_idx] @ coef
        true_resid = float(np.linalg.norm(b - fit) / max(np.linalg.norm(b), 1e-300))
        true_svals = np.linalg.svd(an[:, true_idx], compute_uv=False)
        true_condition = float(true_svals[0] / true_svals[-1]) if true_svals[-1] > 0 else float("inf")
    else:
        true_resid, true_condition = None, None

    true_corr = []
    if an.size and true_present:
        for name in true_present:
            i = valid_names.index(name)
            for j, other in enumerate(valid_names):
                if other in true_present or j == i:
                    continue
                true_corr.append((abs(float((an[:, i] @ an[:, j]))), name, other))
        true_corr.sort(reverse=True)
    return {
        "n_rows": int(a.shape[0]),
        "n_columns": int(a.shape[1]),
        "valid_columns": int(an.shape[1]),
        "rank_machine": rank_machine,
        "rank_rel_1e-6": rank_1e6,
        "rank_rel_1e-8": rank_1e8,
        "condition_rel_1e-6": cond_eff,
        "max_abs_cosine": pairs[0][0] if pairs else None,
        "top_column_correlations": [
            {"abs_cosine": c, "left": left, "right": right} for c, left, right in pairs[:10]
        ],
        "true_present": true_present,
        "true_missing": true_missing,
        "true_support_relative_residual": true_resid,
        "true_support_condition": true_condition,
        "top_true_to_wrong_correlations": [
            {"abs_cosine": c, "true": left, "wrong": right} for c, left, right in true_corr[:10]
        ],
    }


def _support_eval(p: G.LazyL1, support: tuple[int, ...], true: dict[str, float]) -> dict:
    return G.support_eval(p.a, p.b, support, p.names, true)


def _select(a: np.ndarray, b: np.ndarray, names: list[str], true: dict[str, float]) -> dict:
    p = G.LazyL1(a, b, names)
    t0 = time.perf_counter()
    path, _ = p.path_supports()
    l1, criterion_l1 = G.choose_support(p, path)
    l1_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    path_i, _ = p.irl1_supports()
    irl1, criterion_i = G.choose_support(p, path_i)
    irl1_time = time.perf_counter() - t0
    criterion_s = N.Criterion(p.a, p.b)
    stridge = N.m_stridge(criterion_s)
    return {
        "GML1": {**_support_eval(p, l1, true), "time_s": l1_time, "criterion_calls": criterion_l1.calls},
        "GML1_IRL1": {**_support_eval(p, irl1, true), "time_s": irl1_time, "criterion_calls": criterion_i.calls},
        "PDE-FIND_STRidge": {**_support_eval(p, stridge, true), "criterion_calls": criterion_s.calls},
    }


def run_case(case_name: str, case: dict, layers: list[str], noise: list[float], seeds: int, do_selection: bool) -> dict:
    all_terms = G.build_terms(True)
    all_names = [G.term_name(term) for term in all_terms]
    if case["target"] == "u_tt":
        all_names.append("u_t")
    out = {"pde": case["pde"], "true": case["true"], "layers": {}}
    for layer in layers:
        wanted, spec = library_names(layer)
        wanted = list(wanted)
        if case["target"] == "u_tt" and "u_t" not in wanted:
            wanted.append("u_t")
        layer_out = {"spec": spec, "rows": []}
        for sigma in noise:
            seed_values = [0] if sigma == 0 else list(range(seeds))
            rows = []
            for seed in seed_values:
                a_full, b, names_full, meta = G.make_data(case, sigma, seed, all_terms)
                # make_data returns u_t in names for wave; keep that exact order.
                a, names, missing = _column_subset(a_full, names_full, wanted)
                metrics = _matrix_metrics(a, b, names, case["true"])
                row = {"sigma": sigma, "seed": seed, "missing_from_source": missing, "geometry": metrics}
                if do_selection:
                    true_available = {k: v for k, v in case["true"].items() if k in names}
                    row["selection"] = _select(a, b, names, true_available)
                rows.append(row)
            layer_out["rows"].append({"sigma": sigma, "n": len(rows), "raw": rows})
        out["layers"][layer] = layer_out
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--layers", nargs="+", default=["L13", "L43", "L113", "L121_union"],
                    choices=("L13", "L43", "L113", "L121_union"))
    ap.add_argument("--noise", nargs="+", type=float, default=DEFAULT_NOISE)
    ap.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--select", action="store_true", help="also run L1/IRL1/STRidge selection")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    cases = dict(N.CASES)
    if args.only:
        cases = {name: cases[name] for name in args.only}
    if args.smoke:
        cases = {next(iter(cases)): next(iter(cases.values()))}
        args.noise = [0.0]
        args.seeds = 1
    t0 = time.perf_counter()
    report = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "method": "same weak-form data chain; library geometry plus optional GML1/IRL1/STRidge selection",
        "support_tol": SUPPORT_TOL,
        "layers": args.layers,
        "noise": args.noise,
        "seeds": args.seeds,
        "cases": {},
    }
    for name, case in cases.items():
        report["cases"][name] = run_case(name, case, args.layers, args.noise, args.seeds, args.select)
        print(name, "done", flush=True)
    report["wall_time_s"] = time.perf_counter() - t0
    out = Path(args.output) if args.output else Path(__file__).with_name("library_effect_audit_result.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
