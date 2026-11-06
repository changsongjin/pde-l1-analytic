"""Sparse outlier recovery with the same weak-form L1 data chain.

The first outlier experiment contaminates weak-equation rows directly.  This
matches the proposed model

    min 1/2 ||A beta + e - b||_2^2 + lam ||beta||_1 + gamma ||e||_1.

The PDE support and the error support are both read from the joint L1 solution;
least-squares refitting is used only for residual reporting.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

import gml1_maxlib_compare as G
import noise_inverse as N
from lasso_exact import lars_lasso_path, lasso_at


SUPPORT_TOL = 1e-5
DEFAULT_FRACTIONS = [0.0, 0.01, 0.05, 0.10]
DEFAULT_AMPLITUDES = [5.0, 10.0]
DEFAULT_SEEDS = 3
IRL1_RATIO = 1e-3
IRL1_EPS = 1e-8
IRL1_MAX_ITER = 8
DEFAULT_LAYERS = ["L13", "L113"]
DEFAULT_GAMMA = 2.0


def _historical_names() -> list[str]:
    # Keep the audit layers consistent with library_effect_audit.py without
    # importing that experiment's result file.
    names = [G.term_name(term) for term in G.build_terms(False)]
    return names


def layer_names(layer: str) -> list[str]:
    if layer == "L13":
        terms = [(0, 0), (0, 1), (0, 2), (0, 3)] + [(k, m) for k in (1, 2, 3) for m in (1, 2, 3)]
        out = []
        for k, m in terms:
            if m == 0:
                out.append("1")
            elif k == 0:
                out.append("u" if m == 1 else f"u^{m}")
            elif m == 1:
                out.append("u_" + "x" * k)
            else:
                out.append(f"(u^{m})_" + "x" * k)
        return out
    if layer == "L113":
        return _historical_names()
    raise ValueError(layer)


def _subset(a: np.ndarray, all_names: list[str], wanted: list[str]) -> tuple[np.ndarray, list[str]]:
    pos = {name: i for i, name in enumerate(all_names)}
    missing = [name for name in wanted if name not in pos]
    if missing:
        raise RuntimeError(f"library terms missing from generated source: {missing}")
    return a[:, [pos[name] for name in wanted]], list(wanted)


def inject_outliers(b: np.ndarray, fraction: float, amplitude: float, seed: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    rng = np.random.default_rng(seed)
    n = len(b)
    count = int(round(fraction * n))
    if count == 0:
        return b.copy(), np.zeros(n), []
    idx = np.sort(rng.choice(n, count, replace=False))
    scale = max(float(np.linalg.norm(b) / np.sqrt(n)), 1e-12)
    delta = np.zeros(n)
    delta[idx] = amplitude * scale * rng.choice([-1.0, 1.0], size=count)
    return b + delta, delta, [int(i) for i in idx]


def _support_metrics(got: list[int], true: set[int], n: int) -> dict:
    got_set = set(got)
    tp = len(got_set & true)
    fp = len(got_set - true)
    fn = len(true - got_set)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "support": sorted(got_set),
        "true_count": len(true),
        "extra": fp,
        "missed": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-300),
        "n": n,
    }


def _refit_residual(a: np.ndarray, b: np.ndarray, beta_idx: list[int], error_idx: list[int]) -> float:
    blocks = []
    if beta_idx:
        blocks.append(a[:, beta_idx])
    if error_idx:
        blocks.append(np.eye(len(b))[:, error_idx])
    if not blocks:
        fit = np.zeros_like(b)
    else:
        x = np.column_stack(blocks)
        fit = x @ np.linalg.lstsq(x, b, rcond=None)[0]
    return float(np.linalg.norm(b - fit) / max(np.linalg.norm(b), 1e-300))


def _select_joint_path(x: np.ndarray, b: np.ndarray, weights: np.ndarray, p: int, gamma_weight: float) -> dict:
    """Select one weighted-L1 path by residual plus support-size score."""
    n = len(b)
    knots = lars_lasso_path(x, b, weights, lam_min_ratio=0.0, max_steps=max(500, p + n + 20))
    candidates = []
    bnorm = max(float(np.linalg.norm(b)), 1e-300)
    for lam, coef in knots:
        beta = coef[:p]
        err = coef[p:]
        beta_idx = [int(i) for i in np.where(np.abs(beta) > SUPPORT_TOL)[0]]
        err_idx = [int(i) for i in np.where(np.abs(err) > SUPPORT_TOL)[0]]
        residual = float(np.linalg.norm(b - x @ coef) / bnorm)
        score = residual + len(beta_idx) / max(p, 1) + len(err_idx) / max(n, 1)
        candidates.append((score, float(lam), beta_idx, err_idx, residual))
    chosen = min(candidates, key=lambda row: row[0])
    return {
        "lambda": chosen[1],
        "score": chosen[0],
        "beta_support": chosen[2],
        "error_support": chosen[3],
        "l1_residual": chosen[4],
        "n_knots": len(knots),
        "gamma_weight": gamma_weight,
    }


def joint_l1(a: np.ndarray, b: np.ndarray, gamma_weight: float = 1.0) -> dict:
    """Return a joint ordinary weighted-L1 path."""
    col_norm = np.maximum(np.linalg.norm(a, axis=0), 1e-300)
    an = a / col_norm
    n, p = an.shape
    x = np.column_stack([an, np.eye(n)])
    weights = np.concatenate([np.ones(p), np.full(n, float(gamma_weight))])
    out = _select_joint_path(x, b, weights, p, gamma_weight)
    out["method"] = "joint_L1"
    return out


def joint_irl1(a: np.ndarray, b: np.ndarray, gamma_weight: float = 1.0) -> dict:
    """Iteratively reweighted joint L1 for both PDE and error supports."""
    col_norm = np.maximum(np.linalg.norm(a, axis=0), 1e-300)
    an = a / col_norm
    n, p = an.shape
    x = np.column_stack([an, np.eye(n)])
    weights = np.concatenate([np.ones(p), np.full(n, float(gamma_weight))])
    outer = []
    previous = None
    for iteration in range(IRL1_MAX_ITER):
        lam_max = float(np.max(np.abs(x.T @ b) / np.maximum(weights, 1e-300)))
        knots = lars_lasso_path(x, b, weights, lam_min_ratio=0.0, max_steps=max(500, p + n + 20))
        coef = lasso_at(knots, lam_max * IRL1_RATIO)
        beta_idx = tuple(int(i) for i in np.where(np.abs(coef[:p]) > SUPPORT_TOL)[0])
        error_idx = tuple(int(i) for i in np.where(np.abs(coef[p:]) > SUPPORT_TOL)[0])
        outer.append({"iter": iteration, "lambda": lam_max * IRL1_RATIO,
                      "beta_support": list(beta_idx), "error_support": list(error_idx)})
        current = (beta_idx, error_idx)
        weights = 1.0 / (np.abs(coef) + IRL1_EPS)
        weights[p:] *= float(gamma_weight)
        if current == previous:
            break
        previous = current
    out = _select_joint_path(x, b, weights, p, gamma_weight)
    out["method"] = "joint_IRL1"
    out["outer"] = outer
    return out


def baseline_l1(a: np.ndarray, b: np.ndarray, names: list[str], true: dict[str, float]) -> dict:
    p = G.LazyL1(a, b, names)
    supports, _ = p.path_supports()
    chosen, criterion = G.choose_support(p, supports)
    ev = G.support_eval(p.a, p.b, chosen, p.names, true)
    return {**ev, "criterion_calls": criterion.calls}


def run_one(case: dict, layer: str, fraction: float, amplitude: float, seed: int, gamma_weight: float) -> dict:
    full_terms = G.build_terms(False)
    a_full, b, full_names, meta = G.make_data(case, 0.0, seed, full_terms)
    wanted = layer_names(layer)
    if case["target"] == "u_tt":
        wanted.append("u_t")
    a, names = _subset(a_full, full_names, wanted)
    b_obs, delta, true_error = inject_outliers(b, fraction, amplitude, seed + 10000)
    true_names = {key: value for key, value in case["true"].items() if key in names}
    base = baseline_l1(a, b_obs, names, true_names)
    t0 = time.perf_counter()
    joint = joint_l1(a, b_obs, gamma_weight)
    joint["time_s"] = time.perf_counter() - t0
    true_beta = {names.index(key) for key in true_names}

    t0 = time.perf_counter()
    robust = joint_irl1(a, b_obs, gamma_weight)
    robust["time_s"] = time.perf_counter() - t0
    for result in (joint, robust):
        beta_eval = _support_metrics(result["beta_support"], true_beta, len(names))
        error_eval = _support_metrics(result["error_support"], set(true_error), len(b))
        result.update({"beta_metrics": beta_eval, "error_metrics": error_eval})
        result["refit_relative_residual"] = _refit_residual(a, b_obs, result["beta_support"], result["error_support"])
    return {
        "case": case["pde"],
        "layer": layer,
        "sigma": 0.0,
        "fraction": fraction,
        "amplitude_rms": amplitude,
        "seed": seed,
        "n_rows": int(len(b)),
        "library_size": len(names),
        "true_support": sorted(true_names),
        "outlier_rows": true_error,
        "baseline_GML1": base,
        "joint_L1": joint,
        "joint_IRL1": robust,
        "weak_data": meta,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--layers", nargs="+", default=DEFAULT_LAYERS, choices=("L13", "L113"))
    ap.add_argument("--fractions", nargs="+", type=float, default=DEFAULT_FRACTIONS)
    ap.add_argument("--amplitudes", nargs="+", type=float, default=DEFAULT_AMPLITUDES)
    ap.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--gamma", type=float, default=DEFAULT_GAMMA,
                    help="relative L1 weight for sparse error rows after column normalization")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    cases = dict(N.CASES)
    if args.only:
        cases = {name: cases[name] for name in args.only}
    if args.smoke:
        cases = {next(iter(cases)): next(iter(cases.values()))}
        args.layers = [args.layers[0]]
        args.fractions = [0.05]
        args.amplitudes = [10.0]
        args.seeds = 1
    t0 = time.perf_counter()
    rows = []
    for cname, case in cases.items():
        for layer in args.layers:
            for fraction in args.fractions:
                for amplitude in args.amplitudes:
                    for seed in range(args.seeds):
                        row = run_one(case, layer, fraction, amplitude, seed, args.gamma)
                        rows.append(row)
                        print(cname, layer, fraction, amplitude, seed, flush=True)
    report = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "method": "joint weighted L1: min 0.5||A beta + e - b||^2 + lambda||beta||_1 + gamma||e||_1",
        "support_tol": SUPPORT_TOL,
        "layers": args.layers,
        "fractions": args.fractions,
        "amplitudes_rms": args.amplitudes,
        "n_seeds": args.seeds,
        "gamma_weight": args.gamma,
        "rows": rows,
        "wall_time_s": time.perf_counter() - t0,
        "outlier_scope": "sparse contamination was injected in weak-equation right-hand-side rows; point anomalies in raw u(x,t) are a separate follow-up",
    }
    out = Path(args.output) if args.output else Path(__file__).with_name("outlier_l1_compare_result.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
