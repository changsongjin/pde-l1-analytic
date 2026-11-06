"""End-to-end probe with sparse point anomalies in the observed field u(x,t)."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
from numpy.polynomial import polynomial as Poly
from scipy.signal import correlate

import gml1_maxlib_compare as G
import noise_inverse as N
from outlier_l1_compare import (
    DEFAULT_GAMMA,
    _refit_residual,
    _support_metrics,
    baseline_l1,
    joint_irl1,
    joint_l1,
    layer_names,
)


SUPPORT_TOL = 1e-5
DEFAULT_FRACTIONS = [0.0, 0.0001, 0.0005, 0.001]
DEFAULT_AMPLITUDES = [5.0, 10.0]
DEFAULT_SEEDS = 3


def _bump(n_half: int, h: float, max_d: int, power: int = G.P_TEST) -> list[np.ndarray]:
    s = np.linspace(-1.0, 1.0, 2 * n_half + 1)
    a = n_half * h
    q = Poly.polypow([1.0, 0.0, -1.0], power)
    out = []
    for _ in range(max_d + 1):
        out.append(Poly.polyval(s, q) / a**len(out))
        q = Poly.polyder(q)
    return out


def make_point_data(case: dict, terms: list[dict], fraction: float, amplitude: float, seed: int):
    rng = np.random.default_rng(seed)
    x = np.linspace(*case["x"])
    t = np.linspace(*case["t"])
    dx, dt = x[1] - x[0], t[1] - t[0]
    tt, xx = np.meshgrid(t, x, indexing="ij")
    hx, ht = case["half"]
    px = _bump(hx, dx, max(term["k"] for term in terms))
    pt = _bump(ht, dt, 2)
    blocks_a, blocks_b, outliers, affected = [], [], [], []
    row_offset = 0
    for traj_id, fn in enumerate(case["traj"]):
        u0 = fn(xx.ravel(), tt.ravel()).reshape(xx.shape)
        u = u0.copy()
        count = int(round(fraction * u.size))
        point_indices = np.sort(rng.choice(u.size, count, replace=False)) if count else np.array([], dtype=int)
        scale = max(float(np.std(u0)), 1e-12)
        deltas = amplitude * scale * rng.choice([-1.0, 1.0], size=count) if count else np.array([])
        if count:
            u.flat[point_indices] += deltas
        centers_t = np.arange(ht, u.shape[0] - ht, ht)
        centers_x = np.arange(hx, u.shape[1] - hx, hx)
        for flat, delta in zip(point_indices, deltas):
            ti, xi = np.unravel_index(int(flat), u.shape)
            outliers.append({"trajectory": traj_id, "t_index": int(ti), "x_index": int(xi), "delta": float(delta)})
            for ri, ct in enumerate(centers_t):
                if abs(int(ct) - int(ti)) > ht:
                    continue
                for ci, cx in enumerate(centers_x):
                    if abs(int(cx) - int(xi)) <= hx:
                        affected.append(row_offset + ri * len(centers_x) + ci)

        def integ(field: np.ndarray, kx: int, kt: int = 0) -> np.ndarray:
            kernel = np.outer(pt[kt], px[kx])
            full = correlate(field, kernel, mode="valid", method="fft") * dx * dt
            return full[::ht, ::hx].ravel()

        cols = [(-1) ** term["k"] * integ(G._base_field(term, u, xx), term["k"], 0) for term in terms]
        b = integ(u, 0, 2) if case["target"] == "u_tt" else -integ(u, 0, 1)
        if case["target"] == "u_tt":
            cols.append(-integ(u, 0, 1))
        a = np.column_stack(cols)
        norm = max(float(np.linalg.norm(b)), 1e-15)
        blocks_a.append(a / norm)
        blocks_b.append(b / norm)
        row_offset += len(b)
    a = np.vstack(blocks_a)
    b = np.concatenate(blocks_b)
    names = [G.term_name(term) for term in terms]
    if case["target"] == "u_tt":
        names.append("u_t")
    return a, b, names, sorted(set(affected)), outliers, {"x": list(case["x"]), "t": list(case["t"]), "rows": int(len(b))}


def _subset(a: np.ndarray, names: list[str], wanted: list[str]) -> tuple[np.ndarray, list[str]]:
    pos = {name: i for i, name in enumerate(names)}
    if any(name not in pos for name in wanted):
        raise RuntimeError("requested library term is missing from generated source")
    return a[:, [pos[name] for name in wanted]], list(wanted)


def run_one(case: dict, layer: str, fraction: float, amplitude: float, seed: int, gamma: float) -> dict:
    terms = G.build_terms(False)
    a_full, b, full_names, affected, outliers, meta = make_point_data(case, terms, fraction, amplitude, seed)
    wanted = layer_names(layer)
    if case["target"] == "u_tt":
        wanted.append("u_t")
    a, names = _subset(a_full, full_names, wanted)
    true = {key: value for key, value in case["true"].items() if key in names}
    base = baseline_l1(a, b, names, true)
    results = {"joint_L1": joint_l1(a, b, gamma), "joint_IRL1": joint_irl1(a, b, gamma)}
    true_beta = {names.index(key) for key in true}
    for result in results.values():
        result["beta_metrics"] = _support_metrics(result["beta_support"], true_beta, len(names))
        result["affected_row_metrics"] = _support_metrics(result["error_support"], set(affected), len(b))
        result["refit_relative_residual"] = _refit_residual(a, b, result["beta_support"], result["error_support"])
    return {
        "case": case["pde"], "layer": layer, "fraction": fraction, "amplitude_rms": amplitude, "seed": seed,
        "library_size": len(names), "n_rows": len(b), "true_support": sorted(true),
        "raw_outliers": outliers, "affected_weak_rows": affected, "baseline_GML1": base,
        **results, "weak_data": meta,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--layers", nargs="+", default=["L13", "L113"], choices=("L13", "L113"))
    ap.add_argument("--fractions", nargs="+", type=float, default=DEFAULT_FRACTIONS)
    ap.add_argument("--amplitudes", nargs="+", type=float, default=DEFAULT_AMPLITUDES)
    ap.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    cases = dict(N.CASES)
    if args.only:
        cases = {name: cases[name] for name in args.only}
    if args.smoke:
        cases = {next(iter(cases)): next(iter(cases.values()))}
        args.layers = [args.layers[0]]
        args.fractions = [0.001]
        args.amplitudes = [10.0]
        args.seeds = 1
    t0 = time.perf_counter()
    rows = []
    for cname, case in cases.items():
        for layer in args.layers:
            for fraction in args.fractions:
                for amplitude in args.amplitudes:
                    for seed in range(args.seeds):
                        rows.append(run_one(case, layer, fraction, amplitude, seed, args.gamma))
                        print(cname, layer, fraction, amplitude, seed, flush=True)
    report = {
        "python": platform.python_version(), "numpy": np.__version__,
        "method": "raw-field point anomalies; weak matrix A and b are built from contaminated u(x,t)",
        "support_tol": SUPPORT_TOL, "gamma_weight": args.gamma, "layers": args.layers,
        "fractions": args.fractions, "amplitudes_rms": args.amplitudes, "n_seeds": args.seeds,
        "rows": rows, "wall_time_s": time.perf_counter() - t0,
        "interpretation": "affected_weak_rows are a diagnostic footprint, not an exact point-outlier support because raw anomalies also contaminate A",
    }
    out = Path(args.output) if args.output else Path(__file__).with_name("point_outlier_probe_result.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
