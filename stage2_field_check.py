"""Independent development-field diagnostics; no derivatives enter regression.

Finite differences call only stage2_data.field_value. Parameter IDs 0..5 are
the development set; reserved final-test trajectories/noise are never generated.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import sympy as sp

import stage2_data as D
from stage0_evaluation import strict_json_write


ROOT = Path(__file__).resolve().parent
OFFSETS = tuple(range(-3, 4))
LEVELS = (1., .5, .25)
# Physical step lengths, fixed before observing the diagnostic errors.
STEPS = {"heat": (.025, .0005), "wave": (.025, .01),
         "advection_diffusion": (.1, .01), "burgers": (.04, .01),
         "kdv": (.2, .02), "logheat": (.025, .0005)}
FINE_RELATIVE_TOL = 1e-4
CONVERGENCE_FLOOR = 1e-7


def stencil(order):
    coefficients = sp.finite_diff_weights(order, OFFSETS, 0)[order][-1]
    for degree in range(len(OFFSETS)):
        expected = math.factorial(order) if degree == order else 0
        assert sum(c * q**degree for c, q in zip(coefficients, OFFSETS)) == expected
    return np.array([float(c) for c in coefficients])


WEIGHTS = {r: stencil(r) for r in (1, 2, 3)}


def derivative(case, p, x, t, axis, order, h):
    values = np.stack([D.field_value(case, x + q*h if axis == "x" else x,
                                     t + q*h if axis == "t" else t, p)
                       for q in OFFSETS])
    # Subtracting the central value limits cancellation; all derivative weights sum to zero.
    return np.tensordot(WEIGHTS[order], values - values[3], axes=(0, 0)) / h**order


def pde_difference(case, p, x, t, hx, ht):
    u = D.field_value(case, x, t, p)
    ux = derivative(case, p, x, t, "x", 1, hx)
    uxx = derivative(case, p, x, t, "x", 2, hx)
    target = derivative(case, p, x, t, "t", 2 if case == "wave" else 1, ht)
    if case in ("heat", "wave"):
        rhs = uxx
    elif case == "advection_diffusion":
        rhs = -.4*ux + .05*uxx
    elif case == "burgers":
        rhs = -u*ux + .1*uxx
    elif case == "kdv":
        rhs = -6*u*ux - derivative(case, p, x, t, "x", 3, hx)
    elif case == "logheat":
        rhs = uxx + ux**2
    else:
        raise ValueError(case)
    norm = max(float(np.linalg.norm(target)), float(np.linalg.norm(rhs)), 1e-14)
    return {"hx": hx, "ht": ht, "relative_residual": float(np.linalg.norm(target-rhs) / norm),
            "maximum_absolute_residual": float(np.max(np.abs(target-rhs))),
            "target_rms": float(np.sqrt(np.mean(target**2))),
            "rhs_rms": float(np.sqrt(np.mean(rhs**2))),
            "normalization": "max(||time derivative||_2,||PDE spatial RHS||_2,1e-14)"}


def distinct_initial_conditions(case):
    x = np.linspace(D.CASES[case]["x"][0], D.CASES[case]["x"][1], 401)
    fields = [D.field_value(case, x, np.zeros_like(x), D.trajectory_parameters(case, i)) for i in range(6)]
    rows = []
    for i, j in itertools.combinations(range(6), 2):
        norm = max(float(np.linalg.norm(fields[i])), float(np.linalg.norm(fields[j])), 1e-14)
        rows.append({"ids": [i, j], "relative_initial_field_distance": float(np.linalg.norm(fields[i]-fields[j])/norm)})
    parameter_strings = {json.dumps(D.trajectory_parameters(case, i), sort_keys=True) for i in range(6)}
    return {"parameter_sets_distinct": len(parameter_strings) == 6,
            "minimum_pairwise_initial_field_distance": min(r["relative_initial_field_distance"] for r in rows),
            "pairwise": rows,
            "interpretation": "distinct initial conditions; not a proof of statistical independence or identifiability"}


def positivity():
    spec = D.CASES["logheat"]
    xx, tt = np.meshgrid(np.linspace(*spec["x"]), np.linspace(*spec["t"]))
    rows = []
    for i in range(6):
        p = D.trajectory_parameters("logheat", i)
        lower = 1 - float(np.sum(np.abs(p["coefficients"])))
        value = D.field_value("logheat", xx, tt, p)
        minimum = float(np.min(np.exp(value)))
        assert lower >= .45 - 1e-14 and minimum >= lower - 1e-14
        rows.append({"trajectory_id": i, "analytic_w_lower_bound_for_t_nonnegative": lower,
                     "minimum_sampled_positive_heat_field": minimum,
                     "identity": "u=log(w), w_t=w_xx -> u_t=u_xx+u_x^2"})
    return rows


def frequency_diagnostics(case):
    spec = D.CASES[case]
    hx, ht = spec["half"]
    dx = (spec["x"][1]-spec["x"][0])/(spec["x"][2]-1)
    dt = (spec["t"][1]-spec["t"][0])/(spec["t"][2]-1)
    sx, st = np.linspace(-1., 1., 2*hx+1), np.linspace(-1., 1., 2*ht+1)
    px, pt = (1-sx*sx)**8, (1-st*st)**8
    ox, ot = sx*hx*dx, st*ht*dt
    centers = (spec["t"][0]+ht*dt, spec["t"][1]-ht*dt)
    rows = []
    for i in range(6):
        p = D.trajectory_parameters(case, i)
        frequencies = []
        for mode in range(1, 7):
            k = mode*np.pi
            spatial_transfer = float(abs(np.trapezoid(px*np.cos(k*ox), ox)/np.trapezoid(px, ox)))
            if case == "heat":
                amplitude = abs(p["cos_coefficients"][mode-1])
                center_decay = [float(np.exp(-k*k*c)) for c in centers]
                temporal_average = [float(np.trapezoid(pt*np.exp(-k*k*(c+ot)), ot)/np.trapezoid(pt, ot))
                                    for c in centers]
                frequencies.append({"mode": mode, "initial_amplitude": amplitude,
                                    "center_decay_first_last": center_decay,
                                    "weak_field_amplitude_proxy_first_last": [amplitude*spatial_transfer*q for q in temporal_average],
                                    "spatial_window_transfer": spatial_transfer})
            else:
                amplitude = math.hypot(p["cos_coefficients"][mode-1], p["sin_coefficients"][mode-1])
                temporal_transfer = float(abs(np.trapezoid(pt*np.cos(k*ot), ot)/np.trapezoid(pt, ot)))
                frequencies.append({"mode": mode, "initial_temporal_envelope": amplitude,
                                    "weak_field_envelope_proxy": amplitude*spatial_transfer*temporal_transfer,
                                    "spatial_window_transfer": spatial_transfer,
                                    "temporal_window_transfer": temporal_transfer})
        rows.append({"trajectory_id": i, "modes": frequencies})
    return {"case": case, "test_function_power": 8, "weak_time_centers_first_last": centers,
            "spatial_half_width": hx*dx, "temporal_half_width": ht*dt,
            "rows": rows, "interpretation": "field-average Fourier transfer proxy; does not measure full design rank, noise SNR, or unique recoverability"}


def main():
    started = time.perf_counter()
    if not json.loads((ROOT/"stage1_grammar_result.json").read_text(encoding="utf-8"))["all_checks_passed"]:
        raise RuntimeError("stage1 grammar has not passed")
    checks, failures = [], []
    for case, spec in D.CASES.items():
        hx0, ht0 = STEPS[case]
        rng = np.random.default_rng(62000 + list(D.CASES).index(case))
        x = rng.uniform(spec["x"][0]+3*hx0, spec["x"][1]-3*hx0, 128)
        # Early/mid-time points retain signal; all stencils remain within t>=0.
        t = rng.uniform(spec["t"][0]+3*ht0, min(spec["t"][1]-3*ht0, .8*spec["t"][1]), 128)
        for trajectory_id in range(6):
            p = D.trajectory_parameters(case, trajectory_id)
            levels = [pde_difference(case, p, x, t, hx0*q, ht0*q) for q in LEVELS]
            coarse, fine = levels[0]["relative_residual"], levels[-1]["relative_residual"]
            # A numerical floor permits nonmonotonic roundoff after convergence.
            converges = fine <= CONVERGENCE_FLOOR or fine <= coarse/4
            passed = fine <= FINE_RELATIVE_TOL and converges
            row = {"case": case, "trajectory_id": trajectory_id, "levels": levels,
                   "fine_over_coarse_error_ratio": fine/max(coarse, 1e-300),
                   "convergence_or_numerical_floor": converges, "passed": passed}
            checks.append(row)
            if not passed:
                failures.append({"case": case, "trajectory_id": trajectory_id, "finest_residual": fine})
    distinct = {case: distinct_initial_conditions(case) for case in D.CASES}
    for case, row in distinct.items():
        if not row["parameter_sets_distinct"] or row["minimum_pairwise_initial_field_distance"] <= 1e-6:
            failures.append({"case": case, "reason": "initial conditions not sufficiently distinct"})
    report = {"stage": 2, "component": "independent field correctness diagnostics", "date": "2026-11-06",
              "python": platform.python_version(), "numpy": np.__version__, "sympy": sp.__version__,
              "all_checks_passed": not failures, "failures": failures, "field_checks": checks,
              "finite_difference_protocol": {"offsets": OFFSETS, "weights": {str(r): w.tolist() for r,w in WEIGHTS.items()},
                                              "levels": LEVELS, "finest_relative_tolerance": FINE_RELATIVE_TOL,
                                              "convergence_floor": CONVERGENCE_FLOOR,
                                              "comparison": "finest<=coarsest/4 OR finest<=numerical floor",
                                              "reference": "independent temporal and spatial finite differences of field values only"},
              "distinct_initial_conditions": distinct, "logheat_positivity": positivity(),
              "frequency_diagnostics": {c: frequency_diagnostics(c) for c in ("heat", "wave")},
              "source_hashes": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                for name in ("stage2_field_check.py", "stage2_data.py", "redo_two.py", "inverse_irl1.py")},
              "test_data_generated": False, "derivatives_used_for_regression": False,
              "limitations": ["Finite-difference consistency is a numerical diagnostic, not a proof for every point.",
                              "Distinct initial fields or six Fourier modes do not prove sparse identifiability.",
                              "No noisy field derivatives, method selection, or test results are evaluated here."],
              "elapsed_seconds": time.perf_counter()-started}
    strict_json_write(ROOT/"stage2_field_check_result.json", report)
    print(json.dumps({"all_checks_passed": not failures, "checks": len(checks), "failures": failures,
                      "max_finest_relative_residual": max(r["levels"][-1]["relative_residual"] for r in checks),
                      "elapsed_seconds": report["elapsed_seconds"]}))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
