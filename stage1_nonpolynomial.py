"""The frozen 54 nonpolynomial weak columns, evaluated from raw observations.

No reconstructed derivative enters this API. All spatial derivatives are moved
onto the existing v2 discrete weak kernel, on the same centers and 15-point
discarded halo as the polynomial library. Reference derivatives below are used
only by the standalone numerical self-check, never by matrix construction.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import stage1_jet_preprocess as V1
from stage0_evaluation import array_hash, strict_json_write
from stage1_grammar import enumerate_jet, nonpolynomial_recipes
from stage1_weak_reduction_v2 import discrete_bump, weak_reduced_matrix

ROOT = Path(__file__).resolve().parent
HALO = 15
ACCURACY_TOL = 1e-4
REFINEMENT_FLOOR = 1e-7
CONSTANT_ABSOLUTE_TOL = 1e-8
ROUNDOFF_FACTOR = 256.


def _raw_function(raw, family, parameter):
    """Only the declared function values, with no clipping or overflow repair."""
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            if family == "sin": value = np.sin(parameter * raw)
            elif family == "cos": value = np.cos(parameter * raw)
            elif family == "exp": value = np.exp(parameter * raw)
            elif family == "inverse_quadratic": value = (1. + raw * raw) ** (-parameter)
            else: raise ValueError("undeclared nonpolynomial family")
    except FloatingPointError as error:
        raise ValueError(f"nonpolynomial numerical overflow/invalid value: {family}({parameter})") from error
    if not np.isfinite(value).all():
        raise ValueError("non-finite function value; entire 54-column construction rejected")
    return value


def weak_nonpolynomial_matrix(x, t, u, target_order=1, half=(12, 10), stride=None):
    """Return A[rows,54], b[rows], metadata; no normalization or support choice.

    A_j = (-1)^k integral f(u_raw) phi_x^(k) phi_t dx dt.
    The 54 recipes and their order come from stage1_grammar. The unchanged
    target supports u_t and u_tt; adding u_t to the wave library is a caller's
    separate, existing convention. This function always returns exactly 54.
    """
    if not all(np.isrealobj(v) for v in (x, t, u)):
        raise ValueError("real observations and real grids required")
    x, t, u = (np.asarray(v, dtype=float) for v in (x, t, u))
    if not all(np.isfinite(v).all() for v in (x, t, u)):
        raise ValueError("finite raw observations and grids required")
    if target_order not in (1, 2): raise ValueError("target_order must be 1 or 2")
    stride = tuple(half if stride is None else stride)
    half = tuple(half)
    if (len(half) != 2 or len(stride) != 2
            or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v <= 0
                   for v in (*half, *stride))):
        raise ValueError("half and stride must be two positive integers")
    x, t, u, dx, dt = V1._grid(x, t, u, half, 2 * HALO + 1)
    recipes = nonpolynomial_recipes()
    if len(recipes) != 54 or len({r["name"] for r in recipes}) != 54:
        raise ValueError("frozen nonpolynomial catalog no longer contains 54 unique recipes")
    raw = u[:, HALO:-HALO]
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            px, kernel_x = discrete_bump(half[0], dx, 4)
            pt, kernel_t = discrete_bump(half[1], dt, target_order)
            target = (-1)**target_order * V1._integrate(raw, px[0], pt[target_order], dx, dt, stride)
            matrix = np.empty((len(target), 54))
            cached_key, field = None, None
            value_ranges = []
            for j, recipe in enumerate(recipes):
                identity = (recipe["family"], recipe["parameter"])
                if identity != cached_key:
                    field = _raw_function(raw, *identity)
                    value_ranges.append({"family":identity[0], "parameter":identity[1],
                                         "minimum":float(field.min()), "maximum":float(field.max())})
                    cached_key = identity
                k = recipe["derivative_order"]
                matrix[:, j] = (-1)**k * V1._integrate(field, px[k], pt[0], dx, dt, stride)
    except FloatingPointError as error:
        raise ValueError("nonpolynomial weak construction overflowed; no columns removed") from error
    if not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise ValueError("non-finite weak matrix/target; no columns removed")
    hx, ht = half
    sx, st = stride
    return matrix, target, {
        "preprocessing":"frozen54_raw_function_values_with_v2_moment_consistent_weak_derivatives",
        "input_observation_hash":array_hash(u), "raw_shape":list(u.shape), "rows":len(target), "columns":54,
        "recipes":recipes, "names":[r["name"] for r in recipes], "target_order":target_order,
        "target_source":"original_observations_with_moment_consistent_weak_time_derivative",
        "half":list(half), "stride":list(stride), "test_power":V1.TEST_POWER,
        "discarded_halo_x":HALO, "raw_footprint_half_x":HALO+hx, "raw_footprint_half_t":ht,
        "center_x_indices":list(range(HALO+hx, len(x)-HALO-hx, sx)),
        "center_t_indices":list(range(ht, len(t)-ht, st)),
        "derivative_reconstruction":"none", "all_function_fields_source":"raw_observations_only",
        "max_test_function_derivative":4, "kernel_x":kernel_x, "kernel_t":kernel_t,
        "function_value_ranges":value_ranges, "overflow_policy":"reject entire construction; no clipping or column deletion",
        "matrix_hash":array_hash(matrix), "target_hash":array_hash(target),
    }


def _affine_reference(raw, slope, recipe):
    """Independent closed-form d_x^k f(c+s*x+d*t), verification only."""
    family, p, k = recipe["family"], recipe["parameter"], recipe["derivative_order"]
    if family == "sin": value = p**k * np.sin(p * raw + k * np.pi / 2)
    elif family == "cos": value = p**k * np.cos(p * raw + k * np.pi / 2)
    elif family == "exp": value = p**k * np.exp(p * raw)
    elif family == "inverse_quadratic":
        v = 1. + raw * raw
        if k == 0: value = v**(-p)
        elif k == 1: value = -2*p*raw*v**(-p-1)
        elif k == 2: value = -2*p*v**(-p-1) + 4*p*(p+1)*raw**2*v**(-p-2)
        elif k == 3: value = 12*p*(p+1)*raw*v**(-p-2) - 8*p*(p+1)*(p+2)*raw**3*v**(-p-3)
        else: raise ValueError("reference derivative outside frozen catalog")
    else: raise ValueError("unknown reference family")
    return slope**k * value


def _diagnostic_field(x, t):
    xx, tt = np.meshgrid(x, t)
    slope = 1.2 / (x[-1] - x[0])
    u = .2 + slope * (xx - (x[0] + x[-1]) / 2) + .1 * (tt-t[0]) / (t[-1]-t[0])
    return u, slope


def _accuracy_case(x, t, half, target_order):
    u, slope = _diagnostic_field(x, t)
    computed, target, meta = weak_nonpolynomial_matrix(x, t, u, target_order, half)
    dx, dt = x[1]-x[0], t[1]-t[0]
    px, _ = discrete_bump(half[0], dx, 0)
    pt, _ = discrete_bump(half[1], dt, 0)
    reference = np.column_stack([V1._integrate(_affine_reference(u[:, HALO:-HALO], slope, r),
                               px[0], pt[0], dx, dt, half) for r in nonpolynomial_recipes()])
    denominators = np.linalg.norm(reference, axis=0)
    errors = np.linalg.norm(computed-reference, axis=0) / np.maximum(denominators, 1e-30)
    _, old_target, old_meta = weak_reduced_matrix(x, t, u, enumerate_jet(0, 0, 0, 0),
                                                 target_order=target_order, half=half)
    return {"nx":len(x), "nt":len(t), "half":list(half), "physical_radius_x":float(half[0]*dx),
            "physical_radius_t":float(half[1]*dt), "max_relative_error":float(errors.max()),
            "worst_column":meta["names"][int(np.argmax(errors))], "all_column_relative_errors":errors.tolist(),
            "reference_column_norms":denominators.tolist(), "max_absolute_error":float(np.abs(computed-reference).max()),
            "target_bitwise_equal_v2":bool(np.array_equal(target, old_target)),
            "centers_equal_v2":all(meta[k] == old_meta[k] for k in ("center_x_indices", "center_t_indices")),
            "input_hash":array_hash(u), "metadata":meta}


def _constant_case(x, t, half, target_order):
    value = .7
    u = np.full((len(t), len(x)), value)
    computed, _, _ = weak_nonpolynomial_matrix(x, t, u, target_order, half)
    dx, dt = x[1]-x[0], t[1]-t[0]
    px, _ = discrete_bump(half[0], dx, 4)
    pt, _ = discrete_bump(half[1], dt, 0)
    rows = []
    for j, recipe in enumerate(nonpolynomial_recipes()):
        k = recipe["derivative_order"]
        if k == 0: continue
        magnitude = abs(float(_raw_function(np.array([value]), recipe["family"], recipe["parameter"])[0]))
        bound = ROUNDOFF_FACTOR*np.finfo(float).eps*magnitude*np.abs(px[k]).sum()*np.abs(pt[0]).sum()*dx*dt
        error = float(np.abs(computed[:, j]).max())
        rows.append({"name":recipe["name"], "max_absolute_error":error, "roundoff_envelope":float(bound),
                     "within_roundoff_envelope":bool(error <= bound)})
    return {"constant_value":value, "expected_derivative_integral":0., "rows":rows,
            "max_absolute_error":max(r["max_absolute_error"] for r in rows),
            "all_within_roundoff_envelope":all(r["within_roundoff_envelope"] for r in rows),
            "absolute_floor_not_relative_error":True}


def main():
    from stage2_data import CASES
    start = time.perf_counter()
    checks, rows = {}, []
    with threadpool_limits(limits=1):
        for case, spec in CASES.items():
            x, t = np.linspace(*spec["x"]), np.linspace(*spec["t"])
            half = spec["half"]
            target_order = 2 if spec["target"] == "u_tt" else 1
            actual = _accuracy_case(x, t, half, target_order)
            # Spatial refinement keeps the physical weak half-width, temporal
            # grid, and 15 discarded samples fixed. No kernel is retuned.
            refined_x = np.linspace(x[0], x[-1], 2*(len(x)-1)+1)
            refined = _accuracy_case(refined_x, t, (2*half[0], half[1]), target_order)
            constant = _constant_case(x, t, half, target_order)
            passed = {
                "all54_actual_accuracy":actual["max_relative_error"] < ACCURACY_TOL,
                "all54_refined_accuracy":refined["max_relative_error"] < ACCURACY_TOL,
                "refinement_or_already_below_floor":refined["max_relative_error"] < max(
                    .75*actual["max_relative_error"], REFINEMENT_FLOOR),
                "same_physical_radius":bool(np.isclose(actual["physical_radius_x"], refined["physical_radius_x"], rtol=1e-12, atol=0)),
                "same_target_and_centers":all(v["target_bitwise_equal_v2"] and v["centers_equal_v2"] for v in (actual, refined)),
                "constant_absolute_floor":constant["max_absolute_error"] < CONSTANT_ABSOLUTE_TOL,
                "constant_roundoff_envelope":constant["all_within_roundoff_envelope"],
            }
            rows.append({"case_grid":case, "target_order":target_order, "actual":actual, "refined":refined,
                         "constant":constant, "checks":passed, "all_checks_passed":all(passed.values())})
            print(case, passed, "max_error", actual["max_relative_error"], flush=True)
        x, t = np.linspace(-1., 1., 81), np.linspace(0., 1., 25)
        rejections = []
        for label, value in (("exp_overflow", 1000.), ("nan_observation", np.nan), ("infinite_observation", np.inf)):
            try:
                weak_nonpolynomial_matrix(x, t, np.full((len(t), len(x)), value), half=(12, 6))
            except ValueError as error:
                rejections.append({"case":label, "rejected":True, "error":str(error)})
            else: rejections.append({"case":label, "rejected":False})
    checks["six_actual_and_refined_grids"] = len(rows) == 6 and all(r["all_checks_passed"] for r in rows)
    checks["overflow_and_nonfinite_rejected"] = all(r["rejected"] for r in rejections)
    checks["exactly54_declared_recipes"] = len(nonpolynomial_recipes()) == 54
    sources = ["stage1_nonpolynomial.py", "stage1_grammar.py", "stage1_weak_reduction_v2.py",
               "stage1_weak_reduction.py", "stage1_jet_preprocess.py", "stage2_data.py", "stage0_evaluation.py"]
    report = {"stage":"1_nonpolynomial_weak_supplement", "all_checks_passed":all(checks.values()), "checks":checks,
              "source_sha256":{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
              "recipes":nonpolynomial_recipes(), "columns":54, "target_orders_checked":[1, 2], "rows":rows,
              "thresholds_frozen_before_first_run":{"relative_accuracy":ACCURACY_TOL,
                  "refinement_roundoff_floor":REFINEMENT_FLOOR, "constant_absolute_error":CONSTANT_ABSOLUTE_TOL,
                  "roundoff_envelope_multiplier":ROUNDOFF_FACTOR},
              "rejection_checks":rejections, "elapsed_s":time.perf_counter()-start,
              "reference":"independent closed-form chain derivatives for affine synthetic field, verification only",
              "no_pde_fitting_run":True, "test_trajectories_used":False,
              "limitations":["This API adds exactly the frozen 54 functions, not every possible nonpolynomial function.",
                  "Affine clean fixtures test the kernels and signs on actual grids, not all PDE fields or high-frequency signals.",
                  "Fixed physical-width refinement is spatial; the temporal grid is unchanged.",
                  "Constant derivative columns retain measured floating-point floors; no centering or threshold deletion is performed.",
                  "Nonlinear functions of noisy raw observations may be biased; this check does not establish noise robustness.",
                  "Finite output is not an identifiability or conditioning guarantee; no support/complexity/selection policy is added here.",
                  "Maximum1212 (wave1213 with existing u_t column) requires downstream integration checks before experiments."]}
    strict_json_write(ROOT/"stage1_nonpolynomial_result.json", report)
    print({"all_checks_passed":report["all_checks_passed"], "elapsed_s":report["elapsed_s"]})
    if not report["all_checks_passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
