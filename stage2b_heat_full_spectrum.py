"""Prepared full-resolved-spectrum evaluator; main runs numerical checks only.

Applicable subset: u_t = c_const + c0*u + c2*u_xx + c4*u_xxxx + c6*u_xxxxxx,
under an explicitly supplied polynomial-of-Dirichlet-Laplacian realization.
All nx-2 resolved sine modes are used. Unsupported candidates are reported,
never removed from the research library or reclassified as statistical failures.
No benchmark candidate comparison or final-test data generation occurs here.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import scipy
from numpy.polynomial import polynomial as Poly
from scipy.fft import dst, idst
from scipy.linalg import expm
from scipy.special import logsumexp

from stage0_evaluation import array_hash, canonical_with_scale, strict_json_write

ROOT = Path(__file__).resolve().parent
BOUNDARY = "dirichlet_laplacian_polynomial"
SCHEMES = ("raw_all_modes", "frozen_tikhonov_dx2")
ORDERS = {"u": 0, "u_xx": 2, "u_xxxx": 4, "u_xxxxxx": 6}
SUPPORTED = {"1", *ORDERS}
LOG_MAX = math.log(np.finfo(float).max)
LOG_MIN_SUBNORMAL = math.log(np.nextafter(0., 1.))


def _grid(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 4 or not np.isfinite(x).all():
        raise ValueError("need finite endpoint-including spatial grid with at least four nodes")
    dx = float(x[1]-x[0])
    if dx <= 0 or not np.allclose(np.diff(x), dx, rtol=1e-12, atol=1e-14):
        raise ValueError("uniform increasing spatial grid required")
    k = np.arange(1, len(x)-1)*np.pi/(x[-1]-x[0])
    return x, dx, k


def operator_spec(coefficients, *, boundary_realization):
    """No candidate is simplified away based on a zero LS readout coefficient."""
    if boundary_realization != BOUNDARY:
        return {"status": "unsupported_boundary_realization", "required": BOUNDARY}
    canonical, unsupported = {}, []
    for name, coefficient in coefficients.items():
        if not math.isfinite(float(coefficient)):
            raise ValueError("non-finite candidate coefficient")
        try:
            key, scale = canonical_with_scale(name)
        except ValueError:
            unsupported.append(name)
            continue
        if key not in SUPPORTED:
            unsupported.append(name)
            continue
        canonical[key] = canonical.get(key, 0.) + float(coefficient)*scale
    if unsupported:
        return {"status": "unsupported_candidate_structure", "unsupported_terms": unsupported,
                "candidate_support_retained": list(coefficients),
                "selection_action": "none; do not treat unavailable rollout as failure or silently drop this candidate"}
    growth = np.zeros(4)
    for key, order in ORDERS.items():
        growth[order//2] = (-1)**(order//2)*canonical.get(key, 0.)
    return {"status": "supported", "physical_coefficients": canonical,
            "original_selected_support": list(coefficients),
            "growth_polynomial_in_k_squared": growth.tolist(),
            "constant_forcing": canonical.get("1", 0.), "boundary_realization": BOUNDARY,
            "higher_order_boundary_assumption": "powers of the Dirichlet Laplacian; u=0 alone is insufficient to define a general fourth/sixth-order PDE"}


def prepare_initialization(x, observed_initial, *, scheme, boundary_realization):
    """Use only t0. Identical scheme and coefficients serve every method/model."""
    x, dx, k = _grid(x)
    u0 = np.asarray(observed_initial, dtype=float)
    if boundary_realization != BOUNDARY or scheme not in SCHEMES:
        raise ValueError("explicit supported boundary realization and frozen IC scheme required")
    if u0.shape != x.shape or not np.isfinite(u0).all():
        raise ValueError("initial observation shape/values invalid")
    raw = dst(u0[1:-1], type=1, norm="ortho")
    gains = np.ones_like(k) if scheme == SCHEMES[0] else 1./(1.+(2*dx*k)**4)
    if not np.all(gains > 0):
        raise ValueError("initial reconstruction must retain every resolved mode")
    coefficients = gains*raw
    return {"scheme": scheme, "boundary_realization": BOUNDARY,
            "x_hash": array_hash(x), "observed_initial_hash": array_hash(u0),
            "mode_count": len(k), "modal_coefficients": coefficients,
            "raw_modal_coefficients": raw, "gains": gains,
            "prepared_coefficients_hash": array_hash(coefficients),
            "metadata": {"mode_indices": [1, len(k)], "mode_count": len(k),
                "no_hard_mode_cutoff": True, "gain_min": float(gains.min()), "gain_max": float(gains.max()),
                "boundary_observations": [float(u0[0]), float(u0[-1])],
                "boundary_handling": "known homogeneous boundary values enforced; noisy endpoint observations retained in future scoring",
                "smoothing_rule": "identity" if scheme == SCHEMES[0] else "gain(k)=1/(1+(2*dx*k)^4)",
                "regularization_length": 0. if scheme == SCHEMES[0] else 2*dx,
                "fit_information": "initial observations and spatial grid only; no PDE truth, selected model, future observations, or noise-level tuning",
                "forecast_role": "separate frozen initialization ablation, never a per-method optimization"}}


def _signed_log(values):
    values = np.asarray(values, dtype=float)
    logs = np.full(values.shape, -np.inf)
    nonzero = values != 0
    logs[nonzero] = np.log(np.abs(values[nonzero]))
    return logs, np.sign(values)


def _signed_add(log_a, sign_a, log_b, sign_b):
    log_a, sign_a, log_b, sign_b = np.broadcast_arrays(log_a, sign_a, log_b, sign_b)
    return logsumexp(np.stack([log_a, log_b]), b=np.stack([sign_a, sign_b]), axis=0, return_sign=True)


def modal_solution_log(initial, forcing, rates, elapsed):
    """Signed logs of a'=s*a+f, with stable small-exponent and equilibrium forms."""
    initial, forcing, rates = map(np.asarray, (initial, forcing, rates))
    elapsed = np.asarray(elapsed)
    if initial.shape != forcing.shape or initial.shape != rates.shape:
        raise ValueError("modal dimensions differ")
    with np.errstate(over="raise", invalid="raise"):
        z = elapsed[:, None]*rates[None, :]
    la, sa = _signed_log(initial)
    lf, sf = _signed_log(forcing)
    # For |s*t|<=.5, t*exprel(s*t) avoids division by small s.
    log_integral = np.full(z.shape, -np.inf)
    small = np.abs(z) <= .5
    positive_time = np.broadcast_to(elapsed[:, None] > 0, z.shape)
    small_nonzero_time = small & positive_time
    exprel = np.ones(z.shape)
    nonzero_z = small & (z != 0)
    exprel[nonzero_z] = np.expm1(z[nonzero_z])/z[nonzero_z]
    times = np.broadcast_to(elapsed[:, None], z.shape)
    log_integral[small_nonzero_time] = np.log(times[small_nonzero_time])+np.log(exprel[small_nonzero_time])
    logs, signs = _signed_add(la[None, :]+z, sa[None, :], lf[None, :]+log_integral, sf[None, :])
    # Large |s*t|: exp(s*t)*(a0+f/s)-f/s preserves exact equilibria instead
    # of subtracting two huge separately evaluated exponentials.
    nonzero_rate = rates != 0
    lb = np.full(rates.shape, -np.inf)
    sb = np.zeros(rates.shape)
    lb[nonzero_rate] = lf[nonzero_rate]-np.log(np.abs(rates[nonzero_rate]))
    sb[nonzero_rate] = sf[nonzero_rate]*np.sign(rates[nonzero_rate])
    lc, sc = _signed_add(la, sa, lb, sb)
    large_logs, large_signs = _signed_add(lc[None, :]+z, sc[None, :], lb[None, :], -sb[None, :])
    logs[~small], signs[~small] = large_logs[~small], large_signs[~small]
    if np.isnan(logs).any() or np.isposinf(logs).any() or not np.isfinite(signs).all():
        raise ValueError("modal signed-log propagation exceeded supported numerical range")
    return logs, signs


def _positive_roots(coefficients):
    coefficients = np.trim_zeros(np.asarray(coefficients), "b")
    if len(coefficients) < 2:
        return []
    roots = Poly.polyroots(coefficients)
    return sorted(float(r.real) for r in roots if r.real > 0 and abs(r.imag) <= 1e-10*max(1., abs(r.real)))


def spectrum_summary(growth, k, horizon):
    growth = np.trim_zeros(np.asarray(growth), "b")
    if not len(growth):
        growth = np.array([0.])
    with np.errstate(over="raise", invalid="raise"):
        rates = Poly.polyval(k*k, growth)
    if not np.isfinite(rates).all():
        raise ValueError("growth polynomial exceeds float64 numerical range")
    degree, leading = len(growth)-1, float(growth[-1])
    if degree == 0:
        classification = "bounded_zero_order_growth" if leading > 0 else "bounded_zero_order_non_growth"
        max_rate = leading
    elif leading > 0:
        classification, max_rate = "unbounded_high_frequency_growth", None
    else:
        critical = [0., *_positive_roots(Poly.polyder(growth))]
        max_rate = float(max(Poly.polyval(z, growth) for z in critical))
        classification = "high_frequency_damped_with_finite_band_growth" if max_rate > 0 else "high_frequency_damped_no_positive_growth"
    bands = []
    bounds = [0., *_positive_roots(growth), math.inf]
    for lower, upper in zip(bounds[:-1], bounds[1:]):
        sample = max(2*lower, lower+1) if math.isinf(upper) else (lower+upper)/2
        if Poly.polyval(sample, growth) > 0:
            bands.append({"k_lower": math.sqrt(lower), "k_upper": None if math.isinf(upper) else math.sqrt(upper)})
    return {"continuum_classification": classification, "continuum_max_growth_rate": max_rate,
        "continuum_max_growth_status": "unbounded" if max_rate is None else "finite",
        "positive_growth_bands": bands, "resolved_mode_count": len(k),
        "resolved_mode_indices": [1, len(k)], "largest_resolved_wavenumber": float(k[-1]),
        "resolved_positive_growth_modes": (np.flatnonzero(rates > 0)+1).tolist(),
        "resolved_max_growth_rate": float(rates.max()), "resolved_min_growth_rate": float(rates.min()),
        "resolved_max_log_amplification_at_horizon": float(rates.max()*horizon),
        "interpretation": "bounded-frequency/zero-order growth is not by itself ill-posedness; unbounded positive high-frequency rates are a distinct continuum issue"}, rates


def _metric(log_residual_squared, log_observation_squared):
    if not np.isfinite(log_observation_squared):
        return {"status": "zero_observation_norm_relative_error_undefined", "relative_error": None,
                "log_relative_error": None, "log_residual_squared": float(log_residual_squared) if np.isfinite(log_residual_squared) else None}
    if np.isneginf(log_residual_squared):
        return {"status": "zero_residual", "relative_error": 0., "log_relative_error": None}
    value = float(.5*(log_residual_squared-log_observation_squared))
    return {"status": "finite" if value <= LOG_MAX else "above_float64_range_use_log",
            "relative_error": float(np.exp(value)) if value <= LOG_MAX else None,
            "log_relative_error": value}


def evaluate_operator(coefficients, x, t, observations, prepared_initial, *, boundary_realization,
                      return_prediction=False):
    """Full-grid spectral forecast and Parseval score, without model selection.

    Returns (JSON-safe report, prediction_or_None). Missing prediction arrays
    indicate overflow protection; log-domain forecast errors still use all modes.
    """
    spec = operator_spec(coefficients, boundary_realization=boundary_realization)
    if spec["status"] != "supported":
        return {**spec, "error_metric": None, "eligible_for_whole_library_ranking": False}, None
    x, dx, k = _grid(x)
    t, observations = np.asarray(t, dtype=float), np.asarray(observations, dtype=float)
    if (t.ndim != 1 or len(t) < 2 or not np.all(np.diff(t) > 0) or not np.isfinite(t).all()
            or observations.shape != (len(t), len(x)) or not np.isfinite(observations).all()):
        raise ValueError("invalid time grid or observation field")
    if (prepared_initial["x_hash"] != array_hash(x)
            or prepared_initial["observed_initial_hash"] != array_hash(observations[0])
            or prepared_initial["boundary_realization"] != BOUNDARY
            or prepared_initial["mode_count"] != len(k)
            or prepared_initial["scheme"] not in SCHEMES
            or array_hash(prepared_initial["modal_coefficients"]) != prepared_initial["prepared_coefficients_hash"]):
        raise ValueError("initial state was prepared from a different grid/observation/boundary contract")
    elapsed = t-t[0]
    summary, rates = spectrum_summary(spec["growth_polynomial_in_k_squared"], k, float(elapsed[-1]))
    forcing = spec["constant_forcing"]*dst(np.ones(len(k)), type=1, norm="ortho")
    logs, signs = modal_solution_log(prepared_initial["modal_coefficients"], forcing, rates, elapsed)
    observed_modes = dst(observations[1:, 1:-1], type=1, norm="ortho", axis=1)
    observed_logs, observed_signs = _signed_log(observed_modes)
    residual_logs, _ = _signed_add(logs[1:], signs[1:], observed_logs, -observed_signs)
    boundary_logs, _ = _signed_log(observations[1:, [0, -1]])
    # Orthonormal DST-I preserves the interior Euclidean norm. Endpoint errors
    # are included explicitly because the forecast enforces zero boundary data.
    numerator = float(logsumexp(np.concatenate([2*residual_logs.ravel(), 2*boundary_logs.ravel()])))
    target_logs, _ = _signed_log(observations[1:])
    denominator = float(logsumexp(2*target_logs.ravel()))
    metric = _metric(numerator, denominator)
    maximum_log = float(np.max(logs))
    modal_overflow = np.any(logs > LOG_MAX, axis=1)
    first_overflow = int(np.flatnonzero(modal_overflow)[0]) if modal_overflow.any() else None
    # The inverse orthogonal transform can add modes. This conservative bound
    # prevents overflow even when every individual coefficient is representable.
    safe_field = maximum_log + math.log(max(1, len(k))) < LOG_MAX-2
    prediction = None
    if return_prediction and safe_field:
        with np.errstate(under="ignore"):
            values = signs*np.exp(logs)
        prediction = np.zeros_like(observations)
        prediction[:, 1:-1] = idst(values, type=1, norm="ortho", axis=1)
        if not np.isfinite(prediction).all():
            raise ValueError("inverse transform violated the precomputed safe bound")
    report = {"status": "supported_full_resolved_spectral_forecast", "operator": spec,
        "initialization_scheme": prepared_initial["scheme"], "initialization": prepared_initial["metadata"],
        "prepared_initial_hash": prepared_initial["prepared_coefficients_hash"],
        "raw_observation_hash": array_hash(observations), "spectrum": summary,
        "error_metric": metric, "metric_domain": "all future time rows and all observed spatial nodes; t0 excluded",
        "metric_method": "signed log-domain modal propagation and orthonormal Parseval identity, including boundary observation errors",
        "first_sampled_modal_float64_overflow_time": float(elapsed[first_overflow]) if first_overflow is not None else None,
        "maximum_modal_log_absolute_amplitude": maximum_log if np.isfinite(maximum_log) else None,
        "maximum_modal_amplitude_status": "zero_all_modes" if np.isneginf(maximum_log) else "finite_log_value",
        "prediction_field_safe_to_materialize": bool(safe_field), "prediction_array_returned": prediction is not None,
        "decayed_modal_values_below_float64_subnormal_count": int(np.sum(np.isfinite(logs) & (logs < LOG_MIN_SUBNORMAL))),
        "underflow_policy": "all modal logs retained in score; only optional real-valued materialization may round negligible decayed amplitudes to zero",
        "modes_dropped": 0, "mode_count": len(k), "frequency_cutoff_applied": False,
        "boundary_hypothesis": BOUNDARY, "future_data_used_for_fit": False,
        "whole_library_comparison_policy": "report supported/unsupported coverage separately; do not exclude unsupported candidates to select a preferred model",
        "continuum_claim": "This is the full finite observed-grid spectral realization, not proof that an unstable continuum PDE is well posed."}
    return report, prediction


def main():
    checks, diagnostics = {}, {}
    x, t = np.linspace(0, 1, 65), np.linspace(0, .02, 21)
    u0 = np.sin(np.pi*x)+.15*np.sin(7*np.pi*x)+.01*np.sin(31*np.pi*x)
    fields = (np.exp(-np.pi**2*t[:, None])*np.sin(np.pi*x)[None, :]
              +.15*np.exp(-(7*np.pi)**2*t[:, None])*np.sin(7*np.pi*x)[None, :]
              +.01*np.exp(-(31*np.pi)**2*t[:, None])*np.sin(31*np.pi*x)[None, :])
    initial = prepare_initialization(x, fields[0], scheme=SCHEMES[0], boundary_realization=BOUNDARY)
    result, predicted = evaluate_operator({"u_xx": 1.}, x, t, fields, initial,
                                         boundary_realization=BOUNDARY, return_prediction=True)
    checks["clean_diffusion_exact_all_resolved_modes"] = float(np.max(np.abs(predicted-fields))) < 2e-14
    checks["no_six_mode_truncation"] = result["mode_count"] == 63 and result["modes_dropped"] == 0
    rng = np.random.default_rng(2026110712)
    v = rng.normal(size=63)
    transform = dst(v, type=1, norm="ortho")
    checks["dst_parseval_and_roundtrip"] = (np.allclose(v, idst(transform, type=1, norm="ortho"), rtol=1e-13, atol=1e-13)
                                           and abs(v@v-transform@transform) < 1e-12)
    smooth = prepare_initialization(x, fields[0], scheme=SCHEMES[1], boundary_realization=BOUNDARY)
    checks["frozen_reconstruction_retains_every_mode"] = bool(np.all(smooth["gains"] > 0) and len(smooth["gains"]) == 63)
    unsupported, _ = evaluate_operator({"u_x": 0.}, x, t, fields, initial, boundary_realization=BOUNDARY)
    checks["unsupported_zero_readout_not_silently_removed"] = unsupported["status"] == "unsupported_candidate_structure"
    # Independent dense matrix-exponential reference for a forced/reaction model.
    xx, tt = np.linspace(0, 1, 17), np.linspace(0, .03, 9)
    kk = np.arange(1, 16)*np.pi
    initial_vector = rng.normal(size=15)
    generator = np.zeros((16, 16))
    generator[:15, :15] = np.diag(-.3-.02*kk**2)
    generator[:15, 15] = .4*dst(np.ones(15), type=1, norm="ortho")
    exact_modes = np.array([(expm(generator*time)@np.r_[initial_vector, 1.])[:15] for time in tt])
    exact_field = np.zeros((len(tt), len(xx)))
    exact_field[:, 1:-1] = idst(exact_modes, type=1, norm="ortho", axis=1)
    prepared = prepare_initialization(xx, exact_field[0], scheme=SCHEMES[0], boundary_realization=BOUNDARY)
    forced, forecast = evaluate_operator({"1": .4, "u": -.3, "u_xx": .02}, xx, tt, exact_field, prepared,
                                         boundary_realization=BOUNDARY, return_prediction=True)
    checks["affine_modal_solution_matches_independent_expm"] = np.max(np.abs(forecast-exact_field)) < 5e-13
    noisy = exact_field+.001*rng.standard_normal(exact_field.shape)
    prepared_noisy = prepare_initialization(xx, noisy[0], scheme=SCHEMES[0], boundary_realization=BOUNDARY)
    scored, forecast = evaluate_operator({"1": .4, "u": -.3, "u_xx": .02}, xx, tt, noisy, prepared_noisy,
                                        boundary_realization=BOUNDARY, return_prediction=True)
    direct_error = float(np.linalg.norm(forecast[1:]-noisy[1:])/np.linalg.norm(noisy[1:]))
    checks["log_parseval_error_matches_direct_full_grid_error"] = abs(scored["error_metric"]["relative_error"]-direct_error) < 1e-13
    equilibrium_logs, equilibrium_signs = modal_solution_log(np.array([-1.]), np.array([1.]), np.array([1.]), np.array([0., 1000.]))
    checks["equilibrium_no_catastrophic_large_exp_cancellation"] = np.all(equilibrium_logs == 0) and np.all(equilibrium_signs == -1)
    zero_logs, zero_signs = modal_solution_log(np.array([.2]), np.array([.3]), np.array([0.]), np.array([0., 2.]))
    checks["zero_growth_forcing_limit"] = np.allclose(zero_signs*np.exp(zero_logs), [[.2], [.8]], rtol=1e-14)
    unstable, unsafe_array = evaluate_operator({"u_xx": 1., "u_xxxx": .001}, x, t, fields, initial,
                                               boundary_realization=BOUNDARY, return_prediction=True)
    checks["unstable_growth_uses_finite_log_error_no_overflow"] = (unstable["error_metric"]["status"] == "above_float64_range_use_log"
        and math.isfinite(unstable["error_metric"]["log_relative_error"]) and unsafe_array is None and unstable["modes_dropped"] == 0)
    finite_band, _ = spectrum_summary([0., -1., .02, -.00001], np.arange(1, 64)*np.pi, .02)
    reaction, _ = spectrum_summary([.2], np.arange(1, 64)*np.pi, .02)
    checks["bounded_growth_not_confused_with_high_frequency_illposedness"] = (finite_band["continuum_classification"] == "high_frequency_damped_with_finite_band_growth"
        and reaction["continuum_classification"] == "bounded_zero_order_growth")
    diagnostics.update({"clean_diffusion_max_absolute_field_error": float(np.max(np.abs(predicted-fields))),
        "forced_model_reference_error": forced["error_metric"], "parseval_direct_error": direct_error,
        "unstable_numerical_fixture": unstable, "finite_band_fixture": finite_band,
        "frozen_initial_reconstruction": smooth["metadata"]})
    report = {"stage": "2b_dynamic_validation_preparation", "all_checks_passed": bool(all(checks.values())),
        "scope": "numerical self-checks and frozen evaluation protocol only; no research-candidate comparison executed",
        "checks": {key: bool(value) for key, value in checks.items()}, "diagnostics": diagnostics,
        "python_dependency_versions": {"numpy": np.__version__, "scipy": scipy.__version__},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "supported_candidate_terms": sorted(SUPPORTED), "initialization_schemes": list(SCHEMES),
        "boundary_realization": BOUNDARY, "all_resolved_modes_used": True,
        "new_validation_data_generated": False, "final_test_data_generated_or_used": False,
        "reserved_final_test_ids": list(range(6, 12)), "reserved_final_test_noise_seeds": ">=1000",
        "proposed_development_holdout_ids_pending_root_coordination": list(range(12, 18)),
        "method_comparison_started": False, "gate_policy": "research comparisons require root confirmation that numerical tail gate is resolved and common holdout/protocol are frozen",
        "limits": ["Odd derivatives, variable coefficients, nonlinear terms, and unprovided boundary realizations are unsupported, not failed algorithms.",
            "Reporting only supported candidates is a coverage analysis; it must not silently change selection over the complete library.",
            "High-order sine propagation assumes powers of the Dirichlet Laplacian, not merely u=0 as the only boundary condition.",
            "Frozen IC smoothing attenuates every mode continuously; it does not remove noise bias or make an ill-posed continuum operator well posed.",
            "No model-specific or truth-tuned initial reconstruction is allowed; raw and frozen reconstruction are separate identical-input ablations.",
            "Full observed-grid dynamics still represent a finite spectral discretization; continuum high-frequency behavior is reported separately.",
            "The score compares future noisy observations, excludes initialization rows, and includes boundary measurement discrepancies."],
        "primary_implementation_references": ["https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dst.html",
                                               "https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html"]}
    strict_json_write(ROOT/"stage2b_heat_full_spectrum_result.json", report)
    print({"all_checks_passed": report["all_checks_passed"], "checks": report["checks"],
           "unstable_log_relative_error": unstable["error_metric"]["log_relative_error"],
           "method_comparison_started": False})
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
