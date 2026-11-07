"""Post-selection heat operator stability and conditional six-mode prediction.

Reads frozen selected coefficients only. No alpha, support, coefficient, or
weight is retuned. Sine-mode predictions assume a polynomial of the Dirichlet
Laplacian; they are not a general high-order PDE rollout with unspecified extra
boundary conditions. Future validation observations are used only for scoring.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from numpy.polynomial import polynomial as Poly
from scipy.fft import dst

from stage0_data_bundle import load_bundle
from stage0_evaluation import array_hash, canonical_term, strict_json_write

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "stage2a_heat_noise_pilot_result.json"
ORDERS = {"u_xx": 2, "u_xxxx": 4, "u_xxxxxx": 6}
N_MODES = 6


def _strict_load(path):
    def reject(token):
        raise ValueError("non-finite JSON token " + token)
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def _positive_real_roots(coefficients):
    coefficients = np.trim_zeros(np.asarray(coefficients, dtype=float), "b")
    if len(coefficients) < 2:
        return []
    roots = Poly.polyroots(coefficients)
    return sorted(float(z.real) for z in roots if z.real > 0 and abs(z.imag) <= 1e-10*max(1, abs(z.real)))


def operator_spectrum(coefficient_map, x, horizon):
    """s(k) = sum c_(2r)*(-1)^r*k^(2r), with z=k^2 >=0."""
    x = np.asarray(x)
    length, dx = float(x[-1]-x[0]), float(x[1]-x[0])
    if length <= 0 or not np.allclose(np.diff(x), dx):
        raise ValueError("uniform increasing observation grid required")
    growth = np.zeros(4)
    for name, value in coefficient_map.items():
        if name not in ORDERS:
            raise ValueError("only linear constant even-spatial-derivative operators are supported")
        r = ORDERS[name]//2
        growth[r] += (-1)**r*float(value)
    growth = np.trim_zeros(growth, "b")
    if not len(growth):
        growth = np.zeros(1)
    leading = float(growth[-1])
    roots = _positive_real_roots(growth[1:])
    bounds = [0., *roots, math.inf]
    positive_bands = []
    for lower, upper in zip(bounds[:-1], bounds[1:]):
        midpoint = max(2*lower,lower+1) if math.isinf(upper) else (lower+upper)/2
        if Poly.polyval(midpoint, growth) > 0:
            positive_bands.append({"k_lower_exclusive": math.sqrt(lower),
                                   "k_upper_exclusive": None if math.isinf(upper) else math.sqrt(upper),
                                   "upper_status": "unbounded" if math.isinf(upper) else "finite"})
    modes = np.arange(1, len(x)-1)
    k = modes*np.pi/length
    rates = Poly.polyval(k*k,growth)
    positive = modes[rates > 0]
    maximum_index = int(np.argmax(rates))
    # For a negative leading coefficient, extrema over all k>=0 occur at zero
    # or nonnegative roots of ds/dz. Positive leading coefficient is unbounded.
    if leading > 0:
        continuum_max, continuum_status = None, "unbounded_positive_high_frequency_growth"
    elif leading < 0:
        extrema = [0., *_positive_real_roots(Poly.polyder(growth))]
        continuum_max = float(max(Poly.polyval(z,growth) for z in extrema))
        continuum_status = "high_frequency_damped_with_finite_unstable_band" if continuum_max > 0 else "all_nonzero_frequencies_non_growing"
    else:
        continuum_max, continuum_status = 0., "zero_operator_neutral"
    max_log = float(np.max(rates)*horizon)
    safe_exponent = max_log <= math.log(np.finfo(float).max)
    return {"growth_polynomial_in_k_squared": growth.tolist(),
        "formula": "s(k)=-c2*k^2+c4*k^4-c6*k^6; physical fitted coefficients",
        "continuum": {"leading_power_of_k": 2*(len(growth)-1), "leading_coefficient": leading,
            "classification": continuum_status, "positive_growth_bands": positive_bands,
            "max_growth_rate": continuum_max,
            "max_growth_status": "unbounded" if continuum_max is None else "finite"},
        "observed_grid_sine_spectrum": {"dx": dx, "domain_length": length,
            "sampling_nyquist_wavenumber": float(np.pi/dx),
            "nonzero_resolvable_sine_modes": [1, int(modes[-1])],
            "largest_nonzero_resolvable_sine_wavenumber": float(k[-1]),
            "nyquist_note": "mode nx-1 vanishes at every grid point for endpoint-including Dirichlet sine samples; excluded",
            "positive_growth_mode_indices": positive.tolist(), "positive_growth_mode_count": len(positive),
            "mode_of_maximum_growth": int(modes[maximum_index]),
            "maximum_growth_rate": float(rates[maximum_index]),
            "minimum_growth_rate": float(np.min(rates)), "horizon": horizon,
            "largest_modal_log_amplification": max_log,
            "largest_modal_amplification": float(np.exp(max_log)) if safe_exponent else None,
            "amplification_status": "finite_float64" if safe_exponent else "exceeds_float64_range_use_log",
            "interpretation": "finite grid has finitely many bounded rates even when the continuous operator has unbounded high-frequency growth"}}


def six_mode_prediction(coefficient_map, x, t, fields, indices):
    length = x[-1]-x[0]
    modes = np.arange(1,N_MODES+1)
    k = modes*np.pi/length
    basis = np.sin(np.outer(x-x[0],k))
    rates = sum(float(c)*(-1)**(ORDERS[name]//2)*k**ORDERS[name]
                for name,c in coefficient_map.items())
    elapsed = np.asarray(t)-t[0]
    exponents = elapsed[:,None]*rates[None,:]
    if np.max(exponents) > math.log(np.finfo(float).max)-10:
        return {"status":"prediction_overflow_under_conditional_spectrum","growth_rates":rates.tolist()}
    per_trajectory=[]
    total_error2=total_observed2=0.
    for index in indices:
        observed=fields[index]
        initial=np.linalg.lstsq(basis,observed[0],rcond=None)[0]
        prediction=(np.exp(exponents)*initial[None,:])@basis.T
        # Exclude the t0 rows used to construct the initial condition.
        residual=prediction[1:]-observed[1:]
        observed_norm=np.linalg.norm(observed[1:])
        error=np.linalg.norm(residual)
        projected=(np.linalg.lstsq(basis,observed[1:].T,rcond=None)[0].T)@basis.T
        per_trajectory.append({"trajectory":index,"initial_six_mode_coefficients":initial.tolist(),
            "initial_observation_hash":array_hash(observed[0]),
            "prediction_hash":array_hash(prediction),
            "relative_error_to_full_future_observations":float(error/observed_norm),
            "relative_error_to_projected_future_observations":float(np.linalg.norm(prediction[1:]-projected)/np.linalg.norm(projected)),
            "future_observation_energy_outside_six_modes_fraction":float(np.linalg.norm(observed[1:]-projected)**2/observed_norm**2)})
        total_error2+=error**2
        total_observed2+=observed_norm**2
    return {"status":"computed_under_explicit_spectral_assumptions","modes":modes.tolist(),
        "wavenumbers":k.tolist(),"growth_rates":rates.tolist(),
        "future_time_rows_scored":len(t)-1,"initial_time_excluded_from_score":True,
        "relative_error_to_full_future_observations_pooled":float(math.sqrt(total_error2/total_observed2)),
        "per_validation_trajectory":per_trajectory,
        "initialization":"least-squares projection of validation t0 observations onto fixed six sine modes",
        "coefficient_source":"frozen training-only LS readout on the selected L1 support",
        "boundary_realization":"polynomial of the Dirichlet Laplacian; higher-order powers implicitly impose corresponding even-derivative endpoint conditions",
        "not_general_PDE_rollout":True,"future_observations_used_for_parameter_fit":False}


def resolved_initial_spectrum_risk(coefficient_map,x,t,fields,indices):
    """Log-domain growth of ALL resolved sine coefficients in observed t0.

    A modal overflow time is a numerical risk diagnostic, not an asserted exact
    time when the sum of modes overflows. No unstable exponential is evaluated.
    """
    modes=np.arange(1,len(x)-1)
    k=modes*np.pi/(x[-1]-x[0])
    rates=sum(float(c)*(-1)**(ORDERS[name]//2)*k**ORDERS[name]
              for name,c in coefficient_map.items())
    horizon=float(t[-1]-t[0])
    log_max=math.log(np.finfo(float).max)
    rows=[]
    for index in indices:
        # DST-I on interior nodes has coefficient normalization 1/(nx-1).
        initial=dst(fields[index][0,1:-1],type=1)/(len(x)-1)
        valid=np.abs(initial)>0
        positive=valid & (rates>0)
        log_amplitudes=np.log(np.abs(initial[valid]))+rates[valid]*horizon
        if np.any(positive):
            overflow_times=(log_max-np.log(np.abs(initial[positive])))/rates[positive]
            first=int(np.argmin(overflow_times))
            earliest=float(overflow_times[first])
            mode=int(modes[positive][first])
            observed_times=np.asarray(t)-t[0]
            after=observed_times[observed_times>=earliest]
            first_grid_time=float(after[0]) if len(after) else None
        else:
            earliest=mode=first_grid_time=None
        rows.append({"trajectory":index,
            "initial_observed_energy_in_modes_above_six_fraction":float(np.sum(initial[6:]**2)/np.sum(initial**2)),
            "maximum_modal_log_absolute_amplitude_at_horizon":float(np.max(log_amplitudes)),
            "any_modal_amplitude_exceeds_float64_at_horizon":bool(np.max(log_amplitudes)>log_max),
            "first_modal_float64_overflow_time_from_t0":earliest,
            "mode_of_first_modal_overflow":mode,
            "first_observed_time_at_or_after_modal_overflow":first_grid_time})
    return {"modes":[1,int(modes[-1])],"computed_in_log_domain_only":True,
        "positive_rate_interpretation":"positive growth at the stated wavenumbers; growth alone is not a universal physical-invalidity test",
        "per_validation_trajectory":rows,
        "caveat":"Six-mode projection filters initial high-frequency observation noise; these unresolved-in-the-six-mode-test disturbances can grow in the full resolved spectral realization."}


def main():
    report=_strict_load(SOURCE)
    source_hash=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if not report["execution_complete"] or report["test_data_used"]:
        raise ValueError("need completed development report with no test data used")
    rows=[]
    for dataset in report["rows"]:
        if dataset["case"]!="heat" or dataset["sigma"] not in (.01,.05):
            continue
        tag=f"heat_sigma_{repr(float(dataset['sigma'])).replace('.','p')}_seed_{dataset['seed']}"
        x,t,fields,manifest=load_bundle(ROOT/"stage2_data"/(tag+".npz"))
        if manifest["npz_sha256"]!=dataset["raw_bundle_sha256"]:
            raise ValueError("observation bundle changed since selection")
        for layer,layer_result in dataset["layers"].items():
            for method,result in layer_result["methods"].items():
                evaluation=result["evaluation"]
                selected=evaluation["primary_selected_candidate_id"]
                row={"sigma":dataset["sigma"],"seed":dataset["seed"],"layer":layer,"method":method,
                     "path_status":result["path"]["status"],"primary_selected_candidate_id":selected,
                     "raw_bundle_sha256":manifest["npz_sha256"]}
                if selected is None:
                    row["status"]="ineligible_incomplete_path_no_primary_model"
                    rows.append(row)
                    continue
                if not result["path"]["all_solutions_certified"] or not evaluation["selection_eligible"]:
                    raise ValueError("uncertified path supplied a primary model")
                candidate=next(c for c in evaluation["candidates"] if c["candidate_id"]==selected)
                coefficients={canonical_term(n):float(c) for n,c in zip(candidate["support"],candidate["coefficients"])}
                row.update({"selected_support":candidate["support"],"frozen_physical_coefficients":coefficients,
                            "weak_validation_error":candidate["validation_error"],"complexity":candidate["complexity"],
                            "selection_rule":evaluation["selection_rule"],
                            "existing_post_selection_truth_diagnostic":candidate["truth_evaluation_after_selection"]})
                if not coefficients or not set(coefficients)<=set(ORDERS):
                    row["status"]="outside_even_spatial_derivative_only_diagnostic_scope"
                    rows.append(row)
                    continue
                row.update({"status":"diagnosed_without_refitting_or_reselection",
                    "spectrum":operator_spectrum(coefficients,x,float(t[-1]-t[0])),
                    "conditional_six_mode_prediction":six_mode_prediction(coefficients,x,t,fields,[4,5]),
                    "all_resolved_initial_spectrum_risk":resolved_initial_spectrum_risk(coefficients,x,t,fields,[4,5])})
                rows.append(row)
    grid=np.linspace(0,1,257)
    diffusion=operator_spectrum({"u_xx":1},grid,.08)
    antidiffusion=operator_spectrum({"u_xx":1,"u_xxxx":1e-4},grid,.08)
    bounded=operator_spectrum({"u_xx":1,"u_xxxx":3,"u_xxxxxx":1},grid,.08)
    checks={"all_eight_method_layer_noise_records_present":len(rows)==8,
        "diffusion_modes_non_growing":diffusion["observed_grid_sine_spectrum"]["positive_growth_mode_count"]==0,
        "positive_fourth_order_unbounded":antidiffusion["continuum"]["classification"]=="unbounded_positive_high_frequency_growth",
        "sixth_order_can_have_bounded_unstable_band":bounded["continuum"]["classification"]=="high_frequency_damped_with_finite_unstable_band",
        "incomplete_model_not_substituted":sum(r["status"]=="ineligible_incomplete_path_no_primary_model" for r in rows)==1,
        "source_report_unchanged":hashlib.sha256(SOURCE.read_bytes()).hexdigest()==source_hash}
    result={"stage":"2a_post_selection_conditional_diagnostic","all_checks_passed":all(checks.values()),
        "checks":checks,"source_report":SOURCE.name,"source_sha256":source_hash,
        "alpha_unchanged":report["alpha_frozen"],"no_model_or_parameter_reselection":True,
        "test_data_used":False,"rows":rows,
        "diagnostic_only":"A lower weak validation residual does not establish better trajectory prediction or a well-posed continuum operator.",
        "assumptions_and_limits":["Only certified selected models composed of u_xx/u_xxxx/u_xxxxxx are analyzed.",
            "Fourier growth is exact for the stated constant-coefficient spectral realization; no claim is made for unsupported nonlinear structures.",
            "A fourth/sixth order PDE generally needs additional boundary conditions: u=0 alone does not define the complete problem.",
            "Sine predictions assume the operator is a polynomial of the Dirichlet Laplacian, imposing the corresponding spectral boundary realization.",
            "Restricting to six modes discards possible unstable unresolved modes; a good conditional score does not make the full continuous PDE stable.",
            "Six-mode initialization filters most high-frequency observation noise, whereas the full resolved initial-spectrum risk retains it and is evaluated only in the log domain.",
            "The initial condition comes from noisy validation t0 observations; future validation values are used only for evaluation.",
            "This is a post-selection development diagnostic, not independent final-test evidence or a general PDE rollout.",
            "The validation observations are noisy, so their discrepancy also includes measurement noise."]}
    strict_json_write(ROOT/"stage2a_heat_structure_diagnostic_result.json",result)
    for row in rows:
        if "spectrum" in row:
            print(row["sigma"],row["layer"],row["method"],row["spectrum"]["continuum"]["classification"],
                  "six_mode_error",row["conditional_six_mode_prediction"].get("relative_error_to_full_future_observations_pooled"))
    print("all_checks_passed",result["all_checks_passed"])
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__=="__main__":
    main()
