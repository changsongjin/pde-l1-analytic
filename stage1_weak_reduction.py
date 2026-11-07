"""Exact integration by parts of D<=6 jet columns before numerical evaluation.

Each original canonical column remains one free coefficient. Reduced density
terms are a constrained linear combination, never additional free columns.
Only observed u and local-polynomial jets through order 3 enter construction.
This is a preprocessing correction/ablation, not a new sparse estimator.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np
import sympy as sp

import stage1_jet_preprocess as V1
from stage0_evaluation import array_hash, strict_json_write

ROOT = Path(__file__).resolve().parent


def _key(a, b, alpha, phi_order=0):
    return (int(a), int(b), *map(int, alpha), int(phi_order))


def _add(out, source, coefficient=Fraction(1)):
    for key, value in source.items():
        out[key] += coefficient*value
        if out[key] == 0:
            del out[key]


def differentiate_density(density):
    """Independent formal product rule on x, spatial u jets, and phi jets."""
    result = defaultdict(Fraction)
    for key, coefficient in density.items():
        a, b, alpha, phi = key[0], key[1], key[2:9], key[-1]
        if a:
            result[_key(a-1, b, alpha, phi)] += coefficient*a
        for order, exponent in enumerate(alpha):
            if not exponent:
                continue
            if order == 6:
                raise ValueError("certificate derivative would exceed order 6")
            powers = list(alpha)
            powers[order] -= 1
            powers[order+1] += 1
            result[_key(a, b, powers, phi)] += coefficient*exponent
        result[_key(a, b, alpha, phi+1)] += coefficient
    return {key: value for key, value in result.items() if value}


@lru_cache(maxsize=None)
def _reduce(key):
    a, b, alpha, phi = key[0], key[1], list(key[2:9]), key[-1]
    order = max((r for r, n in enumerate(alpha) if n), default=0)
    # Pure linear derivative: repeated integration by parts until no u derivative
    # remains. Product-rule expansion of x^a is part of the exact identity.
    if order and sum(alpha) == 1:
        lower = list(alpha)
        lower[order] = 0
        lower[order-1] = 1
        flux = {_key(a, b, lower, phi): Fraction(1)}
        children = {_key(a, b, lower, phi+1): Fraction(-1)}
        if a:
            children[_key(a-1, b, lower, phi)] = Fraction(-a)
    # u^m u_x has an exact primitive even for m>0.
    elif alpha[1] == 1 and not any(alpha[2:]):
        primitive = [alpha[0]+1, 0, 0, 0, 0, 0, 0]
        c = Fraction(1, alpha[0]+1)
        flux = {_key(a, b, primitive, phi): c}
        children = {_key(a, b, primitive, phi+1): -c}
        if a:
            children[_key(a-1, b, primitive, phi)] = -a*c
    elif order > 3:
        if alpha[order] != 1 or sum(r*n for r, n in enumerate(alpha)) > 6:
            raise ValueError("high-order reduction requires total derivative order <=6")
        rest = list(alpha)
        rest[order] -= 1
        lower = list(rest)
        lower[order-1] += 1
        flux = {_key(a, b, lower, phi): Fraction(1)}
        children = defaultdict(Fraction)
        children[_key(a, b, lower, phi+1)] -= 1
        if a:
            children[_key(a-1, b, lower, phi)] -= a
        for r, n in enumerate(rest):
            if n:
                powers = list(lower)
                powers[r] -= 1
                powers[r+1] += 1
                children[_key(a, b, powers, phi)] -= n
    else:
        return ((key, Fraction(1)),), ()
    reduced, total_flux = defaultdict(Fraction), defaultdict(Fraction, flux)
    for child, coefficient in children.items():
        if not coefficient:
            continue
        child_reduced, child_flux = _reduce(child)
        _add(reduced, dict(child_reduced), coefficient)
        _add(total_flux, dict(child_flux), coefficient)
    return tuple(sorted(reduced.items())), tuple(sorted(total_flux.items()))


def reduce_term(term):
    """Return (reduced_density, boundary_flux) with exact Fraction coefficients.

    Key = (x power, t power, alpha0,...,alpha6, phi derivative order).
    original*phi0 = reduced_density + D_x(boundary_flux) exactly.
    """
    alpha = tuple(term["alpha"])
    if len(alpha) != 7 or any(int(n) != n or n < 0 for n in [term["a"], term["b"], *alpha]):
        raise ValueError("need nonnegative integer powers and seven spatial jet exponents")
    if sum(r*n for r, n in enumerate(alpha)) > 6:
        raise ValueError("this reduction is declared only for total derivative order <=6")
    reduced, flux = _reduce(_key(term["a"], term["b"], alpha))
    return dict(reduced), dict(flux)


def weak_reduced_matrix(x, t, u, terms, target_order=1, half=(12, 10),
                        window=V1.DEFAULT_WINDOW, polyorder=V1.DEFAULT_POLYORDER, stride=None):
    x, t, u, dx, dt = V1._grid(x, t, u, half, window)
    if target_order not in (1, 2) or not terms:
        raise ValueError("target_order must be 1/2 and terms nonempty")
    stride = tuple(stride or half)
    identities = [reduce_term(term)[0] for term in terms]
    max_order = max(max((r for r,n in enumerate(key[2:9]) if n), default=0)
                    for reduced in identities for key in reduced)
    if max_order > 3:
        raise AssertionError("high spatial derivative survived exact reduction")
    max_phi = max(key[-1] for reduced in identities for key in reduced)
    jets, reconstruction = V1.spatial_jets(u, dx, window, polyorder, max_order)
    sg = window//2
    raw, xx, tt = u[:, sg:-sg], x[None, sg:-sg], t[:, None]
    px, pt = V1._bump(half[0], dx, max_phi), V1._bump(half[1], dt, target_order)
    target = (-1)**target_order * V1._integrate(raw, px[0], pt[target_order], dx, dt, stride)

    @lru_cache(maxsize=128)
    def integral(key):
        a, b, alpha, phi = key[0], key[1], key[2:9], key[-1]
        field = np.ones_like(raw)
        if a:
            field *= xx**a
        if b:
            field *= tt**b
        # No reconstructed derivative is necessary for a pure power of u.
        # Using raw u here is the ordinary weak nonlinear transform, whose
        # measurement-noise bias is still reported rather than assumed absent.
        pure_power = not any(alpha[1:])
        for r, exponent in enumerate(alpha):
            if exponent:
                field *= (raw if pure_power else jets[r])**exponent
        return V1._integrate(field, px[phi], pt[0], dx, dt, stride)

    matrix = np.empty((len(target), len(terms)))
    for j, reduced in enumerate(identities):
        matrix[:, j] = sum(float(coefficient)*integral(key) for key, coefficient in reduced.items())
    if not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise ValueError("non-finite reduced jet library; no columns silently removed")
    hx, ht = half
    sx, st = stride
    cache = integral.cache_info()
    return matrix, target, {"input_observation_hash": array_hash(u), "raw_shape": list(u.shape),
        "rows": len(target), "columns": len(terms), "target_order": target_order,
        "target_source": "original_observations_weak_time_derivative",
        "half": list(half), "stride": list(stride), "test_power": V1.TEST_POWER,
        "reconstruction": reconstruction, "raw_footprint_half_x": sg+hx, "raw_footprint_half_t": ht,
        "center_x_indices": list(range(sg+hx, len(x)-sg-hx, sx)),
        "center_t_indices": list(range(ht, len(t)-ht, st)),
        "boundary_policy": "discard reconstruction plus weak halo; bump derivatives vanish at endpoints",
        "preprocessing": "exact_IBP_to_spatial_order_at_most_3_plus_raw_pure_weak_powers",
        "max_reconstructed_order": max_order, "max_test_function_derivative": max_phi,
        "largest_bound_combination": max(map(len, identities)),
        "free_columns_unchanged": True, "evaluated_integral_cache_hits": cache.hits,
        "evaluated_integral_cache_misses": cache.misses, "integral_cache_max_vectors": 128,
        "matrix_hash": array_hash(matrix), "target_hash": array_hash(target)}


def _sympy_check(term):
    """Second symbolic implementation: SymPy chain rule, not sparse derivative."""
    x, t = sp.symbols("x t")
    u = sp.symbols("u0:7")
    phi = sp.symbols("p0:7")
    def expression(density):
        return sum(sp.Rational(c.numerator, c.denominator)*x**k[0]*t**k[1]
                   *sp.prod(v**n for v,n in zip(u, k[2:9]))*phi[k[-1]] for k,c in density.items())
    original = {_key(term["a"], term["b"], term["alpha"]): Fraction(1)}
    reduced, flux = reduce_term(term)
    f = expression(flux)
    derivative = sp.diff(f, x) + sum(sp.diff(f,u[r])*u[r+1] for r in range(6)) + sum(sp.diff(f,phi[r])*phi[r+1] for r in range(6))
    return sp.expand(expression(original)-expression(reduced)-derivative) == 0


def _column_perturbations(clean, noisy):
    norms = np.linalg.norm(clean, axis=0)
    good = norms > max(float(np.max(norms))*1e-12, 1e-14)
    ratios = np.linalg.norm(noisy-clean, axis=0)[good]/norms[good]
    return {"median": float(np.median(ratios)), "q90": float(np.quantile(ratios,.9)),
            "maximum": float(np.max(ratios)), "columns_with_nonvanishing_clean_reference": int(np.sum(good)),
            "near_zero_clean_columns": int(np.sum(~good)),
            "near_zero_reference_policy": "excluded only from relative diagnostic, retained in matrices",
            "frobenius_relative": float(np.linalg.norm(noisy-clean)/np.linalg.norm(clean))}


def main():
    from stage1_grammar import enumerate_jet
    terms = enumerate_jet(6,8,6,2)
    checks = {}
    max_order, max_phi, max_combinations = 0, 0, 0
    for term in terms:
        reduced, flux = reduce_term(term)
        reconstruction = defaultdict(Fraction, reduced)
        _add(reconstruction, differentiate_density(flux))
        original = {_key(term["a"],term["b"],term["alpha"]): Fraction(1)}
        if dict(reconstruction) != original:
            raise AssertionError("formal IBP identity failed: " + term["name"])
        max_order = max(max_order, max(max((r for r,n in enumerate(k[2:9]) if n), default=0) for k in reduced))
        max_phi = max(max_phi, max(k[-1] for k in reduced))
        max_combinations = max(max_combinations, len(reduced))
    checks["all_1158_exact_formal_identities"] = len(terms) == 1158 and max_order <= 3 and max_phi <= 6
    probes = [t for t in terms if (t["a"],t["b"],tuple(t["alpha"])) in {
        (2,0,(0,0,0,0,0,0,1)), (0,0,(1,0,0,0,0,0,1)), (1,1,(2,1,0,0,0,1,0)),
        (0,0,(4,0,1,0,1,0,0)), (0,0,(0,0,0,2,0,0,0)), (2,0,(5,1,0,0,0,0,0))}]
    checks["independent_sympy_chain_rule"] = len(probes) == 6 and all(_sympy_check(t) for t in probes)
    # Endpoint values and first five derivatives vanish exactly for the bump.
    s = sp.symbols("s")
    checks["boundary_flux_zero"] = all(sp.diff((1-s*s)**8,s,r).subs(s,end) == 0 for r in range(6) for end in (-1,1))
    clean_rows = []
    for nx in (129,257,513):
        x, t = np.linspace(-np.pi,np.pi,nx), np.linspace(0,1,81)
        xx,tt = np.meshgrid(x,t)
        field = np.exp(-tt)*(np.sin(xx)+.15*np.cos(2*xx)+.03*np.sin(3*xx))
        # Hold a common physical weak width while refining the x grid.
        half = (12*(nx-1)//128,10)
        a1,b1,_ = V1.weak_jet_matrix(x,t,field,probes,half=half)
        a2,b2,_ = weak_reduced_matrix(x,t,field,probes,half=half)
        differences = np.linalg.norm(a2-a1,axis=0)/np.maximum(np.linalg.norm(a1,axis=0),1e-30)
        clean_rows.append({"nx":nx,"half":list(half),"relative_column_differences":differences.tolist(),
                           "same_target_exact":bool(np.array_equal(b1,b2))})
    checks["smooth_integral_refinement"] = all(f < c for f,c in zip(clean_rows[-1]["relative_column_differences"],clean_rows[0]["relative_column_differences"]) if c > 1e-10)
    checks["same_weak_target"] = all(row["same_target_exact"] for row in clean_rows)
    x,t = np.linspace(-np.pi,np.pi,201),np.linspace(0,1,101)
    xx,tt = np.meshgrid(x,t)
    field = np.exp(-tt)*(np.sin(xx)+.15*np.cos(2*xx)+.03*np.sin(3*xx))
    rng = np.random.default_rng(6031)
    observed = field+.01*np.std(field)*rng.standard_normal(field.shape)
    start = time.perf_counter()
    a1,b1,m1 = V1.weak_jet_matrix(x,t,field,terms)
    n1,c1,_ = V1.weak_jet_matrix(x,t,observed,terms)
    time_v1 = time.perf_counter()-start
    start = time.perf_counter()
    a2,b2,m2 = weak_reduced_matrix(x,t,field,terms)
    n2,c2,_ = weak_reduced_matrix(x,t,observed,terms)
    time_v2 = time.perf_counter()-start
    checks["all_columns_preserved"] = a1.shape == a2.shape == n1.shape == n2.shape and a1.shape[1] == 1158
    checks["paired_noise_same_input_target"] = m1["input_observation_hash"] == m2["input_observation_hash"] and np.array_equal(b1,b2) and np.array_equal(c1,c2)
    pure_index = next(i for i,t in enumerate(terms) if tuple(t["alpha"]) == (0,0,0,0,0,0,1) and t["a"]==t["b"]==0)
    direct = V1.pure_weak_column(x,t,field,1,6)
    checks["pure_sixth_derivative_matches_weak"] = np.allclose(a2[:,pure_index],direct,rtol=1e-12,atol=1e-12)
    report = {"stage":1,"all_checks_passed":all(checks.values()),"checks":checks,
        "identity_count":len(terms),"max_reconstructed_order":max_order,"max_test_function_derivative":max_phi,
        "largest_bound_combination":max_combinations,"sympy_probe_names":[t["name"] for t in probes],
        "smooth_integral_refinement":clean_rows,
        "paired_noise_fixture":{"sigma":.01,"seed":6031,"input_hash":array_hash(observed),
            "full_jet_v1":_column_perturbations(a1,n1),"reduced_jet_v2":_column_perturbations(a2,n2),
            "v1_clean_and_noisy_seconds":time_v1,"v2_clean_and_noisy_seconds":time_v2,
            "v2_metadata":m2},
        "method_scope":"exact weak integration by parts preprocessing; same canonical columns and L1 threshold",
        "boundary_assumptions":"compact polynomial bump power8, derivatives through order5 vanish at x endpoints",
        "termination":"D<=6 and highest r>3 implies unique factor; remaining derivative budget<=2, so other factors cannot exceed3; main order lowers each step",
        "limitations":["Remaining products of reconstructed order<=3 jets still amplify noise and may be biased.",
            "Pure raw u powers also have nonlinear observation-noise bias.",
            "One smooth trajectory/noise seed is a preprocessing diagnostic, not algorithm ranking or final robustness evidence.",
            "No analytical derivatives are used in either matrix-construction function.",
            "Relative per-column perturbations exclude near-zero clean reference norms only from that diagnostic, never from fitting."]}
    strict_json_write(ROOT/"stage1_weak_reduction_result.json",report)
    print({"all_checks_passed":report["all_checks_passed"],"checks":checks,
           "noise_v1":report["paired_noise_fixture"]["full_jet_v1"],"noise_v2":report["paired_noise_fixture"]["reduced_jet_v2"]})
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
