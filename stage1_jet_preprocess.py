"""Frozen local-polynomial jets plus weak integration of jet monomials.

No analytical derivatives enter matrix construction. Raw fields use (nt, nx)
layout. Space reconstruction uses a 31-point degree-8 fit (including the zeroth
jet); no temporal reconstruction is used. Every weak window is contained inside
the reconstruction-valid domain. Synthetic derivative truth below is used ONLY
for preprocessing verification, never for fitting the research benchmarks.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from numpy.polynomial import polynomial as Poly
from scipy.ndimage import correlate1d
from scipy.signal import correlate

from stage0_evaluation import array_hash, strict_json_write

ROOT = Path(__file__).resolve().parent
DEFAULT_WINDOW = 31
DEFAULT_POLYORDER = 8
TEST_POWER = 8


def polynomial_kernels(window=DEFAULT_WINDOW, polyorder=DEFAULT_POLYORDER, max_order=6):
    if window % 2 != 1 or window <= polyorder or not 0 <= max_order <= polyorder:
        raise ValueError("need odd window > polyorder >= max_order >= 0")
    z = np.linspace(-1.0, 1.0, window)
    design = np.vander(z, polyorder+1, increasing=True)
    # SVD least squares on dimensionless coordinates; avoids dx**r in the fit.
    inverse = np.linalg.lstsq(design, np.eye(window), rcond=None)[0]
    kernels = np.array([math.factorial(r) * inverse[r] for r in range(max_order+1)])
    target = np.zeros((max_order+1, polyorder+1))
    for r in range(max_order+1):
        target[r, r] = math.factorial(r)
    residual = kernels @ design - target
    moment_relative = np.max(np.abs(residual) / np.maximum(1.0, np.abs(target)), axis=1)
    if np.max(moment_relative) > 1e-8:
        raise ValueError("local polynomial moment accuracy inadequate")
    return kernels, {"moment_max_absolute_error": float(np.max(np.abs(residual))),
                     "moment_max_scaled_error_by_order": moment_relative.tolist(),
                     "dimensionless_design_condition": float(np.linalg.cond(design))}


def spatial_jets(u, dx, window=DEFAULT_WINDOW, polyorder=DEFAULT_POLYORDER, max_order=6):
    u = np.asarray(u, dtype=float)
    if u.ndim != 2 or u.shape[1] < window or not np.isfinite(u).all() or dx <= 0:
        raise ValueError("invalid observed field/grid for jet reconstruction")
    kernels, audit = polynomial_kernels(window, polyorder, max_order)
    h = window // 2
    radius = h * float(dx)
    jets = np.stack([correlate1d(u, kernel / radius**r, axis=1)[:, h:-h]
                     for r, kernel in enumerate(kernels)])
    return jets, {"window": window, "polyorder": polyorder, "max_order": max_order,
                  "spatial_reconstruction_halo": h, "dx": float(dx),
                  "temporal_reconstruction": "none", "zeroth_jet": "local_polynomial_smoothing",
                  "input_shape": list(u.shape), "jet_shape": list(jets.shape), **audit}


def _grid(x, t, u, half, window):
    x, t, u = np.asarray(x), np.asarray(t), np.asarray(u)
    if x.ndim != 1 or t.ndim != 1 or u.shape != (len(t), len(x)) or len(x) < 2 or len(t) < 2:
        raise ValueError("expected x,t vectors and u with shape (nt,nx)")
    dx, dt = float(x[1]-x[0]), float(t[1]-t[0])
    if dx <= 0 or dt <= 0 or not np.allclose(np.diff(x), dx, rtol=1e-10, atol=1e-13) or not np.allclose(np.diff(t), dt, rtol=1e-10, atol=1e-13):
        raise ValueError("strictly increasing uniform x,t grids required")
    hx, ht = half
    if min(hx, ht) < 1 or len(x) < window + 2*hx or len(t) < 2*ht+1:
        raise ValueError("grid too small for reconstruction and weak window halos")
    return x, t, u, dx, dt


def _bump(half, spacing, max_order):
    z = np.linspace(-1.0, 1.0, 2*half+1)
    poly = Poly.polypow([1.0, 0.0, -1.0], TEST_POWER)
    derivatives = []
    for r in range(max_order+1):
        derivatives.append(Poly.polyval(z, poly) / (half*spacing)**r)
        poly = Poly.polyder(poly)
    return derivatives


def _integrate(field, px, pt, dx, dt, stride):
    sx, st = stride
    if min(sx, st) < 1:
        raise ValueError("weak stride must be positive")
    return (correlate(field, np.outer(pt, px), mode="valid", method="fft")
            [::st, ::sx].ravel() * dx * dt)


def weak_jet_matrix(x, t, u, terms, target_order=1, half=(12, 10),
                    window=DEFAULT_WINDOW, polyorder=DEFAULT_POLYORDER, stride=None):
    """Return physical A,b,metadata; no column/target normalization is fitted here.

    Terms follow stage1_grammar: integer a,b, and alpha=(a0,...,a6).
    Uses all declared columns, including exact zero or nearly dependent columns.
    """
    x, t, u, dx, dt = _grid(x, t, u, half, window)
    if target_order not in (1, 2) or not terms:
        raise ValueError("target_order must be 1/2 and terms nonempty")
    stride = tuple(stride or half)
    max_order = max(max((r for r, n in enumerate(term["alpha"]) if n), default=0) for term in terms)
    jets, reconstruction = spatial_jets(u, dx, window, polyorder, max_order)
    sg = window // 2
    raw = u[:, sg:-sg]
    xx, tt = x[None, sg:-sg], t[:, None]
    px, pt = _bump(half[0], dx, 0), _bump(half[1], dt, target_order)
    target = (-1)**target_order * _integrate(raw, px[0], pt[target_order], dx, dt, stride)
    matrix = np.empty((len(target), len(terms)), dtype=float)
    for j, term in enumerate(terms):
        if any(int(n) != n or n < 0 for n in [term["a"], term["b"], *term["alpha"]]):
            raise ValueError("nonnegative integer exponents required")
        field = np.ones_like(raw)
        if term["a"]:
            field *= xx ** term["a"]
        if term["b"]:
            field *= tt ** term["b"]
        for r, exponent in enumerate(term["alpha"]):
            if exponent:
                field *= jets[r] ** exponent
        matrix[:, j] = _integrate(field, px[0], pt[0], dx, dt, stride)
    if not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise ValueError("non-finite jet library; retain explicit failure rather than dropping columns")
    hx, ht = half
    sx, st = stride
    return matrix, target, {"input_observation_hash": array_hash(u), "raw_shape": list(u.shape),
                           "rows": len(target), "columns": len(terms), "target_order": target_order,
                           "target_source": "original_observations_weak_time_derivative",
                           "half": list(half), "stride": list(stride), "test_power": TEST_POWER,
                           "reconstruction": reconstruction,
                           "raw_footprint_half_x": sg+hx, "raw_footprint_half_t": ht,
                           "center_x_indices": list(range(sg+hx, len(x)-sg-hx, sx)),
                           "center_t_indices": list(range(ht, len(t)-ht, st)),
                           "boundary_policy": "discard reconstruction halo plus weak halo",
                           "matrix_hash": array_hash(matrix), "target_hash": array_hash(target)}


def pure_weak_column(x, t, u, m, k, a=0, b=0, half=(12, 10),
                     window=DEFAULT_WINDOW, stride=None):
    """D_x^k(x^a t^b u^m) on exactly the jet pipeline's weak centers.

    This is a comparator column, not appended freely beside the canonical jet
    columns. Combine jet columns using exact grammar identities for comparison.
    """
    x, t, u, dx, dt = _grid(x, t, u, half, window)
    if not 0 <= k < TEST_POWER or min(m, a, b) < 0:
        raise ValueError("unsupported weak derivative or exponent")
    sg = window//2
    field = x[None, sg:-sg]**a * t[:, None]**b * u[:, sg:-sg]**m
    px, pt = _bump(half[0], dx, k), _bump(half[1], dt, 0)
    return (-1)**k * _integrate(field, px[k], pt[0], dx, dt, tuple(stride or half))


def main():
    from stage1_grammar import enumerate_jet
    checks, rows = {}, []
    kernels, moments = polynomial_kernels()
    checks["all_jet_moments"] = max(moments["moment_max_scaled_error_by_order"]) < 1e-8
    x = np.linspace(-1, 1, 129)
    poly = x**8 + x**3 + x
    jets, _ = spatial_jets(poly[None, :], x[1]-x[0])
    coeff = [0, 1, 0, 1, 0, 0, 0, 0, 1]
    halo = DEFAULT_WINDOW//2
    xc = x[halo:-halo]
    polynomial_errors = []
    for r in range(7):
        exact = Poly.polyval(xc, Poly.polyder(coeff, r))
        polynomial_errors.append(float(np.linalg.norm(jets[r, 0]-exact)/np.linalg.norm(exact)))
    checks["degree8_polynomial_all_orders"] = max(polynomial_errors) < 5e-6
    for n in (129, 257, 513):
        grid = np.linspace(0, 2*np.pi, n)
        h = grid[1]-grid[0]
        field = np.sin(grid) + .1*np.cos(2*grid)
        jets, _ = spatial_jets(field[None, :], h)
        interior = grid[15:-15]
        errors = []
        for r in range(7):
            exact = np.sin(interior+r*np.pi/2) + .1*2**r*np.cos(2*interior+r*np.pi/2)
            errors.append(float(np.linalg.norm(jets[r,0]-exact)/np.linalg.norm(exact)))
        rows.append({"n_x": n, "relative_errors_by_order": errors})
    # Fixed point windows converge in physical width; high-order roundoff is
    # separately visible, not hidden by claiming every refinement must improve.
    checks["fourier_refinement_129_to257"] = all(
        fine < coarse for fine, coarse in zip(rows[1]["relative_errors_by_order"], rows[0]["relative_errors_by_order"]))
    checks["sixth_derivative_clean_accuracy"] = rows[1]["relative_errors_by_order"][6] < .01
    x, t = np.linspace(-np.pi, np.pi, 201), np.linspace(0, 1, 101)
    xx, tt = np.meshgrid(x, t)
    clean = np.exp(-tt)*np.sin(xx) + .25*np.exp(-4*tt)*np.cos(2*xx)
    terms = enumerate_jet(2, 2, 2, 0)
    a, target, meta = weak_jet_matrix(x, t, clean, terms)
    all_terms = enumerate_jet(3, 3, 3, 0)
    bigger, target_bigger, _ = weak_jet_matrix(x, t, clean, all_terms)
    lookup = {tuple((v["a"], v["b"], *v["alpha"])): j for j,v in enumerate(all_terms)}
    cols = [lookup[(v["a"], v["b"], *v["alpha"])] for v in terms]
    checks["same_jet_pipeline_small_library"] = np.array_equal(a, bigger[:, cols]) and np.array_equal(target, target_bigger)
    weak_comparisons = []
    for order in (0, 1, 2):
        col = next(j for j,v in enumerate(terms) if v["alpha"][order] == 1 and sum(v["alpha"]) == 1)
        pure = pure_weak_column(x, t, clean, 1, order)
        weak_comparisons.append({"derivative_order": order,
            "jet_vs_pure_weak_relative_difference": float(np.linalg.norm(a[:,col]-pure)/np.linalg.norm(pure))})
    checks["same_centers_pure_weak_clean_agreement"] = max(v["jet_vs_pure_weak_relative_difference"] for v in weak_comparisons) < .002
    rng = np.random.default_rng(6031)
    noisy = clean + .01*np.std(clean)*rng.standard_normal(clean.shape)
    noisy_jets, _ = spatial_jets(noisy, x[1]-x[0])
    clean_jets, _ = spatial_jets(clean, x[1]-x[0])
    noise_errors = [float(np.linalg.norm(noisy_jets[r]-clean_jets[r])/np.linalg.norm(clean_jets[r])) for r in range(7)]
    checks["finite_noise_diagnostic"] = all(np.isfinite(noise_errors))
    report = {"stage": 1, "all_checks_passed": all(checks.values()), "checks": checks,
              "frozen_protocol": {"window": DEFAULT_WINDOW, "polyorder": DEFAULT_POLYORDER,
                  "max_order": 6, "test_power": TEST_POWER, "derivatives_from": "observations only",
                  "time_reconstruction": "none; original observations in weak target",
                  "hyperparameter_selection": "fixed before benchmark results; no truth-tuned changes"},
              "moments": moments, "polynomial_relative_errors_by_order": polynomial_errors,
              "fourier_refinement": rows, "weak_comparisons": weak_comparisons,
              "same_jet_pipeline_fixture": meta,
              "sigma_1pct_jet_relative_perturbations": noise_errors,
              "interpretation": "Higher derivative noise amplification is a preprocessing limitation, not an L1 optimization failure.",
              "limitations": ["Fixed 31-point window has grid-dependent physical bandwidth.",
                  "Products of reconstructed noisy jets are biased; weak integration does not automatically remove that bias.",
                  "Clean synthetic accuracy does not establish support recovery under noise.",
                  "Validation equation residual is separate from integrating a learned PDE on a held-out initial condition."]}
    strict_json_write(ROOT / "stage1_jet_preprocess_result.json", report)
    print({"all_checks_passed": report["all_checks_passed"], "checks": checks,
           "noise_perturbations": noise_errors})
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
