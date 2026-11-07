"""Independent curved-field verification of all frozen nonpolynomial columns.

SymPy differentiates the full composed functions only in this diagnostic.
The production constructor receives only sampled scalar field observations.
Both coarse and refined grids must meet the predeclared 1% column tolerance.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import sympy as sp
from threadpoolctl import threadpool_limits

import stage1_jet_preprocess as V1
from stage0_evaluation import array_hash, strict_json_write
from stage1_grammar import nonpolynomial_recipes
from stage1_nonpolynomial import HALO, weak_nonpolynomial_matrix
from stage1_weak_reduction_v2 import discrete_bump

ROOT = Path(__file__).resolve().parent
RELATIVE_TOLERANCE = .01
NEAR_ZERO_REFERENCE_SCALE = 1e-12
GRIDS = ((257, 16), (513, 32))


def _reference_functions():
    """Symbolic full composition, independent of production raw-function code."""
    x, t = sp.symbols("x t", real=True)
    field = sp.exp(-t) * (sp.sin(sp.pi*x)/4 + sp.Rational(3, 25)*sp.sin(3*sp.pi*x)
                         + sp.Rational(3, 100)*sp.cos(6*sp.pi*x)) + t/10
    formulas, expressions = [], []
    for recipe in nonpolynomial_recipes():
        p, family = recipe["parameter"], recipe["family"]
        if family == "sin": composed = sp.sin(p*field)
        elif family == "cos": composed = sp.cos(p*field)
        elif family == "exp": composed = sp.exp(p*field)
        elif family == "inverse_quadratic": composed = (1+field**2)**(-p)
        else: raise ValueError("undeclared symbolic family")
        derivative = sp.diff(composed, x, recipe["derivative_order"])
        formulas.append(sp.lambdify((x, t), derivative, modules="numpy", cse=True))
        expressions.append({"name":recipe["name"], "sympy_expression":str(derivative)})
    return formulas, expressions, str(field)


def _grid_check(nx, hx, functions):
    x, t = np.linspace(0., 1., nx), np.linspace(0., .08, 129)
    xx, tt = np.meshgrid(x, t)
    # Sampling uses the predeclared field, without any analytical derivatives.
    u = np.exp(-tt)*(.25*np.sin(np.pi*xx)+.12*np.sin(3*np.pi*xx)+.03*np.cos(6*np.pi*xx))+.1*tt
    half = (hx, 8)
    matrix, target, meta = weak_nonpolynomial_matrix(x, t, u, target_order=1, half=half)
    dx, dt = x[1]-x[0], t[1]-t[0]
    px, _ = discrete_bump(hx, dx, 0)
    pt, _ = discrete_bump(8, dt, 0)
    xref, tref = xx[:, HALO:-HALO], tt[:, HALO:-HALO]
    references = []
    for function in functions:
        strong = np.broadcast_to(np.asarray(function(xref, tref), dtype=float), xref.shape)
        if not np.isfinite(strong).all(): raise ValueError("non-finite independent symbolic reference")
        references.append(V1._integrate(strong, px[0], pt[0], dx, dt, half))
    reference = np.column_stack(references)
    difference = matrix-reference
    reference_norm = np.linalg.norm(reference, axis=0)
    absolute_l2 = np.linalg.norm(difference, axis=0)
    relative = absolute_l2 / np.maximum(reference_norm, np.finfo(float).tiny)
    constant_scale = np.linalg.norm(V1._integrate(np.ones_like(xref), px[0], pt[0], dx, dt, half))
    near_zero_threshold = NEAR_ZERO_REFERENCE_SCALE*max(1., float(constant_scale))
    columns = [{"name":r["name"], "family":r["family"], "parameter":r["parameter"],
                "derivative_order":r["derivative_order"], "reference_l2_norm":float(reference_norm[j]),
                "absolute_l2_error":float(absolute_l2[j]),
                "maximum_absolute_error":float(np.max(np.abs(difference[:, j]))),
                "relative_l2_error":float(relative[j]),
                "near_zero_reference":bool(reference_norm[j] <= near_zero_threshold)}
               for j, r in enumerate(nonpolynomial_recipes())]
    worst = int(np.argmax(relative))
    actual_x = np.asarray(meta["center_x_indices"])
    actual_t = np.asarray(meta["center_t_indices"])
    expected_shape = (len(actual_x)*len(actual_t), 54)
    checks = {"all54_present":matrix.shape == expected_shape,
              "every_column_relative_accuracy":bool(np.max(relative) < RELATIVE_TOLERANCE),
              "finite_matrix_target_reference":bool(np.isfinite(matrix).all() and np.isfinite(target).all()
                                                     and np.isfinite(reference).all())}
    return {"nx":nx, "nt":len(t), "half":list(half), "stride":list(half), "halo_x":HALO,
            "physical_radius_x":float(hx*dx), "physical_radius_t":float(8*dt),
            "maximum_relative_error":float(np.max(relative)), "worst_column":columns[worst],
            "maximum_absolute_error":float(np.max(np.abs(difference))),
            "near_zero_reference_threshold":near_zero_threshold,
            "near_zero_columns":[r for r in columns if r["near_zero_reference"]],
            "columns":columns, "input_hash":array_hash(u), "reference_hash":array_hash(reference),
            "metadata":meta, "checks":checks, "all_checks_passed":all(checks.values())}


def main():
    start = time.perf_counter()
    functions, expressions, field = _reference_functions()
    symbolic_seconds = time.perf_counter()-start
    rows = []
    with threadpool_limits(limits=1):
        for nx, hx in GRIDS:
            row = _grid_check(nx, hx, functions)
            rows.append(row)
            print({"nx":nx, "all_checks_passed":row["all_checks_passed"],
                   "max_relative_error":row["maximum_relative_error"],
                   "worst_column":row["worst_column"]["name"]}, flush=True)
    checks = {"both_grids_all54_accuracy":all(r["all_checks_passed"] for r in rows),
              "same_physical_radius":rows[0]["physical_radius_x"] == rows[1]["physical_radius_x"],
              "frozen_54_functions":len(functions) == len(nonpolynomial_recipes()) == 54}
    sources = ["stage1_nonpolynomial_curved_check.py", "stage1_nonpolynomial.py", "stage1_grammar.py",
               "stage1_weak_reduction_v2.py", "stage1_jet_preprocess.py", "stage0_evaluation.py"]
    report = {"stage":"1_nonpolynomial_curved_field_verification", "all_checks_passed":all(checks.values()),
              "checks":checks, "field_symbolic":field,
              "protocol_frozen_before_first_run":{"x":[0., 1.], "t":[0., .08, 129],
                  "grids":[{"nx":nx,"hx":hx} for nx,hx in GRIDS], "ht":8, "halo_x":HALO,
                  "relative_tolerance_each_column_both_grids":RELATIVE_TOLERANCE,
                  "near_zero_reference_scale":NEAR_ZERO_REFERENCE_SCALE,
                  "near_zero_column_policy":"report absolute and relative errors; retain every column in the same gate",
                  "refinement_policy":"fixed physical spatial radius; no per-column monotonicity requirement"},
              "source_sha256":{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
              "numpy_version":np.__version__, "sympy_version":sp.__version__,
              "symbolic_derivative_expressions":expressions, "rows":rows,
              "symbolic_reference_setup_s":symbolic_seconds, "elapsed_s":time.perf_counter()-start,
              "no_pde_fitting_run":True, "no_analytic_derivatives_in_production_api":True,
              "test_data_used":False, "no_function_or_kernel_change":True,
              "limitations":["A curved synthetic fixture tests composed derivatives, not all real PDE trajectories or noise levels.",
                  "The independent strong-form reference uses the same zeroth sampled quadrature; continuous integral accuracy is not claimed.",
                  "Spatial refinement retains 15 discarded samples, so sampled centers differ; errors refer to each grid's own aligned reference.",
                  "Every one of the54 columns remains in the gate, including any near-zero reference column.",
                  "No L1 fitting, support selection, complexity change, or noise-robustness claim is made."]}
    strict_json_write(ROOT/"stage1_nonpolynomial_curved_check_result.json", report)
    print({"all_checks_passed":report["all_checks_passed"], "elapsed_s":report["elapsed_s"]})
    if not report["all_checks_passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
