"""Auditable evaluation contracts for new PDE experiments; old results untouched.

Arrays in evaluation APIs are physical (possibly common row-weighted) designs.
Train-only column and target norms define the solver coordinates. Support is
always selected from the L1 coefficients at 1e-5, never from the LS readout.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import sympy as sp

SUPPORT_TOL = 1e-5
ROOT = Path(__file__).resolve().parent
X, T = sp.symbols("x t")
U = sp.symbols("u " + " ".join("u_" + "x" * k for k in range(1, 13)))
UT, UTT = sp.symbols("u_t u_tt")
SYMBOLS = (X, T, *U, UT, UTT)
ALLOWED_NAMES = {str(s): s for s in SYMBOLS}
ALLOWED_NAMES.update({"u" + "x" * k: U[k] for k in range(1, 13)})


def _expression(source):
    source = str(source).strip()
    if len(source) > 2048:
        raise ValueError("term expression too long")
    derivative = re.fullmatch(r"\((.*)\)_(x{1,12})", source)
    if derivative:
        p = _expression(derivative[1])
        for _ in derivative[2]:
            if p.has(UT, UTT, U[-1]):
                raise ValueError("mixed time/spatial or spatial order >12 unsupported")
            p = sp.expand(sp.diff(p, X) + sum(sp.diff(p, U[r]) * U[r+1]
                                             for r in range(12)))
        return p
    source = re.sub(r"(?<=[\w)])\s+(?=[A-Za-z(])", "*", source)
    source = source.replace("^", "**")
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise ValueError("unsupported polynomial syntax: " + source) from exc
    if sum(1 for _ in ast.walk(tree)) > 256:
        raise ValueError("term AST too large")

    def visit(node):
        if isinstance(node, ast.Name) and node.id in ALLOWED_NAMES:
            return ALLOWED_NAMES[node.id]
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            if not math.isfinite(node.value):
                raise ValueError("non-finite term coefficient")
            return sp.Rational(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return (-1 if isinstance(node.op, ast.USub) else 1) * visit(node.operand)
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div) and right.is_number and right != 0:
                return left / right
            if isinstance(node.op, ast.Pow) and right.is_Integer and 0 <= right <= 12:
                return left ** int(right)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "mul" and len(node.args) == 2 and not node.keywords):
            return visit(node.args[0]) * visit(node.args[1])
        raise ValueError("unsupported polynomial term: " + source)

    return sp.expand(visit(tree.body))


def canonical_with_scale(name):
    """Return exact polynomial column up to scalar, and that signed scalar.

    A derivative of a product remains ONE constrained polynomial column. This
    routine does not turn its expansion into independent fitted coefficients.
    Unknown functions/operators fail explicitly rather than guessing a label.
    """
    p = sp.Poly(_expression(name), *SYMBOLS)
    if p.is_zero:
        raise ValueError("zero expression is not a candidate column")
    scale = p.LC()
    key = str(sp.expand(p.as_expr() / scale)).replace("**", "^").replace(" ", "")
    return key, float(scale)


def canonical_term(name):
    return canonical_with_scale(name)[0]


def canonical_coefficients(coefficients):
    """Coefficients refer to input expression columns; absorb exact scale."""
    out = {}
    for name, value in coefficients.items():
        key, scale = canonical_with_scale(name)
        out[key] = out.get(key, 0.0) + float(value) * scale
    return out


def support_metrics(got_names, true_names):
    got = {canonical_term(s) for s in got_names}
    true = {canonical_term(s) for s in true_names}
    tp = len(got & true)
    precision = tp / len(got) if got else float(not true)
    recall = tp / len(true) if true else 1.0
    f1 = 2 * tp / (len(got) + len(true)) if got or true else 1.0
    return {"exact": got == true, "precision": precision, "recall": recall,
            "f1": f1, "extra": sorted(got - true), "missed": sorted(true - got),
            "support_canonical": sorted(got), "true_support_canonical": sorted(true)}


def normalize_training(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 2 or b.shape != (a.shape[0],) or not a.size:
        raise ValueError("invalid training design/target shape")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("non-finite training design/target")
    norms = np.linalg.norm(a, axis=0)
    # Exact zero columns are retained, with unit scale; no near-zero pruning.
    scales = np.where(norms == 0, 1.0, norms)
    target_scale = float(np.linalg.norm(b)) or 1.0
    metadata = {"column_scale": scales.tolist(), "target_scale": target_scale,
                "exact_zero_columns": np.flatnonzero(norms == 0).tolist(),
                "fit_split": "train", "support_tol": SUPPORT_TOL}
    return a / scales, b / target_scale, metadata


def physical_coefficients(beta, scales):
    beta = np.asarray(beta, dtype=float)
    columns = np.asarray(scales["column_scale"], dtype=float)
    target = float(scales["target_scale"])
    if (beta.shape != columns.shape or not np.all(np.isfinite(beta))
            or not np.all(np.isfinite(columns)) or np.any(columns <= 0)
            or not math.isfinite(target) or target <= 0):
        raise ValueError("invalid solver coefficients or training scales")
    return beta * target / columns


def coefficient_errors(estimated, truth):
    estimated, truth = canonical_coefficients(estimated), canonical_coefficients(truth)
    denom = sum(v*v for v in truth.values())
    if denom == 0:
        raise ValueError("relative coefficient error undefined for zero true vector")
    union = set(estimated) | set(truth)
    return {"physical_coef_rel_error_union": math.sqrt(sum(
                (estimated.get(k, 0.0) - truth.get(k, 0.0))**2 for k in union) / denom),
            "physical_coef_rel_error_true_only": math.sqrt(sum(
                (estimated.get(k, 0.0) - v)**2 for k, v in truth.items()) / denom)}


def evaluate_support(beta, a_train, b_train, a_val, b_val, names, truth, scales):
    beta = np.asarray(beta, dtype=float)
    a_train, b_train = np.asarray(a_train), np.asarray(b_train)
    a_val, b_val = np.asarray(a_val), np.asarray(b_val)
    if (beta.shape != (len(names),) or a_train.ndim != 2 or a_val.ndim != 2
            or a_train.shape[1] != len(names) or a_val.shape[1] != len(names)
            or b_train.shape != (a_train.shape[0],) or b_val.shape != (a_val.shape[0],)):
        raise ValueError("coefficient/design/name dimension mismatch")
    if not all(np.all(np.isfinite(v)) for v in (beta, a_train, b_train, a_val, b_val)):
        raise ValueError("non-finite evaluation input")
    canonical = [canonical_term(s) for s in names]
    if len(set(canonical)) != len(canonical):
        raise ValueError("exact duplicate canonical columns must be resolved before fitting")
    support = np.flatnonzero(np.abs(beta) > SUPPORT_TOL)
    fit = np.linalg.lstsq(a_train[:, support], b_train, rcond=None)
    readout = np.zeros(len(names))
    readout[support] = fit[0]
    got = [names[i] for i in support]
    val_scale = float(np.linalg.norm(b_val))
    residual = float(np.linalg.norm(a_val @ readout - b_val))
    # Empty/zero-target validation has an absolute error with explicit status.
    result = {"support_indices": support.tolist(), "support": got,
              "support_tol": SUPPORT_TOL, "support_source": "normalized_L1_beta_only",
              "ls_rank": int(fit[2]), "ls_coefficient_map": {names[i]: float(readout[i]) for i in support},
              "l1_physical_coefficient_map": {names[i]: float(c) for i, c in
                                              enumerate(physical_coefficients(beta, scales))},
              "validation_residual_norm": residual,
              "validation_relative_error": residual/val_scale if val_scale else None,
              "validation_status": "relative" if val_scale else "zero_target_absolute_only",
              "validation_metric_kind": "equation_residual_not_trajectory_rollout",
              "complexity": sum(term_complexity(s) for s in got),
              **support_metrics(got, truth),
              **coefficient_errors({names[i]: readout[i] for i in support}, truth)}
    return result


def term_complexity(name):
    """Fixed polynomial expression cost: sum monomial costs plus additions.

    Monomial cost = 1 + max(field degree-1,0) + total derivative order
    + 2*coordinate degree. No library size or known support enters this rule.
    """
    key = canonical_term(name)
    poly = sp.Poly(_expression(key), *SYMBOLS)
    cost = 0
    for powers, _ in poly.terms():
        field_degree = sum(powers[2:])
        derivative_order = sum(r * powers[2+r] for r in range(13)) + powers[-2] + 2*powers[-1]
        cost += 1 + max(field_degree-1, 0) + derivative_order + 2*sum(powers[:2])
    return cost + len(poly.terms()) - 1


def pareto_front(candidates):
    """Candidate dictionaries need dataset_id, validation_error, complexity."""
    if not candidates:
        return []
    ids = {c["dataset_id"] for c in candidates}
    if len(ids) != 1:
        raise ValueError("Pareto comparison requires exactly one common validation dataset")
    points = [(float(c["validation_error"]), float(c["complexity"])) for c in candidates]
    if any(not math.isfinite(x) or not math.isfinite(y) or min(x, y) < 0 for x, y in points):
        raise ValueError("invalid Pareto point")
    return [c for i, c in enumerate(candidates) if not any(
        e <= points[i][0] and k <= points[i][1] and (e < points[i][0] or k < points[i][1])
        for j, (e, k) in enumerate(points) if i != j)]


def array_hash(array):
    a = np.asarray(array)
    if a.dtype.kind not in "biufc" or not np.all(np.isfinite(a)):
        raise ValueError("hash requires a finite numeric array")
    a = np.ascontiguousarray(a, dtype=a.dtype.newbyteorder("<"))
    header = json.dumps({"shape": list(a.shape), "dtype": a.dtype.str}, sort_keys=True).encode()
    return hashlib.sha256(header + b"\n" + a.tobytes()).hexdigest()


def validate_split_manifest(splits):
    """Whole trajectories only. Block splits require raw-footprint audit upstream.

    splits maps train/validation/test to lists of {trajectory_id, observation_hash}.
    Independent noise versions of one trajectory must remain in one split.
    """
    if set(splits) != {"train", "validation", "test"} or any(not v for v in splits.values()):
        raise ValueError("three nonempty trajectory splits are required")
    all_ids, all_hashes = set(), set()
    for records in splits.values():
        ids = {r["trajectory_id"] for r in records}
        hashes = {r["observation_hash"] for r in records}
        if len(ids) != len(records) or all_ids & ids or all_hashes & hashes:
            raise ValueError("trajectory or raw observation reused across splits")
        all_ids.update(ids)
        all_hashes.update(hashes)
    return True


def assert_same_input_manifests(manifests):
    if len(manifests) < 2:
        raise ValueError("same-input check needs at least two methods")
    values = [json.dumps(m, sort_keys=True, allow_nan=False) for m in manifests.values()]
    if len(set(values)) != 1:
        raise ValueError("methods have different raw inputs, trajectories, or splits")
    return True


def strict_json_write(path, data):
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    text = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False, default=convert)
    Path(path).write_text(text + "\n", encoding="utf-8")


def _must_fail(fn):
    try:
        fn()
    except (ValueError, TypeError):
        return True
    raise AssertionError("invalid input unexpectedly accepted")


def main():
    import gml1_maxlib_compare as G
    import noise_inverse as N
    checks = {}
    pairs = [("uxx", "u_xx"), ("mul(mul(u,u_x),u_x)", "u*u_x^2"),
             ("u ux", "u*u_x"), ("(u^2)_x", "u*u_x"),
             ("(u^2)_xx", "u*u_xx+u_x^2")]
    checks["canonical_alias_product_derivative"] = all(canonical_term(a) == canonical_term(b) for a, b in pairs)
    checks["derivative_scale_preserved"] = canonical_with_scale("(u^2)_xx")[1] == 2
    checks["compound_column_not_split"] = canonical_term("(u^2)_xx") != canonical_term("u_x^2")
    checks["unknown_syntax_rejected"] = all(_must_fail(lambda s=s: canonical_term(s))
                                            for s in ["sin(u)", "__import__('os')", "u/u_x"])
    checks["empty_metrics_explicit"] = (support_metrics([], [])["f1"] == 1
                                        and support_metrics([], ["u"])["f1"] == 0
                                        and support_metrics(["u"], [])["f1"] == 0)
    a, b, names, _ = G.make_data(N.CASES["heat"], 0.0, 0, G.build_terms(False))
    an, bn, scales = normalize_training(a, b)
    index = names.index("u_xx")
    beta = np.zeros(len(names))
    beta[index] = np.linalg.lstsq(an[:, [index]], bn, rcond=None)[0][0]
    result = evaluate_support(beta, a, b, a, b, names, N.CASES["heat"]["true"], scales)
    old = G.support_eval(G.LazyL1(a, b, names).a, b, (index,), names, N.CASES["heat"]["true"])
    checks["heat_physical_coefficient"] = result["physical_coef_rel_error_union"] < 1e-7 and old["coef_rel_err"] > 0.4
    checks["scale_prediction_identity"] = np.allclose(a @ physical_coefficients(beta, scales),
                                                      scales["target_scale"] * (an @ beta), rtol=1e-12, atol=1e-12)
    # A selected term with zero LS readout remains in support, by construction.
    zz = np.eye(2)
    _, _, ss = normalize_training(zz, np.array([1.0, 0.0]))
    retained = evaluate_support(np.array([1.0, 2e-5]), zz, [1., 0.], zz, [1., 0.], ["u", "u_x"], {"u": 1}, ss)
    checks["ls_readout_never_changes_support"] = retained["support"] == ["u", "u_x"] and retained["ls_coefficient_map"]["u_x"] == 0
    checks["false_coefficient_in_error"] = coefficient_errors({"u": 1, "u_x": 2}, {"u": 1})["physical_coef_rel_error_union"] == 2
    checks["scalar_equivalent_coefficient_error"] = coefficient_errors({"u*u_x": -1}, {"(u^2)_x": -.5})["physical_coef_rel_error_union"] == 0
    checks["fixed_complexity"] = term_complexity("u_xx") == 3 and term_complexity("uxx") == 3
    points = [{"dataset_id": "seed_0", "validation_error": e, "complexity": c} for e, c in [(1, 1), (.5, 2), (2, 2)]]
    checks["common_data_pareto"] = len(pareto_front(points)) == 2
    checks["cross_seed_pareto_rejected"] = _must_fail(lambda: pareto_front(points + [{**points[0], "dataset_id": "seed_1"}]))
    hash0 = array_hash(np.arange(6).reshape(2, 3))
    checks["hash_layout_and_shape"] = hash0 == array_hash(np.asfortranarray(np.arange(6).reshape(2, 3))) and hash0 != array_hash(np.arange(6))
    manifest = {s: [{"trajectory_id": s, "observation_hash": array_hash(np.array([i]))}]
                for i, s in enumerate(["train", "validation", "test"])}
    checks["split_disjoint_and_equal_inputs"] = validate_split_manifest(manifest) and assert_same_input_manifests({"L1": manifest, "IRL1": manifest})
    bad = {**manifest, "test": manifest["train"]}
    checks["split_leakage_rejected"] = _must_fail(lambda: validate_split_manifest(bad))
    checks["missing_trajectory_rejected"] = _must_fail(lambda: assert_same_input_manifests({"L1": manifest, "DLGA": bad}))
    checks["nonfinite_output_rejected"] = _must_fail(lambda: strict_json_write(ROOT / "stage0_should_never_be_written.json", {"bad": float("inf")}))
    label_rows = []
    previous = json.loads((ROOT / "dlga_same_data_result.json").read_text(encoding="utf-8"))
    for case, group in previous["cases"].items():
        for level in group["rows"]:
            for row in level["raw"]:
                try:
                    corrected = support_metrics(row["support"], row["true_support"])
                    label_rows.append({"case": case, "sigma": level["sigma"], "old_exact": row["exact"],
                                       "canonical_metrics": corrected, "status": "label_audit_only"})
                except ValueError as exc:
                    label_rows.append({"case": case, "sigma": level["sigma"], "status": "unsupported", "reason": str(exc)})
    checks["historical_dlga_terms_supported"] = all(r["status"] != "unsupported" for r in label_rows)
    inputs = [{"case": name, "weak_pipeline_trajectories": len(case["traj"]),
               "historical_dlga_trajectories": 1,
               "same_full_raw_trajectory_input": len(case["traj"]) == 1,
               "strict_ranking_eligible": False,
               "reason": "first trajectory only" if len(case["traj"]) > 1 else
                         "single trajectory agrees in count only; historical noise arrays/stride not shared"}
              for name, case in N.CASES.items()]
    report = {"phase": 0, "all_checks_passed": all(checks.values()), "checks": checks,
              "support_tolerance": SUPPORT_TOL, "heat_scale_reproduction": {
                  "old_coefficient_relative_error": old["coef_rel_err"],
                  "physical_coefficient": result["ls_coefficient_map"]["u_xx"],
                  "corrected_physical_relative_error": result["physical_coef_rel_error_union"],
                  "diagnostic_reuses_training_rows": True},
              "dlga_label_reaudit": label_rows, "historical_raw_input_contract": inputs,
              "metric_conventions": {"both_supports_empty_f1": 1, "empty_prediction_nonempty_truth_precision": 0,
                  "empty_truth_recall": 1, "coefficient_error": "union of canonical predicted and true columns",
                  "validation": "held-out equation residual; trajectory rollout is a separate future metric",
                  "complexity": "fixed symbolic expression cost, independent of library size",
                  "canonicalization": "exact scalar aliases only; constrained sums remain individual columns"},
              "limitations": ["Historical DLGA results remain ineligible for strict same-input rankings.",
                  "No native DLGA training rerun in this evaluation repair.",
                  "Manifest validation is for whole trajectories; buffered block footprints need separate checks.",
                  "Polynomial canonicalizer explicitly rejects nonpolynomial functions; later extension needs a declared parser.",
                  "This script validates metrics, not scientific recovery or a held-out benchmark."]}
    strict_json_write(ROOT / "stage0_evaluation_result.json", report)
    print(json.dumps({"all_checks_passed": report["all_checks_passed"], "checks": checks}, indent=2))
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
