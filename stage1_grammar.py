"""Complete finite jet grammar, with exact derivative-form mappings.

No observations, known support, or correlations are used to generate columns.
Run only after the stage-0 numerical/evaluation gate has been reviewed.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import platform
import time
from collections import Counter
from pathlib import Path


MAX_JET_ORDER = 6
LAYERS = {
    "J8": (2, 2, 2, 0),
    "J18": (3, 3, 3, 0),
    "J38": (4, 4, 4, 0),
    "J114": (4, 4, 4, 1),
    "J1158": (6, 8, 6, 2),
}


def _budget(R, P, D, A):
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (R, P, D, A)):
        raise ValueError("Budgets must be nonnegative integers")
    if R > MAX_JET_ORDER:
        raise ValueError("This frozen grammar supports spatial jet order at most 6")


def key(term):
    """Fixed-length key: coordinate powers followed by u,...,D_x^6 u powers."""
    return (term["a"], term["b"], *term["alpha"])


def term_name(a, b, alpha):
    factors = []
    for base, power in [("x", a), ("t", b)] + [
        ("u" if r == 0 else "u_" + "x" * r, power)
        for r, power in enumerate(alpha)
    ]:
        if power:
            factors.append(base if power == 1 else f"{base}^{power}")
    return " ".join(factors) or "1"


def term_complexity(a, b, alpha):
    """Frozen library-size-independent cost, shared with stage-0 evaluation."""
    return 1 + max(sum(alpha) - 1, 0) + sum(r * p for r, p in enumerate(alpha)) + 2 * (a + b)


def _term(a, b, alpha):
    alpha = tuple(alpha) + (0,) * (MAX_JET_ORDER + 1 - len(alpha))
    return {
        "a": a, "b": b, "alpha": list(alpha), "name": term_name(a, b, alpha),
        "complexity": term_complexity(a, b, alpha), "power": sum(alpha),
        "total_derivative_order": sum(r * p for r, p in enumerate(alpha)),
        "max_derivative_order": max((r for r, p in enumerate(alpha) if p), default=0),
        "domain": "finite real x,t and jet variables", "rule": "jet-polynomial-v1",
    }


def enumerate_jet(R=6, P=8, D=6, A=2):
    """Enumerate exactly a+b<=A, sum(alpha)<=P, sum(r*alpha_r)<=D."""
    _budget(R, P, D, A)
    powers = []

    def visit(r, remaining_power, remaining_deriv, prefix):
        if r > R:
            powers.append(tuple(prefix))
            return
        limit = remaining_power if r == 0 else min(remaining_power, remaining_deriv // r)
        for power in range(limit + 1):
            visit(r + 1, remaining_power - power, remaining_deriv - r * power, prefix + [power])

    visit(0, P, D, [])
    return [_term(a, b, alpha) for a in range(A + 1) for b in range(A + 1 - a) for alpha in powers]


def count_dp(R, P, D, A):
    """Independent generating-function DP by exact field/derivative degrees."""
    _budget(R, P, D, A)
    coefficients = {(0, 0): 1}
    for r in range(R + 1):
        updated = Counter()
        for (power, derivative), multiplicity in coefficients.items():
            for exponent in range(P - power + 1):
                new_derivative = derivative + r * exponent
                if new_derivative <= D:
                    updated[(power + exponent, new_derivative)] += multiplicity
        coefficients = updated
    return math.comb(A + 2, 2) * sum(coefficients.values())


def count_by_partitions(R, P, D, A):
    """Independent enumeration of derivative partitions, with u powers summed."""
    _budget(R, P, D, A)
    bins = [0] * (D + 1)
    for tail in itertools.product(*(range(min(P, D // r) + 1) for r in range(1, R + 1))):
        degree, derivative = sum(tail), sum(r * p for r, p in enumerate(tail, 1))
        if degree <= P and derivative <= D:
            bins[derivative] += P - degree + 1
    return {"without_coordinates": bins, "coordinate_factor": math.comb(A + 2, 2),
            "count": math.comb(A + 2, 2) * sum(bins)}


def differentiate(poly):
    """Exact spatial total derivative of a sparse polynomial in canonical jets.

    poly maps (a,b,alpha_0,...,alpha_6) to integer coefficients. A nonzero
    alpha_6 would need order 7 and is rejected; no term is silently dropped.
    """
    out = Counter()
    for term_key, coefficient in poly.items():
        a, b, *alpha = term_key
        if alpha[-1]:
            raise ValueError("Derivative needs u_xxxxxxx outside the frozen jet order")
        if a:
            out[(a - 1, b, *alpha)] += coefficient * a
        for r, power in enumerate(alpha[:-1]):
            if power:
                new_alpha = alpha.copy()
                new_alpha[r] -= 1
                new_alpha[r + 1] += 1
                out[(a, b, *new_alpha)] += coefficient * power
    return {k: v for k, v in out.items() if v}


def expand_dx(a, b, m, k):
    """Return exact jet-basis map of D_x^k(x^a t^b u^m)."""
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (a, b, m, k)):
        raise ValueError("Powers and derivative orders must be nonnegative integers")
    if k > MAX_JET_ORDER:
        raise ValueError("Derivative order exceeds the frozen grammar")
    poly = {(a, b, m, *([0] * MAX_JET_ORDER)): 1}
    for _ in range(k):
        poly = differentiate(poly)
    return poly


def enumerate_divergence(P=8, K=6, A=2):
    """All declared weak-form recipes, including symbolic zeros and aliases.

    These are constrained directions in the jet basis, NOT additional free
    columns alongside that basis. Each nonzero recipe stores beta_jet=T*z.
    """
    _budget(K, P, K, A)
    terms = []
    for a in range(A + 1):
        for b in range(A + 1 - a):
            for m in range(P + 1):
                for k in range(K + 1):
                    expansion = expand_dx(a, b, m, k)
                    terms.append({
                        "a": a, "b": b, "m": m, "k": k,
                        "name": f"D_x^{k}(x^{a} t^{b} u^{m})",
                        "identically_zero": not bool(expansion),
                        "jet_expansion": [{"key": list(q), "coefficient": c} for q, c in sorted(expansion.items())],
                    })
    return terms


def nonpolynomial_recipes():
    """Finite supplement; no unlisted compositions or continuous parameters."""
    result = []
    for family, parameters, max_derivative in [
        ("sin", (1, 2, 3), 4), ("cos", (1, 2, 3), 4),
        ("exp", (-2, -1, 1, 2), 3), ("inverse_quadratic", (1, 2), 3),
    ]:
        for parameter in parameters:
            base = f"(1+u^2)^(-{parameter})" if family == "inverse_quadratic" else f"{family}({parameter}u)"
            for k in range(max_derivative + 1):
                result.append({"family": family, "parameter": parameter, "derivative_order": k,
                               "name": base if k == 0 else f"D_x^{k}({base})",
                               "domain": "all real u; reject numerical overflow",
                               "rule": "nonpolynomial-supplement-v1"})
    return result


def _symbolic_check():
    """Independent SymPy chain-rule validation; diagnostic use only."""
    import sympy as sp
    x, t = sp.symbols("x t")
    u = sp.Function("u")(x)
    jets = sp.symbols("j0:7")
    checks = [(0, 0, 2, 2), (1, 1, 3, 3), (2, 0, 1, 6), (0, 2, 8, 6), (2, 0, 0, 3)]
    for a, b, m, k in checks:
        expected = sp.diff(x**a * t**b * u**m, x, k)
        expected = expected.xreplace({sp.diff(u, x, r): jets[r] for r in range(7)})
        actual = 0
        for term_key, coefficient in expand_dx(a, b, m, k).items():
            aa, bb, *alpha = term_key
            actual += coefficient * x**aa * t**bb * math.prod(j**p for j, p in zip(jets, alpha))
        assert sp.expand(expected - actual) == 0, (a, b, m, k)
    return {"sympy_version": sp.__version__, "checked_recipes": checks, "passed": True}


def self_check():
    from stage0_evaluation import term_complexity as evaluated_complexity
    low = enumerate_jet(2, 2, 2, 0)
    manual = {"1", "u", "u^2", "u_x", "u u_x", "u_x^2", "u_xx", "u u_xx"}
    assert {t["name"] for t in low} == manual
    layers, previous = {}, set()
    for label, budget in LAYERS.items():
        terms = enumerate_jet(*budget)
        term_keys = {key(t) for t in terms}
        partitions = count_by_partitions(*budget)
        assert len(terms) == len(term_keys) == count_dp(*budget) == partitions["count"]
        assert previous <= term_keys
        previous = term_keys
        layers[label] = {"R_P_D_A": list(budget), "count": len(terms), "counts_check": partitions,
                         "nested_in_next": True, "max_complexity": max(t["complexity"] for t in terms)}
    layers["J1158"]["nested_in_next"] = None
    full = enumerate_jet()
    assert all(t["complexity"] == evaluated_complexity(t["name"]) for t in full)
    full_keys = {key(t) for t in full}
    divergence = enumerate_divergence()
    assert all(tuple(v["key"]) in full_keys for t in divergence for v in t["jet_expansion"])
    assert len(full) == 1158
    assert len(nonpolynomial_recipes()) == 54
    # Brute-force low-budget check is independent of the pruned recursive generator.
    for R, P, D, A in [(0, 0, 0, 0), (1, 2, 2, 1), (2, 3, 3, 2), (3, 3, 2, 0)]:
        expected = {(a, b, *alpha, *([0] * (MAX_JET_ORDER - R)))
                    for a in range(A + 1) for b in range(A + 1 - a)
                    for alpha in itertools.product(range(P + 1), repeat=R + 1)
                    if sum(alpha) <= P and sum(r * p for r, p in enumerate(alpha)) <= D}
        assert {key(t) for t in enumerate_jet(R, P, D, A)} == expected
    truth = {"heat": {"u_xx"}, "wave": {"u_xx"},
             "advection_diffusion": {"u_x", "u_xx"}, "burgers": {"u u_x", "u_xx"},
             "kdv": {"u u_x", "u_xxx"}, "logheat": {"u_x^2", "u_xx"}}
    names = {t["name"] for t in full}
    assert all(s <= names for s in truth.values())
    # A grammar-limited negative example, never a data- or truth-driven deletion.
    negative_names = {t["name"] for t in enumerate_jet(1, 2, 1, 0)}
    assert "u_x^2" not in negative_names and "u_xx" not in negative_names
    assert "u_x^2" in {t["name"] for t in low}
    polynomial_hash = hashlib.sha256(json.dumps(full, sort_keys=True, ensure_ascii=True).encode()).hexdigest()
    return {
        "status": "passed", "all_checks_passed": True,
        "stage0_complexity_agreement_terms": len(full),
        "layers": layers, "manual_low_order_names": sorted(manual),
        "polynomial_catalog_sha256": polynomial_hash, "polynomial_terms": full,
        "weak_divergence_recipes": divergence,
        "weak_divergence_summary": {"recipes": len(divergence),
                                    "symbolic_zeros": sum(t["identically_zero"] for t in divergence),
                                    "interpretation": "constraint maps in the polynomial basis; not extra free columns"},
        "nonpolynomial_recipes": nonpolynomial_recipes(),
        "nonpolynomial_status": "54 finite weak-form recipes declared; numerical supplement not validated here",
        "true_support_coverage": {case: sorted(s) for case, s in truth.items()},
        "negative_example": {"budget_R_P_D_A": [1, 2, 1, 0], "missing_logheat_terms": ["u_x^2", "u_xx"],
                             "interpretation": "grammar exclusion, not a solver failure"},
        "symbolic_identity_check": _symbolic_check(),
        "support_convention": "canonical jet monomials; Burgers nonlinear coefficient -1, KdV -6",
        "wave_ut_nuisance": "outside this spatial-jet grammar; adding it requires explicit +1-column declaration",
        "memory_estimates_MiB_float64": {"design_1000_rows": 8 * 1000 * 1158 / 2**20,
                                           "design_10000_rows": 8 * 10000 * 1158 / 2**20,
                                           "gram": 8 * 1158**2 / 2**20},
        "limitations": ["Completeness applies only to the declared finite grammar.",
                        "No numerical data reconstruction, support recovery, or identifiability claim is made.",
                        "Derivative-form sparsity and jet-monomial sparsity are different model classes.",
                        "Near-correlated columns are retained; no observed-data pruning occurs."],
    }


def main():
    started = time.perf_counter()
    result = self_check()
    result.update({"date": "2026-11-06", "python": platform.python_version(),
                   "elapsed_seconds": time.perf_counter() - started,
                   "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    path = Path(__file__).with_name("stage1_grammar_result.json")
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": result["status"], "counts": {k: v["count"] for k, v in result["layers"].items()},
                      "elapsed_seconds": result["elapsed_seconds"], "result": str(path)}))


if __name__ == "__main__":
    main()
