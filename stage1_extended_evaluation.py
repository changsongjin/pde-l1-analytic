"""Strict evaluation for the unchanged jet catalog plus 54 frozen recipes.

Only exact recipe names are new accepted inputs. Polynomial parsing and scalar
normalization stay in stage0_evaluation. A nonpolynomial derivative is one
constrained column, never separate fitted chain-rule summands. SymPy expansion
below is standalone audit code, not a matrix constructor or fitting routine.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np

import stage0_evaluation as OLD
from stage1_grammar import enumerate_jet, nonpolynomial_recipes

ROOT = Path(__file__).resolve().parent
SUPPORT_TOL = OLD.SUPPORT_TOL
RECIPES = {r['name']: r for r in nonpolynomial_recipes()}
if len(RECIPES) != 54:
    raise ValueError('Expected exactly 54 unique frozen nonpolynomial names.')


@lru_cache(maxsize=4096)
def canonical_with_scale(name):
    """Return (column identity, scalar); unknown functional syntax is rejected.

    NP keys are output identifiers, not an additional accepted expression
    syntax. Only exact source-catalog names enter the nonpolynomial branch.
    """
    if not isinstance(name, str):
        raise TypeError('Candidate name must be a string.')
    recipe = RECIPES.get(name)
    if recipe is None:
        return OLD.canonical_with_scale(name)
    return (f"NP:{recipe['family']}:{recipe['parameter']}:Dx{recipe['derivative_order']}", 1.)


def canonical_term(name):
    return canonical_with_scale(name)[0]


@lru_cache(maxsize=4096)
def term_complexity(name):
    """Frozen, library-independent expression cost; not representation invariant.

    Polynomial costs are unchanged. sin/cos/exp cost 2+k; fixed nonzero
    argument parameters identify shape and cost no extra operation, consistent
    with scalar normalization in the old polynomial contract. Rational cost
    4+m+k retains old cost 4 for 1+u^2, then m inverse/power operations. Outer
    derivatives cost k. A recipe's expanded summands are not separate columns.
    """
    recipe = RECIPES.get(name)
    if recipe is None:
        return OLD.term_complexity(name)
    k = recipe['derivative_order']
    return 4 + recipe['parameter'] + k if recipe['family'] == 'inverse_quadratic' else 2 + k


def canonical_coefficients(coefficients):
    out = {}
    for name, value in coefficients.items():
        key, scale = canonical_with_scale(name)
        value = float(value) * scale
        total = out.get(key, 0.) + value
        if not math.isfinite(value) or not math.isfinite(total):
            raise ValueError('Non-finite physical coefficient or canonical sum.')
        out[key] = total
    return out


def support_metrics(got_names, true_names):
    """Support lists come from normalized L1 beta, including LS-zero terms."""
    got = {canonical_term(name) for name in got_names}
    true = {canonical_term(name) for name in true_names}
    tp = len(got & true)
    return {'exact': got == true,
            'precision': tp / len(got) if got else float(not true),
            'recall': tp / len(true) if true else 1.,
            'f1': 2 * tp / (len(got) + len(true)) if got or true else 1.,
            'extra': sorted(got - true), 'missed': sorted(true - got),
            'support_canonical': sorted(got), 'true_support_canonical': sorted(true)}


def coefficient_errors(estimated, truth):
    """Physical-coordinate errors on the full union, with true-only auxiliary."""
    estimated, truth = canonical_coefficients(estimated), canonical_coefficients(truth)
    denominator = math.hypot(*truth.values())
    if not denominator or not math.isfinite(denominator):
        raise ValueError('Relative error needs a finite nonzero truth-vector norm.')
    union = set(estimated) | set(truth)
    result = {'physical_coef_rel_error_union': math.hypot(*(
        estimated.get(k, 0.) - truth.get(k, 0.) for k in union)) / denominator,
        'physical_coef_rel_error_true_only': math.hypot(*(
        estimated.get(k, 0.) - v for k, v in truth.items())) / denominator}
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError('Physical coefficient error is not representable as finite float64.')
    return result


def _base_expression(recipe, u, sp):
    family, p = recipe['family'], recipe['parameter']
    if family == 'sin': return sp.sin(p*u)
    if family == 'cos': return sp.cos(p*u)
    if family == 'exp': return sp.exp(p*u)
    if family == 'inverse_quadratic': return (1+u*u)**(-p)
    raise ValueError('Undeclared symbolic family.')


def _symbolic_audit():
    import sympy as sp
    jets = sp.symbols('u0:5', real=True)
    x = sp.Symbol('x', real=True)
    ux = sp.Function('u', real=True)(x)
    rows, expressions, signatures = [], [], []
    base_cache, signature_cache = {}, {}
    for recipe in RECIPES.values():
        family_key = (recipe['family'], recipe['parameter'])
        if family_key not in base_cache:
            base_cache[family_key] = _base_expression(recipe, jets[0], sp)
            # Coefficients of this finite Taylor expansion are exact rational
            # numbers. One nonzero 2x2 minor proves nonproportional functions;
            # failure to find one would fail this audit, never assert equality.
            series = sp.Poly(sp.series(base_cache[family_key], jets[0], 0, 14).removeO(), jets[0])
            signature_cache[family_key] = [series.nth(n)*sp.factorial(n) for n in range(14)]
        base, k = base_cache[family_key], recipe['derivative_order']
        expression = base
        for _ in range(k):
            expression = sp.expand(sum(sp.diff(expression, jets[r])*jets[r+1] for r in range(4)))
        independent = sp.diff(_base_expression(recipe, ux, sp), x, k)
        independent = independent.xreplace({sp.diff(ux, x, r): jets[r] for r in range(5)})
        independent_agrees = sp.cancel(sp.expand(expression-independent)) == 0
        polynomial = sp.Poly(expression, *jets[1:])
        weighted_orders = {sum((r+1)*p for r, p in enumerate(powers)) for powers, _ in polynomial.terms()}
        max_order = max((r for r, jet in enumerate(jets) if expression.has(jet)), default=0)
        leading = base if k == 0 else sp.diff(expression, jets[k])
        expected_leading = base if k == 0 else sp.diff(base, jets[0])
        leading_agrees = sp.cancel(sp.expand(leading-expected_leading)) == 0
        signature = signature_cache[family_key][0:13] if k == 0 else signature_cache[family_key][1:14]
        assert any(value != 0 for value in signature)
        # For rational functions, cancellation gives a nonconstant denominator;
        # trig/exp families at nonzero parameter are nonpolynomial analytically.
        if recipe['family'] == 'inverse_quadratic':
            denominator = sp.fraction(sp.cancel(leading))[1]
            not_polynomial = sp.degree(denominator, jets[0]) > 0
            domain_reason = '1+u0^2 is strictly positive for real u0; all chain-rule denominators are its powers.'
        else:
            not_polynomial = bool(leading.has(sp.sin, sp.cos, sp.exp))
            domain_reason = 'sin/cos/exp and their real derivatives are defined for every real u0.'
        real_domain = expression.is_real is True and sp.ask(sp.Q.positive(1+jets[0]**2)) is True
        row = {'name': recipe['name'], 'canonical_key': canonical_term(recipe['name']),
               'family': recipe['family'], 'parameter': recipe['parameter'],
               'derivative_order': k, 'complexity': term_complexity(recipe['name']),
               'jet_expansion': str(expression), 'leading_jet_coefficient': str(leading),
               'maximum_jet_order': max_order, 'weighted_derivative_orders': sorted(weighted_orders),
               'independent_full_derivative_agrees': independent_agrees,
               'leading_coefficient_agrees': leading_agrees, 'all_real_domain': real_domain,
               'domain_reason': domain_reason, 'nonpolynomial_leading_coefficient': bool(not_polynomial),
               'single_constrained_column': True}
        row['all_checks_passed'] = bool(independent_agrees and leading_agrees and max_order == k
                                       and weighted_orders == {k} and real_domain and not_polynomial)
        rows.append(row)
        expressions.append(expression)
        signatures.append(signature)
    by_order = 0
    witnesses = []
    for i, j in itertools.combinations(range(len(rows)), 2):
        if rows[i]['derivative_order'] != rows[j]['derivative_order']:
            # The higher-order column has nonzero f'(u0)*u_k; the other does not.
            by_order += 1
            continue
        left, right = signatures[i], signatures[j]
        witness = next(((a, b, left[a]*right[b]-left[b]*right[a])
                        for a, b in itertools.combinations(range(13), 2)
                        if left[a]*right[b]-left[b]*right[a] != 0), None)
        if witness is None:
            raise AssertionError('No exact nonproportionality witness: '+rows[i]['name']+' / '+rows[j]['name'])
        a, b, determinant = witness
        witnesses.append({'left': rows[i]['name'], 'right': rows[j]['name'],
                          'leading_function_derivative_indices_at_zero': [a, b],
                          'exact_nonzero_determinant': str(determinant)})
    return {'sympy_version': sp.__version__, 'columns': rows,
            'total_np_pairs': len(rows)*(len(rows)-1)//2,
            'pairs_proved_by_distinct_maximum_jet_order': by_order,
            'same_order_pairs_proved_by_exact_taylor_minors': len(witnesses),
            'same_order_witnesses': witnesses,
            'all54_expression_checks_passed': all(r['all_checks_passed'] for r in rows),
            'no_np_pair_proportional': by_order+len(witnesses) == len(rows)*(len(rows)-1)//2,
            'no_np_column_proportional_to_any_jet_polynomial': all(r['nonpolynomial_leading_coefficient'] for r in rows),
            'polynomial_separation_argument': 'A polynomial jet column has polynomial coefficients in u0. '
                'For k>0 each NP column has nonpolynomial f\'(u0) multiplying its unique highest jet u_k; '
                'for k=0 its value f(u0) is nonpolynomial. Coordinate powers cannot change this, since NP '
                'columns have no x or t dependence. Rational coefficients have uncancelled complex poles; '
                'nonzero-frequency trig and nonzero-scale exp are nonpolynomial.',
            'scope': 'No pairwise proportional columns; this is not full linear independence or numerical identifiability.'}


def _must_reject(function):
    try:
        function()
    except (TypeError, ValueError):
        return True
    return False


def _evaluation_checks():
    # Exercise the actual shared scorer on a synthetic design, without a solver
    # or PDE data. No truth reaches candidate generation or score selection.
    from stage2a_baseline import _candidate_records
    checks = {}
    terms = enumerate_jet()
    checks['all1158_polynomial_keys_scales_costs_unchanged'] = all(
        canonical_with_scale(t['name']) == OLD.canonical_with_scale(t['name'])
        and term_complexity(t['name']) == t['complexity'] == OLD.term_complexity(t['name']) for t in terms)
    keys = [canonical_term(t['name']) for t in terms] + [canonical_term(n) for n in RECIPES]
    checks['all1212_unique_column_keys'] = len(keys) == len(set(keys)) == 1212
    invalid = ['sin(u)', 'sin(4u)', 'cos(0u)', 'exp(3u)', '(1+u^2)^(-3)',
               'D_x^5(sin(1u))', 'D_x^4(exp(1u))', 'sin(1u)+cos(1u)',
               '2*sin(1u)', ' sin(1u)', 'np.sin(u)', '__import__("os")']
    checks['unlisted_functions_compositions_and_aliases_rejected'] = all(
        _must_reject(lambda n=n: canonical_term(n)) for n in invalid)
    checks['fixed_predeclared_cost_examples'] = [term_complexity(n) for n in
        ['u', 'u_xx', '1+u^2', 'sin(1u)', 'sin(3u)', 'exp(-2u)',
         'D_x^4(cos(2u))', '(1+u^2)^(-1)', 'D_x^3((1+u^2)^(-2))']] == [1, 3, 4, 2, 2, 2, 6, 5, 9]
    checks['constrained_derivative_not_split'] = canonical_term('D_x^2(sin(1u))') not in {
        canonical_term('sin(1u)'), canonical_term('u_x^2'), canonical_term('u_xx')}
    checks['polynomial_scalar_absorption_retained'] = coefficient_errors(
        {'(u^2)_x': -.5}, {'u u_x': -1.})['physical_coef_rel_error_union'] == 0.
    checks['false_np_physical_coefficient_counted_on_union'] = coefficient_errors(
        {'u': 1., 'sin(1u)': 2.}, {'u': 1.}) == {
            'physical_coef_rel_error_union': 2., 'physical_coef_rel_error_true_only': 0.}
    checks['empty_support_conventions_unchanged'] = all(support_metrics(a, b) == OLD.support_metrics(a, b)
        for a, b in [([], []), ([], ['u']), (['u'], []), (['uxx'], ['u_xx'])])
    checks['nonfinite_coefficients_rejected'] = all(_must_reject(lambda v=v:
        coefficient_errors({'sin(1u)': v}, {'u': 1.})) for v in [np.nan, np.inf, -np.inf])
    a, b = np.eye(3), np.array([1., 0., 0.])
    names = ['u', 'sin(1u)', 'D_x^2(sin(1u))']
    beta = [1., 2e-5, 1e-5]
    point = {'status': 'certified', 'ratio': .1, 'lambda': .1, 'beta': beta, 'support': [0, 1]}
    path = {'points': [point], 'all_solutions_certified': True}
    args = (a, b, a, b, names, [term_complexity(n) for n in names], 'synthetic_common_validation', [(4, a, b)])
    evaluation = _candidate_records(path, *args)
    selected = evaluation['candidates'][evaluation['primary_selected_candidate_id']]
    checks['fixed_threshold_and_LS_does_not_delete_support'] = (SUPPORT_TOL == 1e-5
        and selected['support'] == ['u', 'sin(1u)'] and selected['coefficients'] == [1., 0.]
        and selected['complexity'] == 3 and not selected['ls_readout_changes_support'])
    checks['zero_LS_false_term_still_support_error'] = not support_metrics(selected['support'], ['u'])['exact']
    checks['truth_absent_during_selection'] = all('truth_evaluation_after_selection' not in c for c in evaluation['candidates'])
    partial = _candidate_records({**path, 'all_solutions_certified': False}, *args)
    checks['uncertified_path_not_primary_rankable'] = not partial['selection_eligible'] and partial['primary_selected_candidate_id'] is None
    return checks, {'evaluation_before_truth': evaluation,
                    'truth_after_selection_only': support_metrics(selected['support'], ['u']),
                    'invalid_name_fixtures': invalid}


def main():
    started = time.perf_counter()
    sources = ['stage0_evaluation.py', 'stage1_grammar.py', 'stage1_nonpolynomial.py',
               'stage1_nonpolynomial_result.json', 'stage1_nonpolynomial_curved_check.py',
               'stage1_nonpolynomial_curved_check_result.json', 'stage2a_baseline.py']
    original_hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}
    checks, examples = _evaluation_checks()
    symbolic = _symbolic_audit()
    checks.update({k: symbolic[k] for k in ('all54_expression_checks_passed', 'no_np_pair_proportional',
                                           'no_np_column_proportional_to_any_jet_polynomial')})
    numerical_gates = {p: json.loads((ROOT/p).read_text(encoding='utf-8'),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Non-strict JSON: '+value)))
        for p in ['stage1_nonpolynomial_result.json', 'stage1_nonpolynomial_curved_check_result.json']}
    checks['upstream_numerical_gates_passed'] = all(r['all_checks_passed'] for r in numerical_gates.values())
    checks['old_sources_and_results_unchanged_during_audit'] = all(
        hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == h for p, h in original_hashes.items())
    report = {'stage': '1_extended_evaluation_and_symbolic_catalog_audit',
              'all_checks_passed': all(checks.values()), 'checks': checks,
              'all_checks_passed_semantics': 'Enables integration of frozen54 into development libraries; '
                  'not a support-recovery, noise-robustness, full-rank, or publication gate.',
              'source_sha256': {**original_hashes, Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
              'complexity_rule_frozen_before_any_expanded_library_fit': {
                  'polynomial': 'unchanged stage0 and stage1 grammar; all1158 checked',
                  'sin_cos_exp': '2 + derivative_order',
                  'inverse_quadratic': '4 + parameter_m + derivative_order',
                  'argument_parameter_cost': 'fixed shape identifier; no extra cost for signed scalar',
                  'rational_reason': 'old cost(1+u^2)=4, then m inverse/power operation costs',
                  'scope': 'fixed expression accounting, not invariant under all alternative representations',
                  'depends_on_data_truth_or_library_size': False},
              'support_tol': SUPPORT_TOL, 'threshold_applied_to': 'normalized L1 coefficients only',
              'coefficient_error_coordinates': 'physical coefficients on canonical support union',
              'canonical_key_policy': 'NP output keys identify exact recipes and are not expression inputs; '
                  'exact whitelist source names only, scalar1; chain-rule summands remain bound.',
              'symbolic_audit': symbolic, 'evaluation_examples': examples,
              'numerical_gate_files_checked_without_rerunning': list(numerical_gates),
              'no_pde_fitting_run': True, 'test_trajectories_used': False,
              'elapsed_s': time.perf_counter()-started,
              'limitations': ['Pairwise nonproportionality does not establish full linear independence or usable conditioning.',
                  'Only the declared 54 recipes are covered; no arbitrary nonpolynomial compositions are accepted.',
                  'All-real mathematical domain does not guarantee float64 representability; matrix builder rejects overflow.',
                  'Canonical coefficient errors compare recipe coordinates, not a basis-independent physical prediction metric.',
                  'A future complexity sensitivity study must be declared separately, without adjusting costs to true support.',
                  'This is a candidate-family extension, not a new L1 statistical method.']}
    OLD.strict_json_write(ROOT/'stage1_extended_evaluation_result.json', report)
    print(json.dumps({'all_checks_passed': report['all_checks_passed'], 'checks': checks,
                      'np_pairs_proved': symbolic['total_np_pairs'], 'elapsed_s': report['elapsed_s']}, indent=2))
    if not report['all_checks_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
