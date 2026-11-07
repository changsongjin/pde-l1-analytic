"""Development pilot: shared saved matrices, L1/IRL1, honest numerical failures.

Only validation weak-equation residuals are scored, never trajectory rollouts.
Truth is appended after candidate generation, fixed selection, and Pareto scans.
Previous scripts/results are not changed. Real runs require stage-2 completion.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from stage0_evaluation import (SUPPORT_TOL, array_hash, canonical_term,
                               coefficient_errors, normalize_training,
                               pareto_front, physical_coefficients,
                               strict_json_write, support_metrics, term_complexity)
from stage2_data import CASES
from stage2_identifiability import save_matrices, stack_split, fit_fixed_support
from stage2a_l1_paths import DEFAULT_RATIOS, solve_l1_path, solve_original_irl1

ROOT = Path(__file__).resolve().parent
FROZEN_ALPHA = .005
SENSITIVITY_ALPHAS = [0., .001, .005, .01, .05]
METRIC = 'held_out_weak_equation_residual_not_trajectory_rollout'


@lru_cache(maxsize=8)
def _symbolic_metadata(names):
    canonical = tuple(canonical_term(name) for name in names)
    if len(set(canonical)) != len(canonical):
        raise ValueError('Library contains exact canonical duplicate columns.')
    return canonical, tuple(term_complexity(name) for name in names)


def _candidate_records(path, a, b, av, bv, names, costs, dataset_id, validation_blocks):
    """No truth argument: fit readouts for certified L1 supports without deleting terms."""
    fits, candidate_rows = {}, []
    for point_index, point in enumerate(path['points']):
        if point['status'] != 'certified':
            continue
        beta = np.asarray(point['beta'])
        support = tuple(int(i) for i in np.flatnonzero(np.abs(beta) > SUPPORT_TOL))
        if list(support) != point['support']:
            raise ValueError('Recorded support disagrees with normalized L1 beta.')
        if support not in fits:
            fit = fit_fixed_support(a, b, av, bv, support)
            coefficient = np.asarray(fit['coefficients'])
            validation_each = []
            for trajectory_id, block_a, block_b in validation_blocks:
                residual = np.linalg.norm(block_a[:, support]@coefficient-block_b)
                validation_each.append({'trajectory': trajectory_id,
                                        'relative_error': float(residual/np.linalg.norm(block_b))})
            fits[support] = {**fit, 'support': [names[i] for i in support],
                             'complexity': int(sum(costs[i] for i in support)),
                             'validation_error': fit['validation_equation_relative_error'],
                             'validation_by_trajectory': validation_each,
                             'dataset_id': dataset_id, 'validation_metric': METRIC,
                             'support_source': 'normalized_L1_beta_only',
                             'ls_readout_changes_support': False,
                             'candidate_id': len(fits)}
        candidate_rows.append({'point_index': point_index, 'lambda': point['lambda'],
                               'ratio': point['ratio'], 'candidate_id': fits[support]['candidate_id']})
    candidates = list(fits.values())
    front = pareto_front(candidates)
    sensitive = []
    for alpha in SENSITIVITY_ALPHAS:
        chosen = min(candidates, key=lambda c: (c['validation_error']+alpha*c['complexity'],
                                                c['complexity'], c['validation_error'],
                                                tuple(c['support_indices']))) if candidates else None
        sensitive.append({'alpha': alpha, 'candidate_id': chosen['candidate_id'] if chosen else None,
                          'score': chosen['validation_error']+alpha*chosen['complexity'] if chosen else None})
    selected = next(row['candidate_id'] for row in sensitive if row['alpha'] == FROZEN_ALPHA)
    complete = path['all_solutions_certified']
    return {'candidates': candidates, 'point_to_candidate': candidate_rows,
            'primary_selected_candidate_id': selected if complete else None,
            'exploratory_best_available_candidate_id': selected,
            'selection_eligible': complete,
            'pareto_candidate_ids': [c['candidate_id'] for c in front],
            'sensitivity': sensitive,
            'selection_rule': {'validation_error_coefficient': 1., 'complexity_coefficient': FROZEN_ALPHA,
                               'alpha_frozen_before_true_support_evaluation': True,
                               'complexity_independent_of_library_size': True},
            'partial_path_policy': 'No primary model or ranking when any requested lambda is unfinished/uncertified.'}


def _add_truth_metrics(evaluation, truth):
    # Deliberately called only after all selections/frontiers have been frozen.
    for candidate in evaluation['candidates']:
        coefficient = dict(zip(candidate['support'], candidate['coefficients']))
        candidate['truth_evaluation_after_selection'] = {
            **support_metrics(candidate['support'], truth), **coefficient_errors(coefficient, truth)}


def run_dataset(case, sigma, seed, layers, ratios, l1_budget_s, irl1_budget_s, progress_dir):
    dataset_start = time.perf_counter()
    load_start = time.perf_counter()
    matrices, targets, meta = save_matrices(case, sigma, seed)
    load_seconds = time.perf_counter()-load_start
    a, b = stack_split(matrices, targets, range(4))
    av, bv = stack_split(matrices, targets, range(4, 6))
    all_names = tuple(meta['names'])
    symbol_start = time.perf_counter()
    canonical, all_costs = _symbolic_metadata(all_names)
    if list(all_costs) != meta['complexities']:
        raise ValueError('Frozen grammar and evaluation complexity rules disagree.')
    symbol_seconds = time.perf_counter()-symbol_start
    dataset_id = f"{case}:sigma={sigma!r}:seed={seed}:validation={array_hash(bv)}"
    record = {'case': case, 'sigma': sigma, 'seed': seed, 'dataset_id': dataset_id,
              'raw_bundle_sha256': meta['raw_bundle_sha256'], 'matrix_metadata': meta,
              'splits': {'train_trajectories': list(range(4)), 'validation_trajectories': [4, 5],
                         'test': 'reserved and not generated or inspected'},
              'input_hashes': {'train_matrix': array_hash(a), 'train_target': array_hash(b),
                               'validation_matrix': array_hash(av), 'validation_target': array_hash(bv)},
              'library_names': list(all_names), 'canonical_library_names': list(canonical),
              'library_complexities': list(all_costs), 'layers': {},
              'time_s': {'matrix_load_or_build_this_run': load_seconds,
                         'matrix_build_recorded_source_sum': sum(r['build_seconds'] for r in meta['trajectories']),
                         'symbolic_metadata_this_run': symbol_seconds},
              'validation_metric': METRIC, 'stage': '2a_development_pilot',
              'test_data_used': False, 'truth_used_for_selection': False}
    frontier_candidates = []
    truth_pending = []
    for layer in layers:
        indices = meta['small_indices'] if layer == 'J18' else list(range(a.shape[1]))
        names = [all_names[i] for i in indices]
        costs = [all_costs[i] for i in indices]
        at, va = a[:, indices], av[:, indices]
        norm_start = time.perf_counter()
        an, bn, scales = normalize_training(at, b)
        normalization_seconds = time.perf_counter()-norm_start
        val_blocks = [(i, matrices[i][:, indices], targets[i]) for i in [4, 5]]
        layer_record = {'indices_in_maximum_library': indices, 'names': names,
                        'n_columns': len(indices), 'complexities': costs,
                        'normalization': scales, 'normalization_seconds': normalization_seconds,
                        'solver_input_hashes': {'A': array_hash(an), 'b': array_hash(bn)},
                        'methods': {}}
        record['layers'][layer] = layer_record
        for method, solver, budget in [('L1', solve_l1_path, l1_budget_s),
                                       ('IRL1_original', solve_original_irl1, irl1_budget_s)]:
            tag = f"{case}_{sigma!r}_{seed}_{layer}_{method}".replace('.', 'p')
            path_file = progress_dir/(tag+'_path_result.json')
            path = solver(an, bn, ratios=ratios, budget_s=budget, progress_path=path_file)
            eval_start = time.perf_counter()
            evaluation = _candidate_records(path, at, b, va, bv, names, costs, dataset_id, val_blocks)
            evaluation_seconds = time.perf_counter()-eval_start
            # Save unthresholded L1 physical coefficients for each candidate path point;
            # support is still determined only in normalized coordinates.
            for point in path['points']:
                if point['beta'] is not None:
                    point['physical_L1_coefficients'] = physical_coefficients(point['beta'], scales).tolist()
            method_record = {'path': path, 'evaluation': evaluation,
                             'time_s': {'solver_with_worker_startup': path['elapsed_s'],
                                        'fixed_support_validation': evaluation_seconds},
                             'full_end_to_end_timing_available': False,
                             'timing_note': 'Raw field-generation time is not in the cache; preprocessing build/load is reported separately.'}
            layer_record['methods'][method] = method_record
            truth_pending.append(evaluation)
            for candidate in evaluation['candidates']:
                frontier_candidates.append({**candidate, 'layer': layer, 'method': method,
                                            'method_path_complete': path['all_solutions_certified']})
            strict_json_write(progress_dir/(tag+'_evaluation_result.json'), method_record)
            print(case, sigma, layer, method, path['status'],
                  'candidates', len(evaluation['candidates']),
                  'seconds', round(path['elapsed_s'], 2), flush=True)
    complete_candidates = [c for c in frontier_candidates if c['method_path_complete']]
    record['common_validation_pareto'] = {
        'dataset_id': dataset_id, 'partial_methods_excluded': True,
        'points': [{k: c[k] for k in ['layer', 'method', 'candidate_id', 'support', 'complexity', 'validation_error']}
                   for c in pareto_front(complete_candidates)]}
    # This is the first point at which true coefficients/support are accessed.
    true_start = time.perf_counter()
    for evaluation in truth_pending:
        _add_truth_metrics(evaluation, CASES[case]['true'])
    record['truth'] = CASES[case]['true']
    record['time_s']['truth_evaluation'] = time.perf_counter()-true_start
    record['time_s']['dataset_total'] = time.perf_counter()-dataset_start
    record['all_methods_numerically_complete'] = all(
        method['path']['all_solutions_certified']
        for layer in record['layers'].values() for method in layer['methods'].values())
    return record


def _self_check():
    # Verify evaluation wiring independently of numerical optimization and PDE data.
    a = np.eye(2)
    b = np.array([1., 0.])
    points = [{'status': 'certified', 'ratio': .1, 'lambda': .1, 'beta': [1., 2e-5], 'support': [0, 1]},
              {'status': 'certified', 'ratio': .01, 'lambda': .01, 'beta': [1., 0.], 'support': [0]}]
    complete = {'points': points, 'all_solutions_certified': True}
    ev = _candidate_records(complete, a, b, a, b, ['u', 'u_x'], [1, 2], 'common', [(4, a, b)])
    checks = {'ls_zero_does_not_delete_L1_support': ev['candidates'][0]['support'] == ['u', 'u_x']
              and ev['candidates'][0]['coefficients'][1] == 0.,
              'fixed_score_selects_simpler_equal_error': ev['primary_selected_candidate_id'] == 1,
              'pareto_uses_same_validation_data': ev['pareto_candidate_ids'] == [1],
              'truth_absent_until_selection_complete': all('truth_evaluation_after_selection' not in c for c in ev['candidates'])}
    _add_truth_metrics(ev, {'u': 1.})
    checks['support_failure_not_hidden_by_zero_LS_coefficient'] = not ev['candidates'][0]['truth_evaluation_after_selection']['exact']
    partial = _candidate_records({**complete, 'all_solutions_certified': False},
                                a, b, a, b, ['u', 'u_x'], [1, 2], 'common', [(4, a, b)])
    checks['partial_run_has_no_primary_selection'] = partial['primary_selected_candidate_id'] is None
    result = {'scope': 'baseline evaluation API synthetic self-check; no PDE baseline executed',
              'checks': checks, 'all_checks_passed': all(checks.values())}
    strict_json_write(ROOT/'stage2a_baseline_check_result.json', result)
    print(json.dumps(result, indent=2))
    if not result['all_checks_passed']:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', nargs='+', choices=list(CASES), default=list(CASES))
    parser.add_argument('--noise', nargs='+', type=float, default=[0., .01, .05])
    parser.add_argument('--layers', nargs='+', choices=['J18', 'J1158'], default=['J18', 'J1158'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--l1-budget', type=float, default=120.)
    parser.add_argument('--irl1-budget', type=float, default=300.)
    parser.add_argument('--n-lambda', type=int, default=len(DEFAULT_RATIOS))
    parser.add_argument('--output', type=Path, default=ROOT/'stage2a_baseline_result.json')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        _self_check()
        return
    for gate in ['stage0_gate', 'stage1_gate', 'stage2_identifiability', 'stage2a_l1_paths']:
        result = json.loads((ROOT/(gate+'_result.json')).read_text(encoding='utf-8'))
        if not result['all_checks_passed']:
            raise RuntimeError(gate+' gate has not passed; baseline not started.')
    if args.n_lambda < 2:
        raise ValueError('At least two lambda grid points are required.')
    ratios = np.geomspace(1., 1e-6, args.n_lambda)
    sources = ['stage2a_baseline.py', 'stage2a_l1_paths.py', 'stage0_l1_solver.py',
               'stage0_evaluation.py', 'stage2_identifiability.py', 'stage2_data.py',
               'stage1_grammar.py', 'stage1_weak_reduction.py', 'stage1_jet_preprocess.py']
    report = {'stage': '2a_development_baseline', 'python': platform.python_version(),
              'numpy': np.__version__, 'noise': args.noise, 'seed': args.seed, 'layers': args.layers,
              'lambda_ratios': ratios.tolist(), 'alpha_frozen': FROZEN_ALPHA,
              'alpha_sensitivity': SENSITIVITY_ALPHAS, 'validation_metric': METRIC,
              'budgets_s': {'L1': args.l1_budget, 'IRL1_original': args.irl1_budget},
              'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
              'test_data_used': False, 'rows': [], 'execution_complete': False, 'all_checks_passed': False,
              'limits': ['One development noise seed; no statistical robustness ranking or confidence interval.',
                         'Validation equation residuals are not trajectory rollout errors.',
                         'Fixed alpha sensitivity is descriptive; true support does not choose alpha.',
                         'Timeout/uncertified paths remain visible and are ineligible for primary model selection.',
                         'Standard L1 and original IRL1 only; targeted method changes have not been introduced.']}
    progress_dir = ROOT/'stage2a_progress'/args.output.stem
    progress_dir.mkdir(parents=True, exist_ok=True)
    strict_json_write(args.output, report)
    with threadpool_limits(limits=1):
        for case in args.cases:
            for sigma in args.noise:
                row = run_dataset(case, sigma, args.seed, args.layers, ratios,
                                  args.l1_budget, args.irl1_budget, progress_dir)
                report['rows'].append(row)
                strict_json_write(args.output, report)
    report['execution_complete'] = True
    report['all_checks_passed'] = all(row['all_methods_numerically_complete'] for row in report['rows'])
    report['completion_meaning'] = 'All numerical paths complete, not necessarily correct PDE supports.'
    strict_json_write(args.output, report)
    print('execution_complete:', report['execution_complete'], 'all_numerically_complete:', report['all_checks_passed'])


if __name__ == '__main__':
    main()
