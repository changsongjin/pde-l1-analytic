"""Resumable development pilot of all 1212 declared candidates (wave +u_t).

Fixed L1/original IRL1 and the same alpha/grid/support threshold. No final-test
observations, adaptive weight changes, or post-LS deletion. Each initial path is
budgeted separately; failed points remain visible for numerical continuation.
"""
import argparse
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

import stage1_extended_evaluation as E
from stage0_evaluation import array_hash, normalize_training, physical_coefficients, strict_json_write
from stage1_grammar import nonpolynomial_recipes
from stage2_data import CASES
from stage2_identifiability import stack_split
from stage2a_baseline import _candidate_records, FROZEN_ALPHA, SENSITIVITY_ALPHAS, METRIC
from stage2a_l1_paths import DEFAULT_RATIOS, solve_l1_path, solve_original_irl1
from stage2b_compare import validate_path
from stage2c_extended_matrices import save_matrices

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT/'stage2c_maximum_library_pilot_result.json'
PROGRESS = ROOT/'stage2c_progress'
METHOD_BUDGET = 120.
NOISE_LEVELS = [0., .01, .05]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


@lru_cache(maxsize=2)
def feature_metadata(names):
    return [E.canonical_term(n) for n in names], [E.term_complexity(n) for n in names]


def truth_after_selection(evaluation, truth):
    for c in evaluation['candidates']:
        coefficients = dict(zip(c['support'], c['coefficients']))
        c['truth_evaluation_after_selection'] = {**E.support_metrics(c['support'], truth),
                                                **E.coefficient_errors(coefficients, truth)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-seconds', type=float, default=2400.)
    args = parser.parse_args()
    if not 150 <= args.max_seconds <= 3600:
        raise ValueError('Use a bounded 150..3600 second batch; whole user batch is capped at two hours.')
    gate_names = ['stage1_nonpolynomial_result.json', 'stage1_nonpolynomial_curved_check_result.json',
                  'stage1_extended_evaluation_result.json', 'stage2c_extended_matrices_result.json',
                  'stage2a_batch_checkpoint_v2_result.json']
    for name in gate_names:
        if not read(ROOT/name)['all_checks_passed']:
            raise RuntimeError('Required gate not passed: '+name)
    files = ['stage2c_maximum_library_pilot.py', 'stage2c_extended_matrices.py',
        'stage1_nonpolynomial.py', 'stage1_extended_evaluation.py', 'stage1_grammar.py',
        'stage2_data.py', 'stage2_identifiability.py', 'stage1_weak_reduction_v2.py',
        'stage2a_baseline.py', 'stage2a_l1_paths.py', 'stage2b_compare.py', 'stage0_l1_solver.py', 'stage0_evaluation.py', 'lasso_exact.py']
    hashes = {name: digest(ROOT/name) for name in files}
    gates = {name: digest(ROOT/name) for name in gate_names}
    if OUTPUT.exists():
        report = read(OUTPUT)
        if report['source_sha256'] != hashes or report['gate_sources'] != gates:
            raise ValueError('Prior campaign source changed; preserve and version before continuing.')
    else:
        report = {'stage': '2c_maximum_declared_library_development', 'source_sha256': hashes,
            'gate_sources': gates, 'frozen_before_new_maximum_library_outcomes_utc': datetime.now(timezone.utc).isoformat(),
            'cases': list(CASES), 'noise': NOISE_LEVELS, 'seed': 0, 'layers': ['J1212'],
            'methods': ['L1', 'IRL1_original'], 'lambda_ratios': DEFAULT_RATIOS.tolist(),
            'alpha_frozen': FROZEN_ALPHA, 'alpha_sensitivity': SENSITIVITY_ALPHAS,
            'method_initial_budget_s': METHOD_BUDGET, 'rows': [], 'batches': [],
            'validation_metric': METRIC, 'execution_complete': False, 'all_checks_passed': False,
            'support_tolerance': 1e-5, 'test_data_used': False, 'statistical_method_changed': False,
            'order': 'all six clean cases before 1 percent, then 5 percent',
            'complexity_rule': 'preserved polynomial costs plus frozen stage1_extended_evaluation function-expression costs',
            'limits': ['Development seed0 only; not formal paired multi-seed noise robustness.',
                'Weak residual selection is unchanged and can prefer dynamically unstable models.',
                'No unsupported dynamic model is deleted from the feature library.',
                'Numerically incomplete paths have no primary selected model.',
                'Finite complete declared grammar: 1158 polynomials+54 weak nonpolynomial columns; wave adds u_t.']}
    PROGRESS.mkdir(exist_ok=True)
    batch = {'index': len(report['batches']), 'wall_limit_s': args.max_seconds,
             'started_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running'}
    report['batches'].append(batch)
    start = time.perf_counter()
    def save():
        batch['elapsed_s'] = time.perf_counter()-start
        recorded = {(r['case'], r['sigma'], method) for r in report['rows']
                    for method in r['layers']['J1212']['methods']}
        report['unstarted_tasks'] = [{'case': c, 'sigma': s, 'method': m} for s in NOISE_LEVELS
            for c in CASES for m in ('L1', 'IRL1_original') if (c, s, m) not in recorded]
        report['numerically_unfinished_tasks'] = [{'case': r['case'], 'sigma': r['sigma'], 'method': m,
            'certified_points': sum(p['status'] == 'certified' for p in value['path']['points'])}
            for r in report['rows'] for m, value in r['layers']['J1212']['methods'].items()
            if not value['path']['all_solutions_certified']]
        strict_json_write(OUTPUT, report)
    save()
    stopped = False
    with threadpool_limits(limits=1):
        for sigma in NOISE_LEVELS:
            for case in CASES:
                matrices, targets, meta = save_matrices(case, sigma)
                recipes = nonpolynomial_recipes()
                if (meta['supplement_recipes'] != recipes or meta['names'] !=
                        meta['old_polynomial_metadata']['names']+[r['name'] for r in recipes]):
                    raise ValueError('Cached feature names/order do not match the frozen catalog.')
                a, b = stack_split(matrices, targets, range(4))
                av, bv = stack_split(matrices, targets, range(4, 6))
                names = meta['names']; canonical, costs = feature_metadata(tuple(names))
                if len(canonical) != len(set(canonical)):
                    raise ValueError('Exact canonical duplicate candidate.')
                if costs[:meta['n_polynomial_and_nuisance_columns']] != meta['old_polynomial_metadata']['complexities']:
                    raise ValueError('Original feature complexity changed.')
                an, bn, scales = normalize_training(a, b)
                solver_hash = {'A': array_hash(an), 'b': array_hash(bn)}
                inputs = {'train_matrix': array_hash(a), 'train_target': array_hash(b),
                          'validation_matrix': array_hash(av), 'validation_target': array_hash(bv)}
                row = next((r for r in report['rows'] if (r['case'], r['sigma']) == (case, sigma)), None)
                if row is None:
                    row = {'case': case, 'sigma': sigma, 'seed': 0,
                        'dataset_id': f'{case}:sigma={sigma!r}:seed=0:validation={array_hash(bv)}',
                        'raw_bundle_sha256': meta['raw_bundle_sha256'], 'input_hashes': inputs,
                        'library_names': names, 'canonical_library_names': canonical, 'library_complexities': costs,
                        'splits': {'train_trajectories': [0, 1, 2, 3], 'validation_trajectories': [4, 5], 'test': 'reserved'},
                        'layers': {'J1212': {'names': names, 'complexities': costs, 'n_columns': len(names),
                            'indices_in_maximum_library': list(range(len(names))), 'normalization': scales,
                            'solver_input_hashes': solver_hash, 'methods': {}}},
                        'validation_metric': METRIC, 'truth_used_for_selection': False,
                        'time_s': {'supplement_build_recorded': sum(r['build_seconds'] for r in meta['trajectories'])},
                        'test_data_used': False}
                    report['rows'].append(row)
                if row['input_hashes'] != inputs:
                    raise ValueError('Resumption inputs changed.')
                lib = row['layers']['J1212']
                for method, solve in [('L1', solve_l1_path), ('IRL1_original', solve_original_irl1)]:
                    if method in lib['methods']:
                        continue
                    if time.perf_counter()-start+METHOD_BUDGET+20 > args.max_seconds:
                        stopped = True; break
                    tag = f'{case}_{sigma}_J1212_{method}'.replace('.', 'p')
                    pathfile = PROGRESS/(tag+'_path_result.json')
                    jobfile = PROGRESS/(tag+'_job.json')
                    job = {'case': case, 'sigma': sigma, 'seed': 0, 'method': method, 'layer': 'J1212',
                        'solver_input_hashes': solver_hash, 'raw_bundle_sha256': meta['raw_bundle_sha256'],
                        'source_sha256': hashes, 'gate_sources': gates, 'normalization': scales,
                        'lambda_ratios': DEFAULT_RATIOS.tolist(), 'budget_s': METHOD_BUDGET}
                    if jobfile.exists() and read(jobfile) != job:
                        raise ValueError('Saved task inputs/source mismatch.')
                    if not jobfile.exists():
                        strict_json_write(jobfile, job)
                    if pathfile.exists():
                        from stage2b_compare import interrupted_path
                        path = read(pathfile)
                        if path['status'] == 'running':
                            path = interrupted_path(path)
                    else:
                        path = solve(an, bn, ratios=DEFAULT_RATIOS, budget_s=METHOD_BUDGET, progress_path=pathfile)
                    validate_path(path, an, bn)
                    ev = _candidate_records(path, a, b, av, bv, names, costs, row['dataset_id'],
                                            [(i, matrices[i], targets[i]) for i in (4, 5)])
                    for point in path['points']:
                        if point['beta'] is not None:
                            point['physical_L1_coefficients'] = physical_coefficients(point['beta'], scales).tolist()
                    lib['methods'][method] = {'path': path, 'evaluation': ev,
                        'time_s': {'solver_with_worker_startup': path['elapsed_s']},
                        'path_source_file': str(pathfile.relative_to(ROOT)), 'path_source_sha256': digest(pathfile),
                        'full_end_to_end_timing_available': False}
                    save()
                    print(case, sigma, method, path['status'], round(path['elapsed_s'], 2), flush=True)
                if len(lib['methods']) == 2:
                    for record in lib['methods'].values():
                        truth_after_selection(record['evaluation'], CASES[case]['true'])
                    row['truth'] = CASES[case]['true']
                    row['all_methods_numerically_complete'] = all(m['path']['all_solutions_certified'] for m in lib['methods'].values())
                save()
                if stopped: break
            if stopped: break
    batch['status'] = 'budget_checkpoint' if stopped else 'all_initial_attempts_finished'
    report['execution_complete'] = not stopped and len(report['rows']) == 18 and all(len(r['layers']['J1212']['methods']) == 2 for r in report['rows'])
    report['all_checks_passed'] = report['execution_complete'] and all(r['all_methods_numerically_complete'] for r in report['rows'])
    save()
    print(json.dumps({'execution_complete': report['execution_complete'], 'all_checks_passed': report['all_checks_passed'],
        'unstarted': len(report['unstarted_tasks']), 'numerically_unfinished': report['numerically_unfinished_tasks'],
        'batch_elapsed_s': batch['elapsed_s']}))


if __name__ == '__main__':
    main()
