"""Budgeted L1/unaltered-original-IRL1 candidate paths for later stage 2a.

One full-matrix LARS path per fixed weight vector, independent certification at
every requested lambda, fixed-lambda convex QP fallback only when needed.
This file's main runs API verification, not the later research baseline.
Windows callers must use an if __name__ == '__main__' entry-point guard.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import platform
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from lasso_exact import lars_lasso_path, lasso_at
from stage0_l1_solver import SUPPORT_TOL, _inputs, certify_l1, _active_set_qp

DEFAULT_RATIOS = np.geomspace(1., 1e-6, 24)
IRL1_RATIO, IRL1_EPS, IRL1_MAX_ITER = 1e-3, 1e-8, 8


def _path_points(a, b, weights, ratios, send):
    start = time.perf_counter()
    lambda_max = float(np.max(np.abs(a.T@b)/weights))
    if lambda_max <= 0:
        raise ValueError('lambda_max is zero; a relative-lambda path is undefined.')
    limit = max(500, 4*min(a.shape))
    knots = lars_lasso_path(a, b, weights, lam_min_ratio=float(min(ratios)), max_steps=limit)
    metadata = {'lambda_max': lambda_max, 'knots': len(knots), 'max_steps': limit,
                'last_lambda': float(knots[-1][0]), 'path_build_s': time.perf_counter()-start,
                'weights_min': float(weights.min()), 'weights_max': float(weights.max())}
    send('path_metadata', metadata)
    for ratio in ratios:
        point_start = time.perf_counter()
        lam = float(ratio*lambda_max)
        attempts = []
        beta = lasso_at(knots, lam)
        certificate = certify_l1(a, b, beta, lam, weights)
        attempts.append({'solver': 'shared_LARS_path', 'certificate': certificate})
        error = None
        if not certificate['certified']:
            try:
                beta, details = _active_set_qp(a, b, lam, weights)
                certificate = certify_l1(a, b, beta, lam, weights)
                attempts.append({'solver': 'signed_bound_QP_SVD', **details, 'certificate': certificate})
            except (RuntimeError, np.linalg.LinAlgError) as exc:
                error = str(exc)
                attempts.append({'solver': 'signed_bound_QP_SVD', 'error': error})
        row = {'ratio': float(ratio), 'lambda': lam,
               'status': 'certified' if certificate['certified'] else 'uncertified',
               'beta': beta.tolist(),
               'support': np.flatnonzero(np.abs(beta) > SUPPORT_TOL).tolist(),
               'certificate': certificate, 'attempts': attempts,
               'elapsed_s': float(time.perf_counter()-point_start), 'error': error}
        send('point', row)
        yield row


def _worker(conn, method, a, b, weights, ratios):
    def send(kind, value):
        conn.send((kind, value))
    try:
        with threadpool_limits(limits=1):
            if method == 'IRL1_original':
                previous_support = None
                for iteration in range(IRL1_MAX_ITER):
                    # For a fixed weight vector, the outer iteration needs one lambda.
                    messages = []
                    outer = list(_path_points(a, b, weights, [IRL1_RATIO],
                                              lambda kind, value: messages.append((kind, value))))[0]
                    outer['iteration'] = iteration
                    outer['path_metadata'] = messages[0][1]
                    send('outer', outer)
                    if outer['status'] != 'certified':
                        raise RuntimeError('IRL1 outer fixed-weight solution is uncertified.')
                    beta = np.asarray(outer['beta'])
                    weights = 1./(np.abs(beta)+IRL1_EPS)
                    send('weights', weights.tolist())
                    support = tuple(outer['support'])
                    if previous_support == support:
                        send('outer_stop', 'support_unchanged_original_rule')
                        break
                    previous_support = support
                else:
                    send('outer_stop', 'maximum_8_iterations_original_rule')
            for _ in _path_points(a, b, weights, ratios, send):
                pass
        send('done', True)
    except Exception as exc:
        send('fatal', {'type': type(exc).__name__, 'message': str(exc)})
    finally:
        conn.close()


def _save_progress(path, report):
    if path is None:
        return
    target = Path(path)
    temp = target.with_suffix(target.suffix+'.tmp')
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(target)


def _run(method, a, b, ratios, weights, budget_s, progress_path):
    a, b, weights = _inputs(a, b, 1., weights)
    ratios = np.asarray(DEFAULT_RATIOS if ratios is None else ratios, dtype=float)
    if (ratios.ndim != 1 or ratios.size == 0 or not np.isfinite(ratios).all()
            or not np.all((ratios > 0) & (ratios <= 1)) or not np.all(np.diff(ratios) < 0)):
        raise ValueError('ratios must be strictly decreasing, finite, and in (0,1].')
    if not np.isfinite(budget_s) or budget_s <= 0:
        raise ValueError('budget_s must be finite and positive.')
    report = {'method': method, 'shape': list(a.shape), 'ratios': ratios.tolist(),
              'weights': weights.tolist(), 'support_tolerance': SUPPORT_TOL,
              'budget_s': float(budget_s), 'status': 'running', 'points': [], 'outer': [],
              'outer_stop': None, 'path_metadata': None,
              'original_IRL1_settings': {'relative_lambda': IRL1_RATIO, 'epsilon': IRL1_EPS,
                                         'max_iterations': IRL1_MAX_ITER,
                                         'stop_rule': 'support unchanged after positive finite weight update'},
              'no_truth_used': True, 'no_column_pruning': True}
    _save_progress(progress_path, report)
    context = mp.get_context('spawn')
    receiving, sending = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(sending, method, a, b, weights, ratios.tolist()))
    start = time.perf_counter()
    process.start()
    sending.close()
    ended = False
    terminal = 'worker_error'
    try:
        while not ended:
            remaining = float(budget_s)-(time.perf_counter()-start)
            if remaining <= 0:
                terminal = 'budget_exhausted'
                break
            if receiving.poll(min(.2, remaining)):
                try:
                    kind, value = receiving.recv()
                except EOFError:
                    report['error'] = 'Worker closed its pipe without a completion message.'
                    break
                if kind == 'point':
                    report['points'].append(value)
                elif kind == 'outer':
                    report['outer'].append(value)
                elif kind == 'fatal':
                    report['error'] = value
                    terminal = 'worker_error'
                    ended = True
                elif kind == 'done':
                    terminal = 'complete' if all(p['status'] == 'certified' for p in report['points']) else 'partial_failure'
                    ended = True
                else:
                    report[kind] = value
                report['elapsed_s'] = float(time.perf_counter()-start)
                _save_progress(progress_path, report)
            elif not process.is_alive():
                report['error'] = 'Worker exited without returning every requested lambda.'
                break
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=5.)
        receiving.close()
    complete_ratios = {p['ratio'] for p in report['points']}
    for ratio in ratios:
        if float(ratio) not in complete_ratios:
            report['points'].append({'ratio': float(ratio), 'lambda': None,
                                     'status': terminal, 'beta': None, 'support': None,
                                     'certificate': None, 'attempts': [],
                                     'error': 'Requested lambda not completed; do not omit from method failure counts.'})
    report['status'] = terminal
    report['elapsed_s'] = float(time.perf_counter()-start)
    report['all_solutions_certified'] = bool(terminal == 'complete' and len(report['points']) == len(ratios))
    report['warning'] = ('IRL1 outer fixed-weight solves are convex; the outer algorithm is nonconvex. '
                         'An unchanged support is the historical stopping rule, not an IRL1 global-optimality certificate.')
    _save_progress(progress_path, report)
    return report


def solve_l1_path(a, b, ratios=None, weights=None, budget_s=120., progress_path=None):
    """Input frozen normalized columns; return JSON-safe candidate beta/support records.

Never silently discard failed lambda points. Reject partial runs from definitive
method rankings or report their failure rate. progress_path writes each received
checkpoint; the wall budget includes subprocess startup and covers all lambdas.
"""
    return _run('L1', a, b, ratios, weights, budget_s, progress_path)


def solve_original_irl1(a, b, ratios=None, budget_s=300., progress_path=None):
    """Preserve the historical IRL1 settings, then scan its final weighted path."""
    return _run('IRL1_original', a, b, ratios, None, budget_s, progress_path)


def main():
    import gml1_maxlib_compare as G
    import noise_inverse as N
    from stage0_l1_solver import solve_l1
    start = time.perf_counter()
    rng = np.random.default_rng(602)
    a = rng.normal(size=(70, 12))
    a /= np.linalg.norm(a, axis=0)
    beta = np.zeros(12)
    beta[[1, 6]] = [1., -.4]
    b = a@beta + .001*rng.normal(size=70)
    ratios = np.array([1., .1, .01, .001, .0001, .000001])
    checks = []
    synthetic = solve_l1_path(a, b, ratios, budget_s=60)
    comparisons = []
    for point in synthetic['points']:
        direct = solve_l1(a, b, point['lambda'])
        delta = float(np.linalg.norm(a@(np.asarray(point['beta'])-direct['beta'])))
        comparisons.append({'ratio': point['ratio'], 'prediction_difference': delta,
                            'passed': delta < 1e-8 and point['certificate']['certified']})
    checks.append({'name': 'shared_path_matches_certified_fixed_solver',
                   'passed': all(x['passed'] for x in comparisons), 'comparisons': comparisons})
    irl1 = solve_original_irl1(a, b, ratios, budget_s=60)
    checks.append({'name': 'original_IRL1_all_outer_and_final_points_certified',
                   'passed': irl1['all_solutions_certified'] and all(p['certificate']['certified'] for p in irl1['outer'])})
    raw, target, _, _ = G.make_data(N.CASES['heat'], .05, 0, G.build_terms(False))
    matrix = raw/np.linalg.norm(raw, axis=0)
    heat = solve_l1_path(matrix, target, ratios, budget_s=60)
    fallback_used = any(len(p['attempts']) > 1 for p in heat['points'])
    checks.append({'name': 'heat_113_low_lambda_QP_fallback',
                   'passed': heat['all_solutions_certified'] and fallback_used})
    budget = solve_l1_path(a, b, ratios, budget_s=.001)
    checks.append({'name': 'wall_budget_records_every_unfinished_lambda',
                   'passed': budget['status'] == 'budget_exhausted'
                   and len(budget['points']) == len(ratios)
                   and not budget['all_solutions_certified']})
    n, p = 700, 1158
    step_limit = max(500, 4*min(n,p))
    report = {'scope': 'stage2a API verification only; large-grammar baseline not started',
              'python': platform.python_version(), 'numpy': np.__version__,
              'checks': checks, 'all_checks_passed': all(c['passed'] for c in checks),
              'synthetic_L1': synthetic, 'synthetic_IRL1': irl1,
              'heat_113': heat, 'budget_control': budget,
              'large_matrix_resource_estimate': {
                  'n': n, 'p': p, 'input_matrix_MiB': 8*n*p/2**20,
                  'worst_retained_knots_MiB': 8*p*(step_limit+1)/2**20,
                  'one_active_gram_MiB': 8*min(n,p)**2/2**20,
                  'recommended_first_L1_call_budget_s': 120,
                  'recommended_first_IRL1_call_budget_s': 300,
                  'runtime_prediction': 'Unmeasured: depends on active dimension/conditioning. A completed path within this budget is not promised.',
                  'process_memory_note': 'NumPy/sklearn imports and copies add overhead; run one worker at a time.'},
              'elapsed_s': time.perf_counter()-start}
    target = Path(__file__).with_name('stage2a_l1_paths_result.json')
    target.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print('all_checks_passed:', report['all_checks_passed'])
    print('heat path time:', heat['elapsed_s'], 'fallback used:', fallback_used)
    print('checks:', checks)


if __name__ == '__main__':
    main()
