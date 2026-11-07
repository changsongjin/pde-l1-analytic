"""Continue frozen maximum-library final-weight points; no statistical changes."""
import argparse
import copy
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from stage0_evaluation import array_hash, normalize_training, strict_json_write
from stage0_l1_solver import certify_l1, SUPPORT_TOL
from stage2_identifiability import stack_split
from stage2b_numerical_retry import bounded, sha, load, POINT_BUDGET
from stage2c_extended_matrices import save_matrices

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-seconds', type=float, default=1200.)
    parser.add_argument('--output', type=Path, default=ROOT/'stage2c_numerical_retry_result.json')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Retry output exists; preserve it and select another version.')
    source_file = ROOT/'stage2c_maximum_library_pilot_result.json'
    source = load(source_file)
    if not source['execution_complete']:
        raise RuntimeError('Initial campaign must finish before retries compete for solver resources.')
    for name, expected in {**source['source_sha256'], **source['gate_sources']}.items():
        if sha(ROOT/name) != expected:
            raise ValueError('Frozen campaign source/gate changed: '+name)
    result = {'stage': '2c_numerical_retry', 'source_file': source_file.name, 'source_sha256': sha(source_file),
        'source_script_sha256': sha(Path(__file__)),
        'helper_sha256': {name: sha(ROOT/name) for name in ['stage2b_numerical_retry.py',
            'stage2a_clarabel_probe.py', 'stage2a_retry_remaining_v3.py', 'stage2a_retry_remaining_v4.py',
            'stage0_l1_solver.py', 'stage2c_extended_matrices.py']},
        'rows': [], 'execution_complete': False, 'all_checks_passed': False,
        'per_point_budget_s': POINT_BUDGET, 'invocation_budget_s': args.max_seconds,
        'statistical_method_changed': False, 'test_data_used': False, 'no_truth_used': True}
    started = time.perf_counter()
    def save():
        result['elapsed_s'] = time.perf_counter()-started
        strict_json_write(args.output, result)
    save(); stopped = False
    with threadpool_limits(limits=1):
        for dataset in source['rows']:
            for method, record in dataset['layers']['J1212']['methods'].items():
                if record['path']['all_solutions_certified']:
                    continue
                original = record['path']
                if (original['path_metadata'] is None or any(o['status'] != 'certified' for o in original.get('outer', []))
                        or (method != 'L1' and original.get('outer_stop') is None)):
                    result['rows'].append({'case': dataset['case'], 'sigma': dataset['sigma'], 'method': method,
                        'status': 'outer_or_path_build_incomplete', 'path': copy.deepcopy(original),
                        'reason': 'Final weights/path not frozen; needs explicit outer continuation.'})
                    continue
                matrices, targets, meta = save_matrices(dataset['case'], dataset['sigma'], 0)
                a, b = stack_split(matrices, targets, range(4)); a, b, scales = normalize_training(a, b)
                expected = dataset['layers']['J1212']['solver_input_hashes']
                if {'A': array_hash(a), 'b': array_hash(b)} != expected or scales != dataset['layers']['J1212']['normalization']:
                    raise ValueError('Frozen solver input/normalization mismatch.')
                source_path = ROOT/record['path_source_file']
                if sha(source_path) != record['path_source_sha256']:
                    raise ValueError('Original task path file changed.')
                weights = np.asarray(original['weights']); lmax = original['path_metadata']['lambda_max']
                if not np.isclose(lmax, np.max(np.abs(a.T@b)/weights), rtol=1e-14, atol=0):
                    raise ValueError('Frozen lambda maximum mismatch.')
                path = copy.deepcopy(original)
                row = {'case': dataset['case'], 'sigma': dataset['sigma'], 'seed': 0, 'method': method,
                    'layer': 'J1212', 'input_hashes': expected, 'raw_bundle_sha256': meta['raw_bundle_sha256'],
                    'source_path_file': str(source_path.relative_to(ROOT)), 'source_path_sha256': sha(source_path),
                    'path': path, 'original_elapsed_s': original['elapsed_s'], 'attempts': [], 'retry_elapsed_s': 0.}
                result['rows'].append(row)
                inherited = []
                for i, point in enumerate(path['points']):
                    if point['status'] == 'certified':
                        assert certify_l1(a, b, np.asarray(point['beta']), point['lambda'], weights)['certified']
                        inherited.append(i); continue
                    if time.perf_counter()-started+POINT_BUDGET+10 > args.max_seconds:
                        stopped = True; break
                    lam = float(point['ratio']*lmax)
                    if point['lambda'] is not None and not np.isclose(point['lambda'], lam, rtol=1e-14, atol=0):
                        raise ValueError('Original lambda changed.')
                    attempt = {'point_index': i, 'ratio': point['ratio'], 'lambda': lam, 'prior_status': point['status']}
                    row['attempts'].append(attempt)
                    solution = bounded(a, b, lam, weights, attempt, save)
                    if solution is not None:
                        assert solution['support'] == np.flatnonzero(np.abs(solution['beta']) > SUPPORT_TOL).tolist()
                        path['points'][i] = {'ratio': point['ratio'], 'lambda': lam,
                            'status': 'certified' if solution['certificate']['certified'] else 'uncertified',
                            'beta': solution['beta'], 'support': solution['support'], 'certificate': solution['certificate'],
                            'attempts': solution['attempts'], 'elapsed_s': attempt['wall_seconds'], 'error': None,
                            **{k: solution[k] for k in ['alternate_dual', 'previous_stage0_certificate', 'certificate_kind'] if k in solution}}
                    else:
                        path['points'][i]['lambda'] = lam
                        path['points'][i]['status'] = attempt['status']
                    row['retry_elapsed_s'] = sum(t['wall_seconds'] for t in row['attempts'])
                    path['elapsed_s'] = row['original_elapsed_s']+row['retry_elapsed_s']
                    path['all_solutions_certified'] = all(p['status'] == 'certified' for p in path['points'])
                    path['status'] = 'complete' if path['all_solutions_certified'] else 'partial_failure'
                    save()
                    print(dataset['case'], dataset['sigma'], method, point['ratio'], attempt['status'], flush=True)
                assert all(path['points'][i] == original['points'][i] for i in inherited)
                row['inherited_certified_points_unchanged'] = len(inherited)
                if stopped: break
            if stopped: break
    result['execution_complete'] = not stopped
    result['all_checks_passed'] = not stopped and all(r['path']['all_solutions_certified'] for r in result['rows'])
    result['original_campaign_unchanged'] = sha(source_file) == result['source_sha256']
    assert result['original_campaign_unchanged']
    save()


if __name__ == '__main__':
    main()
