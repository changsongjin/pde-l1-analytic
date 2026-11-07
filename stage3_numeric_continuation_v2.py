"""Version 2: preserve the frozen runner array layout.

External continuation adapter for the frozen runner's documented interface.

Calls the already frozen 120-second fixed-weight rescue primitive. This driver
does not select models, read test observations, change source hashes, or solve
unknown IRL1 outer dependencies. Attempt evidence binds this adapter's bytes.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

import stage3_campaign as C
import stage3_data_v2 as DATA
from stage0_evaluation import array_hash, normalize_training
from stage2_identifiability import stack_split
from stage2b_numerical_retry import bounded, POINT_BUDGET

ROOT = Path(__file__).resolve().parent


def point_from_solution(ratio, lam, solution, wall):
    return {'ratio':ratio, 'lambda':lam, 'status':'certified',
        'beta':solution['beta'], 'support':solution['support'], 'certificate':solution['certificate'],
        'attempts':solution['attempts'], 'elapsed_s':wall, 'error':None,
        **{k:solution[k] for k in ['alternate_dual','previous_stage0_certificate','certificate_kind'] if k in solution}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--batch-token', required=True)
    parser.add_argument('--max-seconds', required=True, type=float)
    args = parser.parse_args()
    if not args.execute or not 450 <= args.max_seconds <= 6600:
        raise ValueError('Explicit execution and 450..6600s outer caller budget required.')
    if not args.batch_token.replace('_', '').replace('-', '').isalnum():
        raise ValueError('Batch token must contain letters, digits, underscore or dash only.')
    protocol = DATA.load_frozen_protocol()
    C.protocol_check(protocol)
    index = C.read(C.INDEX)
    if index['batches'][-1]['status'] == 'running':
        raise RuntimeError('Stop the numerical campaign before continuation.')
    if POINT_BUDGET != protocol['retry_point_budget_s']:
        raise ValueError('Frozen per-point budget differs.')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT / f'stage3_numeric_continuation_{stamp}_result.json'
    start = time.perf_counter()
    report = {'stage':'3_external_same_objective_continuation', 'batch_token':args.batch_token,
        'protocol':C.reference(DATA.PROTOCOL_PATH), 'source':C.reference(Path(__file__)),
        'frozen_rescue_primitive':C.reference(ROOT/'stage2b_numerical_retry.py'),
        'input_index':C.reference(C.INDEX), 'rows':[], 'execution_complete':False,
        'statistical_method_changed':False, 'test_data_read':False,
        'scope':'Numerical completion only. The unchanged runner independently verifies and selects after ingesting each immutable request.'}

    def save():
        report['elapsed_s'] = time.perf_counter() - start
        C.atomic_json(output, report)

    save()
    with threadpool_limits(limits=1):
        for tag, dataset in index['datasets'].items():
            directory = C.DIRECTORY / tag
            for value in dataset['methods'].values():
                if value['numerically_complete']:
                    continue
                if time.perf_counter() - start + POINT_BUDGET + 300 > args.max_seconds:
                    report['stop_reason'] = 'batch_budget_checkpoint'
                    save()
                    print(str(output.relative_to(ROOT)), flush=True)
                    return
                if (directory/'selection_manifest.json').exists():
                    raise ValueError('No numerical continuation after model sealing.')
                record = C.read(C.check_reference(value['result']))
                job = C.read(C.check_reference(record['job']))
                original = C.read(C.check_reference(record['path_source']))
                method_dir = C.check_reference(record['job']).parent
                requests = sorted((method_dir/'continuations').glob('*.json'))
                if requests and C.read(requests[-1])['path_source'] != record['path_source']:
                    raise ValueError('Ingest outstanding requests through the frozen campaign first.')
                if original.get('path_metadata') is None or any(p['status'] != 'certified' for p in original.get('outer', [])):
                    report['rows'].append({'job':record['job'], 'status':'pending_outer_dependency_not_guessed'})
                    save()
                    continue
                meta = C.read(directory/'fit_matrices.json')
                matrix_file = directory/'fit_matrices.npz'
                if C.digest(matrix_file) != meta['npz_sha256']:
                    raise ValueError('Saved matrix bytes changed.')
                js = meta['library_indices'][record['layer']]
                with np.load(matrix_file, allow_pickle=False) as arrays:
                    raw_matrices = [arrays[f'a_{i}'].copy() for i in range(6)]
                    aa = [a[:, js] for a in raw_matrices]
                    bb = [arrays[f'b_{i}'].copy() for i in range(6)]
                a, b = stack_split(aa, bb, range(4))
                an, bn, scales = normalize_training(a, b)
                if (scales != job['normalization'] or job['source_sha256'] != protocol['source_sha256']
                        or job['solver_input_hashes'] != {'A':array_hash(an), 'b':array_hash(bn)}):
                    raise ValueError('Frozen fixed-weight solver coordinates changed.')
                C.validate_campaign_path(original, an, bn, record['method'])
                weights = np.asarray(original['weights'])
                lmax = float(np.max(np.abs(an.T @ bn)/weights))
                path = copy.deepcopy(original)
                evidence_dir = method_dir/'numeric_evidence'/stamp
                evidence_dir.mkdir(parents=True, exist_ok=False)
                row = {'job':record['job'], 'parent_method_result':value['result'],
                       'parent_path':record['path_source'], 'attempts':[], 'attempt_evidence':[]}
                report['rows'].append(row)
                wall = 0.
                # Recover evidence from an interrupted adapter that had not yet
                # committed a request. Earlier ingested costs have a different
                # parent path and therefore cannot be counted a second time.
                unfinished_costs = []
                for prior_file in (method_dir/'numeric_evidence').glob('*/attempt_*.json'):
                    prior = C.read(prior_file)
                    if prior.get('parent_path') != record['path_source']:
                        continue
                    row['attempt_evidence'].append(C.reference(prior_file))
                    wall += float(prior.get('wall_seconds', 0.))
                    row['attempts'].append({'point_index':prior['point_index'], 'status':prior['status'],
                        'wall_seconds':prior.get('wall_seconds'), 'inherited_uncommitted_attempt':True})
                    solution = prior.get('returned_solution')
                    i = prior['point_index']
                    if solution and solution['certificate']['certified'] and path['points'][i]['status'] != 'certified':
                        path['points'][i] = point_from_solution(prior['ratio'], prior['lambda'], solution, prior['wall_seconds'])
                    if prior['status'] == 'running' or 'wall_seconds' not in prior:
                        unfinished_costs.append(C.reference(prior_file))
                row['interrupted_cost_records_are_lower_bounds'] = unfinished_costs
                for i, point in enumerate(path['points']):
                    if point['status'] == 'certified':
                        continue
                    # The cap applies across adapter calls carrying the same user batch token.
                    previous = []
                    for old_file in (method_dir/'numeric_evidence').glob('*/attempt_*.json'):
                        old = C.read(old_file)
                        if old['batch_token'] == args.batch_token and old['point_index'] == i:
                            previous.append(old)
                    for trial in range(len(previous), 2):
                        if time.perf_counter() - start + POINT_BUDGET + 300 > args.max_seconds:
                            break
                        lam = float(point['ratio'] * lmax)
                        attempt = {'batch_token':args.batch_token, 'point_index':i, 'trial_in_batch':trial+1,
                            'ratio':point['ratio'], 'lambda':lam, 'parent_path':record['path_source'],
                            'adapter_source':report['source'], 'frozen_primitive':report['frozen_rescue_primitive']}
                        file = evidence_dir/f'attempt_{i:02}_{trial+1}.json'

                        def save_attempt():
                            C.atomic_json(file, attempt)
                            save()

                        solution = bounded(an, bn, lam, weights, attempt, save_attempt)
                        attempt['returned_solution'] = solution
                        save_attempt()
                        wall += attempt['wall_seconds']
                        row['attempts'].append({'point_index':i, 'status':attempt['status'], 'wall_seconds':attempt['wall_seconds']})
                        row['attempt_evidence'].append(C.reference(file))
                        if solution is not None and solution['certificate']['certified']:
                            path['points'][i] = point_from_solution(point['ratio'], lam, solution, attempt['wall_seconds'])
                            break
                        save()
                if not row['attempt_evidence']:
                    row['status'] = 'no_new_attempt_budget_or_cap'
                    save()
                    continue
                path['elapsed_s'] = original['elapsed_s'] + wall
                path['all_solutions_certified'] = all(p['status']=='certified' for p in path['points'] + path.get('outer', []))
                path['status'] = 'complete' if path['all_solutions_certified'] else 'partial_failure'
                C._preserve_certified_points(original, path)
                C.validate_campaign_path(path, an, bn, record['method'])
                path_file = evidence_dir/'continued_path_result.json'
                C.atomic_json(path_file, path)
                request = {'stage':'stage3_same_objective_continuation', 'parent_method_result':value['result'],
                    'protocol':job['protocol'], 'job':record['job'], 'input_hashes':job['input_hashes'],
                    'solver_input_hashes':job['solver_input_hashes'], 'path_source':C.reference(path_file),
                    'incremental_solver_wall_s':wall, 'attempt_evidence':row['attempt_evidence'] + [report['source']],
                    'interrupted_cost_records_are_lower_bounds':unfinished_costs}
                request_file = method_dir/'continuations'/f'{len(requests)+1:04}.json'
                if request_file.exists():
                    raise FileExistsError('Continuation request is immutable.')
                C.atomic_json(request_file, request)
                row.update(status='request_created', request=C.reference(request_file), complete=path['all_solutions_certified'])
                save()
    report['execution_complete'] = True
    report['all_attempted_paths_complete'] = all(r.get('complete', False) for r in report['rows'])
    save()
    print(str(output.relative_to(ROOT)), flush=True)


if __name__ == '__main__':
    main()
