"""Prepared resumable formal-noise runner; no work occurs on import.

Execution requires --execute, explicit requested datasets and an independently
frozen stage3 protocol. This source is preparation until separately reviewed.
Each method has immutable result/path files; the campaign index contains links,
not copies of every path. Partial paths are never silently solved again.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

import stage1_extended_evaluation as E
from stage0_evaluation import array_hash, normalize_training, pareto_front, strict_json_write
from stage0_l1_solver import certify_l1
from stage1_grammar import LAYERS, enumerate_jet, key, nonpolynomial_recipes
from stage1_nonpolynomial import weak_nonpolynomial_matrix
from stage1_weak_reduction_v2 import weak_reduced_matrix
from stage2_data import CASES
from stage2_identifiability import stack_split
from stage2a_baseline import _candidate_records, FROZEN_ALPHA, SENSITIVITY_ALPHAS
from stage2a_l1_paths import (DEFAULT_RATIOS, IRL1_EPS, IRL1_MAX_ITER, IRL1_RATIO,
                              solve_l1_path, solve_original_irl1)
from stage2b_compare import interrupted_path
from stage2a_retry_remaining_v4 import certify_arbitrary_dual

ROOT = Path(__file__).resolve().parent
DIRECTORY = ROOT/'stage3_campaign'
INDEX = ROOT/'stage3_campaign_result.json'
LIBRARIES = ('J18', 'J1158', 'J1212')
METHODS = ('L1', 'IRL1_original')
SCHEDULE_LIBRARIES = ('J1212', 'J18', 'J1158')
FINALIZATION_RESERVE_S = 300.


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def reject(value):
        raise ValueError('Non-finite JSON constant: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=reject)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    strict_json_write(temporary, value)
    temporary.replace(path)


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT)).replace('\\', '/')


def reference(path):
    return {'file': relative(path), 'sha256': digest(path)}


def check_reference(ref):
    path = (ROOT/ref['file']).resolve()
    if not path.is_relative_to(ROOT) or digest(path) != ref['sha256']:
        raise ValueError('Stored artifact changed or escaped workspace: '+str(path))
    return path


def protocol_check(protocol):
    """Pure metadata check; data gateway separately verifies freeze and hashes."""
    expected = {'cases': list(CASES), 'libraries': list(LIBRARIES), 'methods': list(METHODS),
                'fit_ids': list(range(6)), 'train_ids': list(range(4)), 'validation_ids': [4, 5],
                'test_ids': list(range(6, 12)), 'selection_alpha': FROZEN_ALPHA,
                'support_threshold': E.SUPPORT_TOL, 'lambda_ratios': DEFAULT_RATIOS.tolist(),
                'initial_path_budget_s': 120, 'noise_seeds': list(range(1000, 1020)),
                'noise_levels': [0, .01, .03, .05, .07, .10]}
    for name, value in expected.items():
        if protocol.get(name) != value:
            raise ValueError('Runner/protocol configuration mismatch: '+name)
    method = protocol['method_and_selection']
    if (method['alpha_sensitivity'] != SENSITIVITY_ALPHAS
            or method['original_IRL1']['epsilon'] != IRL1_EPS
            or method['original_IRL1']['outer_relative_lambda'] != IRL1_RATIO
            or method['original_IRL1']['max_outer_iterations'] != IRL1_MAX_ITER):
        raise ValueError('Original method/selection settings changed.')
    if (protocol['execution_and_cost']['finalization_reserve_seconds'] != FINALIZATION_RESERVE_S
            or protocol['max_batch_seconds'] != 7200):
        raise ValueError('Runner batch contract differs from protocol.')
    required = {'stage3_campaign.py', 'stage3_data.py', 'stage3_data_v2.py', 'stage1_extended_evaluation.py',
                'stage2a_baseline.py', 'stage2b_compare.py', 'stage0_l1_solver.py',
                'stage2b_heat_full_spectrum.py', 'stage2b_heat_dynamic_compare.py',
                'stage2a_retry_remaining_v4.py'}
    if not required <= set(protocol['source_sha256']):
        raise ValueError('Frozen protocol is missing direct runner dependencies.')


def requested_datasets(protocol, cases, noise, seeds):
    if (not set(cases) <= set(protocol['cases']) or not set(noise) <= set(protocol['noise_levels'])
            or not set(seeds) <= set(protocol['noise_seeds'])):
        raise ValueError('Requested dataset is outside the frozen grid.')
    result = []
    if 0. in noise:
        if 1000 not in seeds:
            raise ValueError('Deterministic clean datasets use seed1000 only.')
        result.extend((case, 0., 1000) for case in protocol['cases'] if case in cases)
    result.extend((case, float(sigma), seed) for seed in protocol['noise_seeds'] if seed in seeds
                  for sigma in protocol['noise_levels'] if sigma and sigma in noise
                  for case in protocol['cases'] if case in cases)
    return result


def dataset_tag(case, sigma, seed):
    return f'{case}_sigma_{sigma!r}_seed_{seed}'.replace('.', 'p')


def _matrix_metadata(case):
    terms = enumerate_jet()
    small_keys = {key(term) for term in enumerate_jet(*LAYERS['J18'])}
    small = [i for i, term in enumerate(terms) if key(term) in small_keys]
    names = [term['name'] for term in terms]
    if case == 'wave':
        small.append(len(names))
        names.append('u_t')
    polynomial_count = len(names)
    names.extend(r['name'] for r in nonpolynomial_recipes())
    indices = {'J18': small, 'J1158': list(range(polynomial_count)), 'J1212': list(range(len(names)))}
    canonical = [E.canonical_term(name) for name in names]
    if len(set(canonical)) != len(names):
        raise ValueError('Exact canonical duplicate library column.')
    return terms, names, [E.term_complexity(name) for name in names], indices


def saved_matrices(bundle, case, sigma, seed, role, directory, protocol_ref, sources):
    """One observed-field maximum design per trajectory; every library slices it."""
    x, t, fields, raw = bundle
    ids = list(range(6)) if role == 'fit' else list(range(6, 12))
    if len(fields) != 6 or [r['trajectory_id'] for r in raw['trajectories']] != ids:
        raise ValueError('Role-aware bundle has wrong trajectory identities.')
    path = directory/f'{role}_matrices.npz'
    metadata_path = path.with_suffix('.json')
    terms, names, costs, indices = _matrix_metadata(case)
    expected = {'case': case, 'sigma': sigma, 'noise_seed': seed, 'role': role,
                'trajectory_ids': ids, 'raw_bundle_sha256': raw['npz_sha256'],
                'protocol': protocol_ref, 'source_sha256': sources,
                'names': names, 'complexities': costs, 'library_indices': indices}
    started = time.perf_counter()
    if path.exists() or metadata_path.exists():
        if not path.exists() or not metadata_path.exists():
            raise ValueError('Incomplete matrix-cache commit; preserve for explicit recovery.')
        meta = read(metadata_path)
        if any(meta.get(k) != v for k, v in expected.items()) or digest(path) != meta['npz_sha256']:
            raise ValueError('Matrix cache source/data/configuration mismatch.')
        with np.load(path, allow_pickle=False) as arrays:
            matrices = [arrays[f'a_{i}'].copy() for i in range(6)]
            targets = [arrays[f'b_{i}'].copy() for i in range(6)]
        action = 'cache_load'
    else:
        matrices, targets, records = [], [], []
        spec = CASES[case]
        target_order = 2 if case == 'wave' else 1
        for uid, field in zip(ids, fields):
            one_start = time.perf_counter()
            a, b, poly_meta = weak_reduced_matrix(x, t, field, terms,
                target_order=target_order, half=spec['half'])
            if case == 'wave':
                _, ut, _ = weak_reduced_matrix(x, t, field, [terms[0]], target_order=1, half=spec['half'])
                a = np.column_stack([a, ut])
            extra, same_b, extra_meta = weak_nonpolynomial_matrix(x, t, field,
                target_order=target_order, half=spec['half'])
            if (not np.array_equal(b, same_b) or extra.shape != (len(b), 54)
                    or poly_meta['input_observation_hash'] != array_hash(field)
                    or extra_meta['input_observation_hash'] != array_hash(field)
                    or any(poly_meta[k] != extra_meta[k] for k in ('center_x_indices', 'center_t_indices'))):
                raise ValueError('Polynomial and supplement chains use different inputs/targets/centers.')
            polynomial_hash = array_hash(a)
            a = np.column_stack([a, extra])
            matrices.append(a)
            targets.append(b)
            records.append({'trajectory_id': uid, 'raw_hash': array_hash(field),
                'matrix_hash': array_hash(a), 'target_hash': array_hash(b), 'polynomial_hash': polynomial_hash,
                'supplement_hash': array_hash(extra), 'shape': list(a.shape),
                'build_seconds': time.perf_counter()-one_start,
                'center_x_indices': poly_meta['center_x_indices'], 'center_t_indices': poly_meta['center_t_indices']})
        directory.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **{f'a_{i}': a for i, a in enumerate(matrices)},
                                **{f'b_{i}': b for i, b in enumerate(targets)})
        temporary.replace(path)
        meta = {**expected, 'npz_sha256': digest(path), 'trajectories': records,
                'target_order': target_order, 'half': list(spec['half']),
                'data_chain': 'observed raw values; v2 weak-reduced jets plus frozen54 raw nonlinear weak columns',
                'no_candidate_pruning': True, 'test_data_used': role == 'test'}
        atomic_json(metadata_path, meta)
        action = 'build'
    for field, a, b, record in zip(fields, matrices, targets, meta['trajectories']):
        if (a.shape != (len(b), len(names)) or array_hash(field) != record['raw_hash']
                or array_hash(a) != record['matrix_hash'] or array_hash(b) != record['target_hash']):
            raise ValueError('Saved raw/matrix/target contract failed.')
    return matrices, targets, meta, {'action': action, 'seconds': time.perf_counter()-started,
                                     'metadata': reference(metadata_path), 'npz': reference(path)}


def validate_campaign_path(path, an, bn, method):
    """Recheck every claimed success; alternate q never bypasses original KKT."""
    if (path['method'] != method or path['shape'] != list(an.shape)
            or path['support_tolerance'] != E.SUPPORT_TOL
            or path['ratios'] != DEFAULT_RATIOS.tolist() or len(path['points']) != 24):
        raise ValueError('Stored method/input/grid identity changed.')
    final_weights = np.asarray(path['weights'])
    if final_weights.shape != (an.shape[1],) or not np.isfinite(final_weights).all() or np.any(final_weights <= 0):
        raise ValueError('Invalid final weights.')
    reviews = []

    def verify(point, weights, ratio, label):
        if point['ratio'] != ratio:
            raise ValueError('Original lambda ordering changed.')
        expected_lambda = ratio*float(np.max(np.abs(an.T@bn)/weights))
        if point['lambda'] is not None and not np.isclose(point['lambda'], expected_lambda, rtol=2e-13, atol=0):
            raise ValueError('Original fixed-lambda objective changed.')
        if point['status'] != 'certified':
            return
        beta = np.asarray(point['beta'])
        if (beta.shape != (an.shape[1],) or not np.isfinite(beta).all() or point['lambda'] is None
                or point['support'] != np.flatnonzero(np.abs(beta) > E.SUPPORT_TOL).tolist()):
            raise ValueError('Certified point has invalid beta/lambda/support.')
        if 'alternate_dual' in point:
            q = np.asarray(point['alternate_dual'])
            if q.shape != bn.shape or not np.isfinite(q).all() or 'previous_stage0_certificate' not in point:
                raise ValueError('Alternate certificate requires saved finite q and preserved original certificate.')
            original, certificate = certify_arbitrary_dual(an, bn, beta, point['lambda'], weights, q)
            if point['previous_stage0_certificate'] != original:
                raise ValueError('Saved previous stage0 certificate differs from the unchanged primal/objective.')
            exact_keys = ('certificate_kind', 'certified', 'original_stage0_certificate_passed',
                          'exact_dual_feasibility', 'exact_min_dual_slack', 'exact_inequalities_checked',
                          'exact_primal_fraction', 'exact_dual_fraction', 'exact_gap_fraction',
                          'exact_direct_minus_stable_gap_is_zero')
            if any(point['certificate'].get(k) != certificate[k] for k in exact_keys):
                raise ValueError('Saved exact certificate differs from independently verified q evidence.')
            reviews.append({'point': label, 'original_stage0_certificate': original,
                            'alternate_exact_dual_certificate': certificate})
        else:
            certificate = certify_l1(an, bn, beta, point['lambda'], weights)
            reviews.append({'point': label, 'original_stage0_certificate': certificate})
        if not certificate['certified'] or point.get('certificate', {}).get('certified') is not True:
            raise ValueError('Claimed certified point fails independent full-column verification: '+label)

    weights = np.ones(an.shape[1])
    outer = path.get('outer', [])
    if method == 'L1' and (outer or not np.array_equal(path['weights'], weights)):
        raise ValueError('Ordinary L1 must use unit weights and no outer iterations.')
    for i, point in enumerate(outer):
        if point['iteration'] != i or i >= IRL1_MAX_ITER or point['ratio'] != IRL1_RATIO:
            raise ValueError('Original IRL1 outer ordering/settings changed.')
        verify(point, weights, IRL1_RATIO, f'outer/{i}')
        if point['status'] != 'certified':
            if i != len(outer)-1:
                raise ValueError('Cannot advance IRL1 after an uncertified outer solve.')
            break
        beta = np.asarray(point['beta'])
        weights = 1./(np.abs(beta)+IRL1_EPS)
    for i, (point, ratio) in enumerate(zip(path['points'], DEFAULT_RATIOS)):
        verify(point, final_weights, float(ratio), f'final/{i}')
    complete = (all(p['status'] == 'certified' for p in path['points'])
                and all(p['status'] == 'certified' for p in outer))
    if bool(path['all_solutions_certified']) != complete:
        raise ValueError('Path completion flag disagrees with point-level certification.')
    if method == 'IRL1_original' and (complete or any(p['status'] == 'certified' for p in path['points'])):
        if not outer or not all(p['status'] == 'certified' for p in outer) or not np.array_equal(weights, final_weights):
            raise ValueError('Final original-IRL1 weights omit or change an update.')
        supports = [tuple(p['support']) for p in outer]
        first_stable = next((i for i in range(1, len(supports)) if supports[i] == supports[i-1]), None)
        if first_stable is not None:
            if first_stable != len(outer)-1 or path['outer_stop'] != 'support_unchanged_original_rule':
                raise ValueError('Original IRL1 continued beyond or changed support stopping rule.')
        elif len(outer) != IRL1_MAX_ITER or path['outer_stop'] != 'maximum_8_iterations_original_rule':
            raise ValueError('Final path started before original IRL1 stopping criterion.')
    return reviews


def _preserve_certified_points(previous, updated):
    if previous['all_solutions_certified']:
        raise ValueError('Already certified complete paths cannot be replaced.')
    for group in ('outer', 'points'):
        new = updated.get(group, [])
        for i, point in enumerate(previous.get(group, [])):
            if point['status'] == 'certified' and (i >= len(new) or new[i] != point):
                raise ValueError('Continuation changed an originally certified point.')
    if previous.get('path_metadata') is not None and previous['weights'] != updated['weights']:
        raise ValueError('Continuation changed final-path weights after that path began.')


def ingest_continuations(directory, method_dir, record, record_ref, job, an, bn, a, b, av, bv,
                         sliced, targets, names, costs, dataset_id):
    """Ordered immutable requests; only same-objective numerical completion.

    External numerical tooling writes continuations/0001.json, etc. Each request
    binds parent_method_result, protocol, job, input_hashes, solver_input_hashes,
    path_source, incremental_solver_wall_s, and attempt_evidence references.
    No externally supplied evaluation or selected support is trusted.
    """
    requests = sorted((method_dir/'continuations').glob('*.json'))
    for number, request_file in enumerate(requests, 1):
        if request_file.name != f'{number:04}.json':
            raise ValueError('Continuation requests must be contiguous immutable 0001.json, ...')
        request = read(request_file)
        if (request['stage'] != 'stage3_same_objective_continuation'
                or request['parent_method_result'] != record_ref or request['protocol'] != job['protocol']
                or request['job'] != record['job'] or request['input_hashes'] != job['input_hashes']
                or request['solver_input_hashes'] != job['solver_input_hashes']):
            raise ValueError('Continuation provenance/objective differs from original job.')
        check_reference(request['parent_method_result'])
        new_path = read(check_reference(request['path_source']))
        old_path = read(check_reference(record['path_source']))
        output = method_dir/f'reconciled_{number:04}_result.json'
        if output.exists():
            restored = read(output)
            if (restored['continuation_request'] != reference(request_file)
                    or restored['parent_method_result'] != record_ref or restored['path_source'] != request['path_source']):
                raise ValueError('Saved reconciliation provenance changed.')
            record, record_ref = restored, reference(output)
            continue
        if (directory/'selection_manifest.json').exists():
            raise ValueError('Selected/sealed models cannot accept a later continuation.')
        _preserve_certified_points(old_path, new_path)
        wall = float(request['incremental_solver_wall_s'])
        if not math.isfinite(wall) or wall < 0 or not request['attempt_evidence']:
            raise ValueError('Continuation needs finite incremental cost and preserved attempt evidence.')
        for ref in request['attempt_evidence']:
            check_reference(ref)
        started = time.perf_counter()
        reviews = validate_campaign_path(new_path, an, bn, job['method'])
        certificate_seconds = time.perf_counter()-started
        started = time.perf_counter()
        evaluation = _candidate_records(new_path, a, b, av, bv, names, costs, dataset_id,
                                        [(i, sliced[i], targets[i]) for i in (4, 5)])
        updated = {**record, 'parent_method_result': record_ref, 'continuation_request': reference(request_file),
                   'path_source': request['path_source'], 'evaluation': evaluation,
                   'numerically_complete': new_path['all_solutions_certified'], 'certificate_review': reviews,
                   'time_s': {'initial_attempt_history': record['time_s'],
                       'incremental_continuation_solver_wall_s': wall,
                       'historical_solver_wall_s': record['time_s']['historical_solver_wall_s']+wall,
                       'independent_certificate_review': certificate_seconds,
                       'training_LS_validation': time.perf_counter()-started}}
        atomic_json(output, updated)
        record, record_ref = updated, reference(output)
    return record, record_ref


def run_method(directory, case, sigma, seed, layer, method, matrices, targets, meta,
               protocol_ref, sources, deadline):
    indices = meta['library_indices'][layer]
    sliced = [a[:, indices] for a in matrices]
    a, b = stack_split(sliced, targets, range(4))
    av, bv = stack_split(sliced, targets, [4, 5])
    names = [meta['names'][i] for i in indices]
    costs = [meta['complexities'][i] for i in indices]
    an, bn, scales = normalize_training(a, b)
    method_dir = directory/f'{layer}_{method}'
    result_path, path_file, job_file = (method_dir/n for n in ('method_result.json', 'path_result.json', 'job.json'))
    dataset_id = f'{case}:sigma={sigma!r}:seed={seed}:validation={array_hash(bv)}'
    job = {'case': case, 'sigma': sigma, 'noise_seed': seed, 'layer': layer, 'method': method,
           'protocol': protocol_ref, 'source_sha256': sources, 'normalization': scales,
           'input_hashes': {'train_matrix': array_hash(a), 'train_target': array_hash(b),
                            'validation_matrix': array_hash(av), 'validation_target': array_hash(bv)},
           'solver_input_hashes': {'A': array_hash(an), 'b': array_hash(bn)},
           'raw_bundle_sha256': meta['raw_bundle_sha256'], 'names': names, 'complexities': costs,
           'lambda_ratios': DEFAULT_RATIOS.tolist(), 'budget_s': 120.,
           'support_threshold': E.SUPPORT_TOL, 'selection_alpha': FROZEN_ALPHA}
    if job_file.exists() and read(job_file) != job:
        raise ValueError('Saved job differs from requested source/data/objective.')
    if result_path.exists():
        record = read(result_path)
        if record['job'] != reference(job_file):
            raise ValueError('Method result/job checksum mismatch.')
        check_reference(record['path_source'])
        record, ref = ingest_continuations(directory, method_dir, record, reference(result_path),
            job, an, bn, a, b, av, bv, sliced, targets, names, costs, dataset_id)
        return record, ref, False
    if not path_file.exists() and time.perf_counter()+120.+FINALIZATION_RESERVE_S > deadline:
        return None, None, False
    if not job_file.exists():
        atomic_json(job_file, job)
    solver_seconds = 0.
    if path_file.exists():
        path = read(path_file)
        if path['status'] == 'running':
            preserved = method_dir/'interrupted_path_result.json'
            if not preserved.exists():
                atomic_json(preserved, interrupted_path(path))
            path_file, path = preserved, read(preserved)
    else:
        solver = solve_l1_path if method == 'L1' else solve_original_irl1
        path = solver(an, bn, ratios=DEFAULT_RATIOS, budget_s=120., progress_path=path_file)
        solver_seconds = path['elapsed_s']
    verify_start = time.perf_counter()
    reviews = validate_campaign_path(path, an, bn, method)
    certify_seconds = time.perf_counter()-verify_start
    evaluate_start = time.perf_counter()
    evaluation = _candidate_records(path, a, b, av, bv, names, costs, dataset_id,
        [(i, sliced[i], targets[i]) for i in (4, 5)])
    record = {'case': case, 'sigma': sigma, 'noise_seed': seed, 'layer': layer, 'method': method,
              'protocol': protocol_ref, 'job': reference(job_file), 'input_hashes': job['input_hashes'],
              'solver_input_hashes': job['solver_input_hashes'], 'normalization': scales,
              'path_source': reference(path_file), 'evaluation': evaluation,
              'certificate_review': reviews,
              'numerically_complete': path['all_solutions_certified'], 'truth_used': False, 'test_data_used': False,
              'time_s': {'new_solver_call_including_worker_startup': solver_seconds,
                         'recorded_path_total': path['elapsed_s'], 'independent_certificate_review': certify_seconds,
                         'historical_solver_wall_s': path['elapsed_s'],
                         'training_LS_validation': time.perf_counter()-evaluate_start},
              'partial_resume_policy': 'Preserve incomplete path; explicit same-objective continuation and separately reviewed merge required.'}
    atomic_json(result_path, record)
    return record, reference(result_path), True


def seal_selection(directory, identity, records, fit_meta, protocol_ref, protocol, data_api):
    path = directory/'selection_manifest.json'
    if len(records) != 6 or not all(r['numerically_complete'] and r['evaluation']['selection_eligible']
                                   for r, _ in records):
        return None
    models = []
    for record, ref in records:
        evaluation = record['evaluation']
        chosen_id = evaluation['primary_selected_candidate_id']
        chosen = next(c for c in evaluation['candidates'] if c['candidate_id'] == chosen_id)
        source = record['path_source']
        path_record = read(check_reference(source))
        if (len(path_record['points']) != 24 or not path_record['all_solutions_certified']
                or any(p['status'] != 'certified' for p in path_record['points']+path_record.get('outer', []))):
            raise ValueError('Incomplete path cannot seal a selected model.')
        model = {'layer': record['layer'], 'method': record['method'], 'status': 'selected',
                 'numerically_complete': True, 'primary_selected': True,
                 'requested_final_points': 24, 'certified_final_points': 24, 'outer_all_certified': True,
                 'support': chosen['support'], 'physical_coefficients': dict(zip(chosen['support'], chosen['coefficients'])),
                 'trainfit_provenance': {'method_result_file': ref['file'], 'method_result_sha256': ref['sha256'],
                     'path_source_file': source['file'], 'path_source_sha256': source['sha256'],
                     'fit_matrix_metadata_file': relative(directory/'fit_matrices.json'),
                     'fit_matrix_metadata_sha256': digest(directory/'fit_matrices.json'),
                     'candidate_id': chosen_id, 'input_hashes': record['input_hashes'],
                     'normalization': record['normalization'], 'coefficient_fit_split': 'train_ids0_3_only',
                     'selected_path_point_indices': [p['point_index'] for p in evaluation['point_to_candidate']
                                                     if p['candidate_id'] == chosen_id]}}
        model['selected_model_sha256'] = data_api.selected_model_digest(model)
        models.append(model)
    manifest = {'stage': 'stage3_sealed_dataset_selection', 'schema_version': 1, 'sealed': True,
                'dataset': identity, 'protocol': protocol_ref, 'source_sha256': protocol['source_sha256'],
                'fit_bundle_sha256': fit_meta['raw_bundle_sha256'], 'train_ids': [0, 1, 2, 3],
                'validation_ids': [4, 5], 'models': models, 'no_test_data_used': True}
    if path.exists():
        if read(path) != manifest:
            raise ValueError('Immutable selected models would change; test access remains blocked.')
    else:
        atomic_json(path, manifest)
    return path


def evaluate_test(bundle, matrices, targets, meta, selection):
    """Frozen coefficients only; truth and optional dynamics never select a model."""
    import stage2b_heat_full_spectrum as D
    from stage2b_heat_dynamic_compare import _aggregate_dynamic
    x, t, observations, raw = bundle
    names = meta['names']
    initial = {scheme: [D.prepare_initialization(x, u[0], scheme=scheme, boundary_realization=D.BOUNDARY)
                       for u in observations] for scheme in D.SCHEMES} if selection['dataset']['case'] == 'heat' else {}
    rows = []
    for model in selection['models']:
        coefficients = model['physical_coefficients']
        indices = [names.index(name) for name in model['support']]
        values = np.asarray([coefficients[name] for name in model['support']])
        weak = []
        for uid, a, b in zip(range(6, 12), matrices, targets):
            error, target = float(np.linalg.norm(a[:, indices]@values-b)), float(np.linalg.norm(b))
            weak.append({'trajectory_id': uid, 'residual_norm': error, 'target_norm': target,
                         'relative_error': error/target if target else None,
                         'status': 'relative' if target else 'zero_target_absolute_only'})
        errors = [r['relative_error'] for r in weak]
        weak_summary = {'trajectory_errors': weak,
                        'rms_relative_error': math.hypot(*errors)/math.sqrt(6) if all(v is not None for v in errors) else None,
                        'aggregation': 'equal test-trajectory RMS of raw weak relative residuals',
                        'coefficient_refit': False, 'support_unchanged': model['support']}
        dynamics = []
        if not initial:
            dynamics.append({'status': 'unsupported_case', 'reason': 'Only declared heat spectral realization is implemented.',
                             'not_counted_as_statistical_failure': True, 'selection_action': 'none'})
        else:
            for scheme in D.SCHEMES:
                forecasts = []
                for uid, u, prepared in zip(range(6, 12), observations, initial[scheme]):
                    forecast, _ = D.evaluate_operator(coefficients, x, t, u, prepared, boundary_realization=D.BOUNDARY)
                    forecasts.append({'trajectory_id': uid, **forecast})
                dynamics.append({'initialization_scheme': scheme, 'trajectories': forecasts,
                                 'summary': _aggregate_dynamic(forecasts), 'affects_selection': False})
        truth = CASES[selection['dataset']['case']]['true']
        rows.append({'layer': model['layer'], 'method': model['method'],
                     'selected_model_sha256': model['selected_model_sha256'], 'support': model['support'],
                     'physical_coefficients': coefficients, 'heldout_weak': weak_summary, 'dynamics': dynamics,
                     'truth_evaluation_after_selection': {**E.support_metrics(model['support'], truth),
                                                          **E.coefficient_errors(coefficients, truth)}})
    return {'stage': '3_frozen_model_test_evaluation', 'dataset': selection['dataset'],
            'protocol': selection['protocol'], 'test_bundle_sha256': raw['npz_sha256'],
            'test_trajectory_ids': list(range(6, 12)), 'rows': rows,
            'no_test_refit_or_selection': True, 'all_selected_supports_preserved': True,
            'uncertainty_scope': 'One noise-seed bundle, conditional on fixed train/validation and six test ICs.',
            'dynamic_scope': 'Subset-specific all-resolved-mode diagnostic, not a general PDE rollout ranking.'}


def run_dataset(case, sigma, seed, protocol, protocol_ref, data_api, deadline, checkpoint):
    identity = {'case': case, 'sigma': sigma, 'noise_seed': seed}
    directory = DIRECTORY/dataset_tag(case, sigma, seed)
    result_path = directory/'dataset_result.json'
    if result_path.exists():
        saved = read(result_path)
        if saved['dataset'] != identity or saved['protocol'] != protocol_ref:
            raise ValueError('Saved dataset belongs to another protocol.')
        for ref in saved['artifacts']:
            check_reference(ref)
        return reference(result_path), 'complete'
    if time.perf_counter()+120.+FINALIZATION_RESERVE_S > deadline:
        return None, 'batch_budget_checkpoint'
    generation_start = time.perf_counter()
    fit_bundle = data_api.get_fit_bundle(case, sigma, seed)
    generation_seconds = time.perf_counter()-generation_start
    matrices, targets, meta, cache = saved_matrices(fit_bundle, case, sigma, seed, 'fit', directory,
                                                   protocol_ref, protocol['source_sha256'])
    records = []
    for layer in SCHEDULE_LIBRARIES:
        for method in METHODS:
            record, ref, new = run_method(directory, case, sigma, seed, layer, method,
                matrices, targets, meta, protocol_ref, protocol['source_sha256'], deadline)
            if record is None:
                return None, 'batch_budget_checkpoint'
            records.append((record, ref))
            checkpoint(identity, layer, method, ref, record['numerically_complete'])
            print(case, sigma, seed, layer, method, 'complete' if record['numerically_complete'] else 'numerically_pending', flush=True)
    manifest_path = seal_selection(directory, identity, records, meta, protocol_ref, protocol, data_api)
    if manifest_path is None:
        return None, 'numerically_pending_no_test_access'
    if time.perf_counter()+FINALIZATION_RESERVE_S > deadline:
        return None, 'sealed_waiting_test_budget'
    # Only this guarded call can produce test trajectories. No truth is passed
    # to the selector, sealing function, or data gateway.
    test_start = time.perf_counter()
    test_bundle = data_api.get_test_bundle(case, sigma, seed, manifest_path)
    test_generation_seconds = time.perf_counter()-test_start
    test_matrices, test_targets, test_meta, test_cache = saved_matrices(test_bundle, case, sigma, seed,
        'test', directory, protocol_ref, protocol['source_sha256'])
    evaluation_start = time.perf_counter()
    selection = read(manifest_path)
    evaluation = evaluate_test(test_bundle, test_matrices, test_targets, test_meta, selection)
    evaluation['selection_manifest'] = reference(manifest_path)
    evaluation_path = directory/'test_evaluation_result.json'
    if evaluation_path.exists():
        if read(evaluation_path) != evaluation:
            raise ValueError('Frozen test evaluation changed; preserve the prior evidence.')
    else:
        atomic_json(evaluation_path, evaluation)
    eligible = [{**candidate, 'layer': record['layer'], 'method': record['method']}
                for record, _ in records for candidate in record['evaluation']['candidates']]
    result = {'stage': '3_completed_dataset', 'dataset': identity, 'protocol': protocol_ref,
              'status': 'complete', 'selection_manifest': reference(manifest_path),
              'method_results': [ref for _, ref in records], 'test_evaluation': reference(evaluation_path),
              'common_validation_pareto': [{k: c[k] for k in ('layer', 'method', 'candidate_id', 'support', 'complexity', 'validation_error')}
                                           for c in pareto_front(eligible)],
              'artifacts': [reference(manifest_path), reference(evaluation_path), cache['metadata'], cache['npz'],
                            test_cache['metadata'], test_cache['npz'], *[ref for _, ref in records],
                            *[r['job'] for r, _ in records], *[r['path_source'] for r, _ in records]],
              'time_s': {'fit_raw_generate_or_load_this_call': generation_seconds, 'fit_matrix': cache,
                         'test_raw_generate_or_load_this_call': test_generation_seconds, 'test_matrix': test_cache,
                         'test_evaluation': time.perf_counter()-evaluation_start},
              'all_six_paths_certified_before_test': True, 'test_never_changed_models': True,
              'support_recovery_is_not_a_runtime_pass_condition': True}
    atomic_json(result_path, result)
    return reference(result_path), 'complete'


def campaign_main(args):
    import stage3_data_v2 as DATA
    protocol = DATA.load_frozen_protocol()
    protocol_check(protocol)
    if not 450 <= args.max_seconds <= protocol['execution_and_cost']['recommended_scheduler_cutoff_seconds']:
        raise ValueError('Batch scheduler budget must be 450..6600 seconds; caller also enforces the user two-hour wall limit.')
    tasks = requested_datasets(protocol, args.cases, args.noise, args.seeds)
    protocol_ref = reference(DATA.PROTOCOL_PATH)
    if INDEX.exists():
        index = read(INDEX)
        if index['protocol'] != protocol_ref or index['source_sha256'] != protocol['source_sha256']:
            raise ValueError('Campaign resume attempted with changed source/protocol.')
    else:
        index = {'stage': '3_systematic_noise_campaign', 'protocol': protocol_ref,
                 'source_sha256': protocol['source_sha256'], 'datasets': {}, 'batches': [],
                 'planned_dataset_count': protocol['planned_grid']['dataset_count'],
                 'planned_method_count': protocol['planned_grid']['method_run_count'],
                 'scope': 'A resumable index; all statistical claims require complete frozen-grid review.'}
    batch = {'started_utc': datetime.now(timezone.utc).isoformat(), 'scheduler_budget_s': args.max_seconds,
             'requested_datasets': [{'case': c, 'sigma': s, 'noise_seed': n} for c, s, n in tasks],
             'status': 'running', 'single_numerical_worker': True}
    index['batches'].append(batch)
    started = time.perf_counter()
    deadline = started+args.max_seconds

    def save():
        batch['elapsed_s'] = time.perf_counter()-started
        index['execution_complete'] = (len(index['datasets']) == index['planned_dataset_count']
            and all(r.get('status') == 'complete' for r in index['datasets'].values()))
        index['all_checks_passed'] = index['execution_complete']
        index['all_checks_passed_semantics'] = 'All planned methods certified and heldout evaluations recorded; not all supports correct or publication-ready.'
        atomic_json(INDEX, index)

    def checkpoint(identity, layer, method, ref, complete):
        tag = dataset_tag(identity['case'], identity['sigma'], identity['noise_seed'])
        row = index['datasets'].setdefault(tag, {'dataset': identity, 'methods': {}, 'status': 'in_progress'})
        row['methods'][f'{layer}/{method}'] = {'result': ref, 'numerically_complete': complete}
        save()

    save()
    try:
        with threadpool_limits(limits=1):
            for case, sigma, seed in tasks:
                tag = dataset_tag(case, sigma, seed)
                row = index['datasets'].setdefault(tag, {'dataset': {'case': case, 'sigma': sigma, 'noise_seed': seed}, 'methods': {}})
                for value in row['methods'].values():
                    check_reference(value['result'])
                result, status = run_dataset(case, sigma, seed, protocol, protocol_ref, DATA, deadline, checkpoint)
                row.update({'status': status, 'result': result})
                save()
                if status in ('batch_budget_checkpoint', 'sealed_waiting_test_budget'):
                    break
        batch['status'] = 'checkpoint_saved'
    except Exception as error:
        batch['status'] = 'stopped_on_contract_error'
        batch['error'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--cases', nargs='+', choices=list(CASES), required=True)
    parser.add_argument('--noise', nargs='+', type=float, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    parser.add_argument('--max-seconds', type=float, required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.execute:
        raise RuntimeError('Preparation only: --execute plus an independently frozen protocol is required.')
    if not math.isfinite(args.max_seconds) or not 450 <= args.max_seconds <= 6600:
        raise ValueError('Scheduler budget must be 450..6600s; default supervisor adds a 300s final reserve.')
    if args.worker:
        campaign_main(args)
        return
    if os.name != 'nt':
        raise RuntimeError('This prepared process-tree watchdog is explicitly Windows-only.')
    # The hidden supervisor owns this child PID and kills only that process tree
    # on a deadline/interrupt. This includes a still-running numerical grandchild.
    # No background helper may survive and compete with the next user batch.
    hard_limit = args.max_seconds+FINALIZATION_RESERVE_S
    started = time.perf_counter()
    child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--worker'],
                             cwd=ROOT, creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        code = child.wait(timeout=hard_limit)
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
        if child.poll() is None:
            subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=15,
                           creationflags=subprocess.CREATE_NO_WINDOW)
            child.wait(timeout=10)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        atomic_json(ROOT/f'stage3_campaign_watchdog_{stamp}_result.json', {
            'status': 'hard_deadline_checkpoint' if isinstance(error, subprocess.TimeoutExpired) else 'user_interrupt_checkpoint',
            'owned_child_pid': child.pid, 'hard_limit_s': hard_limit,
            'elapsed_s': time.perf_counter()-started, 'child_tree_terminated': child.poll() is not None,
            'resume_policy': 'Latest atomic snapshots remain authoritative; no automatic rerun of unfinished paths.'})
        raise SystemExit(124)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
