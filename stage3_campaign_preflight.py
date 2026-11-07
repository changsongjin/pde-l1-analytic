"""Small synthetic contract checks; no PDE data, formal seeds, or solver calls."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np
from threadpoolctl import threadpool_limits

import stage3_campaign as C
from stage0_evaluation import normalize_training
from stage0_l1_solver import certify_l1
from stage2a_retry_remaining_v4 import certify_arbitrary_dual

ROOT = Path(__file__).resolve().parent


def rejects(function):
    try:
        function()
    except (ValueError, RuntimeError, KeyError):
        return True
    return False


def orthogonal_point(a, b, weights, ratio):
    # Closed-form exact solution for this identity-matrix fixture only. This is
    # not a fitting algorithm or a change to the campaign's L1 solver.
    lam = float(ratio*np.max(np.abs(a.T@b)/weights))
    beta = np.sign(b)*np.maximum(np.abs(b)-lam*weights, 0.)
    certificate = certify_l1(a, b, beta, lam, weights)
    assert certificate['certified']
    return {'status': 'certified', 'ratio': float(ratio), 'lambda': lam,
            'beta': beta.tolist(), 'support': np.flatnonzero(np.abs(beta)>1e-5).tolist(),
            'certificate': certificate, 'attempts': [], 'elapsed_s': 0.}


def fixture_path(a, b, method):
    weights = np.ones(a.shape[1])
    outer, previous, stop = [], None, None
    if method == 'IRL1_original':
        for i in range(C.IRL1_MAX_ITER):
            point = orthogonal_point(a, b, weights, C.IRL1_RATIO)
            point['iteration'] = i
            outer.append(point)
            weights = 1./(np.abs(point['beta'])+C.IRL1_EPS)
            if previous == point['support']:
                stop = 'support_unchanged_original_rule'
                break
            previous = point['support']
        else:
            stop = 'maximum_8_iterations_original_rule'
    return {'method': method, 'shape': list(a.shape), 'support_tolerance': 1e-5,
            'ratios': C.DEFAULT_RATIOS.tolist(), 'weights': weights.tolist(),
            'outer': outer, 'outer_stop': stop, 'all_solutions_certified': True, 'status': 'complete',
            'path_metadata': {'lambda_max': float(np.max(np.abs(a.T@b)/weights))},
            'points': [orthogonal_point(a, b, weights, ratio) for ratio in C.DEFAULT_RATIOS], 'elapsed_s': 1.}


def main():
    output = ROOT/'stage3_campaign_preflight_result.json'
    if output.exists():
        raise FileExistsError('Preserve the recorded preflight; use a new version for changes.')
    started = time.perf_counter()
    before = sorted(str(p.relative_to(ROOT)) for p in (ROOT/'stage3_data').glob('**/*.npz'))
    checks = {}
    draft = C.read(ROOT/'stage3_protocol_draft.json')
    checks['protocol_is_still_draft'] = draft['is_frozen'] is False and draft['execution_authorized'] is False
    tasks = C.requested_datasets(draft, draft['cases'], draft['noise_levels'], draft['noise_seeds'])
    checks['grid606_clean_once_and_seed_major'] = (len(tasks) == len(set(tasks)) == 606
        and sum(sigma == 0. for _, sigma, _ in tasks) == 6 and all(seed == 1000 for _, sigma, seed in tasks if sigma == 0.))
    checks['undeclared_request_rejected'] = rejects(lambda: C.requested_datasets(draft, ['heat'], [.02], [1000]))
    checks['clean_duplicate_seed_rejected'] = rejects(lambda: C.requested_datasets(draft, ['heat'], [0.], [1001]))
    a, b = np.eye(3), np.array([2., -.7, .05])
    a, b, scales = normalize_training(a, b)
    names, costs = ['u', 'sin(1u)', 'u_xx'], [1, 2, 3]
    with threadpool_limits(limits=1):
        path = fixture_path(a, b, 'L1')
        checks['all24_original_L1_certificates_rechecked'] = len(C.validate_campaign_path(path, a, b, 'L1')) == 24
        irl1 = fixture_path(a, b, 'IRL1_original')
        checks['original_IRL1_chain_and_stop_rechecked'] = len(C.validate_campaign_path(irl1, a, b, 'IRL1_original')) == 24+len(irl1['outer'])
        altered = copy.deepcopy(irl1)
        altered['weights'][0] *= 1.001
        checks['changed_final_IRL1_weights_rejected'] = rejects(lambda: C.validate_campaign_path(altered, a, b, 'IRL1_original'))
        alternate = copy.deepcopy(path)
        point = alternate['points'][-1]
        beta = np.asarray(point['beta'])
        q = (b-a@beta)*(1.-1e-10)
        old, new = certify_arbitrary_dual(a, b, beta, point['lambda'], np.ones(3), q)
        point.update({'alternate_dual': q.tolist(), 'previous_stage0_certificate': old, 'certificate': new})
        reviews = C.validate_campaign_path(alternate, a, b, 'L1')
        checks['saved_dual_exactly_rechecked'] = reviews[-1]['alternate_exact_dual_certificate']['certified']
        bad = copy.deepcopy(alternate)
        bad['points'][-1]['alternate_dual'] = (2*q).tolist()
        checks['claimed_success_with_infeasible_q_rejected'] = rejects(lambda: C.validate_campaign_path(bad, a, b, 'L1'))
        bad = copy.deepcopy(alternate)
        bad['points'][-1]['previous_stage0_certificate']['certified'] = not old['certified']
        checks['relabelled_previous_certificate_rejected'] = rejects(lambda: C.validate_campaign_path(bad, a, b, 'L1'))
        bad = copy.deepcopy(alternate)
        bad['points'][-1]['certificate']['exact_gap_fraction'] = '0'
        checks['forged_saved_exact_gap_rejected'] = rejects(lambda: C.validate_campaign_path(bad, a, b, 'L1'))
        partial = copy.deepcopy(path)
        partial['points'][-1].update({'status': 'budget_exhausted', 'beta': None, 'support': None, 'certificate': None})
        partial.update({'all_solutions_certified': False, 'status': 'budget_exhausted'})
        C._preserve_certified_points(partial, path)
        changed = copy.deepcopy(path)
        changed['points'][0]['elapsed_s'] = 1.
        checks['every_previously_certified_point_preserved_verbatim'] = rejects(lambda: C._preserve_certified_points(partial, changed))
        with tempfile.TemporaryDirectory(prefix='stage3_campaign_synthetic_', dir=ROOT) as temporary:
            directory = Path(temporary).resolve()
            assert directory.is_relative_to(ROOT.resolve())
            method_dir = directory/'J18_L1'
            def write(file, value):
                C.atomic_json(file, value)
                return C.reference(file)
            protocol_ref = {'file': 'synthetic_only_no_frozen_protocol', 'sha256': 'synthetic'}
            hashes = {'train_matrix': C.array_hash(a), 'train_target': C.array_hash(b),
                      'validation_matrix': C.array_hash(a), 'validation_target': C.array_hash(b)}
            job = {'protocol': protocol_ref, 'method': 'L1', 'input_hashes': hashes,
                   'solver_input_hashes': {'A': C.array_hash(a), 'b': C.array_hash(b)}}
            job_ref = write(method_dir/'job.json', job)
            old_path_ref = write(method_dir/'path_result.json', partial)
            new_path_ref = write(method_dir/'continued_path.json', path)
            partial_ev = C._candidate_records(partial, a, b, a, b, names, costs, 'synthetic', [(4, a, b)])
            record = {'job': job_ref, 'path_source': old_path_ref, 'evaluation': partial_ev,
                      'numerically_complete': False, 'time_s': {'historical_solver_wall_s': 1.}}
            record_ref = write(method_dir/'method_result.json', record)
            request = {'stage': 'stage3_same_objective_continuation', 'parent_method_result': record_ref,
                       'protocol': protocol_ref, 'job': job_ref, 'input_hashes': hashes,
                       'solver_input_hashes': job['solver_input_hashes'], 'path_source': new_path_ref,
                       'incremental_solver_wall_s': .25, 'attempt_evidence': [new_path_ref]}
            write(method_dir/'continuations'/'0001.json', request)
            args = (directory, method_dir, record, record_ref, job, a, b, a, b, a, b,
                    [a]*6, [b]*6, names, costs, 'synthetic')
            restored, restored_ref = C.ingest_continuations(*args)
            checks['partial_unranked_then_same_rule_reconciled'] = (partial_ev['primary_selected_candidate_id'] is None
                and restored['evaluation']['primary_selected_candidate_id'] is not None
                and restored['numerically_complete'] and restored['time_s']['historical_solver_wall_s'] == 1.25)
            checks['continuation_resume_without_resolve_or_refit'] = C.ingest_continuations(*args) == (restored, restored_ref)
            write(directory/'selection_manifest.json', {'sealed': True})
            second = {**request, 'parent_method_result': restored_ref}
            write(method_dir/'continuations'/'0002.json', second)
            checks['new_continuation_after_seal_rejected'] = rejects(lambda: C.ingest_continuations(*args))
    after = sorted(str(p.relative_to(ROOT)) for p in (ROOT/'stage3_data').glob('**/*.npz'))
    checks['no_formal_or_test_bundle_generated'] = before == after
    report = {'stage': '3_campaign_preparation_contract_check', 'all_checks_passed': all(checks.values()),
              'checks': checks, 'all_checks_passed_semantics': 'Small synthetic runner/provenance contracts only; not formal execution authorization.',
              'source_sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                  ['stage3_campaign.py', 'stage3_campaign_preflight.py', 'stage3_data.py', 'stage3_data_v2.py',
                   'stage0_l1_solver.py', 'stage2a_retry_remaining_v4.py', 'stage3_protocol_draft.json']},
              'new_solver_calls': 0, 'formal_or_test_data_generated': False,
              'frozen_protocol_created': False, 'formal_execution_ready': False,
              'watchdog_status': 'Windows process-tree deadline implementation reviewed; no formal child launched by this preflight.',
              'runtime_fixture': '3x3 orthogonal closed-form coefficients only; temporary files stay inside workspace.',
              'elapsed_s': time.perf_counter()-started,
              'outstanding_before_formal_execution': ['Root independent review and immutable protocol/source freeze.',
                  'Final maximum-library development audit and source/version inventory.',
                  'No statistical or publication conclusion follows from these infrastructure checks.']}
    C.atomic_json(output, report)
    print(json.dumps({'all_checks_passed': report['all_checks_passed'], 'checks': checks, 'elapsed_s': report['elapsed_s']}, indent=2))
    if not report['all_checks_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
