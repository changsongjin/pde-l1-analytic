"""Read-only diagnosis of completed dataset rows in development baseline snapshots.

Does not rerun any solver, alter alpha, or turn one-seed observations into a
noise-robustness ranking. A partially written/unfinished source stays explicit.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from stage0_evaluation import canonical_term, strict_json_write

ROOT = Path(__file__).resolve().parent
CLASSIFICATIONS = {
    'numerical_or_budget_incomplete': 'At least one requested lambda is unfinished or uncertified; no primary selection.',
    'evaluation_incomplete': 'Numerical path completed but no valid primary selection is recorded.',
    'selected_true_support': 'The frozen validation-complexity score selected the known support in this development observation.',
    'path_did_not_generate_true_support': 'No certified path candidate has the known support; this does not establish universal non-identifiability.',
    'validation_complexity_tradeoff': 'The path contains the known support, but the frozen score selected another support.',
}


def classify(complete, selected_exact, true_appears):
    if not complete:
        return 'numerical_or_budget_incomplete'
    if selected_exact is None:
        return 'evaluation_incomplete'
    if selected_exact:
        return 'selected_true_support'
    if true_appears:
        return 'validation_complexity_tradeoff'
    return 'path_did_not_generate_true_support'


def _support_metric(candidate, canonical_by_name, truth_canonical):
    got = {canonical_by_name[name] for name in candidate['support']}
    true = set(truth_canonical)
    tp = len(got & true)
    return {'exact': got == true,
            'f1': 2*tp/(len(got)+len(true)) if got or true else 1.,
            'precision': tp/len(got) if got else float(not true),
            'recall': tp/len(true) if true else 1.,
            'extra': sorted(got-true), 'missed': sorted(true-got)}


def _candidate_summary(candidate, alpha, canonical_by_name, truth_canonical):
    if candidate is None:
        return None
    metrics = _support_metric(candidate, canonical_by_name, truth_canonical)
    stored = candidate.get('truth_evaluation_after_selection', {})
    if 'exact' in stored and metrics['exact'] != stored['exact']:
        raise ValueError('Canonical support audit disagrees with saved exact-support metric.')
    return {'candidate_id': candidate['candidate_id'], 'support': candidate['support'],
            **metrics, 'validation_error': candidate['validation_error'],
            'validation_by_trajectory': candidate.get('validation_by_trajectory'),
            'complexity': candidate['complexity'],
            'fixed_score': candidate['validation_error']+alpha*candidate['complexity'],
            'ls_physical_coefficients': dict(zip(candidate['support'], candidate['coefficients'])),
            'physical_coef_rel_error_union': stored.get('physical_coef_rel_error_union'),
            'physical_coef_rel_error_true_only': stored.get('physical_coef_rel_error_true_only')}


def diagnose_method(source, dataset, layer_name, layer, method_name, method, alpha):
    path, evaluation = method['path'], method['evaluation']
    names = dataset['library_names']
    canonical = dataset.get('canonical_library_names')
    if canonical is None:
        canonical = [canonical_term(name) for name in names]
    canonical_by_name = dict(zip(names, canonical))
    truth_canonical = [canonical_term(name) for name in dataset['truth']]
    candidates = {c['candidate_id']: c for c in evaluation['candidates']}
    exact_candidates = [c for c in candidates.values()
                        if _support_metric(c, canonical_by_name, truth_canonical)['exact']]
    selected_id = evaluation['primary_selected_candidate_id']
    selected = _candidate_summary(candidates.get(selected_id), alpha, canonical_by_name, truth_canonical)
    exploratory_id = evaluation.get('exploratory_best_available_candidate_id')
    exploratory = _candidate_summary(candidates.get(exploratory_id), alpha, canonical_by_name, truth_canonical)
    true_summaries = [_candidate_summary(c, alpha, canonical_by_name, truth_canonical) for c in exact_candidates]
    true_best = min(true_summaries, key=lambda c: c['fixed_score']) if true_summaries else None
    all_cert = (path['status'] == 'complete' and path['all_solutions_certified']
                and len(path['points']) == len(path['ratios'])
                and all(p['status'] == 'certified' and p['certificate']['certified'] for p in path['points']))
    recorded_alpha = evaluation['selection_rule']['complexity_coefficient']
    if recorded_alpha != alpha:
        raise ValueError('Source alpha and method selection rule disagree.')
    if not all_cert and selected is not None:
        raise ValueError('Incomplete numerical path must not have a primary model.')
    category = classify(all_cert, selected['exact'] if selected is not None else None, bool(true_summaries))
    comparison = None
    if selected and true_best:
        comparison = {'selected_minus_true_validation_error': selected['validation_error']-true_best['validation_error'],
                      'selected_minus_true_complexity': selected['complexity']-true_best['complexity'],
                      'selected_minus_true_fixed_score': selected['fixed_score']-true_best['fixed_score'],
                      'interpretation': 'Diagnostic only; do not change alpha using this comparison.'}
    point_counts = Counter(point['status'] for point in path['points'])
    exact_ids = {c['candidate_id'] for c in exact_candidates}
    true_ratios = [point['ratio'] for point in evaluation['point_to_candidate'] if point['candidate_id'] in exact_ids]
    certificates = [p['certificate'] for p in path['points'] if p.get('certificate')]
    return {'source': source, 'dataset_id': dataset['dataset_id'], 'case': dataset['case'],
            'sigma': dataset['sigma'], 'seed': dataset['seed'], 'layer': layer_name, 'method': method_name,
            'input_raw_bundle_sha256': dataset['raw_bundle_sha256'],
            'solver_input_hashes': layer['solver_input_hashes'],
            'n_columns': layer['n_columns'], 'path_status': path['status'],
            'all_requested_solutions_certified': bool(all_cert),
            'path_point_status_counts': dict(point_counts), 'n_candidates': len(candidates),
            'time_s': method['time_s'], 'dataset_time_s': dataset['time_s'],
            'numerical_diagnostics': {
                'max_kkt_relative_to_penalty_among_returned_points': max((c['kkt_max_relative_to_penalty'] for c in certificates), default=None),
                'max_relative_dual_gap_among_returned_points': max((c['dual_gap_relative'] for c in certificates), default=None),
                'fallback_points': sum(len(p.get('attempts', [])) > 1 for p in path['points']),
                'outer_iteration_count': len(path.get('outer', [])), 'outer_stop': path.get('outer_stop')},
            'frozen_alpha': alpha, 'selected': selected,
            'partial_path_exploratory_best_only': exploratory if not all_cert else None,
            'true_support_appears_any_certified_candidate': bool(true_summaries),
            'true_support_candidate_lambda_ratios': true_ratios,
            'true_support_candidates': true_summaries,
            'selected_vs_best_true_support_candidate': comparison,
            'classification': category, 'classification_meaning': CLASSIFICATIONS[category],
            'validation_metric': dataset['validation_metric'],
            'noise_ranking_eligible': False,
            'support_definition': 'L1 normalized coefficients at 1e-5; LS readout never changes support'}


def diagnose_sources(paths):
    records, sources, skipped = [], [], []
    for path in paths:
        path = Path(path)
        try:
            raw = path.read_bytes()
            report = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            sources.append({'file': str(path), 'status': 'unreadable_snapshot', 'error': str(exc)})
            continue
        source = {'file': str(path), 'sha256_of_snapshot': hashlib.sha256(raw).hexdigest(),
                  'source_execution_complete': report.get('execution_complete', False),
                  'source_all_numerical_checks_passed': report.get('all_checks_passed', False),
                  'status': 'complete_campaign_snapshot' if report.get('execution_complete') else 'unfinished_campaign_snapshot',
                  'recorded_dataset_rows': len(report.get('rows', [])), 'analyzed_dataset_rows': 0}
        sources.append(source)
        if report.get('stage') != '2a_development_baseline':
            source['status'] = 'unsupported_source'
            continue
        alpha = report['alpha_frozen']
        for dataset in report.get('rows', []):
            expected_layers = report['layers']
            row_ready = (all(layer in dataset.get('layers', {}) for layer in expected_layers)
                         and 'truth' in dataset
                         and all(set(dataset['layers'][layer].get('methods', {})) == {'L1', 'IRL1_original'}
                                 for layer in expected_layers))
            if not row_ready:
                skipped.append({'source': str(path), 'case': dataset.get('case'), 'sigma': dataset.get('sigma'),
                                'reason': 'Dataset row still incomplete; no partial in-flight row analyzed.'})
                continue
            row_records = []
            try:
                for layer_name, layer in dataset['layers'].items():
                    for method_name, method in layer['methods'].items():
                        row_records.append(diagnose_method(str(path), dataset, layer_name, layer, method_name, method, alpha))
            except (KeyError, ValueError, TypeError) as exc:
                skipped.append({'source': str(path), 'case': dataset.get('case'), 'sigma': dataset.get('sigma'),
                                'reason': 'Invalid/incomplete evaluation row: '+str(exc)})
                continue
            source['analyzed_dataset_rows'] += 1
            records.extend(row_records)
    counts = Counter(row['classification'] for row in records)
    duplicate_keys = Counter((r['dataset_id'], r['layer'], r['method'], r['frozen_alpha']) for r in records)
    return {'stage': '2a_development_diagnosis', 'read_only_solver_runs': 0,
            'all_sources_execution_complete': bool(sources) and all(s.get('source_execution_complete', False) for s in sources),
            'diagnosis_complete_for_readable_snapshot_rows': not skipped and all(s['status'] not in ['unreadable_snapshot', 'unsupported_source'] for s in sources),
            'research_stage_complete': False,
            'sources': sources, 'records': records, 'skipped_dataset_rows': skipped,
            'classification_counts_descriptive_only': dict(counts),
            'duplicate_observation_keys': [list(k)+[v] for k,v in duplicate_keys.items() if v > 1],
            'unique_noise_seeds_observed': sorted({r['seed'] for r in records}),
            'limits': ['Development diagnostics only; no final-test observations or confidence intervals.',
                       'One-seed cases do not support a definite noise-robustness ranking.',
                       'True support is used for diagnosis only; alpha and saved selections are never changed.',
                       'A true support missing from this finite lambda path is not proof of non-identifiability.',
                       'A score tradeoff may reflect data/preprocessing/validation error as well as structural complexity.',
                       'Weak equation residuals are not trajectory rollout errors.',
                       'Duplicate source observations are shown explicitly and must not count as independent replicates.']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'stage2a_diagnosis_result.json')
    args = parser.parse_args()
    checks = [classify(False, None, True) == 'numerical_or_budget_incomplete',
              classify(True, None, False) == 'evaluation_incomplete',
              classify(True, True, True) == 'selected_true_support',
              classify(True, False, False) == 'path_did_not_generate_true_support',
              classify(True, False, True) == 'validation_complexity_tradeoff']
    result = diagnose_sources(args.inputs)
    result['classification_self_checks_passed'] = all(checks)
    result['all_checks_passed'] = bool(all(checks) and result['diagnosis_complete_for_readable_snapshot_rows'])
    strict_json_write(args.output, result)
    print(json.dumps({'records': len(result['records']),
                      'classification_counts': result['classification_counts_descriptive_only'],
                      'all_sources_execution_complete': result['all_sources_execution_complete'],
                      'all_checks_passed': result['all_checks_passed']}, indent=2))


if __name__ == '__main__':
    main()
