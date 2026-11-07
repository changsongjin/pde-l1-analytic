"""Read-only paired development diagnosis; never changes selected models."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from stage0_evaluation import strict_json_write
from stage2a_diagnosis import diagnose_method

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=ROOT/'stage2b_compare_result.json')
    parser.add_argument('--output', type=Path, default=ROOT/'stage2b_diagnosis_result.json')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve completed diagnostic; select a new output version.')
    raw = args.input.read_bytes()
    source = json.loads(raw)
    if not source['execution_complete']:
        raise RuntimeError('All declared tasks must first be attempted; partial numerical paths remain permitted.')
    records, pairs = [], []
    for row in source['rows']:
        for lname, layer in row['layers'].items():
            grouped = {}
            for name, method in layer['methods'].items():
                record = diagnose_method(args.input.name, row, lname, layer, name, method, source['alpha_frozen'])
                record['cohort'] = row['cohort']
                record['historical_cost_retained'] = method['time_s']['solver_with_worker_startup']
                record['origin'] = method['origin']
                records.append(record); grouped[name] = record
            for reference in ('L1', 'IRL1_original'):
                candidate, control = grouped['IRL1_smoothed'], grouped[reference]
                assert candidate['solver_input_hashes'] == control['solver_input_hashes']
                p = {'cohort': row['cohort'], 'case': row['case'], 'sigma': row['sigma'], 'layer': lname,
                    'candidate': 'IRL1_smoothed', 'reference': reference,
                    'both_numerically_complete': candidate['all_requested_solutions_certified'] and control['all_requested_solutions_certified'],
                    'identical_input_hashes': candidate['solver_input_hashes']}
                if p['both_numerically_complete']:
                    a, b = candidate['selected'], control['selected']
                    p.update({'candidate_support': a['support'], 'reference_support': b['support'],
                        'candidate_exact_support': a['exact'], 'reference_exact_support': b['exact'],
                        'candidate_minus_reference_f1': a['f1']-b['f1'],
                        'candidate_minus_reference_validation_error': a['validation_error']-b['validation_error'],
                        'candidate_minus_reference_complexity': a['complexity']-b['complexity'],
                        'candidate_minus_reference_fixed_score': a['fixed_score']-b['fixed_score'],
                        'candidate_solver_seconds': candidate['historical_cost_retained'],
                        'reference_solver_seconds': control['historical_cost_retained'],
                        'runtime_ranking_eligible': False,
                        'runtime_note': 'Reported recorded costs; cached preprocessing and repeated numerical repair require separate accounting.'})
                pairs.append(p)
    keys = {(r['cohort'], r['case'], r['sigma'], r['layer'], r['method']) for r in records}
    assert len(keys) == len(records) == 36
    groups = []
    for cohort in source['cohorts']:
        for method in source['methods']:
            rows = [r for r in records if r['cohort'] == cohort and r['method'] == method]
            groups.append({'cohort': cohort, 'method': method, 'declared_cases_and_libraries': len(rows),
                'classification_counts': dict(Counter(r['classification'] for r in rows)),
                'exact_support_count_descriptive_only': sum(r['selected'] is not None and r['selected']['exact'] for r in rows),
                'numerically_incomplete_count': sum(not r['all_requested_solutions_certified'] for r in rows)})
    report = {'stage': '2b_single_factor_diagnosis', 'all_checks_passed': True,
        'source_file': args.input.name, 'source_sha256': hashlib.sha256(raw).hexdigest(),
        'source_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'read_only_solver_runs': 0, 'records': records, 'paired_records': pairs, 'descriptive_groups': groups,
        'numerically_complete_runs': sum(r['all_requested_solutions_certified'] for r in records),
        'diagnostic_rows': len(records), 'new_true_support_tuning': False, 'formal_test_data_used': False,
        'limits': ['These three deliberately selected development PDE/noise pairs and two library sizes are not iid repetitions.',
            'Different initial conditions AND a new noise seed change together in replication; do not attribute differences solely to one factor.',
            'Per-group counts are descriptive, not general recovery probabilities or a statistically established algorithm ranking.',
            'No primary comparison is made for a numerically incomplete path.',
            'A lower weak-residual score does not imply more accurate dynamics; use separate spectral diagnostics.',
            'Relative epsilon remains fixed at .01; these diagnostic truth metrics do not tune any hyperparameter.']}
    strict_json_write(args.output, report)
    print(json.dumps({'all_checks_passed': True, 'numerically_complete_runs': report['numerically_complete_runs'],
                      'groups': groups}, ensure_ascii=False))


if __name__ == '__main__':
    main()
