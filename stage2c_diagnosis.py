"""Compare the frozen J1158 baseline with the full 1212-feature extension."""
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
    parser.add_argument('--input', type=Path, default=ROOT/'stage2c_maximum_library_pilot_result.json')
    parser.add_argument('--output', type=Path, default=ROOT/'stage2c_diagnosis_result.json')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve the previous diagnostic and choose a new output version.')
    raw = args.input.read_bytes(); source = json.loads(raw)
    baseline_path = ROOT/'stage2a_reconcile_complete_result.json'
    baseline_raw = baseline_path.read_bytes(); baseline = json.loads(baseline_raw)
    old_rows = {(r['case'], r['sigma'], r['seed']): r for r in baseline['rows']}
    records, comparisons, unready = [], [], []
    for row in source['rows']:
        layer = row['layers']['J1212']
        if 'truth' not in row or len(layer['methods']) != 2:
            unready.append({'case': row['case'], 'sigma': row['sigma']}); continue
        old = old_rows[(row['case'], row['sigma'], row['seed'])]
        assert row['raw_bundle_sha256'] == old['raw_bundle_sha256']
        assert row['input_hashes']['train_target'] == old['input_hashes']['train_target']
        assert row['input_hashes']['validation_target'] == old['input_hashes']['validation_target']
        for method, value in layer['methods'].items():
            current = diagnose_method(args.input.name, row, 'J1212', layer, method, value, source['alpha_frozen'])
            reference = diagnose_method(baseline_path.name, old, 'J1158', old['layers']['J1158'], method,
                                        old['layers']['J1158']['methods'][method], baseline['alpha_frozen'])
            current['selected_nonpolynomial_terms'] = ([n for n in current['selected']['support'] if n in
                {n for n, k in zip(row['library_names'], row['canonical_library_names']) if k.startswith('NP:')}]
                if current['selected'] else None)
            records.append(current)
            pair = {'case': row['case'], 'sigma': row['sigma'], 'seed': row['seed'], 'method': method,
                'same_raw_inputs_and_targets': True, 'previous_library': 'J1158', 'expanded_library': 'J1212',
                'both_paths_complete': current['all_requested_solutions_certified'] and reference['all_requested_solutions_certified'],
                'original': reference['selected'], 'expanded': current['selected'],
                'expanded_nonpolynomial_terms': current['selected_nonpolynomial_terms']}
            if pair['both_paths_complete']:
                a, b = pair['expanded'], pair['original']
                pair['expanded_minus_original'] = {'validation_error': a['validation_error']-b['validation_error'],
                    'complexity': a['complexity']-b['complexity'], 'f1': a['f1']-b['f1'],
                    'fixed_score': a['fixed_score']-b['fixed_score']}
            comparisons.append(pair)
    expected = 36
    report = {'stage': '2c_library_extension_diagnosis', 'source_file': args.input.name,
        'source_sha256': hashlib.sha256(raw).hexdigest(),
        'baseline_source_sha256': hashlib.sha256(baseline_raw).hexdigest(),
        'source_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_execution_complete': source['execution_complete'], 'records': records,
        'paired_library_comparisons': comparisons, 'unready_datasets': unready,
        'all_checks_passed': not unready and len(records) == expected,
        'numerically_complete_runs': sum(r['all_requested_solutions_certified'] for r in records),
        'classification_counts_descriptive_only': dict(Counter(r['classification'] for r in records)),
        'selected_nonpolynomial_model_count': sum(bool(r['selected_nonpolynomial_terms']) for r in records),
        'test_data_used': False, 'solver_calls': 0, 'selection_or_support_changed': False,
        'limits': ['The same raw data and preserved feature block isolate this declared library extension; design dimensions intentionally differ.',
            'All six known PDEs lie in the polynomial library; this tests extra functional distractors, not recovery of an omitted true nonpolynomial mechanism.',
            'These mixed-case development counts are not general success probabilities or a noise-robustness ranking.',
            'Physical coefficient errors use canonical polynomial keys and the 54 exact nonpolynomial recipe keys.',
            'Numerical incompletion is distinct from a statistically wrong support; incomplete paths have no primary model.',
            'No alpha, coefficient, support, or L1 weight is altered by this diagnostic.']}
    strict_json_write(args.output, report)
    print(json.dumps({k: report[k] for k in ['all_checks_passed', 'numerically_complete_runs',
                                           'classification_counts_descriptive_only', 'selected_nonpolynomial_model_count']}))


if __name__ == '__main__':
    main()
