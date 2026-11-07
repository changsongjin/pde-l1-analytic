"""Read-only numerical/selection gate for the completed 36-record ablation.

Numerical completeness is separate from correct support, method superiority,
formal statistical robustness, maximal-library completion or publication.
"""
import json
from pathlib import Path
from stage0_evaluation import strict_json_write
from stage2b_independent_review import read, digest

ROOT=Path(__file__).resolve().parent


def main():
    output=ROOT/'stage2b_batch_checkpoint_result.json'
    if output.exists():raise FileExistsError('Preserve completed checkpoint; choose a new version.')
    filenames=['stage2b_reconcile_complete_result.json','stage2b_complete_diagnosis_result.json',
               'stage2b_independent_review_v2_result.json','stage2b_numerical_retry_result.json',
               'stage2b_heat_dynamic_compare_v2_result.json']
    baseline,diagnosis,review,retry,dynamic=[read(ROOT/n) for n in filenames]
    expected={(c,p,s,l,m) for c in ['development_ids0_5_seed0','independent_development_ids12_17_seed1']
        for p,s in [('heat',.01),('burgers',.05),('logheat',.05)]
        for l in ['J18','J1158'] for m in ['L1','IRL1_original','IRL1_smoothed']}
    records=[];outer_count=0
    for row in baseline['rows']:
        for lname,lib in row['layers'].items():
            for name,method in lib['methods'].items():
                path=method['path'];evaluation=method['evaluation']
                points=path['points'];outer=path.get('outer',[]);outer_count+=len(outer)
                certified=[p['status']=='certified' and p['certificate']['certified']
                    and p['certificate']['tolerances']=={'kkt_scaled_abs':1e-8,'kkt_relative':1e-5,'gap_relative':1e-9}
                    for p in points]
                outer_ok=all(p['status']=='certified' and p['certificate']['certified'] for p in outer)
                complete=(path['all_solutions_certified'] and path['status']=='complete' and len(points)==24
                    and all(certified) and outer_ok and evaluation['selection_eligible']
                    and evaluation['primary_selected_candidate_id'] is not None)
                records.append({'cohort':row['cohort'],'case':row['case'],'sigma':row['sigma'],'layer':lname,'method':name,
                    'numerically_complete':complete,'certified_lambda_count':sum(certified),'outer_solution_count':len(outer),
                    'recorded_solver_seconds_including_historical_attempts':method['time_s']['solver_with_worker_startup'],
                    'new_solver_seconds_this_comparison':method['new_solver_seconds_this_comparison']})
    keys=[tuple(r[k] for k in ['cohort','case','sigma','layer','method']) for r in records]
    ledger=baseline['numerical_completion']['cost_ledger']
    sources_ok=all(digest(ROOT/n)==h for n,h in baseline['numerical_completion']['source_hashes'].items())
    checks={'unique_complete_36_record_grid':len(keys)==36 and set(keys)==expected,
        'all_864_fixed_weight_final_points_certified':sum(r['certified_lambda_count'] for r in records)==864,
        'all_outer_points_and_primary_selections_complete':all(r['numerically_complete'] for r in records),
        'full_independent_review_passed':review['all_checks_passed'] and review['new_solver_calls']==0,
        'independent_review_binds_current_complete_source':review['runner_review']['runner_sha256']==digest(ROOT/filenames[0]),
        'independent_review_all_864_final_certificates':sum(r['final_points_recertified'] for r in review['runner_review']['records'])==864,
        'diagnosis_binds_current_complete_source':diagnosis['source_sha256']==digest(ROOT/filenames[0]),
        'diagnosis_all_36_complete':diagnosis['all_checks_passed'] and diagnosis['diagnostic_rows']==36 and diagnosis['numerically_complete_runs']==36,
        'only_2_methods_changed_and_34_identical':baseline['numerical_completion']['changed_method_records']==2
            and baseline['numerical_completion']['unchanged_method_records']==34
            and review['additional_review']['unchanged_method_records']==34,
        'all_4_tail_retries_certified_under_original_tolerances':retry['all_checks_passed']
            and sum(len(r['point_attempts']) for r in retry['rows'])==4
            and all(a['status']=='certified' for r in retry['rows'] for a in r['point_attempts']),
        'source_provenance_intact':sources_ok,
        'new_solver_cost_includes_retries':sum(r['new_solver_seconds_this_comparison'] for r in records)==ledger['new_comparison_solver_seconds_including_retries'],
        'historical_control_costs_retained':sum(r['recorded_solver_seconds_including_historical_attempts'] for r in records)==ledger['recorded_method_solver_seconds_including_historical_controls'],
        'heat_dynamic_source_binding_preserved':dynamic['all_checks_passed'] and review['additional_review']['heat_dynamic_v2_source_binding_valid_via_unchanged_heat_rows'],
        'frozen_alpha_and_support':baseline['alpha_frozen']==.005 and all(
            m['path']['support_tolerance']==1e-5 for r in baseline['rows'] for l in r['layers'].values() for m in l['methods'].values()),
        'no_reserved_test_data_used':not baseline['test_data_used'] and not review['test_data_used'] and not diagnosis['formal_test_data_used']}
    result={'stage':'2b_single_epsilon_factor_development_gate','all_checks_passed':all(checks.values()),'checks':checks,
        'sources':[{'file':n,'sha256':digest(ROOT/n)} for n in filenames],
        'source_script_sha256':digest(Path(__file__)),'max_seconds_per_batch_user_constraint':7200,
        'expected_runs':36,'recorded_runs':len(records),'numerically_complete_runs':sum(r['numerically_complete'] for r in records),
        'certified_final_points':sum(r['certified_lambda_count'] for r in records),'certified_outer_points':outer_count,
        'method_runs':records,'unfinished_numerical_runs':[r for r in records if not r['numerically_complete']],
        'cost_ledger':{**ledger,'read_only_independent_review_seconds':review['elapsed_s']},
        'test_data_generated_or_used':False,'support_correctness_not_a_gate':True,
        'descriptive_comparison':{'original_vs_smoothed_selected_support_identical_pairs':sum(
            p['same_selected_support_original_vs_smoothed'] for p in review['additional_review']['paired_irL1_checks']),
            'paired_cases_and_libraries':12,'not_an_independent_statistical_sample':True},
        'interpretation':'All declared numerical paths, weights, candidate mappings and frozen selection rules verified. This is not evidence of a general robustness ranking or publication readiness.',
        'remaining_scope':['54 nonpolynomial supplementary terms and future maximum-library experiments',
            'native comparator pipelines','formal paired noise seeds','independent final-test evaluation',
            'outlier robustness','novelty and publication evidence review']}
    strict_json_write(output,result)
    print(json.dumps({'all_checks_passed':result['all_checks_passed'],'complete_runs':result['numerically_complete_runs'],
        'final_points':result['certified_final_points'],'outer_points':outer_count,'cost_ledger':ledger}))


if __name__=='__main__':main()
