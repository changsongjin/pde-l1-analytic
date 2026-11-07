"""Final independent audit of all 36 frozen stage-2b method records.

Reuses the read-only preflight and full-column certificate scanner, adding
the whole lambda grid, candidate linkage, weights/stopping, source and cost
checks. No solver, new data, support deletion or statistical parameter change.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
import stage2b_independent_review as V1
from stage0_evaluation import array_hash, normalize_training, physical_coefficients, strict_json_write
from stage2_identifiability import stack_split
from stage2a_l1_paths import DEFAULT_RATIOS

ROOT=Path(__file__).resolve().parent


def additional_audit(filename):
    report=V1.read(filename);original=V1.read(ROOT/'stage2b_compare_result.json')
    baseline=V1.read(ROOT/'stage2a_reconcile_complete_result.json')
    retry=V1.read(ROOT/'stage2b_numerical_retry_result.json')
    protocol=V1.read(ROOT/'stage2b_protocol_result.json')
    hashes={**report['source_sha256'],**report['gate_sources'],**report['control_sources'],
            **report['numerical_completion']['source_hashes']}
    if not all(V1.digest(ROOT/name)==h for name,h in hashes.items()):raise ValueError('Source or gate hash changed.')
    if (report['alpha_frozen']!=.005 or report['alpha_sensitivity']!=[0.,.001,.005,.01,.05]
        or report['lambda_ratios']!=DEFAULT_RATIOS.tolist() or report['test_data_used']
        or report['statistical_parameter_tuning_after_outcomes']):raise ValueError('Frozen study contract changed.')
    expected={(c,p,s,l,m) for c in report['cohorts'] for p,s in protocol['cases_and_noise']
              for l in ['J18','J1158'] for m in ['L1','IRL1_original','IRL1_smoothed']}
    observed=set();records=[];paired=[];unchanged=0;new_seconds=0.;recorded_seconds=0.;control_count=0
    for row,old_row in zip(report['rows'],original['rows']):
        ids=list(range(6)) if row['seed']==0 else list(range(12,18))
        if (row['splits']['train_trajectories']!=ids[:4] or row['splits']['validation_trajectories']!=ids[4:]
            or row['test_data_used'] or row['truth_used_for_selection']):raise ValueError('Split or selection contract changed.')
        if row['seed']==0:
            tag=f"{row['case']}_sigma_{repr(float(row['sigma'])).replace('.', 'p')}_seed_0"
            matrix=ROOT/'stage2_matrices'/(tag+'.npz')
        else:
            tag=f"{row['case']}_sigma_{row['sigma']}_seed_1_ids12_17"
            matrix=ROOT/'stage2b_matrices'/(tag+'.npz')
        if 'npz_sha256' in row['matrix_metadata'] and V1.digest(matrix)!=row['matrix_metadata']['npz_sha256']:
            raise ValueError('Matrix NPZ changed.')
        with np.load(matrix,allow_pickle=False) as data:
            matrices=[data[f'a_{i}'].copy() for i in range(6)];targets=[data[f'b_{i}'].copy() for i in range(6)]
        for a0,b0,meta in zip(matrices,targets,row['matrix_metadata']['trajectories']):
            if array_hash(a0)!=meta['matrix_hash'] or array_hash(b0)!=meta['target_hash']:raise ValueError('Cached A/b changed.')
        a,b=stack_split(matrices,targets,range(4))
        for lname,lib in row['layers'].items():
            subset=lib['indices_in_maximum_library'];an,bn,scales=normalize_training(a[:,subset],b)
            if lname=='J18' and subset!=row['matrix_metadata']['small_indices']:raise ValueError('Small library subset changed.')
            if lname=='J1158' and subset!=list(range(a.shape[1])):raise ValueError('Maximum library pruned.')
            choices={}
            for method,record in lib['methods'].items():
                identity=(row['cohort'],row['case'],row['sigma'],lname,method)
                if identity in observed:raise ValueError('Duplicate method identity.')
                observed.add(identity);path=record['path'];weights=np.asarray(path['weights'])
                if (path['ratios']!=DEFAULT_RATIOS.tolist() or len(path['points'])!=24 or path['support_tolerance']!=1e-5
                    or weights.shape!=(an.shape[1],) or not np.isfinite(weights).all() or not np.all(weights>0)):
                    raise ValueError('Path weights/grid/support contract changed.')
                if method=='L1' and (path['outer'] or not np.array_equal(weights,np.ones(an.shape[1]))):
                    raise ValueError('Ordinary L1 has changed weights.')
                if record['origin']=='stage2b_new_method_run' and path['budget_s']!=120.:raise ValueError('New initial budget changed.')
                lmax=float(np.max(np.abs(an.T@bn)/weights))
                if not np.isclose(lmax,path['path_metadata']['lambda_max'],rtol=2e-13,atol=0):raise ValueError('Final lambda scale mismatch.')
                for ratio,point in zip(DEFAULT_RATIOS,path['points']):
                    if point['ratio']!=ratio or not np.isclose(point['lambda'],lmax*ratio,rtol=2e-13,atol=0):raise ValueError('Requested lambda changed.')
                    if point.get('alternate_dual') is not None:raise ValueError('This run requires a new exact-dual-aware review, not original certificate promotion.')
                    if not np.array_equal(physical_coefficients(point['beta'],scales),point['physical_L1_coefficients']):
                        raise ValueError('Physical readout changed normalized L1 coordinates.')
                outer=path.get('outer',[])
                if method!='L1':
                    if not 2<=len(outer)<=8 or [p['iteration'] for p in outer]!=list(range(len(outer))):raise ValueError('Outer iteration count changed.')
                    repeated=[i for i in range(1,len(outer)) if outer[i]['support']==outer[i-1]['support']]
                    if repeated and repeated!=[len(outer)-1]:raise ValueError('Historical support stopping rule changed.')
                    expected_stop='support_unchanged_original_rule' if repeated else 'maximum_8_iterations_original_rule'
                    if path['outer_stop']!=expected_stop or (not repeated and len(outer)!=8):raise ValueError('Outer stop inconsistent.')
                    if method=='IRL1_smoothed':
                        for p in outer:
                            beta=np.asarray(p['beta']);epsilon=max(1e-8,.01*float(np.max(abs(beta))))
                            update=p['weight_update'];w=1./(abs(beta)+epsilon)
                            if (update['input_beta_sha256']!=array_hash(beta) or update['epsilon']!=epsilon
                                or w.max()/w.min()>101.+1e-12):raise ValueError('Relative-epsilon formula/input/bound changed.')
                        if path['epsilon_trajectory']!=[p['weight_update']['epsilon'] for p in outer]:raise ValueError('Epsilon trajectory mismatch.')
                evaluation=record['evaluation'];candidates=evaluation['candidates'];byid={c['candidate_id']:c for c in candidates}
                mapping=evaluation['point_to_candidate']
                if len(mapping)!=24 or [m['point_index'] for m in mapping]!=list(range(24)):raise ValueError('Candidate mapping omits path points.')
                for point,mapping_record in zip(path['points'],mapping):
                    candidate=byid[mapping_record['candidate_id']]
                    if (candidate['support_indices']!=point['support'] or mapping_record['lambda']!=point['lambda']
                        or mapping_record['ratio']!=point['ratio'] or candidate['support']!=[lib['names'][i] for i in point['support']]
                        or candidate['support_source']!='normalized_L1_beta_only' or candidate['ls_readout_changes_support']
                        or candidate['dataset_id']!=row['dataset_id']):raise ValueError('Candidate support/data linkage invalid.')
                for sensitive,alpha in zip(evaluation['sensitivity'],report['alpha_sensitivity']):
                    choice=min(candidates,key=lambda c:(c['validation_error']+alpha*c['complexity'],c['complexity'],c['validation_error'],tuple(c['support_indices'])))
                    if (sensitive['alpha']!=alpha or sensitive['candidate_id']!=choice['candidate_id']
                        or sensitive['score']!=choice['validation_error']+alpha*choice['complexity']):raise ValueError('Sensitivity choice changed.')
                independent_front=[c['candidate_id'] for c in candidates if not any(
                    q['validation_error']<=c['validation_error'] and q['complexity']<=c['complexity']
                    and (q['validation_error']<c['validation_error'] or q['complexity']<c['complexity']) for q in candidates)]
                if independent_front!=evaluation['pareto_candidate_ids']:raise ValueError('Pareto frontier mismatch.')
                choices[method]=byid[evaluation['primary_selected_candidate_id']]['support_indices']
                original_method=old_row['layers'][lname]['methods'][method]
                if 'numerical_completion_provenance' not in record:
                    if record!=original_method:raise ValueError('Unrelated method changed.')
                    unchanged+=1
                else:
                    if path['points'][:22]!=original_method['path']['points'][:22]:raise ValueError('Certified prefix changed.')
                if record['origin']=='reused_stage2a_control':
                    prior_row=next(r for r in baseline['rows'] if (r['case'],r['sigma'],r['seed'])==(row['case'],row['sigma'],0))
                    prior=prior_row['layers'][lname]['methods'][method]
                    if any(record[k]!=v for k,v in prior.items()) or record['new_solver_seconds_this_comparison']!=0:
                        raise ValueError('Historical controls/costs changed.')
                    control_count+=1
                new_seconds+=record['new_solver_seconds_this_comparison'];recorded_seconds+=record['time_s']['solver_with_worker_startup']
                records.append({'identity':list(identity),'all_24_lambda_weight_coordinates_verified':True,
                    'support_candidate_mapping_pareto_and_sensitivity_verified':True,'outer_stop_and_epsilon_verified':True})
            original_outer=lib['methods']['IRL1_original']['path']['outer'][0]
            variant_outer=lib['methods']['IRL1_smoothed']['path']['outer'][0]
            if original_outer['lambda']!=variant_outer['lambda'] or original_outer['beta']!=variant_outer['beta']:
                raise ValueError('Unweighted first outer solve differs.')
            paired.append({'cohort':row['cohort'],'case':row['case'],'sigma':row['sigma'],'layer':lname,
                'same_unweighted_first_outer_beta':True,'same_selected_support_original_vs_smoothed':choices['IRL1_original']==choices['IRL1_smoothed']})
    ledger=report['numerical_completion']['cost_ledger']
    if (observed!=expected or len(observed)!=36 or unchanged!=34 or control_count!=12
        or new_seconds!=ledger['new_comparison_solver_seconds_including_retries']
        or recorded_seconds!=ledger['recorded_method_solver_seconds_including_historical_controls']
        or ledger['retry_solver_seconds']!=sum(r['retry_elapsed_s'] for r in retry['rows'])):raise ValueError('Grid, preserved controls or total costs inconsistent.')
    dynamic=V1.read(ROOT/'stage2b_heat_dynamic_compare_v2_result.json')
    if dynamic['source_sha256']!=V1.digest(ROOT/'stage2b_compare_result.json') or not dynamic['all_checks_passed']:
        raise ValueError('Dynamic diagnostics source mismatch.')
    if [r for r in report['rows'] if r['case']=='heat']!=[r for r in original['rows'] if r['case']=='heat']:
        raise ValueError('Heat models changed after dynamic diagnostics.')
    if not all(V1.digest(ROOT/name)==h for name,h in hashes.items()):raise ValueError('Sources changed during audit.')
    return {'all_checks_passed':True,'records':records,'paired_irL1_checks':paired,'unchanged_method_records':unchanged,
        'historical_control_records_unchanged':control_count,'cost_ledger_verified':ledger,
        'all_12_paired_selected_supports_identical':all(p['same_selected_support_original_vs_smoothed'] for p in paired),
        'heat_dynamic_v2_source_binding_valid_via_unchanged_heat_rows':True,
        'source_hashes':hashes,'limits':'A common selected support in these 12 development pairs is descriptive; it neither proves equal methods nor establishes a noise robustness ranking.'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,default=ROOT/'stage2b_reconcile_complete_result.json')
    parser.add_argument('--output',type=Path,default=ROOT/'stage2b_independent_review_v2_result.json');args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve completed review; choose a new version.')
    start=time.perf_counter()
    with threadpool_limits(limits=1):
        preflight=V1.preflight();scan=V1.scan_runner(args.input);additional=additional_audit(args.input)
    result={'stage':'2b_complete_independent_review','all_checks_passed':preflight['all_checks_passed'] and scan['all_checks_passed'] and additional['all_checks_passed'],
        'preflight':preflight,'runner_review':scan,'additional_review':additional,
        'source_script_sha256':V1.digest(Path(__file__)),'helper_script_sha256':V1.digest(ROOT/'stage2b_independent_review.py'),
        'new_solver_calls':0,'new_data_generated':False,'test_data_used':False,'statistical_parameters_changed':False,
        'elapsed_s':time.perf_counter()-start}
    strict_json_write(args.output,result)
    print(json.dumps({'all_checks_passed':result['all_checks_passed'],'records':len(scan['records']),
        'final_points_recertified':sum(r['final_points_recertified'] for r in scan['records']),
        'all_12_paired_selected_supports_identical':additional['all_12_paired_selected_supports_identical'],'elapsed_s':result['elapsed_s']}))


if __name__=='__main__':main()
