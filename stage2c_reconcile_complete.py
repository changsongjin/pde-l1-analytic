"""Reconcile certified maximum-library continuations without rerunning solvers.

Recheck every final/outer solution, including saved exact feasible duals.
Only repaired method records change; support, score and NP identities retain
the frozen contracts. Original campaigns and failed attempts stay read-only.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
import stage1_extended_evaluation as E
import stage2a_baseline as B
from stage0_evaluation import array_hash, normalize_training, physical_coefficients, strict_json_write
from stage0_l1_solver import certify_l1, SUPPORT_TOL
from stage2_identifiability import stack_split
from stage2a_l1_paths import DEFAULT_RATIOS
from stage2a_retry_remaining_v4 import certify_arbitrary_dual
from stage2a_diagnosis import diagnose_method
from stage2b_compare import digest, load
from stage2c_extended_matrices import save_matrices
from stage2c_maximum_library_pilot import truth_after_selection

ROOT=Path(__file__).resolve().parent


def recheck(a,b,point,weights):
    beta=np.asarray(point['beta'])
    if point['support']!=np.flatnonzero(abs(beta)>SUPPORT_TOL).tolist():raise ValueError('Support changed.')
    old=certify_l1(a,b,beta,point['lambda'],weights)
    if point.get('alternate_dual') is None:
        if not old['certified']:raise ValueError('Claimed original certificate fails.')
        return {'kind':'original_stage0','certificate':old}
    old,new=certify_arbitrary_dual(a,b,beta,point['lambda'],weights,np.asarray(point['alternate_dual']))
    if (not new['certified'] or not new['exact_dual_feasibility']
        or not new['exact_direct_minus_stable_gap_is_zero']
        or new['exact_gap_fraction']!=point['certificate']['exact_gap_fraction']
        or new['exact_min_dual_slack']!=point['certificate']['exact_min_dual_slack']
        or new['tolerances']!=old['tolerances'] or point['previous_stage0_certificate']['certified']):
        raise ValueError('Saved feasible dual failed exact re-verification or lost old false certificate.')
    return {'kind':'root_reviewed_exact_feasible_dual','certificate':new,'original_stage0_recheck':old,
            'old_false_certificate_preserved':True,'beta_and_q_reverified_without_optimization':True}


def audit_path(a,b,path):
    if path['ratios']!=DEFAULT_RATIOS.tolist() or len(path['points'])!=24 or path['support_tolerance']!=1e-5:
        raise ValueError('Frozen path contract changed.')
    weights=np.ones(a.shape[1]);outer_checks=[]
    for outer in path.get('outer',[]):
        lam=.001*float(np.max(abs(a.T@b)/weights))
        if outer['status']!='certified' or not np.isclose(lam,outer['lambda'],rtol=2e-13,atol=0):raise ValueError('Outer solve incomplete/objective changed.')
        outer_checks.append(recheck(a,b,outer,weights))
        weights=1./(abs(np.asarray(outer['beta']))+1e-8)
    if not np.array_equal(weights,path['weights']):raise ValueError('Original final weights changed.')
    maximum=float(np.max(abs(a.T@b)/weights));checks=[]
    for ratio,point in zip(DEFAULT_RATIOS,path['points']):
        if (point['ratio']!=ratio or point['status']!='certified'
            or not np.isclose(point['lambda'],maximum*ratio,rtol=2e-13,atol=0)):
            raise ValueError('Requested lambda incomplete or changed.')
        checks.append(recheck(a,b,point,weights))
    return {'final_points':len(checks),'outer_points':len(outer_checks),
            'alternate_dual_count':sum(c['kind']!='original_stage0' for c in checks),
            'max_reverified_gap_relative':max(c['certificate']['dual_gap_relative'] for c in checks)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,default=ROOT/'stage2c_maximum_library_pilot_result.json')
    parser.add_argument('--retry',type=Path,default=ROOT/'stage2c_numerical_retry_result.json')
    parser.add_argument('--output',type=Path,default=ROOT/'stage2c_reconcile_complete_result.json')
    parser.add_argument('--diagnosis',type=Path,default=ROOT/'stage2c_complete_diagnosis_result.json');args=parser.parse_args()
    if args.output.exists() or args.diagnosis.exists():raise FileExistsError('Preserve completed outputs; choose new versions.')
    start=time.perf_counter();source=load(args.source);retry=load(args.retry)
    if not source['execution_complete'] or not retry['execution_complete'] or not retry['all_checks_passed']:
        raise ValueError('Require finished initial campaign and complete numerical continuation; partial inputs remain preserved.')
    if retry['source_sha256']!=digest(args.source):raise ValueError('Retry is bound to a different source.')
    snapshots={str(args.source.resolve()):digest(args.source),str(args.retry.resolve()):digest(args.retry)}
    for name,h in {**source['source_sha256'],**source['gate_sources'],**retry['helper_sha256']}.items():
        if digest(ROOT/name)!=h:raise ValueError('Source/gate/helper changed: '+name)
    if (source['alpha_frozen']!=B.FROZEN_ALPHA or source['alpha_sensitivity']!=B.SENSITIVITY_ALPHAS
        or source['lambda_ratios']!=DEFAULT_RATIOS.tolist() or source['test_data_used']):raise ValueError('Statistical contract changed.')
    merged=copy.deepcopy(source);repairs={(r['case'],r['sigma'],r['method']):r for r in retry['rows']}
    if len(repairs)!=len(retry['rows']):raise ValueError('Duplicate repair rows.')
    changed=set();audits=[];diagnosis=[];recert_seconds=0.;evaluation_seconds=0.
    with threadpool_limits(limits=1):
        for dataset in merged['rows']:
            case,sigma=dataset['case'],dataset['sigma'];lib=dataset['layers']['J1212']
            poly=ROOT/'stage2_matrices'/f"{case}_sigma_{repr(float(sigma)).replace('.', 'p')}_seed_0.npz"
            extra=ROOT/'stage2c_matrices'/f'{case}_sigma_{sigma}_seed_0_supplement54.npz'
            if not poly.exists() or not extra.exists():raise ValueError('Reconciliation may not regenerate missing matrices.')
            matrices,targets,meta=save_matrices(case,sigma,0)
            a,b=stack_split(matrices,targets,range(4));av,bv=stack_split(matrices,targets,range(4,6))
            an,bn,scales=normalize_training(a,b)
            if (meta['raw_bundle_sha256']!=dataset['raw_bundle_sha256'] or meta['names']!=lib['names']
                or {'train_matrix':array_hash(a),'train_target':array_hash(b),'validation_matrix':array_hash(av),'validation_target':array_hash(bv)}!=dataset['input_hashes']
                or {'A':array_hash(an),'b':array_hash(bn)}!=lib['solver_input_hashes'] or scales!=lib['normalization']
                or an.shape[1]!=(1213 if case=='wave' else 1212)):raise ValueError('Frozen extended data/normalization changed.')
            expected_costs=meta['old_polynomial_metadata']['complexities']+[E.term_complexity(r['name']) for r in meta['supplement_recipes']]
            if expected_costs!=lib['complexities']:raise ValueError('Nonpolynomial or polynomial cost changed.')
            for method,old in list(lib['methods'].items()):
                key=(case,sigma,method);continuation=repairs.get(key)
                if continuation is not None:
                    if old['path']['all_solutions_certified']:raise ValueError('Do not replace an already complete path.')
                    if continuation['input_hashes']!=lib['solver_input_hashes'] or continuation['raw_bundle_sha256']!=dataset['raw_bundle_sha256']:
                        raise ValueError('Retry solver data differs.')
                    if (digest(ROOT/continuation['source_path_file'])!=continuation['source_path_sha256']
                        or old['path_source_sha256']!=continuation['source_path_sha256']):raise ValueError('Original task path changed.')
                    path=copy.deepcopy(old['path']);new=continuation['path'];replaced=[];dual_audit=[]
                    for field in ['ratios','weights','shape','support_tolerance','budget_s','path_metadata','outer','outer_stop']:
                        if path[field]!=new[field]:raise ValueError('Final-weight continuation changed '+field)
                    for i,(prior,point) in enumerate(zip(path['points'],new['points'])):
                        if prior['status']=='certified':
                            if prior!=point:raise ValueError('Previously certified point changed.')
                            continue
                        if prior['ratio']!=point['ratio'] or (prior['lambda'] is not None and prior['lambda']!=point['lambda']):raise ValueError('Continuation lambda changed.')
                        checked=recheck(an,bn,point,np.asarray(path['weights']))
                        replacement=copy.deepcopy(point)
                        replacement['physical_L1_coefficients']=physical_coefficients(point['beta'],scales).tolist()
                        if checked['kind']!='original_stage0':
                            replacement['certificate_reverified_from_saved_q']=checked['certificate']
                            replacement['root_review_accepted']=True
                            replacement['root_review_rule']='stage2a_retry_remaining_v4 arbitrary feasible dual, unchanged full-column KKT and tolerances'
                            dual_audit.append({'point_index':i,**checked})
                        path['points'][i]=replacement;replaced.append(i)
                    if not replaced:raise ValueError('Repair did not replace any missing point.')
                    extra_seconds=continuation['retry_elapsed_s'];prior_seconds=old['time_s']['solver_with_worker_startup']
                    if abs(new['elapsed_s']-prior_seconds-extra_seconds)>1e-9:raise ValueError('Retry cost lost or duplicated.')
                    path.update({'status':'complete','all_solutions_certified':True,'elapsed_s':new['elapsed_s']})
                    eval_start=time.perf_counter()
                    evaluation=B._candidate_records(path,a,b,av,bv,lib['names'],lib['complexities'],dataset['dataset_id'],
                        [(i,matrices[i],targets[i]) for i in (4,5)])
                    elapsed_eval=time.perf_counter()-eval_start;evaluation_seconds+=elapsed_eval
                    truth_after_selection(evaluation,dataset['truth'])
                    lib['methods'][method]={**old,'path':path,'evaluation':evaluation,
                        'time_s':{**old['time_s'],'solver_with_worker_startup':path['elapsed_s'],
                            'completion_fixed_support_validation':elapsed_eval},
                        'completion_provenance':{'retry_file':args.retry.name,'retry_sha256':digest(args.retry),
                            'replaced_point_indices':replaced,'prior_certified_points_unchanged':24-len(replaced),
                            'original_attempt_seconds':prior_seconds,'retry_attempt_seconds':extra_seconds,
                            'same_objective_weights_threshold_and_selection':True,'saved_dual_rechecks':dual_audit}}
                    changed.add(key)
                result=lib['methods'][method]
                checked_start=time.perf_counter();path_audit=audit_path(an,bn,result['path']);recert_seconds+=time.perf_counter()-checked_start
                ev=result['evaluation']
                selected=min(ev['candidates'],key=lambda c:(c['validation_error']+.005*c['complexity'],c['complexity'],c['validation_error'],tuple(c['support_indices'])))
                if not ev['selection_eligible'] or ev['primary_selected_candidate_id']!=selected['candidate_id']:
                    raise ValueError('Frozen primary selector mismatch.')
                for candidate in ev['candidates']:
                    metric=E.support_metrics(candidate['support'],dataset['truth'])
                    if any(candidate['truth_evaluation_after_selection'][k]!=v for k,v in metric.items()):raise ValueError('Extended support metric mismatch.')
                audits.append({'case':case,'sigma':sigma,'method':method,**path_audit})
                diagnosis.append(diagnose_method(args.output.name,dataset,'J1212',lib,method,result,source['alpha_frozen']))
            dataset['all_methods_numerically_complete']=True
    untouched=0
    for before,after in zip(source['rows'],merged['rows']):
        for name,record in after['layers']['J1212']['methods'].items():
            if (after['case'],after['sigma'],name) not in changed:
                if record!=before['layers']['J1212']['methods'][name]:raise ValueError('Unrelated method changed.')
                untouched+=1
    expected={(case,sigma,m) for case in source['cases'] for sigma in source['noise'] for m in source['methods']}
    if ({(r['case'],r['sigma'],r['method']) for r in audits}!=expected or len(audits)!=36
        or changed!=set(repairs) or untouched+len(changed)!=36):raise ValueError('Complete grid/repair scope mismatch.')
    if any(digest(p)!=h for p,h in snapshots.items()):raise ValueError('Source snapshot changed.')
    total=sum(m['time_s']['solver_with_worker_startup'] for d in merged['rows'] for m in d['layers']['J1212']['methods'].values())
    merged.update({'all_checks_passed':True,'execution_complete':True,'numerically_unfinished_tasks':[]})
    merged['numerical_completion']={'sources':snapshots,'source_script_sha256':digest(Path(__file__)),
        'changed_methods':len(changed),'unchanged_methods':untouched,'complete_methods':36,
        'reverified_final_points':sum(a['final_points'] for a in audits),'reverified_outer_points':sum(a['outer_points'] for a in audits),
        'exact_saved_dual_count':sum(a['alternate_dual_count'] for a in audits),'path_audits':audits,
        'new_solver_calls':0,'old_failure_sources_preserved':True,'test_data_used':False,
        'cost_ledger':{'all_initial_and_retry_solver_seconds':total,
            'retry_solver_seconds':sum(r['retry_elapsed_s'] for r in retry['rows']),
            'retry_whole_script_seconds':retry['elapsed_s'],'all_path_reverification_seconds':recert_seconds,
            'new_fixed_support_evaluation_seconds':evaluation_seconds,'reconciliation_seconds':time.perf_counter()-start,
            'note':'Initial timeouts and every retry retained; shared preprocessing and unavailable prior evaluation timing are not invented.'}}
    strict_json_write(args.output,merged)
    diagnostic={'stage':'2c_maximum_library_development_diagnosis','all_checks_passed':True,
        'source_file':args.output.name,'source_sha256':digest(args.output),'records':diagnosis,
        'numerically_complete_runs':36,'classification_counts_descriptive_only':dict(Counter(r['classification'] for r in diagnosis)),
        'new_solver_calls':0,'test_data_used':False,'frozen_alpha':.005,
        'limits':['One development noise seed is not a formal robustness ranking.',
            'All six true structures are polynomial; the 54 NP columns test distracting function families, not recovery of true nonpolynomial PDEs.']}
    strict_json_write(args.diagnosis,diagnostic)
    print(json.dumps({'all_checks_passed':True,'complete_methods':36,'changed_methods':len(changed),
        'unchanged_methods':untouched,'cost_ledger':merged['numerical_completion']['cost_ledger'],
        'diagnosis_counts':diagnostic['classification_counts_descriptive_only']}))


if __name__=='__main__':main()
