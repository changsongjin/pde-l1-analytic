"""Complete the frozen 36-record comparison using four certified tail solves.

Read-only sources; no optimizer, changed statistic, new data or overwrite.
Only two ordinary-L1 method records change; other 34 records stay identical.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
import stage2a_baseline as B
from stage0_evaluation import array_hash, normalize_training, physical_coefficients, strict_json_write
from stage2_identifiability import stack_split
from stage2a_reconcile_remaining import _common_frontier
from stage2b_compare import digest, load, validate_path

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'stage2b_compare_result.json'
RETRY=ROOT/'stage2b_numerical_retry_result.json'
OUTPUT=ROOT/'stage2b_reconcile_complete_result.json'


def without_readout(point):
    value=copy.deepcopy(point);value.pop('physical_L1_coefficients',None);return value


def main():
    if OUTPUT.exists():raise FileExistsError('Preserve completed result; choose a new version.')
    start=time.perf_counter();source=load(SOURCE);retry=load(RETRY)
    hashes={p.name:digest(p) for p in [SOURCE,RETRY]}
    if not source['execution_complete'] or not retry['execution_complete'] or not retry['all_checks_passed']:
        raise ValueError('Initial attempts and numerical continuation must be finished.')
    if source['alpha_frozen']!=B.FROZEN_ALPHA or source['alpha_sensitivity']!=B.SENSITIVITY_ALPHAS:
        raise ValueError('Frozen selector changed.')
    for name,h in {**source['source_sha256'],**source['gate_sources'],**source['control_sources'],
                   **retry['helper_sha256']}.items():
        if digest(ROOT/name)!=h:raise ValueError('Source/gate/helper hash changed: '+name)
    if digest(ROOT/'stage2b_numerical_retry.py')!=retry['source_script_sha256']:raise ValueError('Retry source changed.')
    merged=copy.deepcopy(source);changed=set();audits=[]
    with threadpool_limits(limits=1):
        for continuation in retry['rows']:
            key=(continuation['cohort'],continuation['case'],continuation['sigma'],continuation['layer'],continuation['method'])
            if key in changed or key[3:]!=('J1158','L1'):raise ValueError('Unexpected/duplicate repair.')
            row=next(r for r in merged['rows'] if (r['cohort'],r['case'],r['sigma'])==key[:3])
            lib=row['layers'][key[3]];old=copy.deepcopy(lib['methods'][key[4]])
            for field,kind in [('source_path','path'),('source_job','job')]:
                if digest(ROOT/continuation[field])!=continuation['source_hashes'][kind]:raise ValueError('Fixed task source changed.')
            if old['execution_provenance']['sha256']!=continuation['source_hashes']['path']:raise ValueError('Wrong original path.')
            if old['path']['all_solutions_certified']:raise ValueError('Do not replace an already complete method.')
            tag=f"{row['case']}_sigma_{row['sigma']}_seed_1_ids12_17"
            matrix=ROOT/'stage2b_matrices'/(tag+'.npz')
            if digest(matrix)!=row['matrix_metadata']['npz_sha256']:raise ValueError('Matrix cache changed.')
            with np.load(matrix,allow_pickle=False) as data:
                matrices=[data[f'a_{i}'].copy() for i in range(6)];targets=[data[f'b_{i}'].copy() for i in range(6)]
            a,b=stack_split(matrices,targets,range(4));av,bv=stack_split(matrices,targets,range(4,6))
            if row['input_hashes']!={'train_matrix':array_hash(a),'train_target':array_hash(b),
                    'validation_matrix':array_hash(av),'validation_target':array_hash(bv)}:raise ValueError('Split data changed.')
            subset=lib['indices_in_maximum_library'];at,va=a[:,subset],av[:,subset]
            an,bn,scales=normalize_training(at,b)
            if (scales!=lib['normalization'] or scales!=continuation['normalization']
                or {'A':array_hash(an),'b':array_hash(bn)}!=continuation['input_hashes']
                or continuation['input_hashes']!=lib['solver_input_hashes']
                or continuation['raw_bundle_sha256']!=row['raw_bundle_sha256']):raise ValueError('Normalized inputs changed.')
            path=copy.deepcopy(old['path']);completed=continuation['path'];replaced=[]
            for field in ['ratios','weights','shape','support_tolerance','budget_s','path_metadata']:
                if path[field]!=completed[field]:raise ValueError('Path convention/weights changed: '+field)
            for i,(prior,new) in enumerate(zip(path['points'],completed['points'])):
                if prior['status']=='certified':
                    if without_readout(prior)!=without_readout(new):raise ValueError('Previously certified point changed.')
                else:
                    if prior['ratio']!=new['ratio'] or (prior['lambda'] is not None and prior['lambda']!=new['lambda']):
                        raise ValueError('Requested lambda changed.')
                    path['points'][i]=copy.deepcopy(new)
                    path['points'][i]['physical_L1_coefficients']=physical_coefficients(new['beta'],scales).tolist()
                    replaced.append(i)
            if replaced!=[22,23]:raise ValueError('Unexpected repair scope.')
            path.update({'status':'complete','all_solutions_certified':True,'elapsed_s':completed['elapsed_s']})
            cert_start=time.perf_counter();validate_path(path,an,bn);cert_s=time.perf_counter()-cert_start
            prior_seconds=old['time_s']['solver_with_worker_startup'];extra=continuation['retry_elapsed_s']
            if abs(path['elapsed_s']-prior_seconds-extra)>1e-9:raise ValueError('Solver cost omitted or duplicated.')
            eval_start=time.perf_counter()
            evaluation=B._candidate_records(path,at,b,va,bv,lib['names'],lib['complexities'],row['dataset_id'],
                    [(row['splits']['validation_trajectories'][i-4],matrices[i][:,subset],targets[i]) for i in (4,5)])
            eval_s=time.perf_counter()-eval_start
            B._add_truth_metrics(evaluation,row['truth'])
            provenance={'source_file':RETRY.name,'source_sha256':hashes[RETRY.name],
                'replaced_point_indices':replaced,'inherited_22_certified_points_unchanged':True,
                'all_24_original_certificates_reverified':True,'new_solver_calls_during_reconciliation':0,
                'same_objective_weights_lambda_grid_data_and_selection_rule':True,
                'prior_initial_solver_seconds':prior_seconds,'retry_solver_seconds':extra,
                'initial_and_retry_solver_seconds':path['elapsed_s'],'fixed_alpha':B.FROZEN_ALPHA}
            lib['methods'][key[4]]={**old,'path':path,'evaluation':evaluation,
                'new_solver_seconds_this_comparison':old['new_solver_seconds_this_comparison']+extra,
                'time_s':{**old['time_s'],'solver_with_worker_startup':path['elapsed_s'],
                    'previous_fixed_support_validation':old['time_s']['fixed_support_validation'],
                    'fixed_support_validation':eval_s,'all_24_certificate_reverification':cert_s},
                'numerical_completion_provenance':provenance}
            row['all_methods_numerically_complete']=True
            row['common_validation_pareto']=_common_frontier(row)
            row['time_s']['numerical_retry_solver_seconds']=extra
            row['time_s']['completion_certificate_reverification_seconds']=cert_s
            row['time_s']['completion_evaluation_seconds']=eval_s
            audits.append({'identity':list(key),**provenance});changed.add(key)
    unchanged=0
    for prior,row in zip(source['rows'],merged['rows']):
        for lname,lib in row['layers'].items():
            for method,value in lib['methods'].items():
                key=(row['cohort'],row['case'],row['sigma'],lname,method)
                if key not in changed:
                    if value!=prior['layers'][lname]['methods'][method]:raise ValueError('Unrelated method record changed.')
                    unchanged+=1
    records=[m for r in merged['rows'] for l in r['layers'].values() for m in l['methods'].values()]
    if len(changed)!=2 or unchanged!=34 or len(records)!=36 or not all(m['path']['all_solutions_certified'] for m in records):
        raise ValueError('Expected complete 36-method grid.')
    if any(digest(ROOT/name)!=h for name,h in hashes.items()):raise ValueError('Source files changed during reconciliation.')
    merged.update({'execution_complete':True,'all_checks_passed':True,'uncertified_jobs':[],
        'completion_meaning':'36 frozen-rule development paths numerically complete; no general algorithm ranking established.'})
    merged['numerical_completion']={'source_hashes':hashes,'source_script_sha256':digest(Path(__file__)),
        'changed_method_records':2,'unchanged_method_records':34,'records':audits,'old_partial_sources_preserved':True,
        'cost_ledger':{'new_comparison_solver_seconds_including_retries':sum(m['new_solver_seconds_this_comparison'] for m in records),
            'recorded_method_solver_seconds_including_historical_controls':sum(m['time_s']['solver_with_worker_startup'] for m in records),
            'retry_solver_seconds':sum(r['retry_elapsed_s'] for r in retry['rows']),
            'retry_whole_script_seconds':retry['elapsed_s'],
            'reconciliation_seconds':time.perf_counter()-start,
            'note':'All original attempts and numerical retries retained. Historical controls are not zero-cost methods; unavailable raw-generation timing is not invented.'}}
    strict_json_write(OUTPUT,merged)
    print(json.dumps({'all_checks_passed':True,'complete_records':36,'unchanged_records':34,
                     'cost_ledger':merged['numerical_completion']['cost_ledger']}))


if __name__=='__main__':main()
