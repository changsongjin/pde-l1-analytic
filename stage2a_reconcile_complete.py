"""Finalize 72 frozen-rule development runs after root-reviewed dual certificates.

Only two dual certificates and their containing ordinary-L1 evaluation change.
No L1 solve, data generation, alpha tuning, or beta/support changes occur here.
All historical sources are read-only and the merged output is write-once.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

import stage2a_baseline as B
from stage0_evaluation import SUPPORT_TOL,array_hash,normalize_training,physical_coefficients,strict_json_write
from stage0_l1_solver import certify_l1
from stage2_identifiability import save_matrices,stack_split
from stage2a_reconcile_remaining import _common_frontier
from stage2a_retry_remaining_v4 import certify_arbitrary_dual

ROOT=Path(__file__).resolve().parent
OUTPUT=ROOT/'stage2a_reconcile_complete_result.json'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def reject(value):raise ValueError('nonfinite JSON '+value)
    return json.loads(Path(path).read_text(encoding='utf-8'),parse_constant=reject)


def point_without_readout(point):
    result=copy.deepcopy(point);result.pop('physical_L1_coefficients',None)
    return result


def main():
    if OUTPUT.exists():raise ValueError('write-once output exists; use a new version')
    start=time.perf_counter()
    files=[ROOT/name for name in ['stage2a_heat_pilot_result.json','stage2a_heat_reconciled_result.json',
                                  'stage2a_other_cases_reconciled_result.json']]
    originals=[read(path) for path in files]
    hashes={str(path):digest(path) for path in files}
    for report in originals:
        if (not report['execution_complete'] or report['alpha_frozen']!=B.FROZEN_ALPHA
                or report['alpha_sensitivity']!=B.SENSITIVITY_ALPHAS
                or report['lambda_ratios']!=originals[0]['lambda_ratios']):
            raise ValueError('incomplete campaign or changed frozen selector/grid')
    # Verify the complete chain binding v4 to the same original other-case data.
    chain=[]
    for name in ['stage2a_retry_remaining_v4_result.json','stage2a_retry_remaining_v3_result.json',
                 'stage2a_retry_remaining_v2_result.json']:
        path=ROOT/name;value=read(path);hashes[str(path)]=digest(path)
        parent=ROOT/value['source_file']
        if value['source_sha256']!=digest(parent):raise ValueError('continuation source hash mismatch')
        hashes[str(parent)]=digest(parent)
        chain.append({'file':name,'sha256':digest(path),'parent_file':parent.name,'parent_sha256':digest(parent)})
        for field,filename in value.items():
            if field.endswith('_file') and field!='source_file' and isinstance(filename,str):
                hash_field=field[:-5]+'_sha256'
                if hash_field in value:
                    predecessor=ROOT/filename
                    if value[hash_field]!=digest(predecessor):raise ValueError('additional predecessor mismatch')
                    hashes[str(predecessor)]=digest(predecessor)
    v4=read(ROOT/'stage2a_retry_remaining_v4_result.json')
    if not v4['all_checks_passed'] or not v4['execution_complete']:raise ValueError('v4 mathematical verification incomplete')
    for old in originals:
        manifests=[]
        if 'reconciliation' in old:manifests.append(old['reconciliation']['source_hashes'])
        if 'numerical_reconciliation' in old:manifests.append(old['numerical_reconciliation']['sources'])
        for manifest in manifests:
            for filename,expected in manifest.items():
                if digest(filename)!=expected:raise ValueError('reconciled campaign provenance mismatch')
                hashes[filename]=expected
    merged=copy.deepcopy(originals[0])
    merged['rows']=[copy.deepcopy(row) for old in originals for row in old['rows']]
    row_keys=[(r['case'],r['sigma'],r['seed']) for r in merged['rows']]
    if len(row_keys)!=18 or len(set(row_keys))!=18:raise ValueError('development dataset grid duplicated/incomplete')
    merged.update({'noise':[0.,.01,.05],'cases':list(dict.fromkeys(r['case'] for r in merged['rows'])),
                   'all_checks_passed':False,'execution_complete':False})
    before_rows=copy.deepcopy(merged['rows'])
    matching=[i for i,r in enumerate(merged['rows']) if (r['case'],r['sigma'],r['seed'])==('advection_diffusion',.05,0)]
    if len(matching)!=1:raise ValueError('target dataset ambiguous')
    index=matching[0];row=merged['rows'][index];layer=row['layers']['J1158'];old_method=copy.deepcopy(layer['methods']['L1'])
    continuation=next(r for r in v4['rows'] if (r['case'],r['sigma'],r['seed'],r['layer'],r['method'])
                       ==('advection_diffusion',.05,0,'J1158','L1'))
    cache=ROOT/'stage2_matrices/advection_diffusion_sigma_0p05_seed_0.npz'
    if not cache.exists():raise ValueError('reconciliation may not generate missing matrices')
    with threadpool_limits(limits=1):
        matrices,targets,meta=save_matrices(row['case'],row['sigma'],row['seed'])
        a,b=stack_split(matrices,targets,range(4));av,bv=stack_split(matrices,targets,range(4,6))
        subset=layer['indices_in_maximum_library'];at,va=a[:,subset],av[:,subset]
        an,bn,scales=normalize_training(at,b)
        actual={'A':array_hash(an),'b':array_hash(bn)}
        if (actual!=layer['solver_input_hashes'] or actual!=continuation['input_hashes']
                or scales!=layer['normalization'] or meta['raw_bundle_sha256']!=row['raw_bundle_sha256']
                or array_hash(av)!=row['input_hashes']['validation_matrix']
                or array_hash(bv)!=row['input_hashes']['validation_target']):
            raise ValueError('training/validation/scales/raw inputs changed')
        path=copy.deepcopy(old_method['path']);new_path=continuation['path']
        for field in ['ratios','weights','shape','support_tolerance']:
            if path[field]!=new_path[field]:raise ValueError('path conventions changed')
        weights=np.asarray(path['weights']);new_certificates=[];inherited_count=0
        recert_start=time.perf_counter()
        for j,(old,new,ratio) in enumerate(zip(path['points'],new_path['points'],path['ratios'])):
            if old['ratio']!=ratio or new['ratio']!=ratio or old['lambda']!=new['lambda']:
                raise ValueError('lambda/grid changed')
            beta=np.asarray(new['beta'])
            if new['support']!=np.flatnonzero(np.abs(beta)>SUPPORT_TOL).tolist():raise ValueError('support changed')
            if old['status']=='certified':
                if point_without_readout(old)!=point_without_readout(new):raise ValueError('old certified point changed')
                certificate=certify_l1(an,bn,beta,new['lambda'],weights)
                if not certificate['certified']:raise ValueError('inherited point no longer certifies')
                inherited_count+=1
                continue
            if old['beta']!=new['beta'] or old['support']!=new['support']:raise ValueError('v4 altered primal coefficients/support')
            old_check,new_check=certify_arbitrary_dual(an,bn,beta,new['lambda'],weights,np.asarray(new['alternate_dual']))
            if (old_check['certified'] or not new_check['certified']
                    or new_check['exact_gap_fraction']!=new['certificate']['exact_gap_fraction']
                    or new_check['exact_min_dual_slack']!=new['certificate']['exact_min_dual_slack']
                    or new_check['tolerances']!=old['certificate']['tolerances']):
                raise ValueError('saved alternate-dual certificate failed exact independent recheck')
            replacement=copy.deepcopy(new)
            replacement['root_review_required']=False
            replacement['root_review_accepted']=True
            replacement['certificate_reverified_from_saved_q']=new_check
            replacement['physical_L1_coefficients']=physical_coefficients(beta,scales).tolist()
            path['points'][j]=replacement
            new_certificates.append({'ratio':ratio,'old_certificate':old_check,'new_certificate':new_check,
                                     'beta_and_support_unchanged':True})
        if len(new_certificates)!=2 or inherited_count!=22:raise ValueError('unexpected target modification count')
        recert_s=time.perf_counter()-recert_start
        path.update({'status':'complete','all_solutions_certified':True,
                     'implementation':'unchanged_L1_path_plus_root_reviewed_exact_feasible_dual_certificates',
                     'alternate_certificate_root_review_accepted':True})
        eval_start=time.perf_counter()
        evaluation=B._candidate_records(path,at,b,va,bv,layer['names'],layer['complexities'],row['dataset_id'],
                                         [(i,matrices[i][:,subset],targets[i]) for i in [4,5]])
        eval_s=time.perf_counter()-eval_start
        extra=float(continuation['v4_certificate_seconds'])
        before_solver=float(old_method['time_s']['solver_with_worker_startup'])
        expected_prior=float(continuation['path']['elapsed_s'])-extra
        if abs(before_solver-expected_prior)>1e-8:raise ValueError('historical solver cost mismatch')
        path['elapsed_s']=before_solver+extra
        layer['methods']['L1']={**old_method,'path':path,'evaluation':evaluation,
            'time_s':{**old_method['time_s'],'solver_with_worker_startup':before_solver+extra,
                      'previous_fixed_support_validation':old_method['time_s']['fixed_support_validation'],
                      'fixed_support_validation':eval_s,'saved_q_exact_reverification':recert_s},
            'completion_reconciliation':{'source_hashes':hashes,'alternate_certificate_root_review_accepted':True,
                'certificates_rechecked':new_certificates,'previously_certified_points_unchanged':inherited_count,
                'original_stage0_certificate_false_preserved':True,'same_beta_support_grid_weights':True,
                'fixed_alpha':B.FROZEN_ALPHA,'same_validation_data':True,'no_solver_rerun':True,
                'historical_solver_cost_before_v4_s':before_solver,'v4_certificate_attempts_wall_s':extra,
                'historical_solver_and_certificate_total_s':before_solver+extra},
            'timing_note':'All original attempts and v2/v3/v4 numerical work retained; exact re-verification and evaluation overhead reported separately.'}
        row['common_validation_pareto']=_common_frontier(row)
        B._add_truth_metrics(evaluation,row['truth'])
        row['all_methods_numerically_complete']=True
        row['time_s']['before_complete_reconciliation_dataset_total']=row['time_s']['dataset_total']
        row['time_s']['v4_certificate_attempts']=extra
        row['time_s']['complete_reconciliation_reverification']=recert_s
        row['time_s']['complete_reconciliation_fixed_support_evaluation']=eval_s
        row['time_s']['dataset_total']+=extra+recert_s+eval_s
    unaffected=all(r==before_rows[i] for i,r in enumerate(merged['rows']) if i!=index)
    untouched_methods=all(m==before_rows[i]['layers'][lname]['methods'][mname]
        for i,r in enumerate(merged['rows']) for lname,lib in r['layers'].items() for mname,m in lib['methods'].items()
        if (i,lname,mname)!=(index,'J1158','L1'))
    if not unaffected or not untouched_methods:raise ValueError('unrelated rows or methods changed')
    if not all(digest(p)==h for p,h in hashes.items()):raise ValueError('source artifact changed during reconciliation')
    run_count=sum(len(lib['methods']) for r in merged['rows'] for lib in r['layers'].values())
    complete_count=sum(m['path']['all_solutions_certified'] for r in merged['rows'] for lib in r['layers'].values() for m in lib['methods'].values())
    if run_count!=72 or complete_count!=72:raise ValueError('final 72-method grid incomplete')
    total_solver=sum(m['time_s']['solver_with_worker_startup'] for r in merged['rows'] for lib in r['layers'].values() for m in lib['methods'].values())
    merged.update({'execution_complete':True,'all_checks_passed':True,
                   'completion_meaning':'72 frozen-rule development method paths numerically complete; no formal robustness/novelty claim.'})
    merged['complete_reconciliation']={'sources':hashes,'continuation_provenance_chain':chain,
        'source_script_sha256':digest(Path(__file__)),'root_review_accepted':True,
        'newly_complete_method_runs':1,'unchanged_method_runs':71,'total_method_runs':72,
        'unchanged_dataset_rows':17,'source_files_unchanged':True,
        'beta_support_and_selection_rule_unchanged':True,'two_saved_dual_vectors_exactly_reverified':True,
        'cost_ledger':{'all_method_historical_solver_and_certification_seconds':total_solver,
                       'v4_target_method_certificate_seconds':extra,
                       'v4_whole_proof_script_recorded_seconds':v4['elapsed_s'],
                       'this_reconciliation_recorded_seconds':time.perf_counter()-start,
                       'saved_q_reverification_seconds':recert_s,'new_selection_evaluation_seconds':eval_s,
                       'note':'Totals retain every recorded solver retry including v4; raw field-generation time absent from historical caches remains unavailable. Verification overhead is separate from optimizer runtime.'}}
    strict_json_write(OUTPUT,merged)
    selected=evaluation['primary_selected_candidate_id']
    chosen=next(c for c in evaluation['candidates'] if c['candidate_id']==selected)
    print(json.dumps({'all_checks_passed':True,'complete_method_runs':complete_count,
                      'selected_support':chosen['support'],'fixed_alpha':B.FROZEN_ALPHA,
                      'historical_solver_total_s':total_solver,'elapsed_s':time.perf_counter()-start},ensure_ascii=False))


if __name__=='__main__':main()
