"""Read-only full-library audit accepting independently verified saved duals.

The delivered review stays unchanged. Its data/residual/selection checks are
reused with a narrowly scoped certificate adapter; all alternate q vectors
are exactly reverified before the original checker is called. No solver runs.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
import stage2c_review as V1
from stage0_evaluation import strict_json_write
from stage0_l1_solver import certify_l1 as original_certificate
from stage2a_retry_remaining_v4 import certify_arbitrary_dual

ROOT=Path(__file__).resolve().parent
ORIGINAL_REVIEW=V1.review_method


def review_with_saved_duals(record,method,a,b,av,bv,an,bn,names,costs,dataset_id):
    weights=np.asarray(record['path']['weights']);alternates=[]
    for index,point in enumerate(record['path']['points']):
        if point.get('alternate_dual') is None:continue
        beta=np.asarray(point['beta']);old,new=certify_arbitrary_dual(an,bn,beta,point['lambda'],weights,np.asarray(point['alternate_dual']))
        if (not point.get('root_review_accepted',False) or point['previous_stage0_certificate']['certified']
            or not new['certified'] or not new['exact_dual_feasibility']
            or not new['exact_direct_minus_stable_gap_is_zero']
            or new['exact_gap_fraction']!=point['certificate']['exact_gap_fraction']
            or new['exact_min_dual_slack']!=point['certificate']['exact_min_dual_slack']
            or new['tolerances']!=old['tolerances']):raise ValueError('Saved arbitrary dual verification failed.')
        alternates.append((index,point,beta,new,old))

    def verified_certificate(matrix,target,beta,lam,given_weights):
        if matrix is not an or target is not bn:raise ValueError('Unexpected certificate input context.')
        old=original_certificate(matrix,target,beta,lam,given_weights)
        if old['certified']:return old
        for _,point,expected_beta,new,_ in alternates:
            if lam==point['lambda'] and np.array_equal(beta,expected_beta) and np.array_equal(given_weights,weights):
                return new
        return old  # The original caller will reject any other failed point.

    saved=V1.certify_l1
    try:
        V1.certify_l1=verified_certificate
        result=ORIGINAL_REVIEW(record,method,a,b,av,bv,an,bn,names,costs,dataset_id)
    finally:V1.certify_l1=saved
    result['saved_duals_independently_reverified']=[{'point_index':i,'ratio':p['ratio'],
        'exact_dual_feasibility':new['exact_dual_feasibility'],
        'exact_gap_identity_verified':new['exact_direct_minus_stable_gap_is_zero'],
        'dual_gap_relative':new['dual_gap_relative'],
        'old_false_certificate_preserved':not p['previous_stage0_certificate']['certified'],
        'original_stage0_current_recheck':old['certified']} for i,p,_,new,old in alternates]
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',type=Path,default=ROOT/'stage2c_reconcile_complete_result.json')
    parser.add_argument('--output',type=Path,default=ROOT/'stage2c_review_v2_result.json');args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve completed review; choose a new version.')
    start=time.perf_counter()
    old_preflight=V1.read(ROOT/'stage2c_preflight_result.json')
    if old_preflight['source_script_sha256']!=V1.digest(ROOT/'stage2c_review.py'):
        raise ValueError('Original delivered review source changed.')
    with threadpool_limits(limits=1):
        matrices=V1.preflight()
        saved=V1.review_method
        try:
            V1.review_method=review_with_saved_duals
            pilot=V1.scan_pilot(args.pilot)
        finally:V1.review_method=saved
    source=V1.read(args.pilot);completion=source['numerical_completion']
    if not all(V1.digest(p)==h for p,h in completion['sources'].items()):raise ValueError('Reconciliation provenance changed.')
    if completion['source_script_sha256']!=V1.digest(ROOT/'stage2c_reconcile_complete.py'):
        raise ValueError('Reconciliation implementation changed.')
    original=V1.read(ROOT/'stage2c_maximum_library_pilot_result.json');same=changed=0
    for old,row in zip(original['rows'],source['rows']):
        for method,value in row['layers']['J1212']['methods'].items():
            if 'completion_provenance' in value:changed+=1
            else:
                if value!=old['layers']['J1212']['methods'][method]:raise ValueError('Unrelated method changed.')
                same+=1
    if same!=completion['unchanged_methods'] or changed!=completion['changed_methods'] or same+changed!=36:
        raise ValueError('Method preservation count mismatch.')
    result={'stage':'2c_independent_complete_review_with_saved_duals',
        'all_checks_passed':matrices['all_checks_passed'] and pilot['all36_paths_numerically_complete'],
        'campaign_completion_gate':pilot['all36_paths_numerically_complete'],'matrix_preflight':matrices,
        'pilot_review':pilot,'unchanged_method_records':same,'changed_method_records':changed,
        'source_script_sha256':V1.digest(Path(__file__)),
        'helper_script_sha256':{n:V1.digest(ROOT/n) for n in ['stage2c_review.py','stage2a_retry_remaining_v4.py','stage0_l1_solver.py']},
        'new_L1_solver_calls':0,'new_data_generated':False,'test_data_used':False,'elapsed_s':time.perf_counter()-start,
        'certificate_scope':'Original full-column KKT and tolerances, with exact saved feasible-dual lower bounds only where independently verified; old false certificates preserved.',
        'limits':['Numerical completeness and reproducible fixed selection are not evidence of correct support or formal multi-seed robustness.',
                  'All six true PDEs already lie in the polynomial library; supplement is a distracting-function stress test.']}
    strict_json_write(args.output,result)
    print({'all_checks_passed':result['all_checks_passed'],'complete_paths':pilot['recorded_methods'],
        'exact_saved_duals':sum(len(r['saved_duals_independently_reverified']) for r in pilot['rows']),
        'elapsed_s':result['elapsed_s']})


if __name__=='__main__':main()
