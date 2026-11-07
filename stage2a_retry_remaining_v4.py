"""Improve only the dual lower bound for two remaining fixed-L1 candidates.

The primal beta, objective, support threshold and KKT/gap tolerances do not
change. Every saved binary64 dual q is checked with exact rational arithmetic.
An alternate valid certificate never implies the old residual-scaling function
returned True. Root review is required before changing the research stage gate.
"""
from __future__ import annotations

import copy
from fractions import Fraction
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from stage0_evaluation import array_hash, normalize_training, strict_json_write
from stage0_l1_solver import certify_l1, SUPPORT_TOL, KKT_ABS_TOL, KKT_REL_TOL, GAP_REL_TOL
from stage2_identifiability import save_matrices, stack_split

ROOT=Path(__file__).resolve().parent
OUTPUT=ROOT/'stage2a_retry_remaining_v4_result.json'
POINT_BUDGET=90.
DUAL_INWARD_MARGIN=1e-10


def exact_dot(values, other_ratios):
    """Exact dot product of stored binary64 values, using dyadic integers."""
    total, exponent = 0, 0
    for value,(num2,exp2) in zip(values,other_ratios):
        num1,den1=float(value).as_integer_ratio()
        power=den1.bit_length()-1+exp2
        if power>exponent:
            total <<= power-exponent
            exponent=power
        total += (num1*num2) << (exponent-power)
    return Fraction(total,1<<exponent)


def ratios(values):
    result=[]
    for value in values:
        numerator,denominator=float(value).as_integer_ratio()
        result.append((numerator,denominator.bit_length()-1))
    return result


def exact_primal_dual(a,b,beta,q,lam,weights):
    """Exact feasibility and two algebraically equivalent gaps for the saved q.

Each float is interpreted as its exact dyadic rational. This is not a tolerance
check on a rounded A.T@q: all p inequalities are compared exactly. Nonnegative
slacks are never clipped. Residuals are recomputed exactly from A,b,beta.
"""
    start=time.perf_counter()
    active=np.flatnonzero(beta!=0.)
    beta_ratios=ratios(beta[active])
    residual=[Fraction.from_float(float(bi))-exact_dot(ai[active],beta_ratios) for ai,bi in zip(a,b)]
    q_ratios=ratios(q)
    correlations=[exact_dot(a[:,j],q_ratios) for j in range(a.shape[1])]
    penalty=[Fraction.from_float(float(lam))*Fraction.from_float(float(w)) for w in weights]
    slacks=[pen-abs(corr) for pen,corr in zip(penalty,correlations)]
    beta_fraction=[Fraction.from_float(float(v)) for v in beta]
    q_fraction=[Fraction.from_float(float(v)) for v in q]
    half=Fraction(1,2)
    primal=half*sum(r*r for r in residual)+sum(p*abs(v) for p,v in zip(penalty,beta_fraction))
    dual=exact_dot(b,q_ratios)-half*sum(v*v for v in q_fraction)
    direct=primal-dual
    stable=half*sum((r-v)**2 for r,v in zip(residual,q_fraction))
    stable+=sum(abs(v)*(p-(1 if v>0 else -1 if v<0 else 0)*c)
                for v,p,c in zip(beta_fraction,penalty,correlations))
    assert stable==direct, 'Exact primal-dual identity failed'
    return {'primal':primal,'dual':dual,'gap':direct,'stable_gap':stable,
            'min_slack':min(slacks),'feasible':all(s>=0 for s in slacks),
            'n_inequalities':len(slacks),'elapsed_s':time.perf_counter()-start}


def certify_arbitrary_dual(a,b,beta,lam,weights,q):
    """Keep the original beta KKT checks; replace only the feasible dual candidate."""
    original=certify_l1(a,b,beta,lam,weights)
    exact=exact_primal_dual(a,b,beta,q,lam,weights)
    # Identical objective-scale and numerical tolerance values to stage 0.
    scale=max(original['primal_objective'],.5*float(b@b)*1e-12,1e-30)
    allowed=GAP_REL_TOL*scale+original['roundoff_allowance']
    kkt_ok=(original['kkt_max_scaled_abs']<=KKT_ABS_TOL
            and original['kkt_max_relative_to_penalty']<=KKT_REL_TOL)
    gap_ok=Fraction(0)<=exact['gap']<=Fraction.from_float(allowed)
    feasible=exact['feasible']
    new={**original,'certified':bool(kkt_ok and gap_ok and feasible),
         'certificate_kind':'original_KKT_plus_exactly_verified_arbitrary_feasible_dual',
         'original_stage0_certificate_passed':original['certified'],
         'dual_objective':float(exact['dual']),'dual_gap':float(exact['gap']),
         'dual_gap_relative':float(exact['gap']/Fraction.from_float(scale)),
         'dual_scale':None,
         'dual_feasibility_max_violation':float(max(Fraction(0),-exact['min_slack'])),
         'exact_dual_feasibility':bool(feasible),'exact_min_dual_slack':str(exact['min_slack']),
         'min_dual_slack_float':float(exact['min_slack']),
         'exact_inequalities_checked':exact['n_inequalities'],
         'exact_primal_fraction':str(exact['primal']), 'exact_dual_fraction':str(exact['dual']),
         'exact_gap_fraction':str(exact['gap']),
         'exact_direct_minus_stable_gap_is_zero':exact['gap']==exact['stable_gap'],
         'allowed_gap_in_original_units':allowed,
         'exact_verification_s':exact['elapsed_s'],
         'note':'beta and original full-column KKT unchanged; a different feasible dual gives a tighter lower bound.'}
    return original,new


def projected_dual(a,b,beta,lam,weights):
    """Project toward active equalities, then shrink against every dual inequality."""
    residual=b-a@beta
    active=np.flatnonzero(beta!=0.)
    q=residual.copy()
    if active.size:
        columns=a[:,active]
        u,s,vt=np.linalg.svd(columns,full_matrices=False)
        if len(s)!=len(active) or s[-1]<=s[0]*np.finfo(float).eps*max(columns.shape):
            raise RuntimeError('Active dual projection is numerically rank deficient; no constraint removed.')
        target=lam*weights[active]*np.sign(beta[active])
        for _ in range(3):
            # If A_S=U diag(s) V.T, this is its minimum-norm row-space correction.
            q+=u@((vt@(target-columns.T@q))/s)
    max_ratio=float(np.max(np.abs(a.T@q)/(lam*weights)))
    shrink=min(1.,1./max(max_ratio,1e-300))*(1.-DUAL_INWARD_MARGIN)
    q*=shrink
    return q,{'active_equalities':int(active.size),'projection_corrections':3,
              'full_library_shrink':shrink,'fixed_inward_margin':DUAL_INWARD_MARGIN,
              'q_minus_original_residual_norm':float(np.linalg.norm(q-residual))}


def self_checks():
    a=np.eye(3);b=np.array([2.,-.7,.05]);weights=np.array([1.,2.,.5]);lam=.1
    beta=np.sign(b)*np.maximum(np.abs(b)-lam*weights,0.)
    q=b-a@beta
    _,good=certify_arbitrary_dual(a,b,beta,lam,weights,q*(1.-DUAL_INWARD_MARGIN))
    _,bad_dual=certify_arbitrary_dual(a,b,beta,lam,weights,2*q)
    _,bad_primal=certify_arbitrary_dual(a,b,np.zeros(3),lam,weights,q*(1.-DUAL_INWARD_MARGIN))
    dot=exact_dot([1e16,1.,-1e16],ratios([1.,1.,1.]))
    checks={'orthogonal_weighted_L1_certifies':good['certified'],
            'infeasible_dual_rejected':not bad_dual['certified'] and not bad_dual['exact_dual_feasibility'],
            'nonoptimal_primal_rejected':not bad_primal['certified'],
            'exact_cancellation_dot':dot==1,
            'exact_gap_identity':all(x['exact_direct_minus_stable_gap_is_zero'] for x in [good,bad_dual,bad_primal])}
    return {'checks':checks,'all_checks_passed':all(checks.values()),
            'correct_example':good,'infeasible_example':bad_dual,'nonoptimal_example':bad_primal}


def _worker(conn,a,b,point,weights):
    try:
        with threadpool_limits(limits=1):
            beta=np.asarray(point['beta']);lam=float(point['lambda'])
            start=time.perf_counter()
            q,construction=projected_dual(a,b,beta,lam,weights)
            old,new=certify_arbitrary_dual(a,b,beta,lam,weights,q)
            conn.send(('solution',{'old_certificate':old,'new_certificate':new,
                                   'q':q.tolist(),'dual_construction':construction,
                                   'beta_unchanged':True,'elapsed_s':time.perf_counter()-start}))
    except Exception as exc:
        conn.send(('error',{'type':type(exc).__name__,'message':str(exc)}))
    finally:conn.close()


def bounded(a,b,point,weights):
    ctx=mp.get_context('spawn');recv,send=ctx.Pipe(duplex=False)
    proc=ctx.Process(target=_worker,args=(send,a,b,point,weights))
    start=time.perf_counter();proc.start();send.close()
    result={'status':'worker_exited'}
    try:
        remaining=max(.001,POINT_BUDGET-(time.perf_counter()-start))
        if recv.poll(remaining):
            kind,payload=recv.recv()
            result={'status':kind,**payload}
        else:result={'status':'budget_exhausted'}
    finally:
        if proc.is_alive():proc.terminate()
        proc.join(timeout=5.);recv.close()
    result['wall_seconds']=time.perf_counter()-start
    return result


def main():
    source_file=ROOT/'stage2a_retry_remaining_v3_result.json'
    source=json.loads(source_file.read_text(encoding='utf-8'))
    checks=self_checks()
    if not checks['all_checks_passed']:raise RuntimeError('Independent certificate self-check failed')
    report={'scope':'two unchanged primal candidates; alternate exact-verified feasible-dual certificate',
            'source_file':source_file.name,'source_sha256':hashlib.sha256(source_file.read_bytes()).hexdigest(),
            'source_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'rows':copy.deepcopy(source['rows']),'point_reviews':[],'self_checks':checks,
            'per_point_wall_seconds':POINT_BUDGET,'all_checks_passed':False,'execution_complete':False,
            'root_review_required':True,'research_stage_gate_updated':False,
            'test_data_used':False,'no_primal_coefficient_changes':True,'support_tolerance':SUPPORT_TOL,
            'original_tolerances':{'kkt_scaled_abs':KKT_ABS_TOL,'kkt_relative':KKT_REL_TOL,'gap_relative':GAP_REL_TOL},
            'mathematical_identity':'P(beta)-D(q)=0.5||b-A beta-q||^2 + sum_j |beta_j|(lambda*w_j-sign(beta_j)*a_j^T q)',
            'validity_condition':'Every |a_j^T q| <= lambda*w_j; checked exactly for the saved binary64 inputs and q.',
            'reference':'https://stanford.edu/~boyd/papers/pdf/l1_ls.pdf'}
    start=time.perf_counter()
    strict_json_write(OUTPUT,report)
    with threadpool_limits(limits=1):
        for row,old_row in zip(report['rows'],source['rows']):
            pending=[i for i,p in enumerate(row['path']['points']) if p['status']!='certified']
            if not pending:continue
            if row['case']!='advection_diffusion':raise RuntimeError('Unexpected remaining case outside scope')
            matrices,targets,_=save_matrices(row['case'],row['sigma'],row['seed'])
            a,b=stack_split(matrices,targets,range(4));a,b,_=normalize_training(a[:,list(range(a.shape[1]))],b)
            assert {'A':array_hash(a),'b':array_hash(b)}==row['input_hashes']
            weights=np.asarray(row['path']['weights'])
            for i in pending:
                point=row['path']['points'][i]
                review=bounded(a,b,point,weights)
                review.update({'case':row['case'],'ratio':point['ratio'],'lambda':point['lambda'],'point_index':i,
                               'input_hashes':row['input_hashes']})
                report['point_reviews'].append(review)
                if review['status']=='solution' and review['new_certificate']['certified']:
                    # Do not overwrite the historical old certificate. It is kept
                    # verbatim in this new point's audit and remains false.
                    replacement={**point,'status':'certified','certificate':review['new_certificate'],
                                 'certificate_kind':review['new_certificate']['certificate_kind'],
                                 'previous_stage0_certificate':review['old_certificate'],
                                 'alternate_dual':review['q'],
                                 'attempts':point['attempts']+[{'solver':'unchanged_beta_exact_dual_certificate',**review}],
                                 'elapsed_s':point.get('elapsed_s',0.)+review['wall_seconds'],
                                 'error':None,'root_review_required':True}
                    assert replacement['beta']==point['beta'] and replacement['support']==point['support']
                    row['path']['points'][i]=replacement
                row['v4_certificate_seconds']=row.get('v4_certificate_seconds',0.)+review['wall_seconds']
                strict_json_write(OUTPUT,report)
                print(point['ratio'],review['status'],review.get('new_certificate',{}).get('certified'),
                      review.get('new_certificate',{}).get('dual_gap_relative'),flush=True)
            assert all(row['path']['points'][i]==p for i,p in enumerate(old_row['path']['points']) if p['status']=='certified')
            complete=all(p['status']=='certified' for p in row['path']['points'])
            row['path'].update({'all_solutions_certified':complete,'status':'complete' if complete else 'partial_failure',
                                'elapsed_s':old_row['path']['elapsed_s']+row.get('v4_certificate_seconds',0.),
                                'alternate_certificate_root_review_required':True})
    report['execution_complete']=True
    report['all_checks_passed']=all(row['path']['all_solutions_certified'] for row in report['rows'])
    report['elapsed_s']=time.perf_counter()-start
    report['limits']=['Exact rational verification is for the stored binary64 inputs, not unknown real experimental fields.',
                      'This tightens the numerical dual lower bound; it changes neither the L1 estimator nor model selection.',
                      'The old residual-only certificate is retained and is not relabeled as passed.',
                      'Root must review the new certificate before the broader stage gate is updated.']
    strict_json_write(OUTPUT,report)
    print(json.dumps({'all_checks_passed':report['all_checks_passed'],'elapsed_s':report['elapsed_s'],
                      'root_review_required':True},allow_nan=False))


if __name__=='__main__':main()
