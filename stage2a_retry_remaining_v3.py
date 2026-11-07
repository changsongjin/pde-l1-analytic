"""Bounded compensated-residual refinement of only the three remaining L1 points.

No old files, objective, support threshold, lambda, or certificate tolerances
are changed. Windows longdouble equals float64 here, so compensated products
and math.fsum are used explicitly instead of claiming extended precision.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from stage0_evaluation import array_hash, normalize_training, strict_json_write
from stage0_l1_solver import SUPPORT_TOL, certify_l1
from stage2_identifiability import save_matrices, stack_split

ROOT=Path(__file__).resolve().parent
OUTPUT=ROOT/'stage2a_retry_remaining_v3_result.json'
POINT_BUDGET=90.


def accurate_ax_minus(a, x, other):
    """Compensate each IEEE-double product, then fsum products/errors/-other.

Dekker splitting assumes ordinary finite non-overflowing binary64 arithmetic.
The returned residual is double; original certificate arithmetic is unchanged.
"""
    splitter=134217729.
    ca=splitter*a; cx=splitter*x
    if not (np.isfinite(ca).all() and np.isfinite(cx).all()):
        raise ValueError('compensated product split overflow')
    ah=ca-(ca-a); al=a-ah
    xh=cx-(cx-x); xl=x-xh
    product=a*x
    error=((ah*xh-product)+ah*xl+al*xh)+al*xl
    return np.array([math.fsum(itertools.chain(product[i],error[i],[-float(other[i])]))
                     for i in range(a.shape[0])])


def quality(certificate):
    tolerances=certificate['tolerances']
    return max(certificate['kkt_max_scaled_abs']/tolerances['kkt_scaled_abs'],
               certificate['kkt_max_relative_to_penalty']/tolerances['kkt_relative'],
               max(0.,certificate['dual_gap_relative'])/tolerances['gap_relative'])


def refine(a,b,lam,weights,initial,send,max_iterations=100):
    beta=np.asarray(initial).copy()
    initial_cert=certify_l1(a,b,beta,lam,weights)
    best=beta.copy();best_cert=initial_cert
    events=[];last_active=None;factor=None
    start=time.perf_counter()
    send('best',{'beta':best.tolist(),'certificate':best_cert,'iteration':-1})
    for iteration in range(max_iterations):
        current_cert=certify_l1(a,b,beta,lam,weights)
        if current_cert['certified']:
            best,best_cert=beta.copy(),current_cert
            break
        active=np.flatnonzero(beta!=0.)
        signs=np.sign(beta[active])
        active_key=(tuple(active),tuple(signs))
        if active.size:
            if active_key!=last_active:
                signed=a[:,active]*signs
                u,s,vt=np.linalg.svd(signed,full_matrices=False)
                if len(s)!=len(active) or s[-1]<=s[0]*np.finfo(float).eps*max(signed.shape):
                    events.append({'iteration':iteration,'event':'active_numerically_singular','active':len(active)})
                    break
                factor=(signed,s,vt);last_active=active_key
            signed,s,vt=factor
            z=beta[active]*signs
            current=z.copy()
            gradient=-accurate_ax_minus(signed,z,b)
            stationarity=accurate_ax_minus(signed.T,gradient,lam*weights[active])
            z += vt.T@((vt@stationarity)/s**2)
            if np.any(z<=0.):
                bad=np.flatnonzero(z<=0.)
                denominators=current[bad]-z[bad]
                if np.any(denominators<=0.):
                    events.append({'iteration':iteration,'event':'invalid_bound_denominator'})
                    break
                ratios=current[bad]/denominators
                alpha=float(ratios.min());hit=bad[ratios==alpha]
                feasible=current+alpha*(z-current);feasible[hit]=0.
                if np.any(feasible<0.):
                    events.append({'iteration':iteration,'event':'bound_roundoff_negative'})
                    break
                beta[active]=feasible*signs
                events.append({'iteration':iteration,'event':'exact_bound_step','alpha':alpha,
                               'zeroed_indices':active[hit].tolist()})
            else:
                beta[active]=z*signs
                events.append({'iteration':iteration,'event':'compensated_stationarity_correction',
                               'active':len(active),'condition':float(s[0]/s[-1]),
                               'compensated_stationarity_max_abs':float(np.max(np.abs(stationarity)))})
        certificate=certify_l1(a,b,beta,lam,weights)
        objective_allowance=2*initial_cert['roundoff_allowance']
        if certificate['primal_objective']>initial_cert['primal_objective']+objective_allowance:
            events[-1]['objective_increase_vs_initial']=certificate['primal_objective']-initial_cert['primal_objective']
        if (certificate['primal_objective']<=initial_cert['primal_objective']+objective_allowance
                and (certificate['certified'] or quality(certificate)<quality(best_cert))):
            best,best_cert=beta.copy(),certificate
            send('best',{'beta':best.tolist(),'certificate':best_cert,'iteration':iteration})
        if certificate['certified']:
            break
        # Only a genuine omitted-column KKT violation adds a variable. Existing
        # active stationarity can be refined again even when none may enter.
        corr=a.T@(b-a@beta)
        violation=np.abs(corr)-lam*weights
        violation[beta!=0.]=-np.inf
        entering=int(np.argmax(violation))
        if np.isfinite(violation[entering]) and violation[entering]>0.:
            beta[entering]=np.sign(corr[entering])*violation[entering]/float(a[:,entering]@a[:,entering])
            events.append({'iteration':iteration,'event':'exact_coordinate_enter','index':entering,
                           'violation':float(violation[entering])})
        if iteration%10==9:
            send('iteration',{'iteration':iteration,'best_certificate':best_cert,
                              'current_certificate':certificate,'elapsed_s':time.perf_counter()-start})
    return {'beta':best.tolist(),'certificate':best_cert,'initial_certificate':initial_cert,
            'events':events,'elapsed_s':time.perf_counter()-start,
            'objective_change':best_cert['primal_objective']-initial_cert['primal_objective'],
            'prediction_difference':float(np.linalg.norm(a@(best-np.asarray(initial))))}


def _worker(conn,a,b,lam,weights,initial):
    try:
        with threadpool_limits(limits=1):
            result=refine(a,b,lam,weights,initial,lambda kind,value:conn.send((kind,value)))
        conn.send(('solution',result))
    except Exception as exc:
        conn.send(('error',{'type':type(exc).__name__,'message':str(exc)}))
    finally:conn.close()


def bounded_refine(a,b,lam,weights,point,attempt,save):
    context=mp.get_context('spawn');receiver,sender=context.Pipe(duplex=False)
    process=context.Process(target=_worker,args=(sender,a,b,lam,weights,point['beta']))
    start=time.perf_counter();process.start();sender.close()
    result=None
    attempt.update({'status':'running','best_snapshot':None,'wall_budget_s':POINT_BUDGET})
    save()
    try:
        while True:
            remaining=POINT_BUDGET-(time.perf_counter()-start)
            if remaining<=0:attempt['status']='budget_exhausted';break
            if receiver.poll(min(.2,remaining)):
                try:kind,value=receiver.recv()
                except EOFError:attempt['status']='worker_closed';break
                if kind=='best':attempt['best_snapshot']=value
                elif kind=='iteration':attempt['iteration_progress']=value
                elif kind=='solution':result=value;attempt['status']='finished';break
                elif kind=='error':attempt['error']=value;attempt['status']='worker_error';break
                attempt['wall_seconds']=time.perf_counter()-start;save()
            elif not process.is_alive():attempt['status']='worker_exited';break
    finally:
        if process.is_alive():process.terminate()
        process.join(timeout=5.);receiver.close()
    attempt['wall_seconds']=time.perf_counter()-start
    if result is None and attempt['best_snapshot'] is not None:
        result={**attempt['best_snapshot'],'events':[],
                'partial_result_preserved':True,'initial_certificate':point['certificate']}
    if result is not None:
        parent=certify_l1(a,b,np.asarray(result['beta']),lam,weights)
        result['certificate']=parent
        attempt['final_certificate']=parent
        attempt['status']='certified' if parent['certified'] else 'uncertified'
    save()
    return result


def main():
    source_file=ROOT/'stage2a_retry_remaining_v2_result.json'
    source=json.loads(source_file.read_text(encoding='utf-8'))
    # Meaningful arithmetic checks: cancellation and product-rounding remainder.
    cancellation=accurate_ax_minus(np.array([[1e16,1.,-1e16]]),np.ones(3),np.zeros(1))[0]
    product=accurate_ax_minus(np.array([[1.+2**-27]]),np.array([1.-2**-27]),np.ones(1))[0]
    assert cancellation==1. and product==-2**-54
    report={'scope':'three-point bounded numerical refinement only; no statistical estimator change',
            'source_file':source_file.name,'source_sha256':hashlib.sha256(source_file.read_bytes()).hexdigest(),
            'source_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'rows':copy.deepcopy(source['rows']),'point_attempts':[],
            'per_point_wall_seconds':POINT_BUDGET,'execution_complete':False,'all_checks_passed':False,
            'arithmetic_checks':{'cancellation':float(cancellation),'product_roundoff_remainder':float(product)},
            'longdouble_epsilon':float(np.finfo(np.longdouble).eps),'float64_epsilon':float(np.finfo(float).eps),
            'certificate_unchanged':True,'support_tolerance':SUPPORT_TOL,
            'no_true_support_used':True,'test_data_used':False}
    start=time.perf_counter()
    def save():
        report['elapsed_s']=time.perf_counter()-start
        report['pending']=[{'case':row['case'],'ratio':p['ratio'],'status':p['status']}
                           for row in report['rows'] for p in row['path']['points'] if p['status']!='certified']
        strict_json_write(OUTPUT,report)
    with threadpool_limits(limits=1):
        for row,original_row in zip(report['rows'],source['rows']):
            pending=[i for i,p in enumerate(row['path']['points']) if p['status']!='certified']
            if not pending:continue
            if row['case'] not in ['advection_diffusion','logheat']:
                raise RuntimeError('Unexpected pending case outside this three-point scope')
            matrices,targets,meta=save_matrices(row['case'],row['sigma'],row['seed'])
            a,b=stack_split(matrices,targets,range(4))
            a,b,_=normalize_training(a[:,list(range(a.shape[1]))],b)
            assert {'A':array_hash(a),'b':array_hash(b)}==row['input_hashes']
            weights=np.asarray(row['path']['weights'])
            for index in pending:
                point=row['path']['points'][index]
                lam=float(point['lambda'])
                assert point['beta'] is not None
                assert abs(lam-float(row['path']['path_metadata']['lambda_max'])*point['ratio'])<=np.spacing(lam)*4
                attempt={'case':row['case'],'sigma':row['sigma'],'seed':row['seed'],'point_index':index,
                         'ratio':point['ratio'],'lambda':lam,'prior_certificate':point['certificate']}
                report['point_attempts'].append(attempt)
                result=bounded_refine(a,b,lam,weights,point,attempt,save)
                if result is not None:
                    beta=np.asarray(result['beta']);cert=result['certificate']
                    replacement={**point,'status':'certified' if cert['certified'] else 'uncertified',
                                 'beta':beta.tolist(),'support':np.flatnonzero(np.abs(beta)>SUPPORT_TOL).tolist(),
                                 'certificate':cert,'elapsed_s':point.get('elapsed_s',0.)+attempt['wall_seconds'],
                                 'attempts':point.get('attempts',[])+[{'solver':'compensated_residual_SVD_refinement',**result}],
                                 'error':None if cert['certified'] else 'Unchanged full certificate still fails; retain for next batch.'}
                    row['path']['points'][index]=replacement
                row['v3_retry_seconds']=row.get('v3_retry_seconds',0.)+attempt['wall_seconds']
                complete=all(p['status']=='certified' for p in row['path']['points'])
                row['path'].update({'status':'complete' if complete else 'partial_failure','all_solutions_certified':complete,
                                    'elapsed_s':original_row['path']['elapsed_s']+row['v3_retry_seconds']})
                save()
                print(row['case'],point['ratio'],attempt['status'],round(attempt['wall_seconds'],3),flush=True)
            assert all(row['path']['points'][i]==p for i,p in enumerate(original_row['path']['points']) if p['status']=='certified')
    report['execution_complete']=True
    report['all_checks_passed']=all(row['path']['all_solutions_certified'] for row in report['rows'])
    report['limits']=['Only the three previously uncertified fixed-lambda solves were attempted.',
                      'Compensated residuals guide the optimizer; acceptance still uses the unchanged original float64 certificate.',
                      'Uncertified points are not statistical support-recovery failures and remain ineligible for complete-path ranking.']
    save()
    print(json.dumps({'all_checks_passed':report['all_checks_passed'],'pending':report['pending'],
                      'elapsed_s':report['elapsed_s']},allow_nan=False))


if __name__=='__main__':main()
