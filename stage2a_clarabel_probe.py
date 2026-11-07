"""Independent residual-QP check of weighted L1, with a 90 s process budget.

Clarabel is a candidate numerical tool, not a new estimator or replacement of
the benchmark solver. Original objective/KKT/gap/support rules stay unchanged.
https://clarabel.org/stable/python/getting_started_py/
https://clarabel.org/stable/api_settings/
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'.research_deps'))
import clarabel

from stage0_evaluation import array_hash, normalize_training, strict_json_write
from stage0_l1_solver import certify_l1, solve_l1, _coordinate_descent, SUPPORT_TOL
from stage2_identifiability import stack_split
from stage2a_solver_polish import polish_fixed_l1


def clarabel_fixed_l1(a, b, lam, weights, time_limit=60.):
    """Residual QP x=[beta,t,r]; never form A.T A in the supplied objective.

min .5*r.T*r + lambda*w.T*t
s.t. A*beta-r=b, beta-t<=0, -beta-t<=0.
"""
    n, p = a.shape
    ip, inn = sparse.eye(p, format='csc'), sparse.eye(n, format='csc')
    znp, zpn = sparse.csc_matrix((n,p)), sparse.csc_matrix((p,n))
    hessian = sparse.diags(np.r_[np.zeros(2*p), np.ones(n)], format='csc')
    linear = np.r_[np.zeros(p), lam*weights, np.zeros(n)]
    constraints = sparse.vstack([
        sparse.hstack([sparse.csc_matrix(a), znp, -inn]),
        sparse.hstack([ip, -ip, zpn]),
        sparse.hstack([-ip, -ip, zpn])], format='csc')
    rhs = np.r_[b, np.zeros(2*p)]
    cones = [clarabel.ZeroConeT(n), clarabel.NonnegativeConeT(2*p)]
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_threads = 1
    settings.time_limit = min(60., float(time_limit))
    settings.max_iter = 200
    settings.tol_gap_abs = 1e-13
    settings.tol_gap_rel = 1e-12
    settings.tol_feas = 1e-12
    settings.iterative_refinement_abstol = 1e-14
    settings.iterative_refinement_reltol = 1e-14
    start = time.perf_counter()
    solver = clarabel.DefaultSolver(hessian, linear, constraints, rhs, cones, settings)
    solution = solver.solve()
    beta = np.asarray(solution.x[:p])
    metadata = {'solver': 'clarabel', 'version': clarabel.__version__,
                'status': str(solution.status), 'iterations': int(solution.iterations),
                'solve_time_s': float(solution.solve_time),
                'construction_and_solve_s': time.perf_counter()-start,
                'cone_objective': float(solution.obj_val), 'cone_dual_objective': float(solution.obj_val_dual),
                'primal_residual_reported': float(solution.r_prim), 'dual_residual_reported': float(solution.r_dual),
                'n_variables': 2*p+n, 'constraint_shape': list(constraints.shape),
                'constraint_nonzeros': int(constraints.nnz),
                'settings': {'max_threads': 1, 'time_limit': settings.time_limit, 'max_iter': 200,
                             'tol_gap_abs': 1e-13, 'tol_gap_rel': 1e-12, 'tol_feas': 1e-12,
                             'iterative_refinement_abstol': 1e-14, 'iterative_refinement_reltol': 1e-14},
                'note': 'Internal KKT regularization is solver arithmetic; acceptance uses the original unregularized L1 certificate.'}
    return beta, metadata


def solve_and_certify(a, b, lam, weights, send, solver_time_limit=60.):
    start = time.perf_counter()
    raw_beta, external = clarabel_fixed_l1(a, b, lam, weights, time_limit=solver_time_limit)
    certificate = certify_l1(a, b, raw_beta, lam, weights)
    stages = [{'stage': 'raw_clarabel', 'certificate': certificate,
               'exact_nonzeros': int(np.count_nonzero(raw_beta)), 'elapsed_s': time.perf_counter()-start}]
    send('solver_stage', {'external_solver': external, **stages[-1]})
    beta = raw_beta.copy()
    if not certificate['certified']:
        # A genuine proximal-gradient update for the same L1 objective. The
        # optimization threshold lambda*w/L is unrelated to reporting 1e-5.
        prox_start = time.perf_counter()
        lipschitz = float(np.linalg.norm(a, 2)**2)*(1.+1e-12)
        for _ in range(10):
            z = beta - a.T@(a@beta-b)/lipschitz
            beta = np.sign(z)*np.maximum(np.abs(z)-lam*weights/lipschitz, 0.)
        certificate = certify_l1(a, b, beta, lam, weights)
        stages.append({'stage': '10_true_L1_proximal_gradient_steps', 'certificate': certificate,
                       'lipschitz': lipschitz, 'optimization_threshold_range':
                       [float(lam*weights.min()/lipschitz), float(lam*weights.max()/lipschitz)],
                       'exact_nonzeros': int(np.count_nonzero(beta)),
                       'elapsed_s': time.perf_counter()-prox_start})
        send('solver_stage', stages[-1])
    if not certificate['certified']:
        polish_start = time.perf_counter()
        try:
            beta, details = polish_fixed_l1(a,b,lam,weights,beta,max_iterations=50)
            certificate = certify_l1(a,b,beta,lam,weights)
            stages.append({'stage': 'same_objective_warm_QP_polish', 'certificate': certificate,
                           'details': details, 'elapsed_s': time.perf_counter()-polish_start})
        except (RuntimeError, np.linalg.LinAlgError) as exc:
            stages.append({'stage': 'same_objective_warm_QP_polish', 'error': str(exc),
                           'elapsed_s': time.perf_counter()-polish_start})
        send('solver_stage', stages[-1])
    if not certificate['certified']:
        cd_start = time.perf_counter()
        beta, details = _coordinate_descent(a,b,lam,weights,initial=beta,max_iter=2000)
        certificate = certify_l1(a,b,beta,lam,weights)
        stages.append({'stage': 'same_objective_coordinate_descent', 'certificate': certificate,
                       'details': details, 'elapsed_s': time.perf_counter()-cd_start})
        send('solver_stage', stages[-1])
    return {'beta': beta.tolist(), 'support': np.flatnonzero(np.abs(beta)>SUPPORT_TOL).tolist(),
            'external_solver': external, 'stages': stages, 'certificate': certificate,
            'raw_to_final_prediction_difference': float(np.linalg.norm(a@(beta-raw_beta))),
            'raw_to_final_objective_change': certificate['primal_objective']-stages[0]['certificate']['primal_objective'],
            'elapsed_s': time.perf_counter()-start}


def comparison(a, b, lam, weights, result, reference):
    beta = np.asarray(result['beta'])
    ref = np.asarray(reference)
    certificate = certify_l1(a,b,ref,lam,weights)
    current = result['certificate']
    delta = float(np.linalg.norm(a@(beta-ref)))
    bound = float(np.sqrt(2*max(0, current['dual_gap']+certificate['dual_gap'])+1e-12*float(b@b)))
    obj_difference = abs(current['primal_objective']-certificate['primal_objective'])
    return {'reference_certificate': certificate, 'prediction_difference': delta,
            'prediction_bound_from_both_dual_gaps_with_roundoff': bound,
            'objective_absolute_difference': obj_difference,
            'objective_relative_difference': obj_difference/max(certificate['primal_objective'],1e-30),
            'passed': bool(certificate['certified'] and current['certified'] and delta<=bound
                           and obj_difference<=abs(current['dual_gap'])+abs(certificate['dual_gap'])
                           +current['roundoff_allowance']+certificate['roundoff_allowance']),
            'coefficient_equality_required': False}


def _worker(conn):
    def send(kind,value): conn.send((kind,value))
    try:
        with threadpool_limits(limits=1):
            rng = np.random.default_rng(603)
            a = rng.normal(size=(60,12)); a/=np.linalg.norm(a,axis=0)
            truth = np.zeros(12);truth[[1,5,9]]=[1.,-.4,.3]
            b = a@truth+.002*rng.normal(size=60)
            weights=np.geomspace(.3,3.,12)
            lam=float(np.max(np.abs(a.T@b)/weights)*.01)
            direct=solve_l1(a,b,lam,weights)
            result=solve_and_certify(a,b,lam,weights,send,solver_time_limit=10.)
            checked=comparison(a,b,lam,weights,result,direct['beta'])
            send('synthetic',{'shape':list(a.shape),'lambda':lam,'solution':result,'independent_comparison':checked,
                              'all_checks_passed':checked['passed']})
            if not checked['passed']:
                send('done',False);return
            source_path=ROOT/'stage2a_solver_retry_result.json'
            source=json.loads(source_path.read_text(encoding='utf-8'))
            with np.load(ROOT/'stage2_matrices/heat_sigma_0p05_seed_0.npz',allow_pickle=False) as data:
                a,b=stack_split([data[f'a_{i}'] for i in range(4)],[data[f'b_{i}'] for i in range(4)],range(4))
            a,b,_=normalize_training(a[:,list(range(a.shape[1]))],b)
            hashes={'A':array_hash(a),'b':array_hash(b)}
            assert hashes==source['input_hashes']
            reference=source['path']['points'][-1]
            lam=float(reference['lambda']);weights=np.asarray(source['path']['weights'])
            send('heat_input',{'hashes':hashes,'shape':list(a.shape),'lambda':lam,'ratio':reference['ratio'],
                               'reference_result_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest(),
                               'reference_used_only_after_independent_solve':True})
            result=solve_and_certify(a,b,lam,weights,send,solver_time_limit=60.)
            checked=comparison(a,b,lam,weights,result,reference['beta'])
            send('heat',{'solution':result,'independent_comparison':checked,'all_checks_passed':checked['passed']})
            send('done',checked['passed'])
    except Exception as exc:
        send('error',{'type':type(exc).__name__,'message':str(exc)})
    finally:
        conn.close()


def main():
    output=ROOT/'stage2a_clarabel_probe_result.json'
    report={'scope':'independent numerical weighted-L1 check; no method/alpha changes and no benchmark rerun',
            'clarabel_version':clarabel.__version__,'wall_budget_s':90.,'status':'running','all_checks_passed':False,
            'references':['https://clarabel.org/stable/python/getting_started_py/','https://clarabel.org/stable/api_settings/'],
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'dependency_manifest_sha256':hashlib.sha256((ROOT/'requirements-research.txt').read_bytes()).hexdigest(),
            'stages':[], 'test_data_used':False,'reporting_support_tol':SUPPORT_TOL,
            'limits':['Clarabel Solved status is not accepted without the original full-column certificate.',
                      'Interior-point coefficients can be tiny nonzero; only genuine L1 optimization steps may make them zero.',
                      'One synthetic matrix and one development matrix do not establish general solver superiority.']}
    context=mp.get_context('spawn');receiver,sender=context.Pipe(duplex=False)
    process=context.Process(target=_worker,args=(sender,))
    start=time.perf_counter();process.start();sender.close()
    strict_json_write(output,report)
    try:
        while True:
            remaining=90.-(time.perf_counter()-start)
            if remaining<=0:
                report['status']='budget_exhausted';break
            if receiver.poll(min(.2,remaining)):
                try:kind,value=receiver.recv()
                except EOFError:
                    report['status']='worker_closed';break
                if kind=='solver_stage':report['stages'].append(value)
                elif kind=='done':
                    report['status']='complete';report['all_checks_passed']=bool(value);break
                elif kind=='error':report['error']=value;report['status']='error';break
                else:report[kind]=value
                report['elapsed_s']=time.perf_counter()-start
                strict_json_write(output,report)
                print(kind, value.get('stage',value.get('all_checks_passed','received')),flush=True)
            elif not process.is_alive():
                report['status']='worker_exited';break
    finally:
        if process.is_alive():process.terminate()
        process.join(timeout=5.);receiver.close()
    report['elapsed_s']=time.perf_counter()-start
    strict_json_write(output,report)
    print(json.dumps({'status':report['status'],'all_checks_passed':report['all_checks_passed'],
                      'elapsed_s':report['elapsed_s']},allow_nan=False))


if __name__=='__main__':main()
