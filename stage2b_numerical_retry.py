"""Only unfinished frozen stage-2b ordinary-L1 points, after the main worker stops.

Per-point hard wall limit 120 s; all successes, failures and costs are retained.
Uses fixed task path/job sources rather than the mutable campaign report.
Never overwrites the original paths, job manifests or method-comparison file.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from stage0_evaluation import array_hash,normalize_training,strict_json_write
from stage0_l1_solver import certify_l1,SUPPORT_TOL,KKT_ABS_TOL,KKT_REL_TOL
from stage2_identifiability import stack_split
from stage2a_clarabel_probe import solve_and_certify
from stage2a_retry_remaining_v3 import refine
from stage2a_retry_remaining_v4 import projected_dual,certify_arbitrary_dual

ROOT=Path(__file__).resolve().parent
POINT_BUDGET=120.
DEFAULT_PREFIX='independent_development_ids12_17_seed1_burgers_0p05_J1158_L1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    def reject(token):raise ValueError('nonfinite JSON '+token)
    return json.loads(Path(path).read_text(encoding='utf-8'),parse_constant=reject)


def alternate_certificate(a,b,beta,lam,weights):
    old=certify_l1(a,b,beta,lam,weights)
    if old['certified']:return {'accepted':True,'certificate':old,'certificate_kind':'original_stage0'}
    if old['kkt_max_scaled_abs']>KKT_ABS_TOL or old['kkt_max_relative_to_penalty']>KKT_REL_TOL:
        return {'accepted':False,'certificate':old,'reason':'original full-column KKT still fails'}
    try:
        q,construction=projected_dual(a,b,beta,lam,weights)
        original,new=certify_arbitrary_dual(a,b,beta,lam,weights,q)
    except (RuntimeError,np.linalg.LinAlgError) as exc:
        return {'accepted':False,'certificate':old,'reason':str(exc)}
    return {'accepted':new['certified'],'certificate':new,'certificate_kind':new['certificate_kind'],
            'previous_stage0_certificate':original,'alternate_dual':q.tolist(),'dual_construction':construction,
            'root_reviewed_certificate_rule':'stage2a_retry_remaining_v4 exact feasible-dual proof; tolerances and beta KKT unchanged'}


def _worker(conn,a,b,lam,weights):
    def send(kind,value):conn.send((kind,value))
    try:
        with threadpool_limits(limits=1):
            solution=solve_and_certify(a,b,lam,weights,send,solver_time_limit=50.)
            beta=np.asarray(solution['beta'])
            accepted=alternate_certificate(a,b,beta,lam,weights)
            send('candidate',{'beta':beta.tolist(),**accepted})
            attempts=[{'solver':'independent_Clarabel_and_same_objective_polish','solution':solution,
                       'alternate_certificate_review':accepted}]
            if not accepted['accepted']:
                result=refine(a,b,lam,weights,beta,lambda kind,value:send('refinement_'+kind,value),max_iterations=40)
                beta=np.asarray(result['beta'])
                accepted=alternate_certificate(a,b,beta,lam,weights)
                attempts.append({'solver':'compensated_stationarity_SVD_refinement','solution':result,
                                 'alternate_certificate_review':accepted})
            send('solution',{'beta':beta.tolist(),'support':np.flatnonzero(np.abs(beta)>SUPPORT_TOL).tolist(),
                             **accepted,'attempts':attempts,'new_primal_solve_performed':True})
    except Exception as exc:
        send('error',{'type':type(exc).__name__,'message':str(exc)})
    finally:conn.close()


def bounded(a,b,lam,weights,attempt,save):
    ctx=mp.get_context('spawn');recv,send=ctx.Pipe(duplex=False)
    proc=ctx.Process(target=_worker,args=(send,a,b,lam,weights))
    start=time.perf_counter();proc.start();send.close()
    result=None;best=None
    attempt.update({'status':'running','stages':[],'wall_budget_s':POINT_BUDGET});save()
    try:
        while True:
            remaining=POINT_BUDGET-(time.perf_counter()-start)
            if remaining<=0:attempt['status']='budget_exhausted';break
            if recv.poll(min(.2,remaining)):
                try:kind,value=recv.recv()
                except EOFError:attempt['status']='worker_closed';break
                if kind=='solver_stage':attempt['stages'].append(value)
                elif kind in ['candidate','refinement_best']:
                    best=value;attempt['best_candidate_snapshot']=value
                elif kind=='refinement_iteration':attempt['refinement_progress']=value
                elif kind=='solution':result=value;break
                elif kind=='error':attempt['status']='worker_error';attempt['error']=value;break
                attempt['wall_seconds']=time.perf_counter()-start;save()
            elif not proc.is_alive():attempt['status']='worker_exited';break
    finally:
        if proc.is_alive():proc.terminate()
        proc.join(timeout=5.);recv.close()
    attempt['wall_seconds']=time.perf_counter()-start
    if result is None and best is not None:
        beta=np.asarray(best['beta']);cert=certify_l1(a,b,beta,lam,weights)
        result={'beta':beta.tolist(),'support':np.flatnonzero(np.abs(beta)>SUPPORT_TOL).tolist(),
                'certificate':cert,'accepted':cert['certified'],'attempts':[],'partial_candidate_preserved':True}
    if result is not None:
        beta=np.asarray(result['beta'])
        if result.get('alternate_dual') is not None:
            old,cert=certify_arbitrary_dual(a,b,beta,lam,weights,np.asarray(result['alternate_dual']))
            result['previous_stage0_certificate']=old
        else:cert=certify_l1(a,b,beta,lam,weights)
        if cert['certified']!=result['accepted']:raise ValueError('parent/worker acceptance mismatch')
        result['certificate']=cert
        attempt['status']='certified' if cert['certified'] else 'uncertified'
        attempt['certificate']=cert
    save()
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prefixes',nargs='+',default=[DEFAULT_PREFIX])
    parser.add_argument('--output',type=Path,default=ROOT/'stage2b_numerical_retry_result.json')
    args=parser.parse_args()
    if args.output.exists():raise ValueError('retry result exists; preserve it and use a new version')
    report={'scope':'unfinished stage2b ordinary-L1 lambda points only; fixed task manifest sources',
            'per_point_wall_seconds':POINT_BUDGET,'source_script_sha256':sha(Path(__file__)),
            'helper_sha256':{n:sha(ROOT/n) for n in ['stage2a_clarabel_probe.py','stage2a_retry_remaining_v3.py',
                                                    'stage2a_retry_remaining_v4.py','stage0_l1_solver.py']},
            'rows':[],'execution_complete':False,'all_checks_passed':False,
            'no_true_support_used':True,'test_data_used':False,'statistical_method_changed':False}
    started=time.perf_counter()
    def save():
        report['elapsed_s']=time.perf_counter()-started
        strict_json_write(args.output,report)
    with threadpool_limits(limits=1):
        for prefix in args.prefixes:
            path_file=ROOT/'stage2b_progress'/(prefix+'_path_result.json')
            job_file=ROOT/'stage2b_progress'/(prefix+'_job.json')
            original=load(path_file);job=load(job_file)
            fixed_hashes={'path':sha(path_file),'job':sha(job_file)}
            if (original['status']=='running' or job['method']!='L1' or original['method']!='L1'
                    or job['seed']!=1 or job['cohort']!='independent_development_ids12_17_seed1'):
                raise ValueError('unexpected method/cohort or still-running task')
            for name,digest in {**job['source_sha256'],**job['gate_sources']}.items():
                if sha(ROOT/name)!=digest:raise ValueError('frozen job source or gate changed')
            tag=f"{job['case']}_sigma_{job['sigma']}_seed_1_ids12_17"
            matrix_file=ROOT/'stage2b_matrices'/(tag+'.npz');meta=load(matrix_file.with_suffix('.json'))
            if sha(matrix_file)!=meta['npz_sha256'] or meta['raw_bundle_sha256']!=job['raw_bundle_sha256']:
                raise ValueError('new cohort cached matrix/raw hash mismatch')
            with np.load(matrix_file,allow_pickle=False) as data:
                matrices=[data[f'a_{i}'].copy() for i in range(6)];targets=[data[f'b_{i}'].copy() for i in range(6)]
            for a0,b0,record in zip(matrices,targets,meta['trajectories']):
                if array_hash(a0)!=record['matrix_hash'] or array_hash(b0)!=record['target_hash']:raise ValueError('matrix/target hash mismatch')
            a,b=stack_split(matrices,targets,range(4));subset=meta['small_indices'] if job['layer']=='J18' else list(range(a.shape[1]))
            a,b,scales=normalize_training(a[:,subset],b)
            if {'A':array_hash(a),'b':array_hash(b)}!=job['solver_input_hashes'] or scales!=job['normalization']:
                raise ValueError('solver coordinates changed')
            weights=np.asarray(original['weights']);lmax=float(original['path_metadata']['lambda_max'])
            if not np.isclose(lmax,float(np.max(np.abs(a.T@b)/weights)),rtol=1e-14,atol=0):raise ValueError('lambda_max mismatch')
            path=copy.deepcopy(original)
            inherited=[];pending=[]
            for i,point in enumerate(path['points']):
                expected=float(point['ratio']*lmax)
                if point['status']=='certified':
                    cert=certify_l1(a,b,np.asarray(point['beta']),point['lambda'],weights)
                    if not cert['certified'] or point['support']!=np.flatnonzero(np.abs(point['beta'])>SUPPORT_TOL).tolist():
                        raise ValueError('inherited certificate/support invalid')
                    inherited.append(i)
                else:
                    if point['lambda'] is not None and not np.isclose(point['lambda'],expected,rtol=1e-14,atol=0):raise ValueError('lambda changed')
                    point['lambda']=expected;pending.append(i)
            row={'cohort':job['cohort'],'case':job['case'],'sigma':job['sigma'],'seed':job['seed'],
                 'layer':job['layer'],'method':'L1','path':path,'point_attempts':[],
                 'source_path':str(path_file.relative_to(ROOT)),'source_job':str(job_file.relative_to(ROOT)),
                 'source_hashes':fixed_hashes,'input_hashes':job['solver_input_hashes'],
                 'raw_bundle_sha256':job['raw_bundle_sha256'],'normalization':scales,
                 'inherited_certified_point_indices':inherited,'original_elapsed_s':original['elapsed_s'],
                 'lambda_restoration':'recorded lambda_max times frozen ratio, independently checked from identical A,b,weights'}
            report['rows'].append(row);save()
            for i in pending:
                point=path['points'][i];attempt={'point_index':i,'ratio':point['ratio'],'lambda':point['lambda'],
                                               'prior_status':original['points'][i]['status']}
                row['point_attempts'].append(attempt)
                solution=bounded(a,b,point['lambda'],weights,attempt,save)
                if solution is not None:
                    path['points'][i]={'ratio':point['ratio'],'lambda':point['lambda'],
                        'status':'certified' if solution['certificate']['certified'] else 'uncertified',
                        'beta':solution['beta'],'support':solution['support'],'certificate':solution['certificate'],
                        'attempts':solution['attempts'],'elapsed_s':attempt['wall_seconds'],'error':None,
                        **{k:solution[k] for k in ['alternate_dual','previous_stage0_certificate','certificate_kind'] if k in solution}}
                else:path['points'][i]['status']=attempt['status']
                row['retry_elapsed_s']=sum(x['wall_seconds'] for x in row['point_attempts'])
                path['elapsed_s']=row['original_elapsed_s']+row['retry_elapsed_s']
                path['all_solutions_certified']=all(p['status']=='certified' for p in path['points'])
                path['status']='complete' if path['all_solutions_certified'] else 'partial_failure'
                save();print(job['case'],point['ratio'],attempt['status'],round(attempt['wall_seconds'],2),flush=True)
            assert all(path['points'][i]==original['points'][i] for i in inherited)
            if {'path':sha(path_file),'job':sha(job_file)}!=fixed_hashes:raise ValueError('original task sources changed')
            row['original_certified_points_and_sources_unchanged']=True
            row['cost_note']='path elapsed is original initial attempt plus every retry; audit overhead is separately in report elapsed.'
    report['execution_complete']=True
    report['all_checks_passed']=all(r['path']['all_solutions_certified'] for r in report['rows'])
    save()
    print(json.dumps({'all_checks_passed':report['all_checks_passed'],'rows':len(report['rows']),
                      'elapsed_s':report['elapsed_s']},allow_nan=False))


if __name__=='__main__':main()
