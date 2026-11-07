"""Independent read-only source/API/data/optional runner review for stage 2b.

No PDE solve, statistical variant, data generation, alpha change, or successful
path rerun. Use a new output filename for preflight versus final review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import stage2a_l1_paths as ORIGINAL
import stage2b_smoothed_irl1 as VARIANT
import stage2b_heat_full_spectrum as SPECTRUM
from stage0_data_bundle import load_bundle
from stage0_evaluation import SUPPORT_TOL,array_hash,normalize_training,strict_json_write
from stage0_l1_solver import certify_l1
from stage2_identifiability import stack_split

ROOT=Path(__file__).resolve().parent


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def reject(token):raise ValueError('nonfinite JSON '+token)
    return json.loads(Path(path).read_text(encoding='utf-8'),parse_constant=reject)


def data_review(protocol):
    rows=[];seen_ids=set();all_original_hashes=set()
    for case,sigma in protocol['cases_and_noise']:
        tag=f'{case}_sigma_{sigma}_seed_1_ids12_17'
        raw_path=ROOT/'stage2b_data'/(tag+'.npz')
        matrix_path=ROOT/'stage2b_matrices'/(tag+'.npz')
        if not raw_path.exists() or not matrix_path.exists():raise ValueError('review may not generate missing data')
        x,t,observations,manifest=load_bundle(raw_path)
        meta=read(matrix_path.with_suffix('.json'))
        ids=[r['trajectory_id'] for r in manifest['trajectories']]
        splits=[r['split'] for r in manifest['trajectories']]
        if (ids!=list(range(12,18)) or splits!=['train']*4+['validation']*2
                or manifest['seed']!=1 or manifest['test_data_generated_or_used']
                or digest(matrix_path)!=meta['npz_sha256'] or manifest['npz_sha256']!=meta['raw_bundle_sha256']):
            raise ValueError('independent development manifest mismatch')
        for source,sha in meta['source_hashes'].items():
            if digest(ROOT/source)!=sha:raise ValueError('new cohort preprocessing source changed')
        old_tag=f"{case}_sigma_{repr(float(sigma)).replace('.', 'p')}_seed_0"
        _,_,old_observations,_=load_bundle(ROOT/'stage2_data'/(old_tag+'.npz'))
        old_hashes={array_hash(u) for u in old_observations}
        new_hashes={array_hash(u) for u in observations}
        if old_hashes&new_hashes or len(new_hashes)!=6:raise ValueError('reused/duplicate observation field')
        all_original_hashes.update(old_hashes)
        with np.load(matrix_path,allow_pickle=False) as data:
            for i,(u,m) in enumerate(zip(observations,meta['trajectories'])):
                if (array_hash(data[f'a_{i}'])!=m['matrix_hash'] or array_hash(data[f'b_{i}'])!=m['target_hash']
                        or array_hash(u)!=m['raw_hash'] or m['trajectory']!=ids[i]):
                    raise ValueError('raw/matrix/target indexing or hashing mismatch')
        seen_ids.update(ids)
        rows.append({'case':case,'sigma':sigma,'ids':ids,'seed':1,'splits':splits,
                     'old_new_raw_hashes_disjoint':True,'verified_matrix_target_pairs':6,
                     'raw_sha256':manifest['npz_sha256'],'matrix_sha256':digest(matrix_path)})
    return {'all_checks_passed':seen_ids==set(range(12,18)) and not(seen_ids&set(range(6,12))),
            'rows':rows,'reserved_final_test_not_loaded':True,
            'independence_scope':'disjoint generated development trajectories/noise seed, not independent proof of broad method superiority'}


def preflight():
    protocol=read(ROOT/'stage2b_protocol_result.json')
    api=read(ROOT/'stage2b_smoothed_irl1_result.json')
    full=read(ROOT/'stage2b_heat_full_spectrum_result.json')
    files=['stage2b_smoothed_irl1.py','stage2a_l1_paths.py','stage0_l1_solver.py','stage2b_protocol.py',
           'stage2b_data.py','stage2b_heat_full_spectrum.py','stage2b_heat_dynamic_compare.py',
           'stage2b_heat_noise_mechanism.py']
    hashes={name:digest(ROOT/name) for name in files}
    checks={'variant_checked_source_matches':api['variant_source_sha256']==hashes['stage2b_smoothed_irl1.py'] and api['all_checks_passed'],
            'same_fixed_weight_solver_function':VARIANT._path_points is ORIGINAL._path_points,
            'same_outer_lambda_iterations_floor_support':VARIANT.IRL1_RATIO==ORIGINAL.IRL1_RATIO==.001
                and VARIANT.IRL1_MAX_ITER==ORIGINAL.IRL1_MAX_ITER==8
                and VARIANT.EPSILON_FLOOR==ORIGINAL.IRL1_EPS==1e-8 and VARIANT.SUPPORT_TOL==SUPPORT_TOL==1e-5,
            'same_24_lambda_grid':np.array_equal(VARIANT.DEFAULT_RATIOS,ORIGINAL.DEFAULT_RATIOS),
            'frozen_single_epsilon_and_alpha':protocol['epsilon_relative']==VARIANT.RELATIVE_EPSILON==.01
                and protocol['selection_alpha']==VARIANT.FROZEN_SELECTION_ALPHA==.005,
            'full_spectrum_checked_source_matches':full['all_checks_passed'] and full['source_sha256']==hashes['stage2b_heat_full_spectrum.py']}
    examples=[]
    for beta in [np.zeros(5),np.array([2.,-.02,0.,1e-12,.1]),np.array([1e-9,0.,-2e-10])]:
        w,meta=VARIANT.smoothed_weights(beta)
        epsilon=max(1e-8,.01*float(np.max(np.abs(beta))))
        expected=1./(np.abs(beta)+epsilon)
        ok=np.array_equal(w,expected) and np.all(w>0) and np.isfinite(w).all() and w.max()/w.min()<=101.+1e-12
        examples.append({'beta':beta.tolist(),'epsilon':epsilon,'weight_ratio':float(w.max()/w.min()),'passed':bool(ok)})
    checks['independent_epsilon_formula_and_ratio_examples']=all(x['passed'] for x in examples)
    # Short direct API boundary checks; no research candidate or solver used.
    x=np.linspace(0,1,17);t=np.linspace(0,.01,5)
    field=np.sin(np.pi*x)[None,:]*np.exp(-np.pi**2*t[:,None])
    initial=SPECTRUM.prepare_initialization(x,field[0],scheme='raw_all_modes',boundary_realization=SPECTRUM.BOUNDARY)
    report,pred=SPECTRUM.evaluate_operator({'u_xx':1.},x,t,field,initial,boundary_realization=SPECTRUM.BOUNDARY,return_prediction=True)
    changed=field.copy();changed[1:]+=.01
    other,other_pred=SPECTRUM.evaluate_operator({'u_xx':1.},x,t,changed,initial,boundary_realization=SPECTRUM.BOUNDARY,return_prediction=True)
    unsupported,_=SPECTRUM.evaluate_operator({'u_x':0.},x,t,field,initial,boundary_realization=SPECTRUM.BOUNDARY)
    smooth=SPECTRUM.prepare_initialization(x,field[0],scheme='frozen_tikhonov_dx2',boundary_realization=SPECTRUM.BOUNDARY)
    summary,_=SPECTRUM.spectrum_summary([0.,-.1,1e-9],np.arange(1,16)*np.pi,.01)
    checks.update({'future_observations_do_not_change_forecast':np.array_equal(pred,other_pred),
        'future_observations_change_score_only':report['error_metric']!=other['error_metric'],
        'all_resolved_modes_retained':report['mode_count']==15 and report['modes_dropped']==0 and np.all(smooth['gains']>0),
        'unsupported_zero_LS_term_not_deleted':unsupported['status']=='unsupported_candidate_structure',
        'finite_grid_and_continuum_classifications_separate':summary['continuum_classification']=='unbounded_high_frequency_growth'
            and summary['resolved_max_growth_rate']<0})
    data=data_review(protocol)
    checks['independent_data_hash_split_review']=data['all_checks_passed']
    # Budget labels are operational differences, not silently rewritten history.
    baseline=read(ROOT/'stage2a_reconcile_complete_result.json')
    control_budget=[]
    for row in baseline['rows']:
        if [row['case'],row['sigma']] not in protocol['cases_and_noise']:continue
        for layer,lib in row['layers'].items():
            for method,result in lib['methods'].items():
                control_budget.append({'case':row['case'],'layer':layer,'method':method,
                    'historical_declared_initial_budget_s':result['path']['budget_s'],
                    'historical_all_attempts_s':result['time_s']['solver_with_worker_startup']})
    checks['historical_original_IRL1_controls_finished_under_new_initial_budget']=all(
        r['historical_all_attempts_s']<120 for r in control_budget if r['method']=='IRL1_original')
    return {'all_checks_passed':all(checks.values()),'checks':checks,'source_sha256':hashes,
        'protocol_sha256':digest(ROOT/'stage2b_protocol_result.json'),'formula_examples':examples,'data_review':data,
        'statistical_difference_review':{'only_factor':'epsilon update',
            'unchanged':'unweighted first solve; outer lambda 1e-3; 8 iterations; historical support stop after update; final weights include last update; 24-point final path; fixed solver; support 1e-5; selection alpha .005',
            'weight_ratio_proof':'For M=max|beta|, m=min|beta| and epsilon>=.01 M, max(w)/min(w)=(M+epsilon)/(m+epsilon)<=1+M/epsilon<=101.',
            'no_convergence_or_novelty_claim':True},
        'operational_budget_audit':control_budget,
        'boundary_notes':['Full-spectrum forecasting is only for declared constant-coefficient even linear heat operators and optional constant forcing.',
            'Higher-order boundary realization is a polynomial of the Dirichlet Laplacian, not implied by u=0 alone for every PDE.',
            'Unsupported structures remain visible; they cannot be silently removed or counted as forecasting failures.',
            'Raw versus fixed-smoothed initial states are separate common initialization diagnostics; neither changes primary model selection.',
            'Noise/trajectory replication is developmental; reserved test IDs6..11 and seeds>=1000 are not read.'],
        'noise_mechanism_mathematical_review':{
            'identity':'For frozen linear residual operator L and fresh independent eta with E eta=0, Cov eta=sigma^2 I, E||L(u+eta)||^2=||Lu||^2+sigma^2||L||_F^2.',
            'correct_interpretation':'Counterfactual expectation under fresh independent noise redraws AFTER freezing the original selected operator.',
            'not_claimed':'It is not the conditional expectation of the original noise given the event that this model was selected.',
            'overlap':'Overlapping weak windows do not invalidate the trace identity; they do invalidate treating rows as independent repetitions.',
            'MC_SE':'128 independent whole-field noise draws justify the sample SE of their total squared residual energies.',
            'impulse_scope':'One pixel center impulse checks indexing only. The Frobenius identity follows mathematically from translated equal stencils and is corroborated by whole-field MC.',
            'normalization':'Clean target norm is held fixed for this diagnostic; this is not the expectation of the original noisy ratio selection score.',
            'independent_formula_reviewer':'evaluation_contract reported independent IBP sign and trace-variance verification; this review confirms the same algebra and scope.'}}


def scan_runner(path):
    """Only completed dataset rows are eligible; no optimizer is called."""
    start_hash=digest(path);report=read(path)
    rows=[];issues=[]
    expected_methods={'L1','IRL1_original','IRL1_smoothed'}
    for row in report.get('rows',[]):
        if not row.get('all_methods_executed',False):
            issues.append({'case':row['case'],'cohort':row['cohort'],'issue':'dataset row in progress'});continue
        seed=row['seed'];case=row['case'];sigma=row['sigma']
        if seed==0:
            tag=f"{case}_sigma_{repr(float(sigma)).replace('.', 'p')}_seed_0"
            matrix=ROOT/'stage2_matrices'/(tag+'.npz')
        else:
            tag=f'{case}_sigma_{sigma}_seed_1_ids12_17';matrix=ROOT/'stage2b_matrices'/(tag+'.npz')
        with np.load(matrix,allow_pickle=False) as data:
            matrices=[data[f'a_{i}'].copy() for i in range(6)];targets=[data[f'b_{i}'].copy() for i in range(6)]
        a,b=stack_split(matrices,targets,range(4));av,bv=stack_split(matrices,targets,range(4,6))
        if row['input_hashes']!={'train_matrix':array_hash(a),'train_target':array_hash(b),
                               'validation_matrix':array_hash(av),'validation_target':array_hash(bv)}:
            raise ValueError('runner input hash mismatch')
        for lname,lib in row['layers'].items():
            if set(lib['methods'])!=expected_methods:raise ValueError('runner methods incomplete')
            subset=lib['indices_in_maximum_library'];an,bn,scales=normalize_training(a[:,subset],b)
            if scales!=lib['normalization'] or {'A':array_hash(an),'b':array_hash(bn)}!=lib['solver_input_hashes']:
                raise ValueError('runner normalization mismatch')
            for method,record in lib['methods'].items():
                path_record=record['path'];weights=np.ones(an.shape[1]);outer_checks=[]
                for outer in path_record.get('outer',[]):
                    if outer['status']!='certified':outer_checks.append(False);continue
                    lam=.001*float(np.max(np.abs(an.T@bn)/weights))
                    if not np.isclose(lam,outer['lambda'],rtol=2e-13,atol=0):raise ValueError('outer lambda changed')
                    cert=certify_l1(an,bn,np.asarray(outer['beta']),outer['lambda'],weights)
                    outer_checks.append(cert['certified'])
                    beta=np.asarray(outer['beta'])
                    epsilon=max(1e-8,.01*float(np.max(np.abs(beta)))) if method=='IRL1_smoothed' else 1e-8
                    if method=='IRL1_smoothed':
                        if not np.array_equal(weights,np.asarray(outer['weights_used'])) or outer['weight_update']['epsilon']!=epsilon:
                            raise ValueError('smoothed outer weight update mismatch')
                    weights=1./(np.abs(beta)+epsilon)
                if path_record.get('outer') and all(outer_checks) and not np.array_equal(weights,np.asarray(path_record['weights'])):
                    raise ValueError('final weights do not match final outer update')
                weights=np.asarray(path_record['weights']);certified=[]
                for point in path_record['points']:
                    if point['status']!='certified':certified.append(False);continue
                    beta=np.asarray(point['beta'])
                    cert=certify_l1(an,bn,beta,point['lambda'],weights)
                    if point['support']!=np.flatnonzero(np.abs(beta)>SUPPORT_TOL).tolist():raise ValueError('support mismatch')
                    certified.append(cert['certified'])
                evaluation=record['evaluation'];candidates=evaluation['candidates']
                for candidate in candidates:
                    support=candidate['support_indices'];coef=np.asarray(candidate['coefficients'])
                    error=float(np.linalg.norm(av[:,subset][:,support]@coef-bv)/np.linalg.norm(bv))
                    if not np.isclose(error,candidate['validation_error'],rtol=1e-11,atol=1e-12):raise ValueError('validation error changed')
                    if candidate['complexity']!=sum(lib['complexities'][i] for i in support):raise ValueError('complexity mismatch')
                selected=evaluation['primary_selected_candidate_id']
                complete=path_record['all_solutions_certified']
                score_ok=selected is None
                if complete:
                    chosen=min(candidates,key=lambda c:(c['validation_error']+.005*c['complexity'],c['complexity'],
                                                       c['validation_error'],tuple(c['support_indices'])))
                    score_ok=selected==chosen['candidate_id']
                if not score_ok:raise ValueError('fixed validation selector inconsistent')
                rows.append({'case':case,'sigma':sigma,'seed':seed,'cohort':row['cohort'],'layer':lname,'method':method,
                             'outer_solutions_recertified':all(outer_checks),'final_points_recertified':sum(certified),
                             'requested_points':len(path_record['points']),'numerically_complete':bool(complete and all(certified) and all(outer_checks)),
                             'fixed_selection_reproduced':score_ok,'weights_chain_verified':True,
                             'historical_or_current_solver_seconds':record['time_s']['solver_with_worker_startup']})
    return {'runner_file':str(path),'runner_sha256':start_hash,'source_snapshot_unchanged':digest(path)==start_hash,
            'runner_execution_complete':report.get('execution_complete',False),'records':rows,'issues':issues,
            'all_checks_passed':bool(report.get('execution_complete',False) and len(rows)==36 and not issues
                and all(r['numerically_complete'] and r['fixed_selection_reproduced'] for r in rows) and digest(path)==start_hash)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--runner',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'stage2b_independent_review_result.json')
    args=parser.parse_args()
    if args.output.exists():raise ValueError('review output exists; preserve it and choose a new version')
    with threadpool_limits(limits=1):
        checked=preflight()
        runner=scan_runner(args.runner) if args.runner else None
    result={'stage':'2b_independent_read_only_review','preflight':checked,'runner_review':runner,
            'all_preflight_checks_passed':checked['all_checks_passed'],'runner_scan_complete':runner is not None and runner['runner_execution_complete'],
            'all_checks_passed':bool(checked['all_checks_passed'] and runner is not None and runner['all_checks_passed']),
            'source_script_sha256':digest(Path(__file__)),'new_solver_calls':0,'new_data_generated':False,
            'test_data_used':False,'statistical_parameters_changed':False,
            'completion_meaning':'Preflight alone is not a completed method comparison; complete runner audit requires all declared runs and fixed-rule checks.'}
    strict_json_write(args.output,result)
    print(json.dumps({'all_preflight_checks_passed':result['all_preflight_checks_passed'],
                      'runner_scan_complete':result['runner_scan_complete'],'all_checks_passed':result['all_checks_passed']}))


if __name__=='__main__':main()
