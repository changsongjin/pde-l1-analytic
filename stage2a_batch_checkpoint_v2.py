"""Read-only gate for the complete 72-run development baseline and its diagnosis."""
import hashlib
import json
from pathlib import Path

from stage0_evaluation import strict_json_write

ROOT=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    output=ROOT/'stage2a_batch_checkpoint_v2_result.json'
    if output.exists():raise ValueError('write-once checkpoint already exists')
    source=ROOT/'stage2a_reconcile_complete_result.json'
    diagnosis_file=ROOT/'stage2a_complete_diagnosis_result.json'
    baseline=json.loads(source.read_text(encoding='utf-8'))
    diagnosis=json.loads(diagnosis_file.read_text(encoding='utf-8'))
    expected={(case,sigma,0,layer,method) for case in
              ['heat','wave','advection_diffusion','burgers','kdv','logheat']
              for sigma in [0.,.01,.05] for layer in ['J18','J1158'] for method in ['L1','IRL1_original']}
    records=[];alternate=[]
    for row in baseline['rows']:
        for lname,layer in row['layers'].items():
            for method,result in layer['methods'].items():
                path=result['path']
                key=(row['case'],row['sigma'],row['seed'],lname,method)
                point_checks=[]
                for point in path['points']:
                    cert=point.get('certificate') or {}
                    ok=(point['status']=='certified' and cert.get('certified',False)
                        and cert['tolerances']=={'kkt_scaled_abs':1e-8,'kkt_relative':1e-5,'gap_relative':1e-9})
                    point_checks.append(ok)
                    if cert.get('certificate_kind'):
                        verified=point.get('certificate_reverified_from_saved_q',{})
                        accepted=(point.get('root_review_accepted',False) and verified.get('certified',False)
                                  and verified.get('exact_dual_feasibility',False)
                                  and verified.get('exact_direct_minus_stable_gap_is_zero',False)
                                  and not point['previous_stage0_certificate']['certified'])
                        alternate.append({'case':row['case'],'sigma':row['sigma'],'layer':lname,'method':method,
                                          'ratio':point['ratio'],'accepted':accepted,
                                          'old_certificate_still_false':not point['previous_stage0_certificate']['certified'],
                                          'new_dual_gap_relative':verified.get('dual_gap_relative')})
                complete=(path['all_solutions_certified'] and path['status']=='complete'
                          and len(path['points'])==24 and all(point_checks)
                          and result['evaluation']['selection_eligible'])
                records.append({'case':key[0],'sigma':key[1],'seed':key[2],'layer':key[3],'method':key[4],
                                'numerically_complete':complete,'certified_lambda_count':sum(point_checks),
                                'requested_lambda_count':len(path['points']),
                                'historical_solver_and_certification_seconds':result['time_s']['solver_with_worker_startup']})
    keys=[tuple(r[k] for k in ['case','sigma','seed','layer','method']) for r in records]
    total=sum(r['historical_solver_and_certification_seconds'] for r in records)
    checked_sources=all(sha(filename)==digest for filename,digest in baseline['complete_reconciliation']['sources'].items())
    diag_sources=diagnosis['sources']
    checks={'complete_unique_72_run_grid':len(keys)==72 and set(keys)==expected,
            'all_1728_fixed_weight_path_points_certified':sum(r['certified_lambda_count'] for r in records)==1728,
            'all_paths_and_primary_selections_complete':all(r['numerically_complete'] for r in records),
            'two_alternate_duals_independently_rechecked_and_root_accepted':len(alternate)==2 and all(a['accepted'] for a in alternate),
            'saved_source_provenance_intact':checked_sources,
            'frozen_selection_rule':baseline['alpha_frozen']==.005 and baseline['alpha_sensitivity']==[0.,.001,.005,.01,.05],
            'diagnosis_uses_exactly_this_complete_source':len(diag_sources)==1 and diag_sources[0]['sha256_of_snapshot']==sha(source),
            'diagnosis_all_72_records_complete':diagnosis['all_checks_passed'] and len(diagnosis['records'])==72
                and diagnosis['all_sources_execution_complete'],
            'historical_cost_including_v4_matches':abs(total-baseline['complete_reconciliation']['cost_ledger']['all_method_historical_solver_and_certification_seconds'])<1e-8,
            'no_test_data_used':not baseline['test_data_used']}
    report={'stage':'2a','all_checks_passed':all(checks.values()),'checks':checks,
            'sources':[{'file':source.name,'sha256':sha(source)},{'file':diagnosis_file.name,'sha256':sha(diagnosis_file)}],
            'max_seconds_per_batch_user_constraint':7200,'expected_runs':72,'recorded_runs':len(records),
            'numerically_complete_runs':sum(r['numerically_complete'] for r in records),
            'method_runs':records,'unfinished_numerical_runs':[r for r in records if not r['numerically_complete']],
            'accepted_alternate_dual_certificates':alternate,
            'cost_ledger':baseline['complete_reconciliation']['cost_ledger'],
            'test_data_generated_or_used':False,'new_statistical_L1_variant_tested':False,
            'support_correctness_not_a_gate':True,
            'next_stage_authorization_boundary':'Numerical gate is complete. Root/user task scope controls stage 2b; this file does not choose or alter a statistical method.',
            'pending_scope':['targeted L1 development ablations and independent validation','native comparison pipelines',
                             'held-out PDE rollout','54 nonpolynomial supplementary terms','formal >=20 paired noise seeds',
                             'publication novelty/evidence review','later raw-field outliers'],
            'interpretation':'72 complete numerical development paths; no claim of universal recovery, noise ranking, final-test performance, or publication readiness.'}
    strict_json_write(output,report)
    print(json.dumps({'all_checks_passed':report['all_checks_passed'],'complete_runs':report['numerically_complete_runs'],
                      'alternate_dual_certificates':len(alternate),'historical_solver_and_certificate_seconds':total}))


if __name__=='__main__':main()
