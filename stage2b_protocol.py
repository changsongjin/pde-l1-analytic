"""Freeze a single-factor development protocol before new method outcomes."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from stage0_evaluation import strict_json_write

ROOT = Path(__file__).resolve().parent
PROTOCOL = {
    'stage': '2b_single_factor_development',
    'cases_and_noise': [['heat', .01], ['burgers', .05], ['logheat', .05]],
    'layers': ['J18', 'J1158'],
    'methods': ['L1', 'IRL1_original', 'IRL1_smoothed'],
    'only_statistical_change': 'IRL1 epsilon=max(1e-8,0.01*max(abs(beta))) instead of fixed 1e-8',
    'epsilon_relative': .01, 'epsilon_floor': 1e-8,
    'outer_lambda_ratio': .001, 'outer_max_iterations': 8,
    'outer_stopping': 'historical unchanged support; not a convergence certificate',
    'lambda_ratios': {'count': 24, 'start': 1., 'stop': 1e-6, 'spacing': 'geometric'},
    'support_tolerance': 1e-5, 'selection_alpha': .005,
    'selection_metric': 'validation weak residual plus alpha times fixed grammar complexity',
    'first_diagnostic': {'trajectory_ids': list(range(6)), 'noise_seed': 0,
                         'controls': 'reuse hash-verified completed stage2a results'},
    'independent_development_replication': {'trajectory_ids': list(range(12, 18)),
        'train_ids': list(range(12, 16)), 'validation_ids': [16, 17], 'noise_seed': 1,
        'parameter_seed': '50000+100*case_index+trajectory_id',
        'noise_seed_sequence': '[20261106,case_index,noise_seed,trajectory_id]',
        'status': 'development replication; not the reserved final test'},
    'final_test': {'trajectory_ids': list(range(6, 12)), 'noise_seeds': '>=1000',
                   'generate_or_inspect': False},
    'per_method_initial_wall_budget_s': 120.,
    'batch_wall_limit_s': 7200,
    'numerical_retries': 'same objective and thresholds, all failure evidence and extra cost retained',
    'dynamic_diagnostic': {
        'does_not_change_primary_selection': True,
        'scope': 'supported constant-coefficient even linear heat operators only',
        'boundary_realization': 'polynomial of Dirichlet Laplacian; additional high-order BC implicit',
        'modes': 'all nx-2 resolvable interior sine modes',
        'initializations': ['raw observed t0', 'fixed g(k)=1/(1+(2*dx*k)^4) spatial smoothing'],
        'unsupported_models': 'report unsupported; do not count as failed or silently remove from ranking'},
    'prohibited_tuning': 'No epsilon/alpha/noise/trajectory choice after inspecting new variant outcomes',
}


def main():
    output = ROOT / 'stage2b_protocol_result.json'
    if output.exists():
        raise FileExistsError('Frozen protocol exists; preserve it.')
    audit = json.loads((ROOT/'stage_execution_audit_result.json').read_text(encoding='utf-8'))
    old = json.loads((ROOT/'stage0_gate_result.json').read_text(encoding='utf-8'))
    expected = {**old['historical_script_sha256'], **audit['new_scripts']}
    changed = [name for name, sha in expected.items()
               if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != sha]
    if changed:
        raise RuntimeError('Previously delivered scripts changed: '+repr(changed))
    strict_json_write(output, {**PROTOCOL, 'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'previous_script_sha256': expected, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'all_checks_passed': True, 'new_method_PDE_outcomes_seen': False})
    print('Stage2b protocol frozen; existing scripts unchanged.')


if __name__ == '__main__':
    main()
