"""Post-selection linear-noise propagation, without changing fit or selection.

For iid raw-field noise, E||L(u+eta)||^2=||Lu||^2+s^2||L||_F^2.
Overlapping windows may be correlated: this expectation uses only the trace,
not independent-row error bars. The denominator here is the CLEAN weak target
norm; this is a diagnostic, not the original noisy ratio selection score.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import correlate
from threadpoolctl import threadpool_limits

from stage0_evaluation import strict_json_write, array_hash
from stage1_grammar import enumerate_jet
from stage1_weak_reduction_v2 import discrete_bump, weak_reduced_matrix
from stage2_data import get_bundle, field_value, CASES

ROOT = Path(__file__).resolve().parent
ORDERS = {'u': 0, 'u_xx': 2, 'u_xxxx': 4, 'u_xxxxxx': 6}


def residual_kernel(coefficients, x, t):
    hx, ht = CASES['heat']['half']
    dx, dt = x[1]-x[0], t[1]-t[0]
    px, _ = discrete_bump(hx, dx, max(ORDERS[name] for name in coefficients))
    pt, _ = discrete_bump(ht, dt, 1)
    spatial = sum(value*px[ORDERS[name]] for name, value in coefficients.items())
    # A*c - b: b = - integral(u * phi_t).
    return (np.outer(pt[0], spatial)+np.outer(pt[1], px[0]))*dx*dt


def apply_kernel(field, kernel):
    hx, ht = CASES['heat']['half']
    return correlate(field[:, 15:-15], kernel, mode='valid', method='fft')[::ht, ::hx].ravel()


def main():
    source = ROOT/'stage2a_heat_reconciled_result.json'
    records = json.loads(source.read_text(encoding='utf-8'))['rows']
    dataset = next(r for r in records if r['sigma'] == .01)
    frozen = []
    for name, method in dataset['layers']['J1158']['methods'].items():
        evaluation = method['evaluation']
        assert evaluation['selection_eligible']
        chosen = next(c for c in evaluation['candidates'] if c['candidate_id'] == evaluation['primary_selected_candidate_id'])
        frozen.append({'method': name, 'candidate': chosen})
    terms = {term['name']: term for term in enumerate_jet()}
    x, t, observed, manifest = get_bundle('heat', .01, 0)
    xx, tt = np.meshgrid(x, t)
    report = {'stage': '2b_postselection_noise_mechanism', 'rows': [], 'checks': [],
        'source_file': source.name, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'source_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'raw_bundle_sha256': manifest['npz_sha256'], 'all_checks_passed': False,
        'no_coefficients_support_or_selection_changed': True, 'final_test_data_used': False,
        'noise_expectation': 'iid zero-mean raw noise; variance sigma^2*std(clean)^2',
        'normalization': 'fixed clean target energy per trajectory; NOT expectation of noisy ratio score',
        'row_dependence': 'overlap is allowed for trace expectation; no independent-row confidence intervals',
        'truth_access': 'clean field is used only for this post-selection diagnostic, never fit or selection'}
    with threadpool_limits(limits=1):
        for model in frozen:
            chosen = model['candidate']
            coefficients = dict(zip(chosen['support'], chosen['coefficients']))
            if any(name not in ORDERS for name in coefficients):
                report['rows'].append({'method': model['method'], 'status': 'unsupported_linear_structure', 'support': list(coefficients)})
                continue
            kernel = residual_kernel(coefficients, x, t)
            per_trajectory = []
            for i in (4, 5):
                clean = field_value('heat', xx, tt, manifest['trajectories'][i]['parameters'])
                selected_terms = [terms[name] for name in coefficients]
                a, b, _ = weak_reduced_matrix(x, t, clean, selected_terms, half=CASES['heat']['half'])
                direct = a@np.array(list(coefficients.values()))-b
                kernel_residual = apply_kernel(clean, kernel)
                difference = float(np.linalg.norm(direct-kernel_residual)/max(np.linalg.norm(b), 1e-30))
                assert difference < 1e-10
                # Independent stencil/direct correlation check, and exact
                # row-operator Frobenius identity via one raw-pixel impulse.
                hx, ht = CASES['heat']['half']
                impulse = np.zeros_like(clean); impulse[ht, 15+hx] = 1.
                impulse_response = apply_kernel(impulse, kernel)
                check_center = abs(impulse_response[0]-kernel[ht, hx]) < 1e-12*max(1., abs(kernel[ht, hx]))
                assert check_center
                variance = (.01*np.std(clean))**2
                expected_noise_squared = len(b)*variance*float(np.sum(kernel**2))
                signal_squared = float(direct@direct)
                target_squared = float(b@b)
                actual = apply_kernel(observed[i], kernel)
                # MC confirms the full overlapping-window aggregate expectation.
                # Draws are diagnostic fields only, not additional model fits.
                rng = np.random.default_rng(72100+i)
                draws = []
                for _ in range(128):
                    noise = rng.standard_normal(clean.shape)*np.sqrt(variance)
                    noise_response = apply_kernel(noise, kernel)
                    draws.append(float(noise_response@noise_response))
                mean = float(np.mean(draws)); se = float(np.std(draws, ddof=1)/np.sqrt(len(draws)))
                agreement = abs(mean-expected_noise_squared) <= 6*se
                assert agreement
                report['checks'].append({'method': model['method'], 'trajectory': i,
                    'kernel_vs_weak_relative_difference': difference, 'impulse_center_correct': check_center,
                    'analytic_noise_energy': expected_noise_squared, 'MC_mean': mean, 'MC_standard_error': se,
                    'MC_128_whole_fields_agree_within_6_SE': agreement, 'passed': difference < 1e-10 and agreement})
                per_trajectory.append({'trajectory_id': i, 'kernel_sha256': array_hash(kernel),
                    'rows': len(b), 'raw_noise_variance': float(variance),
                    'clean_relative_residual': float(np.sqrt(signal_squared/target_squared)),
                    'noise_expected_squared_relative_residual': expected_noise_squared/target_squared,
                    'expected_squared_relative_residual': (signal_squared+expected_noise_squared)/target_squared,
                    'root_expected_squared_relative_residual': float(np.sqrt((signal_squared+expected_noise_squared)/target_squared)),
                    'actual_observed_residual_with_clean_denominator': float(np.linalg.norm(actual)/np.linalg.norm(b))})
            report['rows'].append({'method': model['method'], 'status': 'diagnosed',
                'support': chosen['support'], 'coefficients': chosen['coefficients'],
                'original_validation_weak_error': chosen['validation_error'],
                'complexity': chosen['complexity'], 'trajectories': per_trajectory,
                'aggregate_root_expected_squared_relative_residual': float(np.sqrt(np.mean([
                    r['expected_squared_relative_residual'] for r in per_trajectory]))),
                'aggregate_clean_relative_residual': float(np.sqrt(np.mean([
                    r['clean_relative_residual']**2 for r in per_trajectory])))})
    report['all_checks_passed'] = all(r['passed'] for r in report['checks'])
    report['limits'] = ['Selected operators are conditioned on the original validation selection; not an independent method comparison.',
        'This analytic linear-kernel noise calculation does not apply unchanged to nonlinear jet products.',
        'No noise-corrected selection score or new hyperparameter is introduced here.',
        'The squared weak residual can favor a kernel with lower noise gain; this alone does not establish dynamic accuracy.']
    strict_json_write(ROOT/'stage2b_heat_noise_mechanism_result.json', report)
    print(json.dumps({'all_checks_passed': report['all_checks_passed'], 'models': [
        {k: r[k] for k in ('method', 'support', 'aggregate_root_expected_squared_relative_residual', 'aggregate_clean_relative_residual')}
        for r in report['rows'] if r['status'] == 'diagnosed']}))


if __name__ == '__main__':
    main()
