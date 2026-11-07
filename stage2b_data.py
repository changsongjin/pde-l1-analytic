"""New independent development trajectories; reserved final test is inaccessible."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from stage0_data_bundle import load_bundle, save_bundle
from stage0_evaluation import array_hash, strict_json_write
from stage1_grammar import enumerate_jet, key, LAYERS
from stage1_weak_reduction_v2 import weak_reduced_matrix
from stage2_data import CASES, field_value, trajectory_parameters
from stage2b_protocol import PROTOCOL

ROOT = Path(__file__).resolve().parent
IDS = list(range(12, 18))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    return {n: digest(ROOT/n) for n in ['stage2b_data.py', 'stage2b_protocol.py', 'stage2_data.py',
        'stage1_grammar.py', 'stage1_jet_preprocess.py', 'stage1_weak_reduction.py',
        'stage1_weak_reduction_v2.py']}


def _parameters(case, trajectory_id):
    # Same distributions/order of RNG draws as the established generator.
    rng = np.random.default_rng(50000+100*list(CASES).index(case)+trajectory_id)
    if case == 'heat':
        coeff = rng.uniform(.35, 1., 6)*rng.choice([-1., 1.], 6)/np.sqrt(np.arange(1, 7))
        return {'cos_coefficients': coeff.tolist(), 'sin_coefficients': [0.]*6}
    if case == 'logheat':
        coeff = rng.uniform(-1., 1., 6)
        coeff *= .55/np.sum(np.abs(coeff))
        return {'coefficients': coeff.tolist(), 'positive_heat_field_lower_bound': .45}
    if case == 'burgers':
        return {'ks': [0., *(-np.sort(rng.uniform(3., 16., 3))).tolist()],
                'x0s': [0., *rng.uniform(.1, 1.2, 3).tolist()], 'nu': .1}
    raise ValueError('Only frozen stage2b cases are enabled.')


def get_bundle(case, sigma, seed=1):
    if [case, sigma] not in PROTOCOL['cases_and_noise'] or seed != 1:
        raise ValueError('Only the frozen development replication is enabled.')
    path = ROOT/'stage2b_data'/f'{case}_sigma_{sigma}_seed_{seed}_ids12_17.npz'
    expected = {'case': case, 'sigma': sigma, 'seed': seed, 'source_hashes': sources()}
    if path.exists():
        loaded = load_bundle(path)
        if any(loaded[3].get(k) != v for k, v in expected.items()):
            raise ValueError('Preserve mismatched cache and version the new attempt.')
        return loaded
    spec = CASES[case]
    x, t = np.linspace(*spec['x']), np.linspace(*spec['t'])
    xx, tt = np.meshgrid(x, t)
    observations, metadata = [], []
    for i in IDS:
        params = _parameters(case, i)
        clean = field_value(case, xx, tt, params)
        rng = np.random.default_rng(np.random.SeedSequence([20261106, list(CASES).index(case), seed, i]))
        observations.append(clean+sigma*np.std(clean)*rng.standard_normal(clean.shape))
        metadata.append({'trajectory_id': i, 'split': 'train' if i < 16 else 'validation',
                         'parameters': params, 'clean_std_for_synthetic_noise': float(np.std(clean))})
    save_bundle(path, x, t, observations, {**expected, 'target': spec['target'],
        'trajectories': metadata, 'role': 'independent_development_replication',
        'noise_model': 'Gaussian sigma*std(clean trajectory) added to observed u',
        'test_data_generated_or_used': False})
    return load_bundle(path)


def save_matrices(case, sigma, seed=1):
    x, t, observations, manifest = get_bundle(case, sigma, seed)
    path = ROOT/'stage2b_matrices'/f'{case}_sigma_{sigma}_seed_{seed}_ids12_17.npz'
    if path.exists():
        meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        if meta['source_hashes'] != sources() or meta['raw_bundle_sha256'] != manifest['npz_sha256']:
            raise ValueError('Matrix source/input cache mismatch')
        if digest(path) != meta['npz_sha256']:
            raise ValueError('Matrix file hash mismatch')
        with np.load(path, allow_pickle=False) as data:
            matrices, targets = [data[f'a_{i}'].copy() for i in range(6)], [data[f'b_{i}'].copy() for i in range(6)]
        for a, b, m in zip(matrices, targets, meta['trajectories']):
            assert array_hash(a) == m['matrix_hash'] and array_hash(b) == m['target_hash']
        return matrices, targets, meta
    terms = enumerate_jet()
    small = {key(term) for term in enumerate_jet(*LAYERS['J18'])}
    matrices, targets, rows = [], [], []
    for i, observed in zip(IDS, observations):
        start = time.perf_counter()
        a, b, meta = weak_reduced_matrix(x, t, observed, terms, target_order=1, half=CASES[case]['half'])
        assert np.isfinite(a).all() and np.isfinite(b).all()
        matrices.append(a); targets.append(b)
        rows.append({'trajectory': i, 'split': 'train' if i < 16 else 'validation', 'rows': len(b),
            'raw_hash': meta['input_observation_hash'], 'matrix_hash': array_hash(a),
            'target_hash': array_hash(b), 'build_seconds': time.perf_counter()-start})
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{f'a_{i}': a for i, a in enumerate(matrices)},
                        **{f'b_{i}': b for i, b in enumerate(targets)})
    metadata = {'case': case, 'sigma': sigma, 'seed': seed, 'source_hashes': sources(),
        'raw_bundle_sha256': manifest['npz_sha256'], 'npz_sha256': digest(path),
        'names': [term['name'] for term in terms], 'complexities': [term['complexity'] for term in terms],
        'small_indices': [i for i, term in enumerate(terms) if key(term) in small],
        'trajectories': rows, 'test_data_used': False, 'half': list(CASES[case]['half']), 'target_order': 1}
    strict_json_write(path.with_suffix('.json'), metadata)
    return matrices, targets, metadata


def main():
    import stage2_field_check as check
    protocol = json.loads((ROOT/'stage2b_protocol_result.json').read_text(encoding='utf-8'))
    assert protocol['all_checks_passed']
    report = {'stage': '2b_data', 'protocol_sha256': digest(ROOT/'stage2b_protocol_result.json'),
        'source_hashes': sources(), 'rows': [], 'field_checks': [], 'all_checks_passed': False,
        'test_data_generated_or_used': False, 'parameter_distribution_equivalence': []}
    output = ROOT/'stage2b_data_result.json'
    with threadpool_limits(limits=1):
        for case, sigma in PROTOCOL['cases_and_noise']:
            identical = all(_parameters(case, i) == trajectory_parameters(case, i) for i in range(6))
            assert identical
            report['parameter_distribution_equivalence'].append({'case': case, 'old_ids_0_5_exact_match': identical})
            spec = CASES[case]; hx, ht = check.STEPS[case]
            rng = np.random.default_rng(63000+list(CASES).index(case))
            qx = rng.uniform(spec['x'][0]+3*hx, spec['x'][1]-3*hx, 128)
            qt = rng.uniform(spec['t'][0]+3*ht, .8*spec['t'][1], 128)
            for i in IDS:
                levels = [check.pde_difference(case, _parameters(case, i), qx, qt, hx*q, ht*q) for q in check.LEVELS]
                coarse, fine = levels[0]['relative_residual'], levels[-1]['relative_residual']
                passed = fine <= check.FINE_RELATIVE_TOL and (fine <= check.CONVERGENCE_FLOOR or fine <= coarse/4)
                report['field_checks'].append({'case': case, 'trajectory_id': i, 'levels': levels, 'passed': passed})
            assert all(r['passed'] for r in report['field_checks'])
            _, _, meta = save_matrices(case, sigma)
            report['rows'].append(meta)
            strict_json_write(output, report)
            print(case, sigma, 'six saved raw/matrix pairs; independent field checks passed', flush=True)
    report['all_checks_passed'] = all(r['passed'] for r in report['field_checks'])
    strict_json_write(output, report)


if __name__ == '__main__':
    main()
