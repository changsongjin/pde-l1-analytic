"""Append all 54 declared weak nonpolynomial columns to preserved jet matrices.

Only the new columns are cached. Loading returns the exact original polynomial
block followed by the supplement; no near-collinearity or zero-column pruning.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from stage0_evaluation import array_hash, strict_json_write
from stage1_grammar import nonpolynomial_recipes
from stage1_nonpolynomial import weak_nonpolynomial_matrix
from stage2_data import CASES, get_bundle
from stage2_identifiability import save_matrices as polynomial_matrices

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    return {n: digest(ROOT/n) for n in ['stage2c_extended_matrices.py', 'stage1_nonpolynomial.py',
        'stage1_grammar.py', 'stage2_data.py', 'stage2_identifiability.py', 'stage1_weak_reduction_v2.py',
        'stage1_weak_reduction.py', 'stage1_jet_preprocess.py']}


def save_matrices(case, sigma, seed=0):
    gate = json.loads((ROOT/'stage1_nonpolynomial_result.json').read_text(encoding='utf-8'))
    if not gate['all_checks_passed']:
        raise RuntimeError('Supplement numerical gate has not passed.')
    poly, targets, old_meta = polynomial_matrices(case, sigma, seed)
    x, t, fields, manifest = get_bundle(case, sigma, seed)
    assert old_meta['raw_bundle_sha256'] == manifest['npz_sha256']
    recipes = nonpolynomial_recipes()
    path = ROOT/'stage2c_matrices'/f'{case}_sigma_{sigma}_seed_{seed}_supplement54.npz'
    expected = {'case': case, 'sigma': sigma, 'seed': seed, 'source_hashes': source_hashes(),
                'raw_bundle_sha256': manifest['npz_sha256'],
                'polynomial_source_metadata_sha256': digest(ROOT/'stage2_matrices'/
                    f"{case}_sigma_{repr(float(sigma)).replace('.', 'p')}_seed_{seed}.json")}
    if path.exists():
        meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        if any(meta[k] != v for k, v in expected.items()) or digest(path) != meta['npz_sha256']:
            raise ValueError('Extended matrix source or cache mismatch; preserve and version.')
        with np.load(path, allow_pickle=False) as data:
            extra = [data[f'additional_{i}'].copy() for i in range(6)]
    else:
        extra, rows = [], []
        for i, field in enumerate(fields):
            start = time.perf_counter()
            a54, b, local = weak_nonpolynomial_matrix(x, t, field,
                target_order=old_meta['target_order'], half=CASES[case]['half'])
            if not np.array_equal(b, targets[i]):
                raise ValueError('New feature family changed the original weak target.')
            if a54.shape != (len(b), 54) or local['input_observation_hash'] != array_hash(field):
                raise ValueError('New feature family row count/raw input mismatch.')
            combined = np.column_stack([poly[i], a54])
            extra.append(a54)
            rows.append({'trajectory': i, 'split': 'train' if i < 4 else 'validation',
                'raw_hash': array_hash(field), 'polynomial_hash': array_hash(poly[i]),
                'supplement_hash': array_hash(a54), 'combined_hash': array_hash(combined),
                'target_hash': array_hash(b), 'target_bitwise_unchanged': True,
                'build_seconds': time.perf_counter()-start, 'supplement_metadata': local})
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{f'additional_{i}': a for i, a in enumerate(extra)})
        meta = {**expected, 'npz_sha256': digest(path), 'trajectories': rows,
            'names': old_meta['names']+[r['name'] for r in recipes],
            'n_polynomial_and_nuisance_columns': poly[0].shape[1], 'n_supplement_columns': 54,
            'n_total_columns': poly[0].shape[1]+54, 'supplement_recipes': recipes,
            'polynomial_block_unchanged': True, 'test_data_used': False,
            'target_order': old_meta['target_order'], 'half': list(CASES[case]['half']),
            'old_polynomial_metadata': old_meta,
            'scope': 'complete declared J1158 plus 54-function supplement; wave adds external nuisance u_t'}
        strict_json_write(path.with_suffix('.json'), meta)
    combined = [np.column_stack([a, z]) for a, z in zip(poly, extra)]
    for a, z, b, record in zip(combined, extra, targets, meta['trajectories']):
        if (array_hash(a) != record['combined_hash'] or array_hash(z) != record['supplement_hash']
                or array_hash(b) != record['target_hash']):
            raise ValueError('Combined matrix or target differs from saved manifest.')
    return combined, targets, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise', nargs='+', type=float, default=[0., .01, .05])
    parser.add_argument('--output', type=Path, default=ROOT/'stage2c_extended_matrices_result.json')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new versioned output for new matrix cohorts.')
    result = {'stage': '2c_maximum_declared_library_matrices', 'all_checks_passed': False,
        'source_sha256': source_hashes(), 'rows': [], 'test_data_used': False,
        'fitting_input': 'observed fields only, shared saved raw inputs and targets',
        'no_candidate_pruning': True, 'expected_columns': {'ordinary_cases': 1212, 'wave_with_u_t': 1213}}
    with threadpool_limits(limits=1):
        for case in CASES:
            for sigma in args.noise:
                start = time.perf_counter()
                matrices, targets, meta = save_matrices(case, sigma)
                assert all(a.shape[1] == (1213 if case == 'wave' else 1212) for a in matrices)
                assert len(set(meta['names'])) == len(meta['names'])
                result['rows'].append({'case': case, 'sigma': sigma, 'seed': 0,
                    'columns': meta['n_total_columns'], 'raw_bundle_sha256': meta['raw_bundle_sha256'],
                    'supplement_npz_sha256': meta['npz_sha256'],
                    'all_targets_bitwise_preserved': all(r['target_bitwise_unchanged'] for r in meta['trajectories']),
                    'all_polynomial_blocks_preserved': all(array_hash(a[:, :meta['n_polynomial_and_nuisance_columns']]) == r['polynomial_hash']
                        for a, r in zip(matrices, meta['trajectories'])),
                    'wall_seconds': time.perf_counter()-start})
                strict_json_write(args.output, result)
                print(case, sigma, 'columns', meta['n_total_columns'], flush=True)
    result['all_checks_passed'] = all(r['all_targets_bitwise_preserved'] and r['all_polynomial_blocks_preserved'] for r in result['rows'])
    result['limits'] = ['Finite declared grammar, not every conceivable differential equation term.',
        'Adding columns does not establish identifiability, good support recovery or a new noise threshold.',
        'Nonlinear f(observed u) remains biased under observation noise; no bias correction was introduced.',
        'No formal held-out test generated and no method fitting run by this script.']
    strict_json_write(args.output, result)


if __name__ == '__main__':
    main()
