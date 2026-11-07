"""Report-only implementation of already frozen stage-3 statistical rules.

No campaign file is read on import or in the synthetic self-check. Formal entry
points reject incomplete seed sets. Observations and selected models are never
generated, fitted or changed here.
"""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binom, binomtest, multinomial

ROOT = Path(__file__).resolve().parent
SEEDS = list(range(1000, 1020))
NOISE = np.array([0., .01, .03, .05, .07, .10])
BOOTSTRAPS = 9999
BOOTSTRAP_SEED = 2026110701


def complete_seeds(seed_ids):
    if list(seed_ids) != SEEDS:
        raise ValueError('Require all 20 complete seeds in frozen order, with paired outcomes.')


def binary(values, shape):
    arr = np.asarray(values)
    if arr.shape != shape or not np.isin(arr, [0, 1]).all():
        raise ValueError('Expected complete binary support-recovery indicators.')
    return arr.astype(float)


@lru_cache(maxsize=84)
def exact_interval(k, n, level):
    interval = binomtest(k, n).proportion_ci(confidence_level=level, method='exact')
    return [float(interval.low), float(interval.high)]


def recovery_interval(seed_ids, values):
    complete_seeds(seed_ids)
    arr = binary(values, (20,))
    count = int(arr.sum())
    return {'successes':count, 'n':20, 'estimate':count/20, 'ci95':exact_interval(count, 20, .95),
            'scope':'pointwise conditional-noise interval; fixed initial conditions'}


def paired_difference_interval(seed_ids, l1, irl1):
    complete_seeds(seed_ids)
    first, second = binary(l1, (20,)), binary(irl1, (20,))
    plus = int(np.count_nonzero((first == 0) & (second == 1)))
    minus = int(np.count_nonzero((first == 1) & (second == 0)))
    lo_p, hi_p = exact_interval(plus, 20, .975)
    lo_m, hi_m = exact_interval(minus, 20, .975)
    return {'n':20, 'irl1_only_successes':plus, 'l1_only_successes':minus,
            'estimate':(plus-minus)/20, 'ci95':[lo_p-hi_m, hi_p-lo_m],
            'method':'frozen Bonferroni difference of two 97.5% exact discordance-probability intervals'}


def paired_auc(seed_ids, indicators, labels):
    """Input [20 seeds,6 noise levels,K frozen library/method labels]."""
    complete_seeds(seed_ids)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError('Unique library/method labels required.')
    arr = binary(indicators, (20, 6, len(labels)))
    if not np.array_equal(arr[:, 0, :], np.broadcast_to(arr[0, 0, :], (20, len(labels)))):
        raise ValueError('The clean endpoint is deterministic and must be shared across seeds.')
    auc = np.trapezoid(arr, x=NOISE, axis=1) / .10
    estimate = auc.mean(axis=0)
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(0, 20, size=(BOOTSTRAPS, 20))
    boot = auc[indices].mean(axis=1)  # Exactly the same resampled seeds for all columns.

    def interval(point, replicas):
        lo, hi = np.quantile(replicas, [.025, .975], method='linear')
        return {'estimate':float(point), 'basic_ci95':[float(2*point-hi), float(2*point-lo)],
                'degenerate_resampling':bool(np.ptp(replicas) == 0),
                'degenerate_interpretation':'zero bootstrap variation is conditional evidence, not proof of zero population uncertainty'}

    per_label = {label:{**interval(estimate[j], boot[:, j]), 'per_seed_auc':auc[:, j].tolist()}
                 for j, label in enumerate(labels)}
    differences = {}
    for j, label in enumerate(labels):
        if label.endswith('/L1'):
            other = label[:-3] + '/IRL1_original'
            if other in labels:
                k = labels.index(other)
                differences[other+' minus '+label] = interval(estimate[k]-estimate[j], boot[:, k]-boot[:, j])
    return {'per_label':per_label, 'paired_method_differences':differences,
            'resamples':BOOTSTRAPS, 'rng_seed':BOOTSTRAP_SEED,
            'scope':'whole-seed paired basic bootstrap; deterministic clean endpoint shared; intervals not truncated'}


def observed_crossings(rates):
    values = np.asarray(rates, dtype=float)
    if values.shape != NOISE.shape or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError('Six finite frozen-grid recovery rates required.')
    exact = NOISE[values == .5].tolist()
    brackets = [[float(NOISE[i]), float(NOISE[i+1])] for i in range(5)
                if (values[i]-.5)*(values[i+1]-.5) < 0]
    return {'observed_exact_half_points':exact, 'strict_crossing_brackets':brackets,
            'crossing_observed':bool(exact or brackets), 'extrapolation':False}


def main():
    path = ROOT/'stage3_statistics_result.json'
    if path.exists():
        raise FileExistsError('Preserve synthetic verification; use a new version for changes.')
    protocol_path = ROOT/'stage3_protocol_result.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    checks = {'frozen_protocol':protocol['is_frozen'] is True,
        'frozen_seed_and_noise_grid':protocol['noise_seeds'] == SEEDS and protocol['noise_levels'] == NOISE.tolist(),
        'frozen_bootstrap_settings':protocol['evaluation_and_statistics']['AUC_uncertainty']['resamples'] == BOOTSTRAPS
            and protocol['evaluation_and_statistics']['AUC_uncertainty']['rng_seed'] == BOOTSTRAP_SEED}
    rejected = 0
    for ids in [SEEDS[:-1], SEEDS[::-1], [1000]*20]:
        try:
            recovery_interval(ids, np.zeros(20))
        except ValueError:
            rejected += 1
    checks['incomplete_or_unpaired_sets_rejected'] = rejected == 3
    checks['exact_endpoints'] = exact_interval(0, 20, .95)[0] == 0 and exact_interval(20, 20, .95)[1] == 1
    min_binomial_coverage = 1.
    for probability in np.linspace(0, 1, 101):
        coverage = sum(binom.pmf(k, 20, probability) for k in range(21)
                       if exact_interval(k, 20, .95)[0] <= probability <= exact_interval(k, 20, .95)[1])
        min_binomial_coverage = min(min_binomial_coverage, float(coverage))
    checks['exact_binomial_enumerated_coverage_ge95'] = min_binomial_coverage >= .95-1e-12
    # Enumerate all discordance-count combinations, combining both agreeing cells.
    counts = [(p, m, 20-p-m) for p in range(21) for m in range(21-p)]
    intervals = [(exact_interval(p, 20, .975)[0]-exact_interval(m, 20, .975)[1],
                  exact_interval(p, 20, .975)[1]-exact_interval(m, 20, .975)[0]) for p, m, _ in counts]
    min_paired_coverage = 1.
    for p_int in range(11):
        for m_int in range(11-p_int):
            p, m = p_int/10, m_int/10
            probs = [p, m, max(0., 1-p-m)]
            coverage = sum(multinomial.pmf(c, 20, probs) for c, (lo, hi) in zip(counts, intervals) if lo <= p-m <= hi)
            min_paired_coverage = min(min_paired_coverage, float(coverage))
    checks['paired_interval_enumerated_coverage_ge95'] = min_paired_coverage >= .95-1e-12
    deterministic = np.ones((20, 6, 2))
    deterministic[:, :, 0] = 0
    auc = paired_auc(SEEDS, deterministic, ['J1212/L1', 'J1212/IRL1_original'])
    checks['shared_clean_and_degenerate_auc'] = (auc['per_label']['J1212/L1']['estimate'] == 0
        and abs(auc['per_label']['J1212/IRL1_original']['estimate']-1) < 1e-14
        and all(r['degenerate_resampling'] for r in auc['per_label'].values()))
    rng = np.random.default_rng(17)
    curves = rng.integers(0, 2, size=(20, 6, 2))
    curves[:, :, 1] = curves[:, :, 0]
    curves[:, 0, :] = 1
    paired = paired_auc(SEEDS, curves, ['J18/L1', 'J18/IRL1_original'])
    diff = next(iter(paired['paired_method_differences'].values()))
    checks['identical_curves_have_zero_paired_bootstrap_variation'] = diff['basic_ci95'] == [0., 0.]
    checks['multiple_crossings_and_no_extrapolation'] = (len(observed_crossings([1, .4, .8, .2, .6, 0])['strict_crossing_brackets']) == 5
        and not observed_crossings([1]*6)['crossing_observed'])
    report = {'stage':'3_frozen_statistics_synthetic_verification', 'all_checks_passed':all(checks.values()),
        'checks':checks, 'minimum_enumerated_binomial_coverage':min_binomial_coverage,
        'minimum_enumerated_paired_coverage':min_paired_coverage,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'protocol_sha256':hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        'created_utc':datetime.now(timezone.utc).isoformat(), 'formal_outcomes_read':False,
        'scope':'Synthetic implementation checks of predeclared rules; finite-grid coverage checks complement the frozen interval construction, not a new statistical method.'}
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(report))
    if not report['all_checks_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
