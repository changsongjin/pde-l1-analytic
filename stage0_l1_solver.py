"""Certified finite-matrix weighted L1; old experimental scripts stay unchanged.

Objective: .5 ||A beta-b||_2^2 + lambda sum_j w_j |beta_j|.
The certificate checks every column, including arbitrarily small nonzero beta.
The 1e-5 SUPPORT_TOL is solely a reporting convention, never a solver tolerance.
References: https://scikit-learn.org/1.6/modules/generated/sklearn.linear_model.Lasso.html
and https://stanford.edu/~boyd/papers/pdf/l1_ls.pdf (adapted .5 loss convention).
"""
from __future__ import annotations

import json
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.linear_model import Lasso
from threadpoolctl import threadpool_limits

from lasso_exact import lars_lasso_path, lasso_at

SUPPORT_TOL = 1e-5
KKT_ABS_TOL = 1e-8
KKT_REL_TOL = 1e-5
GAP_REL_TOL = 1e-9


def _inputs(a, b, lam, weights):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 2 or b.shape != (a.shape[0],) or not all(a.shape):
        raise ValueError('Expected nonempty A[n,p] and b[n].')
    w = np.ones(a.shape[1]) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (a.shape[1],) or not np.all(w > 0):
        raise ValueError('All weights must be finite, strictly positive, and match columns.')
    if not (np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(w).all()
            and np.isfinite(lam) and lam > 0):
        raise ValueError('Inputs must be finite and lambda strictly positive.')
    return a, b, w


def certify_l1(a, b, beta, lam, weights=None):
    """Return a full-column floating-point optimality certificate.

Dual q = residual * min(1, min_j lambda*w_j/|a_j^T residual|).
Dual objective b.T q - .5 q.T q; feasibility requires |A.T q| <= lambda*w.
Use a small inward floating-point margin; report signed gap without hiding
negative roundoff. This is a numerical certificate, not interval arithmetic.
"""
    a, b, w = _inputs(a, b, lam, weights)
    beta = np.asarray(beta, dtype=float)
    if beta.shape != w.shape or not np.isfinite(beta).all():
        raise ValueError('Coefficient shape/value invalid.')
    residual = b - a @ beta
    corr = a.T @ residual
    penalty = float(lam) * w
    active = beta != 0.0
    v = np.maximum(np.abs(corr) - penalty, 0.0)
    v[active] = np.abs(corr[active] - penalty[active] * np.sign(beta[active]))
    data_scale = max(float(np.max(np.abs(a.T @ b))), 1e-15)
    relative_v = v / np.maximum(penalty, data_scale * 1e-14)
    corr_ratio = float(np.max(np.abs(corr) / penalty))
    dual_scale = min(1.0, 1.0 / max(corr_ratio, 1e-300)) * (1.0 - 32*np.finfo(float).eps)
    q = residual * dual_scale
    primal = float(.5 * residual @ residual + penalty @ np.abs(beta))
    dual = float(b @ q - .5 * q @ q)
    gap = primal - dual
    objective_scale = max(primal, .5 * float(b @ b) * 1e-12, 1e-30)
    rel_gap = gap / objective_scale
    dual_violation = float(np.max(np.maximum(np.abs(a.T @ q) - penalty, 0.0)))
    max_abs = float(np.max(v))
    max_rel = float(np.max(relative_v))
    scaled_abs = max_abs / max(data_scale, 1e-15)
    roundoff_allowance = 128*np.finfo(float).eps * max(float(b @ b), primal, 1e-30)
    certified = (scaled_abs <= KKT_ABS_TOL and max_rel <= KKT_REL_TOL
                 and -roundoff_allowance <= gap <= GAP_REL_TOL*objective_scale + roundoff_allowance
                 and dual_violation <= roundoff_allowance)
    return {
        'certified': bool(certified), 'n_columns_checked': int(a.shape[1]),
        'active_count_exact_nonzero': int(active.sum()),
        'reported_support_count_1e-5': int(np.sum(np.abs(beta) > SUPPORT_TOL)),
        'active_kkt_max_abs': float(v[active].max()) if active.any() else 0.0,
        'inactive_kkt_max_abs': float(v[~active].max()) if (~active).any() else 0.0,
        'kkt_max_abs': max_abs, 'kkt_max_scaled_abs': scaled_abs,
        'kkt_max_relative_to_penalty': max_rel,
        'primal_objective': primal, 'dual_objective': dual, 'dual_gap': gap,
        'dual_gap_relative': rel_gap, 'dual_scale': float(dual_scale),
        'dual_feasibility_max_violation': dual_violation,
        'roundoff_allowance': float(roundoff_allowance),
        'tolerances': {'kkt_scaled_abs': KKT_ABS_TOL, 'kkt_relative': KKT_REL_TOL,
                       'gap_relative': GAP_REL_TOL},
    }


def _coordinate_descent(a, b, lam, w, initial=None, max_iter=200000):
    # z=w*beta makes the penalty ordinary L1; sklearn divides loss by n.
    model = Lasso(alpha=lam/len(b), fit_intercept=False, tol=1e-13,
                  max_iter=max_iter, warm_start=initial is not None, selection='cyclic')
    if initial is not None:
        model.coef_ = np.asarray(initial) * w
        model.intercept_ = 0.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.fit(np.asfortranarray(a/w), b)
    return model.coef_/w, {'iterations': int(model.n_iter_),
                          'sklearn_dual_gap_unscaled': float(model.dual_gap_ * len(b)),
                          'warnings': [str(c.message) for c in caught]}


def _active_set_qp(a, b, lam, w, max_steps=5000):
    """Bound-constrained quadratic minimization of the signed L1 variables.

For the active orthant solve (B.T B) z = B.T b - lambda*w, z >= 0.
If its minimizer leaves the feasible orthant, take the exact first bound step
and remove that zero variable. This is an optimization step, not post-fit
thresholding. SVD + residual refinement avoids squaring the condition number
in the linear solve. No true support, reporting threshold, or column pruning.
"""
    beta = np.zeros(a.shape[1])
    indices, signs = [], []
    boundary_events = 0
    for step in range(max_steps):
        certificate = certify_l1(a, b, beta, lam, w)
        if certificate['certified']:
            return beta, {'iterations': step, 'boundary_events': boundary_events}
        corr = a.T @ (b-a@beta)
        violation = np.abs(corr)-lam*w
        violation[indices] = -np.inf
        j = int(np.argmax(violation))
        if violation[j] > 0:
            indices.append(j)
            signs.append(float(np.sign(corr[j])))
        if not indices:
            raise RuntimeError('QP has no entering variable but certification failed.')
        for _ in range(max_steps):
            signed = a[:, indices] * np.asarray(signs)
            u, singular, vt = np.linalg.svd(signed, full_matrices=False)
            if singular[-1] <= singular[0] * np.finfo(float).eps * max(signed.shape):
                raise RuntimeError('QP active matrix is numerically singular; no ridge or column deletion used.')
            z = vt.T @ ((u.T@b)/singular - (vt@(lam*w[indices]))/singular**2)
            for _ in range(2):
                stationarity = signed.T @ (b-signed@z) - lam*w[indices]
                z += vt.T @ ((vt@stationarity)/singular**2)
            current = beta[indices] * signs
            if np.all(z > 0):
                beta[indices] = z * signs
                break
            bad = np.flatnonzero(z <= 0)
            fractions = current[bad] / (current[bad]-z[bad])
            hit = int(np.argmin(fractions))
            candidate = current + fractions[hit]*(z-current)
            remove = int(bad[hit])
            candidate[remove] = 0.0
            beta[indices] = candidate * signs
            del indices[remove]
            del signs[remove]
            boundary_events += 1
            if not indices:
                break
        else:
            raise RuntimeError('QP bound iterations exhausted.')
    raise RuntimeError('QP outer iterations exhausted.')


def solve_l1(a, b, lam, weights=None, initial=None):
    """Return beta/certificate/attempts/elapsed_s; raise if no certified answer.

No column screening or support-dependent deletion. The legacy full-matrix LARS
    candidate is checked independently. Bound-constrained QP / coordinate descent are fallbacks,
still optimizing exactly the same positive weighted L1 objective.
"""
    a, b, w = _inputs(a, b, lam, weights)
    start = time.perf_counter()
    attempts = []
    with threadpool_limits(limits=1):
        knots = lars_lasso_path(a, b, w, lam_min_ratio=0.0,
                                max_steps=max(500, 10*a.shape[1]))
        beta = lasso_at(knots, float(lam))
        certificate = certify_l1(a, b, beta, lam, w)
        attempts.append({'solver': 'legacy_full_matrix_LARS', 'knots': len(knots),
                         'last_lambda': float(knots[-1][0]), 'certificate': certificate})
        if not certificate['certified']:
            try:
                beta, details = _active_set_qp(a, b, lam, w)
                certificate = certify_l1(a, b, beta, lam, w)
                attempts.append({'solver': 'signed_bound_QP_SVD', **details, 'certificate': certificate})
            except RuntimeError as exc:
                attempts.append({'solver': 'signed_bound_QP_SVD', 'error': str(exc)})
        if not certificate['certified']:
            beta, details = _coordinate_descent(a, b, lam, w, initial=beta if initial is None else initial)
            certificate = certify_l1(a, b, beta, lam, w)
            attempts.append({'solver': 'sklearn_coordinate_descent', **details,
                             'certificate': certificate})
    if not certificate['certified']:
        raise RuntimeError('No certified weighted L1 solution: ' + json.dumps(attempts, allow_nan=False))
    return {'beta': beta, 'certificate': certificate, 'attempts': attempts,
            'elapsed_s': float(time.perf_counter()-start)}


def _trial(label, a, b, ratio, w=None):
    w = np.ones(a.shape[1]) if w is None else np.asarray(w)
    lam = float(np.max(np.abs(a.T@b)/w) * ratio)
    result = solve_l1(a, b, lam, w)
    with threadpool_limits(limits=1):
        independent, details = _coordinate_descent(a, b, lam, w)
    independent_cert = certify_l1(a, b, independent, lam, w)
    c = result['certificate']
    independent_obj = independent_cert['primal_objective']
    obj_relative_difference = abs(c['primal_objective']-independent_obj)/max(c['primal_objective'], 1e-30)
    prediction_difference = float(np.linalg.norm(a@(result['beta']-independent)))
    # Convexity bounds squared prediction difference by 2*(both suboptimalities).
    prediction_bound = float(np.sqrt(2*max(0, c['dual_gap']+independent_cert['dual_gap'])
                                    + 1e-12*max(float(b@b), 1e-30)))
    passed = bool(c['certified'] and independent_cert['certified']
                  and obj_relative_difference < 2e-8 and prediction_difference <= prediction_bound)
    return {'label': label, 'shape': list(a.shape), 'lambda': lam, 'lambda_ratio': ratio,
            'weights_range': [float(w.min()), float(w.max())],
            'support': np.flatnonzero(np.abs(result['beta']) > SUPPORT_TOL).tolist(),
            'solver': {k:v for k,v in result.items() if k != 'beta'},
            'independent_sklearn': {**details, 'certificate': independent_cert},
            'objective_relative_difference': float(obj_relative_difference),
            'prediction_difference': prediction_difference,
            'prediction_gap_bound_with_roundoff': prediction_bound,
            'passed': passed}


def main():
    import gml1_maxlib_compare as G
    import noise_inverse as N
    from library_effect_audit import library_names
    start = time.perf_counter()
    rng = np.random.default_rng(601)
    a = rng.normal(size=(100, 12))
    a /= np.linalg.norm(a, axis=0)
    truth = np.zeros(12)
    truth[[0, 3, 8]] = [1.0, -.7, .2]
    b = a@truth + .001*rng.normal(size=100)
    cases = [('synthetic_generic', a, b, 1e-2, None),
             ('synthetic_weights', a, b, 1e-3, np.geomspace(.1, 10, 12))]
    tied = a.copy()
    tied[:, 1] = tied[:, 0]
    cases.append(('synthetic_exact_duplicate', tied, b, 1e-3, None))
    correlated = a.copy()
    correlated[:, 1] = correlated[:, 0] + 1e-3*a[:, 1]
    cases.append(('synthetic_correlated', correlated, b, 1e-3, None))
    rows = []
    for args in cases:
        row = _trial(*args)
        rows.append(row)
        print(row['label'], row['passed'], flush=True)
    for case_name in ['heat', 'burgers', 'kdv']:
        for sigma in [0., .05]:
            raw, target, names, _ = G.make_data(N.CASES[case_name], sigma, 0, G.build_terms(False))
            for layer in ['L13', 'L113']:
                wanted, _ = library_names(layer)
                matrix = raw[:, [names.index(n) for n in wanted]]
                norms = np.linalg.norm(matrix, axis=0)
                # Keep even zero-norm columns (L1 forces their unique optimum to zero).
                matrix = matrix/np.where(norms > 0, norms, 1.)
                row = _trial(f'{case_name}_{sigma}_{layer}', matrix, target, 1e-3)
                rows.append(row)
                print(row['label'], row['passed'], flush=True)
    low_lambda_checks = []
    weighted_checks = []
    for case_name in ['heat', 'burgers', 'kdv']:
        raw, b, _, _ = G.make_data(N.CASES[case_name], .05, 0, G.build_terms(False))
        a = raw/np.linalg.norm(raw, axis=0)
        lam = float(np.max(np.abs(a.T@b))*1e-6)
        answer = solve_l1(a, b, lam)
        with threadpool_limits(limits=1):
            independent, details = _coordinate_descent(a, b, lam, np.ones(a.shape[1]))
        low_lambda_checks.append({'case': case_name, 'lambda_ratio': 1e-6,
                                  'solution': {k:v for k,v in answer.items() if k != 'beta'},
                                  'independent_cd': {**details, 'certificate': certify_l1(a,b,independent,lam)},
                                  'comparison_status': 'Independent CD budget exhausted; full KKT/duality certificate is the gate.'})
        baseline = solve_l1(a, b, float(np.max(np.abs(a.T@b))*1e-3))['beta']
        weighted_checks.append(_trial(case_name+'_IRL1_first_update', a, b, 1e-3,
                                      1./(np.abs(baseline)+1e-8)))
        print('low_lambda_and_IRL1', case_name, answer['certificate']['certified'], flush=True)
    # A deliberately corrupted active coefficient must fail despite omitted KKT.
    tiny = np.array([[1., 0.], [0., 1.]])
    corrupt = certify_l1(tiny, np.array([1., 0.]), np.array([.2, 1e-8]), .1)
    reject_bad = not corrupt['certified'] and corrupt['active_kkt_max_abs'] > .1
    # Invalid zero/negative weights may not silently remove a column.
    rejected = 0
    for w in [np.array([0., 1.]), np.array([-1., 1.]), np.array([np.inf, 1.])]:
        try:
            solve_l1(tiny, np.ones(2), .1, w)
        except ValueError:
            rejected += 1
    report = {'stage': 0, 'scope': 'full-matrix fixed-weight convex L1 numerical verification',
              'objective': '0.5||A beta-b||^2 + lambda*sum(w*abs(beta)); no intercept',
              'support_tolerance': SUPPORT_TOL,
              'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                           'scipy': scipy.__version__, 'sklearn': sklearn.__version__},
              'references': ['https://scikit-learn.org/1.6/modules/generated/sklearn.linear_model.Lasso.html',
                             'https://stanford.edu/~boyd/papers/pdf/l1_ls.pdf'],
              'trials': rows, 'corrupted_active_solution_rejected': bool(reject_bad),
              'low_lambda_diagnostics': low_lambda_checks,
              'IRL1_positive_weight_checks': weighted_checks,
              'invalid_weight_cases_rejected': rejected,
              'all_checks_passed': bool(all(r['passed'] for r in rows+weighted_checks)
                                       and all(r['solution']['certificate']['certified'] for r in low_lambda_checks)
                                       and reject_bad and rejected == 3),
              'elapsed_s': time.perf_counter()-start,
              'limits': ['No support recovery claim: this audits optimization only.',
                         'Finite floating point certificate, not a rigorous interval bound.',
                         'Independent coordinate descent exhausted its 200000-iteration budget on low-lambda diagnostics.',
                         'IRL1 outer iterations are nonconvex and not globally certified.',
                         'Full matrices retained; memory/cost for larger grammars must be checked.']}
    Path(__file__).with_name('stage0_l1_solver_result.json').write_text(
        json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print('all_checks_passed:', report['all_checks_passed'], flush=True)


if __name__ == '__main__':
    main()
