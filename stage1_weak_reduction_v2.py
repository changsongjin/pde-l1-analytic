"""IBP jet columns with discrete, moment-consistent weak derivative kernels.

The analytic compact bump remains (1-z^2)^8. Higher derivative quadrature
weights are minimally corrected to satisfy integration-by-parts moments against
the SAME sampled zeroth bump. These weights are a discretization, not samples
of a different analytic derivative. Both spatial and temporal derivative kernels
follow this rule. Historical v1 is retained, including its coarse-grid failure.
"""
from __future__ import annotations

import math
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.polynomial import polynomial as Poly

import stage1_jet_preprocess as V1
import stage1_weak_reduction as IBP
from stage0_evaluation import array_hash, strict_json_write

ROOT = Path(__file__).resolve().parent
MOMENT_DEGREE = 10


def discrete_bump(half, spacing, max_order, moment_degree=MOMENT_DEGREE):
    if 2*half+1 <= moment_degree or max_order > moment_degree or spacing <= 0:
        raise ValueError("weak kernel grid/degree/order incompatible")
    z = np.linspace(-1,1,2*half+1)
    design = np.vander(z,moment_degree+1,increasing=True)
    base_poly = Poly.polypow([1,0,-1],V1.TEST_POWER)
    base = (1-z*z)**V1.TEST_POWER
    kernels, errors, gains = [base], [], []
    radius = half*spacing
    for order in range(1,max_order+1):
        # Factored Leibniz evaluates the continuous derivative without power-
        # basis cancellation near the compact support boundary.
        sample = np.zeros_like(z)
        for j in range(order+1):
            if j <= V1.TEST_POWER and order-j <= V1.TEST_POWER:
                coefficient = (math.comb(order,j)*(-1)**j
                    *math.factorial(V1.TEST_POWER)//math.factorial(V1.TEST_POWER-j)
                    *math.factorial(V1.TEST_POWER)//math.factorial(V1.TEST_POWER-order+j))
                sample += coefficient*(1-z)**(V1.TEST_POWER-j)*(1+z)**(V1.TEST_POWER-order+j)
        target = np.array([0.0 if q < order else (-1)**order*math.factorial(q)
                           /math.factorial(q-order)*np.dot(base,z**(q-order))
                           for q in range(moment_degree+1)])
        corrected = sample.copy()
        for _ in range(2):
            corrected += np.linalg.lstsq(design.T,target-design.T@corrected,rcond=None)[0]
        # Enforce exact parity; the moment equations and compact bump share it.
        corrected = (corrected+(-1)**order*corrected[::-1])/2
        residual = design.T@corrected-target
        errors.append(float(np.max(np.abs(residual)/np.maximum(1,np.abs(target)))))
        gains.append({"order":order,"l2_continuous_samples":float(np.linalg.norm(sample)),
                      "l2_discrete_weights":float(np.linalg.norm(corrected)),
                      "relative_weight_change":float(np.linalg.norm(corrected-sample)/max(np.linalg.norm(sample),1e-30))})
        kernels.append(corrected/radius**order)
    return kernels,{"moment_degree":moment_degree,"moment_design_condition":float(np.linalg.cond(design)),
                    "scaled_moment_residual_by_derivative":errors,"weight_noise_gains":gains,
                    "rule":"minimum_l2_correction_to_sampled_derivative_with_common_discrete_zeroth_bump_moments"}


def weak_reduced_matrix(x,t,u,terms,target_order=1,half=(12,10),
                        window=V1.DEFAULT_WINDOW,polyorder=V1.DEFAULT_POLYORDER,stride=None):
    x,t,u,dx,dt=V1._grid(x,t,u,half,window)
    if target_order not in (1,2) or not terms:
        raise ValueError("target_order must be 1/2 and terms nonempty")
    stride=tuple(stride or half)
    identities=[IBP.reduce_term(term)[0] for term in terms]
    max_order=max(max((r for r,n in enumerate(key[2:9]) if n),default=0) for reduced in identities for key in reduced)
    max_phi=max(key[-1] for reduced in identities for key in reduced)
    jets,reconstruction=V1.spatial_jets(u,dx,window,polyorder,max_order)
    sg=window//2
    raw,xx,tt=u[:,sg:-sg],x[None,sg:-sg],t[:,None]
    px,kernel_x=discrete_bump(half[0],dx,max_phi)
    pt,kernel_t=discrete_bump(half[1],dt,target_order)
    target=(-1)**target_order*V1._integrate(raw,px[0],pt[target_order],dx,dt,stride)

    @lru_cache(maxsize=128)
    def integral(key):
        a,b,alpha,phi=key[0],key[1],key[2:9],key[-1]
        field=np.ones_like(raw)
        if a: field*=xx**a
        if b: field*=tt**b
        pure=not any(alpha[1:])
        for r,n in enumerate(alpha):
            if n: field*=(raw if pure else jets[r])**n
        return V1._integrate(field,px[phi],pt[0],dx,dt,stride)
    matrix=np.empty((len(target),len(terms)))
    for j,reduced in enumerate(identities):
        matrix[:,j]=sum(float(c)*integral(key) for key,c in reduced.items())
    if not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise ValueError("non-finite matrix; no columns removed")
    hx,ht=half
    sx,st=stride
    cache=integral.cache_info()
    return matrix,target,{"input_observation_hash":array_hash(u),"raw_shape":list(u.shape),
        "rows":len(target),"columns":len(terms),"target_order":target_order,
        "target_source":"original_observations_with_moment_consistent_weak_time_derivative",
        "half":list(half),"stride":list(stride),"test_power":V1.TEST_POWER,
        "reconstruction":reconstruction,"raw_footprint_half_x":sg+hx,"raw_footprint_half_t":ht,
        "center_x_indices":list(range(sg+hx,len(x)-sg-hx,sx)),"center_t_indices":list(range(ht,len(t)-ht,st)),
        "preprocessing":"exact_IBP_to_order3_discrete_moment_consistent_weak_kernels_v2",
        "boundary_policy":"discard reconstruction+weak halo; P8 analytic bump satisfies flux assumptions",
        "kernel_x":kernel_x,"kernel_t":kernel_t,"max_reconstructed_order":max_order,
        "max_test_function_derivative":max_phi,"free_columns_unchanged":True,
        "largest_bound_combination":max(map(len,identities)),"integral_cache_max_vectors":128,
        "cache_hits":cache.hits,"cache_misses":cache.misses,
        "matrix_hash":array_hash(matrix),"target_hash":array_hash(target)}


def _reference_matrix(x,t,terms,half):
    """Synthetic verification ONLY; never imported by benchmark construction."""
    xx,tt=np.meshgrid(x,t)
    jets=np.array([np.exp(-tt)*(np.sin(xx+r*np.pi/2)+.15*2**r*np.cos(2*xx+r*np.pi/2)
                   +.03*3**r*np.sin(3*xx+r*np.pi/2)) for r in range(7)])
    sg=15
    xx,tt=xx[:,sg:-sg],tt[:,sg:-sg]
    jets=jets[:,:,sg:-sg]
    dx,dt=x[1]-x[0],t[1]-t[0]
    px,_=discrete_bump(half[0],dx,0)
    pt,_=discrete_bump(half[1],dt,0)
    columns=[]
    for term in terms:
        field=xx**term["a"]*tt**term["b"]
        for r,n in enumerate(term["alpha"]):
            if n: field=field*jets[r]**n
        columns.append(V1._integrate(field,px[0],pt[0],dx,dt,half))
    return np.column_stack(columns)


def main():
    from stage1_grammar import enumerate_jet
    terms=enumerate_jet(6,8,6,2)
    checks={}
    _,moments=discrete_bump(12,.03,6)
    checks["all_discrete_moments"] = max(moments["scaled_moment_residual_by_derivative"]) < 1e-8
    # Independently check a physical polynomial u=x^q using discrete integration
    # of its known derivative, for all q<=10 and derivative orders through six.
    x=np.linspace(-.5,.5,25)
    dx=x[1]-x[0]
    kernels,_=discrete_bump(12,dx,6)
    polynomial_errors=[]
    for k in range(1,7):
        for q in range(11):
            lhs=(-1)**k*np.dot(x**q,kernels[k])*dx
            rhs=0.0 if q<k else math.factorial(q)/math.factorial(q-k)*np.dot(x**(q-k),kernels[0])*dx
            polynomial_errors.append(abs(lhs-rhs)/max(1,abs(rhs)))
    checks["physical_polynomial_weak_derivative"] = max(polynomial_errors)<1e-7
    refinement=[]
    high=[i for i,v in enumerate(terms) if max((r for r,n in enumerate(v["alpha"]) if n),default=0)>3]
    for nx in (129,257):
        x,t=np.linspace(-np.pi,np.pi,nx),np.linspace(0,1,61)
        xx,tt=np.meshgrid(x,t)
        u=np.exp(-tt)*(np.sin(xx)+.15*np.cos(2*xx)+.03*np.sin(3*xx))
        half=(12*(nx-1)//128,8)
        computed,b,meta=weak_reduced_matrix(x,t,u,terms,half=half)
        reference=_reference_matrix(x,t,terms,half)
        errors=np.linalg.norm(computed-reference,axis=0)/np.maximum(np.linalg.norm(reference,axis=0),1e-30)
        refinement.append({"nx":nx,"half":list(half),"max_all_column_error":float(np.max(errors)),
                           "max_high_column_error":float(np.max(errors[high])),
                           "median_high_column_error":float(np.median(errors[high])),
                           "worst_column":terms[int(np.argmax(errors))]["name"],
                           "all_column_errors":errors.tolist()})
    checks["all1158_clean_accuracy"] = refinement[-1]["max_all_column_error"]<.01
    checks["all_high_columns_clean_accuracy"] = refinement[-1]["max_high_column_error"]<.01
    checks["clean_refinement"] = refinement[-1]["max_all_column_error"]<refinement[0]["max_all_column_error"]
    x,t=np.linspace(-np.pi,np.pi,201),np.linspace(0,1,101)
    xx,tt=np.meshgrid(x,t)
    u=np.exp(-tt)*(np.sin(xx)+.15*np.cos(2*xx)+.03*np.sin(3*xx))
    observed=u+.01*np.std(u)*np.random.default_rng(6031).standard_normal(u.shape)
    timings={}
    start=time.perf_counter()
    old,_,_=V1.weak_jet_matrix(x,t,u,terms)
    old_noisy,_,_=V1.weak_jet_matrix(x,t,observed,terms)
    timings["fulljet_v1_clean_and_noisy_s"]=time.perf_counter()-start
    start=time.perf_counter()
    new,b,meta=weak_reduced_matrix(x,t,u,terms)
    new_noisy,bn,_=weak_reduced_matrix(x,t,observed,terms)
    timings["weak_reduced_v2_clean_and_noisy_s"]=time.perf_counter()-start
    reference=_reference_matrix(x,t,terms,(12,10))
    cleanerr=np.linalg.norm(new-reference,axis=0)/np.maximum(np.linalg.norm(reference,axis=0),1e-30)
    checks["actual_pilot_grid_clean_accuracy"] = np.max(cleanerr)<.01
    checks["preserve_all_columns"] = new.shape==old.shape and new.shape[1]==1158
    report={"stage":1,"all_checks_passed":all(checks.values()),"checks":checks,
        "protocol_change":"all weak derivative quadrature weights enforce common zeroth-kernel moments through degree10; base bump and SG windows unchanged",
        "symbolic_identity_dependency":"stage1_weak_reduction_result.json: 1158 exact identities and independent SymPy checks",
        "rejected_previous_numerical_gate":"stage1_weak_reduction_result.json all_checks_passed only covered identities and convergence, not adequate absolute clean accuracy; x^2 u_xxxxxx error remained29.5% at nx513",
        "moments":moments,"max_physical_polynomial_scaled_error":max(polynomial_errors),
        "clean_refinement":refinement,"pilot_grid_max_clean_column_error":float(np.max(cleanerr)),
        "pilot_worst_clean_column":terms[int(np.argmax(cleanerr))]["name"],
        "paired_noise_fixture":{"sigma":.01,"seed":6031,"input_hash":array_hash(observed),
            "fulljet_v1":IBP._column_perturbations(old,old_noisy),
            "weak_reduced_v2":IBP._column_perturbations(new,new_noisy),"timings":timings,"metadata":meta},
        "limitations":["Moment-consistent kernels are discrete quadrature weights, not pointwise analytic bump derivatives.",
            "One analytic smooth test validates numerics but cannot establish accuracy for every PDE/resolution.",
            "Residual order<=3 derivative products and nonlinear raw powers remain noise biased.",
            "Method comparisons must use identical raw observations and this frozen target/kernel version.",
            "Any accuracy failure is reported explicitly; no columns removed and support threshold remains1e-5."]}
    strict_json_write(ROOT/"stage1_weak_reduction_v2_result.json",report)
    print({"all_checks_passed":report["all_checks_passed"],"checks":checks,
           "max_clean_error":float(np.max(cleanerr)),"noise_old":report["paired_noise_fixture"]["fulljet_v1"],
           "noise_new":report["paired_noise_fixture"]["weak_reduced_v2"]})
    if not report["all_checks_passed"]:raise SystemExit(1)


if __name__=="__main__":main()
