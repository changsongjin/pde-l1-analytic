"""Development-only multi-initial-condition fields, saved once for all methods.

Test trajectory IDs 6..11 and noise seeds >=1000 are reserved and not generated.
Only scalar field VALUES are exposed to the regression/jet reconstruction chain.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import inverse_irl1 as I
import redo_two
from stage0_data_bundle import load_bundle, save_bundle
from stage0_evaluation import strict_json_write

ROOT = Path(__file__).resolve().parent
CASES = {
    "heat": {"x": (0., 1., 257), "t": (0., .08, 129), "half": (16, 8),
             "target": "u_t", "true": {"u_xx": 1.}},
    "wave": {"x": (0., 1., 257), "t": (0., 1., 257), "half": (16, 16),
             "target": "u_tt", "true": {"u_xx": 1.}},
    "advection_diffusion": {"x": (-3., 4., 281), "t": (0., 1., 101), "half": (20, 10),
             "target": "u_t", "true": {"u_x": -.4, "u_xx": .05}},
    "burgers": {"x": (-1., 3., 321), "t": (0., 1., 161), "half": (16, 16),
             "target": "u_t", "true": {"u*u_x": -1., "u_xx": .1}},
    "kdv": {"x": (-10., 12., 441), "t": (0., 1.2, 241), "half": (20, 20),
             "target": "u_t", "true": {"u*u_x": -6., "u_xxx": -1.}},
    "logheat": {"x": (0., 1., 257), "t": (0., .08, 129), "half": (16, 8),
             "target": "u_t", "true": {"u_xx": 1., "u_x^2": 1.}},
}


def trajectory_parameters(case, trajectory_id):
    if not 0 <= trajectory_id < 6:
        raise ValueError("only development trajectories 0..5 are enabled before method freezing")
    seed = 50000 + 100 * list(CASES).index(case) + trajectory_id
    rng = np.random.default_rng(seed)
    if case in ("heat", "wave"):
        coeff = rng.uniform(.35, 1., 6) * rng.choice([-1., 1.], 6) / np.sqrt(np.arange(1, 7))
        return {"cos_coefficients": coeff.tolist(),
                "sin_coefficients": rng.uniform(-.5, .5, 6).tolist() if case == "wave" else [0.]*6}
    if case == "logheat":
        coefficients = rng.uniform(-1., 1., 6)
        coefficients *= .55 / np.sum(np.abs(coefficients))
        return {"coefficients": coefficients.tolist(), "positive_heat_field_lower_bound": .45}
    if case == "advection_diffusion":
        return {"widths": rng.uniform(.35, 1.05, 3).tolist(),
                "centers": rng.uniform(-1., 1., 3).tolist(),
                "amplitudes": rng.uniform(.4, 1.2, 3).tolist()}
    if case == "burgers":
        return {"ks": [0., *(-np.sort(rng.uniform(3., 16., 3))).tolist()],
                "x0s": [0., *rng.uniform(.1, 1.2, 3).tolist()], "nu": .1}
    if case == "kdv":
        return {"a1": float(rng.uniform(.7, 1.25)), "a2": float(rng.uniform(1.6, 2.3)),
                "x01": float(rng.uniform(-3., -1.)), "x02": float(rng.uniform(-9., -6.))}
    raise ValueError(case)


def kdv_value(x, t, p):
    a1, a2 = p["a1"], p["a2"]
    alpha = ((a1-a2)/(a1+a2))**2
    th1 = a1*(x-p["x01"]) - a1**3*t
    th2 = a2*(x-p["x02"]) - a2**3*t
    theta = np.stack([np.zeros_like(th1), th1, th2, np.log(alpha)+th1+th2])
    probs = np.exp(theta - theta.max(axis=0))
    probs /= probs.sum(axis=0)
    q = np.array([0., a1, a2, a1+a2]).reshape((4,) + (1,)*np.ndim(x))
    return 2*((probs*q*q).sum(axis=0) - (probs*q).sum(axis=0)**2)


def field_value(case, x, t, p):
    x, t = np.broadcast_arrays(x, t)
    if case in ("heat", "wave", "logheat"):
        u = np.ones_like(x) if case == "logheat" else np.zeros_like(x)
        for i in range(6):
            k = (i+1)*np.pi
            if case == "logheat":
                u += p["coefficients"][i] * np.cos(k*x)*np.exp(-k*k*t)
            elif case == "heat":
                u += p["cos_coefficients"][i] * np.sin(k*x)*np.exp(-k*k*t)
            else:
                u += np.sin(k*x) * (p["cos_coefficients"][i]*np.cos(k*t)
                                   + p["sin_coefficients"][i]*np.sin(k*t))
        if case == "logheat":
            if np.min(u) <= 0:
                raise ValueError("logheat positive field assumption violated")
            return np.log(u)
        return u
    if case == "advection_diffusion":
        return sum(a * redo_two.gaussian(x.ravel(), t.ravel(), w, x0=c)[0].reshape(x.shape)
                   for a, w, c in zip(p["amplitudes"], p["widths"], p["centers"]))
    if case == "burgers":
        # Only field values leave this historical exact-solution helper.
        return I.cole_hopf(x.ravel(), t.ravel(), p["ks"], p["x0s"], p["nu"])[0].reshape(x.shape)
    if case == "kdv":
        return kdv_value(x, t, p)
    raise ValueError(case)


def bundle_path(case, sigma, seed):
    tag = repr(float(sigma)).replace(".", "p").replace("-", "m")
    return ROOT / "stage2_data" / f"{case}_sigma_{tag}_seed_{seed}.npz"


def get_bundle(case, sigma, seed):
    if sigma < 0 or not np.isfinite(sigma) or not 0 <= seed < 1000:
        raise ValueError("invalid development noise specification")
    path = bundle_path(case, sigma, seed)
    if path.exists():
        loaded = load_bundle(path)
        manifest = loaded[3]
        expected = {"case": case, "sigma": sigma, "seed": seed,
                    "generator_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        if any(manifest.get(k) != v for k, v in expected.items()):
            raise ValueError("cached bundle does not match requested inputs/generator; preserve and version it")
        return loaded
    spec = CASES[case]
    x, t = np.linspace(*spec["x"]), np.linspace(*spec["t"])
    xx, tt = np.meshgrid(x, t)
    params = [trajectory_parameters(case, i) for i in range(6)]
    observed, trajectories = [], []
    for i, param in enumerate(params):
        clean = field_value(case, xx, tt, param)
        rng = np.random.default_rng(np.random.SeedSequence([20261106, list(CASES).index(case), seed, i]))
        noise = rng.standard_normal(clean.shape)
        observed.append(clean + sigma * np.std(clean) * noise)
        trajectories.append({"trajectory_id": f"{case}/development/{i}",
                             "split": "train" if i < 4 else "validation", "parameters": param,
                             "clean_std_for_synthetic_noise": float(np.std(clean))})
    save_bundle(path, x, t, observed,
                {"case": case, "sigma": sigma, "seed": seed, "target": spec["target"],
                 "generator_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 "trajectories": trajectories, "role": "development_only",
                 "test_status": "not generated or inspected; IDs 6..11 and noise seeds >=1000 reserved",
                 "noise_model": "additive Gaussian sigma*std(clean trajectory), applied to observed u",
                 "normalization": "none in raw observations; fit scaling using training matrices only"})
    return load_bundle(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    parser.add_argument("--noise", nargs="+", type=float, default=[0., .01])
    args = parser.parse_args()
    stage1 = json.loads((ROOT / "stage1_grammar_result.json").read_text(encoding="utf-8"))
    if not stage1["all_checks_passed"]:
        raise RuntimeError("stage1 grammar gate has not passed")
    rows = []
    for case in args.cases:
        for sigma in args.noise:
            _, _, fields, manifest = get_bundle(case, sigma, 0)
            assert len(fields) == 6 and all(np.isfinite(u).all() for u in fields)
            rows.append({"case": case, "sigma": sigma, "seed": 0,
                         "file": str(bundle_path(case, sigma, 0).relative_to(ROOT)),
                         "npz_sha256": manifest["npz_sha256"],
                         "train_trajectories": 4, "validation_trajectories": 2})
    # An independent historical implementation checks the generalized tau formula.
    qx = np.linspace(-8., 8., 80)
    qt = np.linspace(0., 1., 80)
    original = redo_two.two_soliton_fields(qx, qt)[0]
    new = kdv_value(qx, qt, {"a1": 1., "a2": 2., "x01": -2., "x02": -8.})
    err = float(np.max(np.abs(original-new)))
    assert err < 1e-12
    result = {"stage": 2, "component": "development field generation", "rows": rows,
              "kdv_generalization_vs_original_max_abs": err, "all_checks_passed": True,
              "test_data_generated": False,
              "limits": ["This is data preparation, not an identifiability or noise robustness result.",
                         "Six trajectories do not establish universal identifiability or narrow confidence intervals.",
                         "Initial-condition changes alone are not evidence of causal identification."]}
    strict_json_write(ROOT / "stage2_data_result.json", result)
    print(json.dumps({"bundles": len(rows), "test_data_generated": False}))


if __name__ == "__main__":
    main()
