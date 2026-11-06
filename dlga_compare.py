"""Small, reproducible DLGA-PDE comparison on the project's inverse cases.

This uses the open-source KD reimplementation of DLGA-PDE.  It is deliberately
separate from the L1 benchmark because DLGA has a neural surrogate and a genetic
search budget.  The JSON records those budgets and the raw returned expression.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

import noise_inverse as N

try:
    import kd
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit("Install external/kd (sail-kd) before running dlga_compare.py") from exc


def field(case, sigma, seed, stride=1):
    x = np.linspace(*case["x"])
    t = np.linspace(*case["t"])
    tt, xx = np.meshgrid(t, x, indexing="ij")
    # The existing analytic trajectories accept flattened arrays in several cases.
    u = case["traj"][0](xx.ravel(), tt.ravel()).reshape(tt.shape).T
    rng = np.random.default_rng(seed)
    u = u + sigma * np.std(u) * rng.standard_normal(u.shape)
    if stride > 1:
        x, t, u = x[::stride], t[::stride], u[::stride, ::stride]
    return x, t, u


def canonical_term(name):
    name = str(name).replace(" ", "")
    if name.startswith("mul(") and name.endswith(")"):
        inside = name[4:-1].split(",")
        if len(inside) == 2 and set(inside) == {"u", "u_x"}:
            return "u*u_x"
    return name


def run_one(case, sigma, seed, cfg, stride=1):
    x, t, u = field(case, sigma, seed, stride=stride)
    lhs = "u_tt" if case["target"] == "u_tt" else "u_t"
    ds = kd.PDEDataset.from_arrays(coords={"x": x, "t": t}, fields={"u": u}, lhs=lhs,
                                   name=f"{case['pde']}|sigma={sigma}|seed={seed}")
    model = kd.Model(
        algorithm="dlga",
        generations=cfg["generations"],
        seed=seed,
        pop_size=cfg["pop_size"],
        library=cfg["library"],
        target_lhs_order=2 if case["target"] == "u_tt" else 1,
        lhs_auto_select=False,
        surrogate_max_epochs=cfg["surrogate_max_epochs"],
        surrogate_patience=cfg["surrogate_patience"],
        max_modules=cfg["max_modules"],
        max_module_length=cfg["max_module_length"],
        epsilon=cfg["epsilon"],
    )
    t0 = time.perf_counter()
    model.fit(ds)
    elapsed = time.perf_counter() - t0
    terms = []
    for name, scalar in model.result_.equation.terms:
        terms.append(canonical_term(name))
    true = {"u*u_x" if k == "(u^2)_x" else k for k in case["true"]}
    got = set(terms)
    return {
        "case": case["pde"], "sigma": sigma, "seed": seed,
        "support": terms, "exact": got == true,
        "extra": len(got - true), "missed": len(true - got),
        "true_support": sorted(true), "equation": str(model.result_.equation),
        "best_expression": str(model.result_.best_expression),
        "score": float(model.result_.best_score), "time_s": elapsed,
        "surrogate": {"epochs": cfg["surrogate_max_epochs"], "patience": cfg["surrogate_patience"]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--noise", nargs="+", type=float, default=[0.0, 0.05])
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--generations", type=int, default=20)
    ap.add_argument("--pop-size", type=int, default=40)
    ap.add_argument("--surrogate-epochs", type=int, default=500)
    ap.add_argument("--surrogate-patience", type=int, default=80)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    cases = dict(N.CASES)
    if args.only:
        cases = {k: v for k, v in cases.items() if k in args.only}
    if args.smoke:
        cases = {next(iter(cases)): next(iter(cases.values()))}
        args.noise = [0.0]
        args.seeds = 1
        args.generations = min(args.generations, 5)
        args.pop_size = min(args.pop_size, 20)
        args.surrogate_epochs = min(args.surrogate_epochs, 100)
        args.surrogate_patience = min(args.surrogate_patience, 20)
        args.stride = max(args.stride, 4)

    cfg = {
        "library": ["u", "u_x", "u_xx", "u_xxx"],
        "generations": args.generations, "pop_size": args.pop_size,
        "surrogate_max_epochs": args.surrogate_epochs,
        "surrogate_patience": args.surrogate_patience,
        "max_modules": 4, "max_module_length": 4, "epsilon": 1e-3,
    }
    report = {
        "python": platform.python_version(), "numpy": np.__version__,
        "algorithm": "DLGA-PDE via sail-kd",
        "config": {**cfg, "stride": args.stride}, "noise": args.noise, "n_seeds": args.seeds, "cases": {},
    }
    for cname, case in cases.items():
        rows = []
        for sigma in args.noise:
            seeds = [0] if sigma == 0 else list(range(args.seeds))
            raw = []
            for seed in seeds:
                try:
                    raw.append(run_one(case, sigma, seed, cfg, stride=args.stride))
                except Exception as exc:  # keep the benchmark auditable
                    raw.append({"case": case["pde"], "sigma": sigma, "seed": seed,
                                "error": f"{type(exc).__name__}: {exc}"})
            good = [r for r in raw if "exact" in r]
            row = {"sigma": sigma, "n": len(seeds), "raw": raw,
                   "exact_rate": sum(int(r["exact"]) for r in good) / len(seeds) if seeds else 0.0}
            rows.append(row)
            print(cname, "sigma", sigma, "exact_rate", row["exact_rate"], flush=True)
        report["cases"][cname] = {"pde": case["pde"], "true": case["true"], "rows": rows}
    out = Path(args.output) if args.output else Path(__file__).with_name("dlga_compare_result.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
