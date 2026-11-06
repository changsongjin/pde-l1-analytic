"""DLGA-PDE on the exact datasets used by sga_compare.py."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import kd
import sga_compare as C
from dlga_compare import canonical_term


DATASETS = C.DATASETS
NOISE = [0.0, 0.05]
STRIDE = 4
CFG = {"generations": 30, "pop_size": 40, "surrogate_max_epochs": 500,
       "surrogate_patience": 80, "max_modules": 4, "max_module_length": 4,
       "epsilon": 1e-3, "library": ["u", "u_x", "u_xx", "u_xxx"]}


def run_one(data, sigma, seed):
    rng = np.random.default_rng(10_000 + seed)
    u = C.noisy(data["u"], sigma, seed)
    x, t = data["x"][::STRIDE], data["t"][::STRIDE]
    u = u[::STRIDE, ::STRIDE]
    ds = kd.PDEDataset.from_arrays(coords={"x": x, "t": t}, fields={"u": u}, lhs="u_t",
                                   name=f"same-data|sigma={sigma}|seed={seed}")
    model = kd.Model(algorithm="dlga", generations=CFG["generations"], seed=seed,
                     pop_size=CFG["pop_size"], library=CFG["library"],
                     target_lhs_order=1, lhs_auto_select=False,
                     surrogate_max_epochs=CFG["surrogate_max_epochs"],
                     surrogate_patience=CFG["surrogate_patience"],
                     max_modules=CFG["max_modules"], max_module_length=CFG["max_module_length"],
                     epsilon=CFG["epsilon"])
    t0 = time.perf_counter()
    model.fit(ds)
    terms = [canonical_term(name) for name, _ in model.result_.equation.terms]
    name_map = {"u ux": "u*u_x", "ux": "u_x", "uxx": "u_xx", "uxxx": "u_xxx"}
    true = {name_map.get(k, k) for k in data["true_strong"]}
    got = set(terms)
    return {"support": terms, "true_support": sorted(true), "exact": got == true,
            "extra": len(got - true), "missed": len(true - got),
            "equation": str(model.result_.equation), "score": float(model.result_.best_score),
            "time_s": time.perf_counter() - t0}


def main():
    report = {"datasets": DATASETS, "noise": NOISE, "stride": STRIDE, "config": CFG, "cases": {}}
    for name in DATASETS:
        data = C.load_dataset(name)
        rows = []
        for sigma in NOISE:
            raw = [run_one(data, sigma, 0)]
            rows.append({"sigma": sigma, "n": 1, "raw": raw, "exact_rate": float(raw[0]["exact"])})
            print(name, sigma, raw[0]["exact"], raw[0]["support"], flush=True)
        report["cases"][name] = {"true_strong": data["true_strong"], "rows": rows}
    out = Path(__file__).with_name("dlga_same_data_result.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
