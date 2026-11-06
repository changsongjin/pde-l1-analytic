"""Same-data comparison on the datasets already used by SGA-PDE.

The SGA result JSON is reused for the SGA column; GML-L1 and STRidge are
rerun on the identical noisy fields and weak-form test-function placement.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.signal import correlate

import gml1_maxlib_compare as G
import noise_inverse as N
import sga_compare as C


DATASETS = C.DATASETS
NOISE = [0.0, 0.01, 0.05]
SEEDS = C.SEEDS


def max_rows(u, x, t, half, terms):
    f = np.asarray(u, float).T
    dx, dt = x[1] - x[0], t[1] - t[0]
    hx, ht = half
    max_k = max(term["k"] for term in terms)
    def bump(n_half, h, max_d):
        s = np.linspace(-1.0, 1.0, 2 * n_half + 1)
        a = n_half * h
        poly = np.polynomial.polynomial.polypow([1.0, 0.0, -1.0], G.P_TEST)
        out, q = [], poly
        for d in range(max_d + 1):
            out.append(np.polynomial.polynomial.polyval(s, q) / a ** d)
            q = np.polynomial.polynomial.polyder(q)
        return out

    px = bump(hx, dx, max_k)
    pt = bump(ht, dt, 1)
    xgrid = np.broadcast_to(x[None, :], f.shape)

    def integ(field, kx, kt=0):
        ker = np.outer(pt[kt], px[kx])
        return (correlate(field, ker, mode="valid", method="fft") * dx * dt)[::ht, ::hx].ravel()

    a = np.column_stack([(-1) ** term["k"] * integ(G._base_field(term, f, xgrid), term["k"], 0)
                         for term in terms])
    b = -integ(f, 0, 1)
    return a, b


def one(data, sigma, seed, terms):
    u = C.noisy(data["u"], sigma, seed)
    a, b = max_rows(u, data["x"], data["t"], data["half"], terms)
    names = [G.term_name(t) for t in terms]
    p = G.LazyL1(a, b, names)
    true = data["true_weak"]
    t0 = time.perf_counter()
    ss, _ = p.path_supports()
    sl, cl = G.choose_support(p, ss)
    l1_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    si, _ = p.irl1_supports()
    sir, cir = G.choose_support(p, si)
    ir_time = time.perf_counter() - t0
    crs = N.Criterion(p.a, p.b)
    sr = N.m_stridge(crs)
    return {
        "GML1": {**G.support_eval(p.a, p.b, sl, p.names, true), "time_s": l1_time},
        "GML1_IRL1": {**G.support_eval(p.a, p.b, sir, p.names, true), "time_s": ir_time},
        "PDE-FIND_STRidge": {**G.support_eval(p.a, p.b, sr, p.names, true), "time_s": 0.0},
    }


def main():
    terms = G.build_terms()
    sga = json.loads((Path(__file__).with_name("sga_compare_result.json")).read_text(encoding="utf-8"))
    report = {
        "datasets": DATASETS, "noise": NOISE, "seeds": SEEDS,
        "data_source": "sga_compare.load_dataset; C.noisy; same fields and weak test windows as SGA-PDE comparison",
        "library_size": len(terms), "cases": {},
    }
    for name in DATASETS:
        data = C.load_dataset(name)
        rows = []
        for sigma in NOISE:
            seeds = [0] if sigma == 0 else SEEDS
            agg = {m: {"exact": 0, "supports": {}} for m in ("GML1", "GML1_IRL1", "PDE-FIND_STRidge")}
            for seed in seeds:
                got = one(data, sigma, seed, terms)
                for m, ev in got.items():
                    agg[m]["exact"] += int(ev["exact"])
                    key = " + ".join(ev["support"])
                    agg[m]["supports"][key] = agg[m]["supports"].get(key, 0) + 1
            sga_vals = []
            for seed in seeds:
                key = f"{name}|{sigma}|{seed}"
                if key in sga["runs"]:
                    sga_vals.append(bool(sga["runs"][key]["sga"]["exact"]))
            row = {"sigma": sigma, "n": len(seeds)}
            for m, v in agg.items():
                row[m] = {"exact_rate": v["exact"] / len(seeds),
                          "supports": dict(sorted(v["supports"].items(), key=lambda kv: -kv[1])[:5])}
            row["SGA-PDE"] = {"exact_rate": sum(sga_vals) / len(sga_vals) if sga_vals else None}
            rows.append(row)
            print(name, sigma, {m: row[m]["exact_rate"] for m in ("GML1", "GML1_IRL1", "PDE-FIND_STRidge", "SGA-PDE")}, flush=True)
        report["cases"][name] = {"pde": data.get("true_strong"), "true_weak": data["true_weak"], "rows": rows}
    out = Path(__file__).with_name("same_data_compare_result.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
