"""Stage-1 implementation and finite-library resource pilot, not recovery claims."""

import json
import time
from pathlib import Path

import numpy as np
import psutil

from stage0_evaluation import strict_json_write
from stage1_grammar import enumerate_jet
from stage1_jet_preprocess import weak_jet_matrix

ROOT = Path(__file__).resolve().parent


def main():
    components = ("stage0_gate", "stage1_grammar", "stage1_jet_preprocess")
    gates = {name: json.loads((ROOT / (name+"_result.json")).read_text(encoding="utf-8"))["all_checks_passed"]
             for name in components}
    if not all(gates.values()):
        raise RuntimeError("prerequisite implementation checks failed")
    x, t = np.linspace(-np.pi, np.pi, 161), np.linspace(0., 1., 101)
    xx, tt = np.meshgrid(x, t)
    observed = np.exp(-tt)*np.sin(xx) + .25*np.exp(-4*tt)*np.cos(2*xx)
    terms = enumerate_jet()
    start = time.perf_counter()
    a, b, meta = weak_jet_matrix(x, t, observed, terms, half=(12, 10))
    seconds = time.perf_counter()-start
    passed = a.shape[1] == 1158 and np.isfinite(a).all() and np.isfinite(b).all()
    report = {"stage": 1, "component_checks": gates,
              "all_checks_passed": bool(passed),
              "max_library_pilot": {"shape": list(a.shape), "build_seconds": seconds,
                                     "matrix_bytes": a.nbytes, "gram_bytes_estimate": 8*a.shape[1]**2,
                                     "available_memory_after": psutil.virtual_memory().available,
                                     "matrix_hash": meta["matrix_hash"]},
              "main_grammar_status": "1158 finite polynomial jet columns ready for development diagnostics",
              "nonpolynomial_supplement_status": "54 recipes declared; numerical extension pending",
              "next_action": "multi-trajectory and matrix-perturbation diagnostics; exact weak reduction pilot",
              "limits": ["Preprocessing implementation checks are not noise robustness validation.",
                         "High-order reconstruction noise is a measured limitation requiring investigation.",
                         "No final test trajectories or final-test noise seeds used."]}
    strict_json_write(ROOT / "stage1_gate_result.json", report)
    print(json.dumps({"all_checks_passed": bool(passed), "shape": list(a.shape), "build_seconds": seconds}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
