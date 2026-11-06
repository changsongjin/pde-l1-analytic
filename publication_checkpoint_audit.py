"""Read-only checks for the exploratory checkpoint before GitHub upload.

Does not edit experiments or rerun their main entry points.  A syntactically
valid script or parseable JSON is not treated as publication validation.
"""

import json
import platform
from pathlib import Path

import numpy as np

import gml1_maxlib_compare as G
import noise_inverse as N


ROOT = Path(__file__).resolve().parent
PREFIXES = ("gml1_", "dlga_", "benchmark_", "same_data_", "maxlib_",
            "library_effect_", "outlier_", "point_outlier_", "stability_pareto_")


def check_serialization():
    scripts, results, nonfinite, errors = [], [], [], []

    def reject_constant(token):
        raise ValueError("non-finite JSON token: " + token)

    for path in sorted(ROOT.iterdir()):
        if not path.is_file() or not path.name.startswith(PREFIXES):
            continue
        try:
            if path.suffix == ".py":
                compile(path.read_text(encoding="utf-8-sig"), str(path), "exec")
                scripts.append(path.name)
            elif path.suffix == ".json":
                source = path.read_text(encoding="utf-8-sig")
                json.loads(source)
                results.append(path.name)
                try:
                    json.loads(source, parse_constant=reject_constant)
                except ValueError as exc:
                    nonfinite.append({"file": path.name, "error": str(exc)})
        except (ValueError, SyntaxError) as exc:
            errors.append({"file": path.name, "error": str(exc)})
    return {"scripts_checked": scripts, "json_checked": results,
            "syntax_or_parse_errors": errors, "nonstandard_nonfinite_json": nonfinite}


def coefficient_check():
    a, b, names, _ = G.make_data(N.CASES["heat"], 0.0, 0, G.build_terms(False))
    problem = G.LazyL1(a, b, names)
    support = (problem.names.index("u_xx"),)
    old = G.support_eval(problem.a, b, support, problem.names, N.CASES["heat"]["true"])
    physical = float(np.linalg.lstsq(a[:, [names.index("u_xx")]], b, rcond=None)[0][0])
    return {"case": "heat sigma=0", "stored_evaluator_relative_error": old["coef_rel_err"],
            "physical_coefficient_from_raw_design": physical,
            "physical_relative_error": abs(physical - 1.0),
            "status": "confirmed normalized-versus-physical coefficient mismatch; support unaffected"}


def label_checks():
    data = json.loads((ROOT / "dlga_same_data_result.json").read_text(encoding="utf-8"))
    aliases = {"ux": "u_x", "uxx": "u_xx", "uxxx": "u_xxx"}
    mismatches = []
    for case, result in data["cases"].items():
        for group in result["rows"]:
            for row in group["raw"]:
                names = row.get("true_support", [])
                if any(name in aliases for name in names):
                    mismatches.append({"case": case, "sigma": group["sigma"],
                                       "true_support": names, "reported_support": row["support"]})
    return {"mixed_true_support_aliases": mismatches,
            "status": "source and saved JSON differ; re-evaluate original expressions with one symbolic canonicalizer"}


def rank_checks():
    data = json.loads((ROOT / "library_effect_audit_result.json").read_text(encoding="utf-8"))
    rows = []
    for case, result in data["cases"].items():
        geometry = result["layers"]["L113"]["rows"][0]["raw"][0]["geometry"]
        rows.append({"case": case, "columns": geometry["n_columns"],
                     "machine_rank": geometry["rank_machine"],
                     "effective_rank_relative_1e-6": geometry["rank_rel_1e-6"],
                     "truncated_condition_relative_1e-6": geometry["condition_rel_1e-6"]})
    return rows


def main():
    report = {
        "date": "2026-11-06", "python": platform.python_version(), "numpy": np.__version__,
        "scope": "checkpoint audit; no new benchmark campaign",
        "serialization": check_serialization(), "coefficient_check": coefficient_check(),
        "dlga_label_check": label_checks(), "rank_definitions": rank_checks(),
        "publication_blockers": [
            "Current coefficient-error summaries compare normalized coefficients to physical truth.",
            "Saved DLGA support aliases and source code are inconsistent; rankings need re-evaluation.",
            "LazyL1 verifies omitted-column KKT gaps, but does not independently certify all active-column stationarity conditions.",
            "MSTLS complexity |S|/p changes when library p changes, confounding a library-size experiment.",
            "Effective rank at 1e-6 and a truncated condition number do not prove sparse non-identifiability.",
            "Current Pareto summary compares only selected models and pools different seed datasets; no common-data validation frontier yet.",
            "Outlier gamma=2 was selected with benchmark truth; sensitivity with one seed is exploratory, not held-out validation.",
            "Row-error weights in code mean gamma/lambda=gamma_weight, not an independently fixed gamma.",
            "DLGA baseline field() uses only the first trajectory; it is not the same multitrajectory input as the weak baseline.",
            "Three noise seeds are a pilot; final curves need uncertainty intervals and a larger paired design.",
        ],
        "release_status": "exploratory checkpoint; not submission-ready evidence",
    }
    out = ROOT / "publication_checkpoint_audit_result.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Checked", len(report["serialization"]["scripts_checked"]), "scripts and",
          len(report["serialization"]["json_checked"]), "JSON files")
    print("Syntax/parse errors:", len(report["serialization"]["syntax_or_parse_errors"]))
    print("Non-finite JSON files:", len(report["serialization"]["nonstandard_nonfinite_json"]))
    print("Confirmed coefficient/label issues; checkpoint remains exploratory")
    print("wrote", out.name)


if __name__ == "__main__":
    main()
