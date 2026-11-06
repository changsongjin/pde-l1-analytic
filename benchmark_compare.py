"""Merge the new L1/PDE-FIND, DLGA and existing SGA benchmark JSON files."""

from __future__ import annotations

import collections
import json
from pathlib import Path


ROOT = Path(__file__).parent


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def main():
    gml = read("gml1_maxlib_compare_all_result.json")
    out = {
        "generated_by": "benchmark_compare.py",
        "scope": {"baseline_cases": list(gml["cases"]), "noise_seeds": 3,
                   "support_rule": "set equality after canonical weak names",
                   "high_noise_policy": "report rate curves; no universal pass gate"},
        "gml1_maxlib": {}, "pde_find_stridge": {}, "dlga_budgeted": {}, "sga_existing": {},
        "same_data_comparison": {}, "same_data_dlga": {},
    }
    for case, data in gml["cases"].items():
        out["gml1_maxlib"][case] = {}
        out["pde_find_stridge"][case] = {}
        for row in data["rows"]:
            sigma = str(row["sigma"])
            out["gml1_maxlib"][case][sigma] = {
                "GML1": row["GML1"]["exact_rate"],
                "GML1_IRL1": row["GML1_IRL1"]["exact_rate"],
            }
            out["pde_find_stridge"][case][sigma] = row["PDE-FIND_STRidge"]["exact_rate"]

    for name in ("dlga_heat_budgeted_result.json", "dlga_wave_budgeted_result.json",
                 "dlga_advection_budgeted_result.json", "dlga_burgers_budgeted_result.json",
                 "dlga_kdv_budgeted_result.json"):
        data = read(name)
        for case, case_data in data["cases"].items():
            out["dlga_budgeted"][case] = {
                str(row["sigma"]): {"exact_rate": row["exact_rate"],
                                     "equations": [item.get("equation") for item in row["raw"]]}
                for row in case_data["rows"]
            }

    for name in ("sga_compare_result.json", "sga_compare_extra_result.json"):
        data = read(name)
        aggregate = collections.defaultdict(lambda: collections.defaultdict(list))
        for key, row in data["runs"].items():
            label, sigma, _ = key.split("|")
            if "sga" in row:
                aggregate[label][str(float(sigma))].append(bool(row["sga"]["exact"]))
        for case, rows in aggregate.items():
            out["sga_existing"][case] = {sigma: sum(values) / len(values)
                                          for sigma, values in rows.items()}

    if (ROOT / "gml1_kdv_nonpoly_result.json").exists():
        data = read("gml1_kdv_nonpoly_result.json")
        row = data["cases"]["kdv"]["rows"][0]
        out["nonpoly_extension"] = {
            "kdv_noiseless": {m: row[m]["exact_rate"] for m in ("GML1", "GML1_IRL1", "PDE-FIND_STRidge")},
            "result_file": "gml1_kdv_nonpoly_result.json",
        }

    if (ROOT / "same_data_compare_result.json").exists():
        data = read("same_data_compare_result.json")
        out["same_data_comparison"] = {
            case: {str(row["sigma"]): {
                "GML1": row["GML1"]["exact_rate"],
                "GML1_IRL1": row["GML1_IRL1"]["exact_rate"],
                "PDE-FIND_STRidge": row["PDE-FIND_STRidge"]["exact_rate"],
                "SGA-PDE": row["SGA-PDE"]["exact_rate"],
            } for row in case_data["rows"]}
            for case, case_data in data["cases"].items()
        }
    if (ROOT / "dlga_same_data_result.json").exists():
        data = read("dlga_same_data_result.json")
        out["same_data_dlga"] = {
            case: {str(row["sigma"]): row["exact_rate"] for row in case_data["rows"]}
            for case, case_data in data["cases"].items()
        }

    path = ROOT / "benchmark_compare_result.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
