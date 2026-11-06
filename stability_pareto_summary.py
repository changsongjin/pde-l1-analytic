"""Summarize support stability and residual-complexity Pareto points."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from outlier_l1_compare import layer_names


ROOT = Path(__file__).parent
SELECTION_FILES = [
    ROOT / "library_effect_selection_heat_wave_result.json",
    ROOT / "library_effect_selection_adv_burgers_kdv_result.json",
]


def term_complexity(name: str) -> int:
    if name == "1":
        return 1
    derivative_match = re.search(r"_x+$", name)
    derivative = len(derivative_match.group(0)) - 1 if derivative_match else 0
    powers = [int(x) for x in re.findall(r"u\^(\d+)", name)]
    power = powers[0] if powers else (1 if "u" in name else 0)
    coordinate = 2 if "x^2" in name else (1 if re.search(r"\bx u", name) else 0)
    nonpoly = 2 if any(tag in name for tag in ("sin", "cos", "exp", "inv")) else 0
    return 1 + derivative + power + 2 * coordinate + nonpoly


def support_complexity(support: list[str], error_count: int = 0) -> int:
    return sum(term_complexity(name) for name in support) + int(error_count)


def stability_summary() -> list[dict]:
    groups = defaultdict(list)
    for path in SELECTION_FILES:
        data = json.loads(path.read_text(encoding="utf-8"))
        for cname, case in data["cases"].items():
            for layer, layer_data in case["layers"].items():
                for noise_group in layer_data["rows"]:
                    for raw in noise_group["raw"]:
                        for method, result in raw["selection"].items():
                            support = tuple(result["support"])
                            groups[(cname, layer, raw["sigma"], method)].append({
                                "support": support,
                                "exact": bool(result["exact"]),
                                "extra": result["extra"],
                                "missed": result["missed"],
                                "complexity": support_complexity(list(support)),
                            })
    out = []
    for (case, layer, sigma, method), rows in sorted(groups.items()):
        counts = Counter(row["support"] for row in rows)
        modal_support, modal_count = counts.most_common(1)[0]
        out.append({
            "case": case, "layer": layer, "sigma": sigma, "method": method,
            "n": len(rows), "exact_rate": sum(row["exact"] for row in rows) / len(rows),
            "modal_stability": modal_count / len(rows),
            "modal_support": list(modal_support), "unique_supports": len(counts),
            "mean_extra": sum(row["extra"] for row in rows) / len(rows),
            "mean_missed": sum(row["missed"] for row in rows) / len(rows),
            "mean_complexity": sum(row["complexity"] for row in rows) / len(rows),
        })
    return out


def pareto_rows() -> list[dict]:
    data = json.loads((ROOT / "outlier_l1_compare_result.json").read_text(encoding="utf-8"))
    groups = defaultdict(list)
    for row in data["rows"]:
        key = (row["case"], row["layer"], row["fraction"], row["amplitude_rms"])
        for method in ("joint_L1", "joint_IRL1"):
            result = row[method]
            names = layer_names(row["layer"])
            if row["case"].startswith("u_tt"):
                names.append("u_t")
            candidate = {
                "method": method,
                "seed": row["seed"],
                "residual": result["refit_relative_residual"],
                "complexity": support_complexity(
                    [names[i] for i in result["beta_support"] if i < len(names)],
                    len(result["error_support"]),
                ),
                "beta_f1": result["beta_metrics"]["f1"],
                "error_f1": result["error_metrics"]["f1"],
            }
            # The support names are recovered from the stored library ordering below.
            candidate["beta_support_indices"] = result["beta_support"]
            groups[key].append(candidate)
    out = []
    for key, candidates in sorted(groups.items()):
        front = []
        for candidate in candidates:
            dominated = any(
                other["residual"] <= candidate["residual"]
                and other["complexity"] <= candidate["complexity"]
                and (other["residual"] < candidate["residual"] or other["complexity"] < candidate["complexity"])
                for other in candidates
            )
            if not dominated:
                front.append(candidate)
        out.append({
            "case": key[0], "layer": key[1], "fraction": key[2], "amplitude_rms": key[3],
            "pareto": front,
        })
    return out


def main() -> None:
    out = {
        "support_complexity": "sum_j [1 + derivative_order + power + 2*coordinate_degree + nonpolynomial_penalty] + error_count",
        "stability": stability_summary(),
        "outlier_pareto": pareto_rows(),
        "note": "The current Pareto summary is a diagnostic layer; support generation still comes from L1/IRL1, never from refit coefficients.",
    }
    path = ROOT / "stability_pareto_summary_result.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", path, "stability rows", len(out["stability"]), "pareto groups", len(out["outlier_pareto"]))


if __name__ == "__main__":
    main()
