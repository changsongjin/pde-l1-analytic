# 噪声实验的误差分析图。只读结果 JSON，不跑实验。输出到 figures_noise/。
# 图 1：反问题各方法选对率随噪声的变化；图 2：系数相对误差随噪声的变化（对数坐标，附斜率 1 参考线）。
# 正问题和 SGA-PDE 的结果文件存在时，另画图 3、图 4。
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures_noise"
OUT.mkdir(exist_ok=True)
SERIES = {"L1": "#2a78d6", "IRL1": "#eb6834", "STRidge": "#1baf7a", "GA": "#eda100", "Exhaustive": "#52514e"}
MARK = {"L1": "o", "IRL1": "s", "STRidge": "^", "GA": "D", "Exhaustive": "x"}
LABEL = {"L1": "plain $\\ell_1$", "IRL1": "reweighted $\\ell_1$", "STRidge": "STRidge", "GA": "GA (ours)", "Exhaustive": "exhaustive"}
TITLE = {"heat": "Heat", "wave": "Wave", "advection_diffusion": "Advection–diffusion", "burgers": "Burgers", "kdv": "KdV"}
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 300})


def style(ax):
    ax.grid(True, color="#e4e3df", linewidth=0.5)
    ax.set_axisbelow(True)


def inverse_figures():
    r = json.loads((ROOT / "noise_inverse_result.json").read_text(encoding="utf-8"))
    cases = list(r["cases"])
    fig, axes = plt.subplots(1, len(cases), figsize=(11, 2.3), sharey=True)
    for ax, c in zip(axes, cases):
        rows = r["cases"][c]["rows"]
        s = np.array([row["sigma"] for row in rows]) * 100
        for m in ["L1", "IRL1", "STRidge", "GA"]:
            ax.plot(s, [row[m]["exact_rate"] for row in rows], color=SERIES[m], marker=MARK[m], markersize=3, label=LABEL[m])
        ax.set_title(TITLE[c], fontsize=9)
        ax.set_xlabel("noise (% of std u)")
        style(ax)
    axes[0].set_ylabel("exact-support rate")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.savefig(OUT / "inverse_success.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(cases), figsize=(11, 2.3), sharey=True)
    for ax, c in zip(axes, cases):
        rows = [row for row in r["cases"][c]["rows"] if row["sigma"] > 0]
        s = np.array([row["sigma"] for row in rows])
        y = np.array([row["IRL1"]["coef_rel_err_median"] for row in rows])
        ax.loglog(s * 100, y, color=SERIES["IRL1"], marker=MARK["IRL1"], markersize=3, label="reweighted $\\ell_1$ (median)")
        ref = y[2] * s / s[2]
        ax.loglog(s * 100, ref, color="#8a8984", linestyle="--", linewidth=0.8, label="slope 1")
        ax.set_title(TITLE[c], fontsize=9)
        ax.set_xlabel("noise (%)")
        style(ax)
    axes[0].set_ylabel("coefficient relative error")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.savefig(OUT / "inverse_coef_error.png")
    plt.close(fig)


def forward_figure():
    f = ROOT / "noise_forward_result.json"
    if not f.exists():
        return
    # 第一步残差 ÷ 噪声本身的残差（门槛/3）。正例应在 1 附近，反例应明显大于门槛倍数。
    r = json.loads(f.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(5, 2.8))
    cases = [("kdv", "KdV", "#2a78d6", "o"), ("burgers", "Burgers", "#1baf7a", "^"),
             ("chafee_infante", "Chafee–Infante", "#eda100", "D"),
             ("kdv_no_tanh2", "KdV, tanh² removed (neg.)", "#eb6834", "s"),
             ("kdv_two_soliton", "two-soliton KdV (neg.)", "#52514e", "x")]
    for name, lab, col, mk in cases:
        runs = [x for x in r["runs"] if x["case"] == name]
        xs = np.array([x["sigma"] * 100 for x in runs])
        ys = np.array([x["stage_a_relative_residual"] / (x["gate"] / 3.0) for x in runs])
        jit = np.exp(0.06 * (np.array([x["seed"] for x in runs]) - 2))
        ax.scatter(xs * jit, ys, s=10, color=col, marker=mk, label=lab, linewidths=0.8)
    for g, ls, txt in [(3.0, "--", "gate = 3× noise (used)"), (1.5, ":", "gate = 1.5× noise")]:
        ax.axhline(g, color="#8a8984", linestyle=ls, linewidth=0.8)
        ax.text(0.085, g * 1.08, txt, fontsize=7, color="#5f5e5a")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("noise on initial/boundary data (% of std)")
    ax.set_ylabel("residual / noise level")
    style(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.savefig(OUT / "forward_gate.png")
    plt.close(fig)


def sga_figure():
    f = ROOT / "sga_compare_result.json"
    if not f.exists():
        return
    r = json.loads(f.read_text(encoding="utf-8"))
    ds_names = {"sga_burgers": "Burg.\n(SGA)", "sga_kdv": "KdV\n(SGA)", "sga_chafee_infante": "C–I\n(SGA)",
                "ours_burgers": "Burg.\n(ours)", "ours_kdv": "KdV\n(ours)"}
    methods = [("SGA-PDE", "#8a8984"), ("IRL1", SERIES["IRL1"]), ("GA", SERIES["GA"]), ("Exhaustive", SERIES["Exhaustive"])]
    fig, axes = plt.subplots(1, len(r["noise"]), figsize=(11, 2.4), sharey=True)
    for ax, s in zip(axes, r["noise"]):
        for j, (m, col) in enumerate(methods):
            ys = []
            for ds in ds_names:
                vs = [v for k, v in r["runs"].items() if k.startswith(f"{ds}|{s}|")]
                if m == "SGA-PDE":
                    ys.append(np.mean([bool(v["sga"] and v["sga"]["exact"]) for v in vs]))
                else:
                    ys.append(np.mean([v["ours"][m]["exact"] for v in vs]))
            ax.bar(np.arange(len(ds_names)) + (j - 1.5) * 0.2, ys, width=0.2, color=col,
                   label=m if m == "SGA-PDE" else LABEL[m])
        ax.set_xticks(np.arange(len(ds_names)))
        ax.set_xticklabels(ds_names.values(), fontsize=6.5)
        ax.set_title(f"noise {s * 100:g}%", fontsize=9)
        style(ax)
    axes[0].set_ylabel("exact-support rate")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.savefig(OUT / "sga_compare.png")
    plt.close(fig)


if __name__ == "__main__":
    inverse_figures()
    forward_figure()
    sga_figure()
    print(sorted(p.name for p in OUT.iterdir()))
