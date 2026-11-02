# 噪声与遗传算法补充实验的中文配图（PPT 用）。只读结果 JSON，输出到 slides/img/。
# 英文论文图由根目录 noise_figures.py 生成，这里是同样数据的中文版，SGA 图另把补跑的两个方程合进来。
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMG = Path(__file__).resolve().parent / "img"
COL = {"SGA": "#8a8984", "L1": "#2a78d6", "IRL1": "#eb6834", "STRidge": "#1baf7a", "GA": "#eda100", "Exhaustive": "#52514e"}
LAB = {"SGA": "SGA-PDE", "L1": "普通 L1", "IRL1": "加权 L1", "STRidge": "STRidge", "GA": "自写遗传算法", "Exhaustive": "穷举"}
MARK = {"L1": "o", "IRL1": "s", "STRidge": "^", "GA": "D"}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei"], "axes.unicode_minus": False,
                     "mathtext.fontset": "dejavusans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 300})


def style(ax):
    ax.grid(True, color="#e4e3df", linewidth=0.5)
    ax.set_axisbelow(True)


def load(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def inverse():
    r = load("noise_inverse_result.json")
    title = {"heat": "热方程", "wave": "波动方程", "advection_diffusion": "平流扩散", "burgers": "Burgers", "kdv": "KdV"}
    fig, axes = plt.subplots(1, 5, figsize=(12, 2.9), sharey=True)
    for ax, c in zip(axes, r["cases"]):
        rows = r["cases"][c]["rows"]
        s = np.array([row["sigma"] for row in rows]) * 100
        for m in ["L1", "IRL1", "STRidge", "GA"]:
            ax.plot(s, [row[m]["exact_rate"] for row in rows], color=COL[m], marker=MARK[m], markersize=3.5,
                    linewidth=1.6 if m == "IRL1" else 1.0, label=LAB[m])
        ax.set_title(title[c], fontsize=11)
        ax.set_xlabel("噪声（%）")
        style(ax)
    axes[0].set_ylabel("选对率（20 个种子）")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.savefig(IMG / "noise_inverse_zh.png")
    plt.close(fig)


def forward():
    r = load("noise_forward_result.json")
    cases = [("kdv", "KdV", "#2a78d6", "o"), ("burgers", "Burgers", "#1baf7a", "^"),
             ("chafee_infante", "Chafee–Infante", "#eda100", "D"),
             ("kdv_no_tanh2", "反例：删掉 tanh²", "#eb6834", "s"), ("kdv_two_soliton", "反例：双孤子", "#52514e", "x")]
    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    for name, lab, col, mk in cases:
        runs = [x for x in r["runs"] if x["case"] == name]
        xs = np.array([x["sigma"] * 100 for x in runs]) * np.exp(0.06 * (np.array([x["seed"] for x in runs]) - 2))
        ys = np.array([x["stage_a_relative_residual"] / (x["gate"] / 3.0) for x in runs])
        ax.scatter(xs, ys, s=14, color=col, marker=mk, label=lab, linewidths=0.9)
    for g, ls, txt in [(3.0, "--", "门槛 = 3 × 噪声（现用）"), (1.5, ":", "门槛 = 1.5 × 噪声")]:
        ax.axhline(g, color="#8a8984", linestyle=ls, linewidth=0.9)
        ax.text(0.085, g * 1.08, txt, fontsize=9, color="#5f5e5a")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("初边值数据的噪声（%）")
    ax.set_ylabel("残差 ÷ 噪声水平")
    style(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=9)
    fig.savefig(IMG / "noise_forward_zh.png")
    plt.close(fig)


def sga():
    a, b = load("sga_compare_result.json"), load("sga_compare_extra_result.json")
    groups = {"Burgers": ("a", ["sga_burgers", "ours_burgers"]), "KdV": ("a", ["sga_kdv", "ours_kdv"]),
              "C–I": ("a", ["sga_chafee_infante"]), "compound": ("b", ["sga_compound"]),
              "divide*": ("b", ["sga_divide"])}
    noise = a["noise"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.0), sharey=True)
    for ax, s in zip(axes, noise):
        for j, m in enumerate(["SGA", "IRL1", "GA", "Exhaustive"]):
            ys = []
            for src, dss in groups.values():
                runs = [v for k, v in (a if src == "a" else b)["runs"].items() if k.split("|")[0] in dss and float(k.split("|")[1]) == s]
                if m == "SGA":
                    ys.append(np.mean([bool(v["sga"] and v["sga"]["exact"]) for v in runs]))
                else:
                    ys.append(np.mean([(v["ours"]["extended"] if src == "b" else v["ours"])[m]["exact"] for v in runs]))
            ax.bar(np.arange(len(groups)) + (j - 1.5) * 0.2, ys, width=0.2, color=COL[m], label=LAB[m])
        ax.set_xticks(np.arange(len(groups)))
        ax.set_xticklabels(groups.keys(), fontsize=8.5)
        ax.set_title(f"噪声 {s * 100:g}%", fontsize=11)
        style(ax)
    axes[0].set_ylabel("选对率")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.savefig(IMG / "noise_sga_zh.png")
    plt.close(fig)


if __name__ == "__main__":
    inverse()
    forward()
    sga()
    from PIL import Image
    for f in ["noise_inverse_zh.png", "noise_forward_zh.png", "noise_sga_zh.png"]:
        print(f, Image.open(IMG / f).size)
