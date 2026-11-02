# 第二版汇报 PPT（build_deck2.js）新增的中文图。只读结果 JSON，输出到 slides/img/。
#   cost_zh.png    各方法单次求解耗时（对数轴）
#   coef_err_zh.png 加权 L1 系数相对误差随噪声（对数-对数，附斜率 1 参考线）
#   robust_zh.png   各方法“选对率 ≥ 0.95 时能承受的最大噪声”热图
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMG = Path(__file__).resolve().parent / "img"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei"], "axes.unicode_minus": False,
                     "mathtext.fontset": "dejavusans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 300})
INV = json.loads((ROOT / "noise_inverse_result.json").read_text(encoding="utf-8"))
SGA = [json.loads((ROOT / f).read_text(encoding="utf-8")) for f in ("sga_compare_result.json", "sga_compare_extra_result.json")]
EQ = {"heat": "热方程", "wave": "波动方程", "advection_diffusion": "平流扩散", "burgers": "Burgers", "kdv": "KdV"}


def cost():
    rows = [(m, float(np.median([row[m]["mean_time_s"] for c in INV["cases"].values() for row in c["rows"]])))
            for m in ["L1", "IRL1", "STRidge", "GA", "Exhaustive"]]
    sga_t = float(np.median([v["sga_raw"]["time_s"] for d in SGA for v in d["runs"].values() if v["sga_raw"]]))
    names = {"L1": "普通 L1", "IRL1": "加权 L1（本文）", "STRidge": "STRidge", "GA": "自写遗传算法", "Exhaustive": "穷举（≤6 项）"}
    labels = [names[m] for m, _ in rows] + ["SGA-PDE"]
    vals = [t for _, t in rows] + [sga_t]
    cols = ["#b8b7b2", "#eb6834", "#b8b7b2", "#eda100", "#52514e", "#8a8984"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals, color=cols, height=0.6)
    for yi, v in zip(y, vals):
        txt = f"{v * 1000:.1f} ms" if v < 1 else f"约 {v / 60:.0f} 分钟（30 分钟限时）"
        ax.text(v * 1.4, yi, txt, va="center", fontsize=10)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 3e5)
    ax.set_xlabel("单次求解耗时（秒，对数轴；中位数）")
    ax.grid(True, axis="x", color="#e4e3df", linewidth=0.5)
    ax.set_axisbelow(True)
    fig.savefig(IMG / "cost_zh.png")
    plt.close(fig)
    return labels, vals


def coef_err():
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    cols = ["#2a78d6", "#1baf7a", "#eda100", "#eb6834", "#52514e"]
    for (c, name), col in zip(EQ.items(), cols):
        rows = [r for r in INV["cases"][c]["rows"] if 0 < r["sigma"] <= 0.07 and r["IRL1"]["coef_rel_err_median"] is not None]
        s = np.array([r["sigma"] for r in rows]) * 100
        ax.loglog(s, [r["IRL1"]["coef_rel_err_median"] for r in rows], marker="o", markersize=3.5, color=col, label=name)
    s = np.array([0.1, 7.0])
    ax.loglog(s, 1e-3 * s / 0.5, color="#8a8984", linestyle="--", linewidth=0.9, label="斜率 1（误差 ∝ 噪声）")
    ax.set_xlabel("噪声（%）")
    ax.set_ylabel("系数相对误差（中位数）")
    ax.grid(True, color="#e4e3df", linewidth=0.5)
    ax.legend(fontsize=9, loc="upper left")
    fig.savefig(IMG / "coef_err_zh.png")
    plt.close(fig)


def robust():
    methods = [("L1", "普通 L1"), ("STRidge", "STRidge"), ("IRL1", "加权 L1"), ("GA", "自写遗传算法"), ("Exhaustive", "穷举")]
    grid = np.zeros((len(methods), len(EQ)))
    for j, c in enumerate(EQ):
        rows = sorted(INV["cases"][c]["rows"], key=lambda r: r["sigma"])
        for i, (m, _) in enumerate(methods):
            best = -1.0
            for r in rows:  # 从 0 起连续保持选对率 ≥ 0.95 的最大噪声
                if r[m]["exact_rate"] >= 0.95:
                    best = r["sigma"] * 100
                else:
                    break
            grid[i, j] = best
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    shown = np.where(grid < 0, np.nan, grid)
    ax.imshow(shown, cmap="Blues", vmin=-2, vmax=8, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            ax.text(j, i, "无噪声也错" if v < 0 else ("仅无噪声" if v == 0 else f"{v:g}%"), ha="center", va="center", fontsize=10,
                    color="#c2410c" if v < 0 else ("white" if v >= 5 else "#141b2d"))
    ax.set_xticks(range(len(EQ)))
    ax.set_xticklabels(EQ.values())
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([n for _, n in methods])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    fig.savefig(IMG / "robust_zh.png")
    plt.close(fig)
    return grid


if __name__ == "__main__":
    print(cost())
    coef_err()
    print(robust())
    from PIL import Image
    for f in ["cost_zh.png", "coef_err_zh.png", "robust_zh.png"]:
        print(f, Image.open(IMG / f).size)
