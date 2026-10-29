# 论文配图。只读结果 JSON 和 inverse_irl1.py 的数据函数，不改任何实验脚本。
# 图 1：反问题，普通 L1 与重加权 L1 沿 λ 路径的 u_x 系数和相对残差（精确同伦路径重算）。
# 图 2：通用库正问题，第一步 A 的 40 个起点相对残差（排序），与门槛 1e-8 对比。
# 运行：在仓库根目录 D:\Miniconda3\envs\py312\python.exe paper\make_figures.py

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import inverse_irl1 as I  # noqa: E402

OUT = Path(__file__).with_name("figures")
OUT.mkdir(exist_ok=True)

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "^", "D", "v"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
    "savefig.bbox": "tight", "savefig.dpi": 300,
})


def style(ax):
    ax.grid(True, which="major", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def inverse_paths(a_raw, b, ux):
    col = np.maximum(np.linalg.norm(a_raw, axis=0), 1e-15)
    a = a_raw / col
    out = {}
    for name, w in (("plain", np.ones(a.shape[1])), ("irl1", I.irl1(a, b)[0])):
        _, _, rows, _ = I.frequency_on_path(a, b, w)
        out[name] = (np.array([r[0] for r in rows]), np.array([r[1] for r in rows]),
                     np.array([abs(r[2][ux]) for r in rows]))
    return out


def figure_inverse():
    cases = [("KdV two-soliton", I.data_kdv_two_soliton()), ("Burgers, Cole–Hopf", I.data_burgers_cole_hopf())]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.2))
    for j, (title, (a_raw, b, names, *_)) in enumerate(cases):
        ux = [i for i, n in enumerate(names) if n.split()[0] == "u_x"][0]
        paths = inverse_paths(a_raw, b, ux)
        lam, _, cx = paths["plain"]
        assert np.all(paths["irl1"][2] == 0.0)  # 重加权后 u_x 在 40 个 λ 上都是精确 0
        ax = axes[j]
        nz = cx > 0
        ax.loglog(lam[nz], cx[nz], color=SERIES[0], marker=MARKERS[0], markersize=3)
        ax.set_title(title, fontsize=8.5, color=INK)
        ax.set_xlabel("$\\lambda/\\lambda_{\\max}$")
        ax.invert_xaxis()
        style(ax)
    axes[0].set_ylabel("plain $\\ell_1$: $|u_x|$ coef.\n(normalized)")
    fig.savefig(OUT / "inverse_path.pdf")
    plt.close(fig)


def figure_gate():
    r = json.loads((ROOT / "forward_generic_v2_result.json").read_text(encoding="utf-8"))
    order = [("kdv", "KdV"), ("burgers", "Burgers"), ("chafee_infante", "Chafee–Infante"),
             ("kdv_no_tanh2", "KdV without $\\tanh^2$"), ("kdv_two_soliton", "KdV two-soliton")]
    fig, ax = plt.subplots(figsize=(5.5, 2.6))
    for k, (key, lab) in enumerate(order):
        y = np.array(r["cases"][key]["stage_a"]["relative_residuals_sorted"])
        ax.semilogy(np.arange(1, y.size + 1), y, color=SERIES[k], marker=MARKERS[k], markersize=3,
                    linewidth=1.2, label=lab)
    ax.axhline(r["gate"], color=INK2, linewidth=1.0, linestyle="--")
    ax.text(40.5, r["gate"] * 2.5, "gate $10^{-8}$", color=INK2, ha="right", va="bottom", fontsize=7.5)
    ax.set_xlabel("random start (sorted by residual)")
    ax.set_ylabel("stage-A relative residual")
    ax.set_xlim(0, 41)
    style(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.savefig(OUT / "gate.pdf")
    plt.close(fig)


if __name__ == "__main__":
    figure_inverse()
    figure_gate()
    print("written:", sorted(p.name for p in OUT.iterdir()))
