# PPT 用的中文版配图。论文图不改：TikZ 图由英文源文件替换文字生成，matplotlib 图复用 paper/make_figures.py 的数据函数。
# 运行：在仓库根目录 D:\Miniconda3\envs\py312\python.exe slides\make_figures_zh.py
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "paper"))
import make_figures as M  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ZH = HERE / "zh"
ZH.mkdir(exist_ok=True)
IMG = HERE / "img"

OVERVIEW = [
    (r"(a) Forward: PDE known, solution unknown", r"(a) 正问题：已知 PDE，求解析解"),
    (r"initial and boundary data", r"初值和边界数据"),
    (r"\textcolor{inkB}{no interior samples}", r"\textcolor{inkB}{不用解的内部采样}"),
    (r"$(k,c,x_0)$ searched", r"$(k,c,x_0)$ 连续搜索"),
    (r"{$\ell_1$ penalty\\ + residual\\ gate}", r"{$\ell_1$ 惩罚\\ + 残差门槛}"),
    (r"(b) Inverse: solution samples known, PDE unknown", r"(b) 反问题：已知解的采样，求方程"),
    (r"{samples of $u(x,t)$\\ and its derivatives\\[1pt]\textcolor{inkB}{(KdV two-soliton)}}",
     r"{$u(x,t)$ 及其导数的采样\\[1pt]\textcolor{inkB}{（KdV 双孤子）}}"),
    (r"{reweighted\\ $\ell_1$}", r"{重加权\\ $\ell_1$}"),
    (r"\textcolor{inkB}{plain $\ell_1$ also keeps $u_x$}", r"\textcolor{inkB}{普通 $\ell_1$ 还会多留 $u_x$}"),
    (r"{selected}", r"{保留}"),
    (r"{set to zero by $\ell_1$}", r"{被 $\ell_1$ 压成 0}"),
]
FLOW = [
    (r"{PDE, initial and\\ boundary data,\\ generic library}", r"{PDE、初边值\\ 通用候选库}"),
    (r"{40 random starts\\ $k\in[0.1,10]$\\ $c\in[-8,8]$, $x_0\in\Omega$\\ LM on $\|r\|^2$}",
     r"{40 个随机起点\\ $k\in[0.1,10]$\\ $c\in[-8,8]$, $x_0\in\Omega$\\ LM 最小化 $\|r\|^2$}"),
    (r"{continue best start\\ (up to 3000\\ LM iterations)}", r"{最好的起点\\ 继续迭代\\（至多 3000 步）}"),
    (r"{relative\\ residual\\ $\le 10^{-8}$?}", r"{相对残差\\ $\le 10^{-8}$？}"),
    (r"{report: library\\ contains no exact\\ solution}", r"{报告：\\ 库中没有\\ 精确解}"),
    (r"{Stage A: locate $(k, c, x_0)$}", r"{第一步：找波参数 $(k, c, x_0)$}"),
    (r"{round 0: plain $\ell_1$\\ at $\lambda = 10^{-3}\lambda_{\max}$}", r"{第 0 轮：普通 $\ell_1$\\ $\lambda = 10^{-3}\lambda_{\max}$}"),
    (r"{reweight\\ $w_j = 1/(|\beta_j|+\varepsilon)$\\ and re-solve $\ell_1$}", r"{改权重\\ $w_j = 1/(|\beta_j|+\varepsilon)$\\ 再解 $\ell_1$}"),
    (r"{40-point $\lambda$ path:\\ keep columns with\\ $|\beta_j|>10^{-5}$ in $>20$}", r"{40 个 $\lambda$ 上\\ $|\beta_j|>10^{-5}$\\ 超过 20 次的项保留}"),
    (r"{Stage C: refit\\ $\xi_S$ and $(k,c,x_0)$\\ on support $S$}", r"{第三步：在支撑 $S$ 上\\ 重拟合 $\xi_S$\\ 和 $(k,c,x_0)$}"),
    (r"{closed-form\\ solution $S$, $\xi_S$,\\ $(k,c,x_0)$}", r"{解析解\\ $S$、$\xi_S$、$(k,c,x_0)$}"),
    (r"{inverse problem:\\ $\Theta(u)$ and $u_t$\\ (no Stage A)}", r"{反问题：\\ $\Theta(u)$ 和 $u_t$\\（跳过第一步）}"),
    (r"{Stage B: select support with $\ell_1$ only}", r"{第二步：只用 $\ell_1$ 选项}"),
    (r"{yes, fix $(k,c,x_0)$}", r"{是，固定 $(k,c,x_0)$}"),
    (r"{no}", r"{否}"),
    (r"{until support repeats}", r"{直到支撑不再变化}"),
]

PREAMBLE = r"""\documentclass[border=4pt]{standalone}
\usepackage[fontset=windows]{ctex}
\xeCJKsetup{CJKglue=\hskip 0pt}
\usepackage{amsmath,amssymb}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,shapes.geometric}
\definecolor{ink}{HTML}{0B0B0B}\definecolor{inkB}{HTML}{52514E}\definecolor{inkM}{HTML}{8A8984}
\definecolor{keep}{HTML}{2A78D6}\definecolor{paleK}{HTML}{CDE2FB}\definecolor{paleG}{HTML}{F0EFEC}
\definecolor{warn}{HTML}{EB6834}\definecolor{paleW}{HTML}{FBE3D8}
\newcommand{\sech}{\operatorname{sech}}
\begin{document}
"""


def tikz_zh(name, pairs):
    src = (ROOT / "paper" / "figures" / f"{name}.tex").read_text(encoding="utf-8")
    for en, zh in pairs:
        if en not in src:
            raise RuntimeError(f"{name}: 找不到 {en}")
        src = src.replace(en, zh)
    # 固定 text width 会把中文行拉开；改成最小宽度，每行按自然宽度排
    src = src.replace("text width=", "minimum width=")
    tex = ZH / f"{name}_zh.tex"
    tex.write_text(PREAMBLE + src + "\n\\end{document}\n", encoding="utf-8")
    subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", tex.name], cwd=ZH, check=True,
                   stdout=subprocess.DEVNULL)
    subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile", f"{name}_zh.pdf", str(IMG / f"{name}_zh")], cwd=ZH, check=True)


def set_zh_font():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei"],
                         "axes.unicode_minus": False, "mathtext.fontset": "dejavusans", "font.size": 9})


def inverse_zh():
    cases = [("KdV 双孤子", M.I.data_kdv_two_soliton()), ("Burgers（Cole–Hopf）", M.I.data_burgers_cole_hopf())]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.4))
    for j, (title, (a_raw, b, names, *_)) in enumerate(cases):
        ux = [i for i, n in enumerate(names) if n.split()[0] == "u_x"][0]
        lam, _, cx = M.inverse_paths(a_raw, b, ux)["plain"]
        nz = cx > 0
        ax = axes[j]
        ax.loglog(lam[nz], cx[nz], color=M.SERIES[0], marker=M.MARKERS[0], markersize=3)
        ax.set_title(title, fontsize=10, color=M.INK)
        ax.set_xlabel("$\\lambda/\\lambda_{\\max}$（从左到右减小）")
        ax.invert_xaxis()
        M.style(ax)
    axes[0].set_ylabel("普通 $\\ell_1$ 的 $u_x$ 系数\n（归一化）")
    fig.savefig(IMG / "inverse_path_zh.png", dpi=300)
    plt.close(fig)


def gate_zh():
    r = json.loads((ROOT / "forward_generic_v2_result.json").read_text(encoding="utf-8"))
    order = [("kdv", "KdV"), ("burgers", "Burgers"), ("chafee_infante", "Chafee–Infante"),
             ("kdv_no_tanh2", "反例：KdV 删掉 $\\tanh^2$"), ("kdv_two_soliton", "反例：KdV 双孤子")]
    fig, ax = plt.subplots(figsize=(5.5, 2.7))
    for k, (key, lab) in enumerate(order):
        y = np.array(r["cases"][key]["stage_a"]["relative_residuals_sorted"])
        ax.semilogy(np.arange(1, y.size + 1), y, color=M.SERIES[k], marker=M.MARKERS[k], markersize=3,
                    linewidth=1.2, label=lab)
    ax.axhline(r["gate"], color=M.INK2, linewidth=1.0, linestyle="--")
    ax.text(40.5, r["gate"] * 2.5, "门槛 $10^{-8}$", color=M.INK2, ha="right", va="bottom", fontsize=8.5)
    ax.set_xlabel("随机起点（按残差排序）")
    ax.set_ylabel("第一步相对残差")
    ax.set_xlim(0, 41)
    M.style(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.5)
    fig.savefig(IMG / "gate_zh.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    tikz_zh("overview", OVERVIEW)
    tikz_zh("flowchart", FLOW)
    set_zh_font()
    inverse_zh()
    gate_zh()
    for f in ZH.glob("*"):
        if f.suffix in (".aux", ".log", ".pdf"):
            f.unlink()
    print("ok")
