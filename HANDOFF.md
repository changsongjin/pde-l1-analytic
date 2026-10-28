# PDE L1 解析解 / 反问题：交接文档

最后更新：2026-10-28  
工作区：`D:\03_项目代码\PDE`  
Python：`D:\Miniconda3\envs\py312\python.exe`（NumPy 2.2.5）  
不要用 `D:\Miniconda3\python.exe`，那个环境没有 NumPy。

本文档给下一任接手人。论文和 PPT **还没写**，按用户条件：实验结果都正确后再写。

---

## 1. 课题一句话

已知 PDE，把很多候选函数列和系数都先设出来，用 **L1 范数** 把大部分系数压成 0，从而确定解析解由哪些项、哪些系数构成。

用户原话：「先把这些项和系数全部设出来，然后 L1 norm 会让大部分的系数为 0。」

方向后来扩成两边：

| 方向 | 已知 | 未知 | 通过标准 |
| --- | --- | --- | --- |
| 正问题 | PDE + 初边值 | 解析解的项和系数 | L1 选出的支撑 = 真项 |
| 反问题 | 解在网格上的值（精确解采样） | 方程的项和系数 | L1 选出的支撑 = 真方程的项 |

---

## 2. 硬约束（不要改）

1. **求解必须是 L1。** 自适应 L1、列归一化可以。序贯阈值（STLSQ）不能当默认求解器，也不能用来绕开 L1。
2. **通过只认支撑。** 丢掉 L1 之后的最小二乘只读系数，不决定谁是 0。
3. **不许靠放宽阈值删掉反问题里很小的 \(u_x\)。** 现用阈值 \(10^{-5}\)。
4. **正问题不提供解的内部采样。** 损失是配点 PDE 残差 + 初边值残差 + \(\lambda\|\xi\|_1\)。
5. **单模态反问题会有更短等价方程。** 热方程、波动、平流扩散必须用两个模态。单条 Burgers 行波精确满足 \(u_t+c u_x=0\)，比真方程更短，只作对照，不要求通过。
6. **已有脚本不改。** 重做写新脚本。
7. **不清楚先问用户。** 需要 skill 可以下载。遇到问题向用户汇报。
8. 用户曾把国防科技大学校外登录账号发在群里。**全组不得写入命令、网址或文件，也不得用来登录。不要复述密码。**

---

## 3. 用户后续要求（按时间）

- 先从顶级会议、期刊、arXiv 调研，再定技术路线。
- 从易到难，先从最简单的一档开始。后来定成正问题 5 个、反问题 5 个。
- 研究陈云天、张东晓是否用遗传算法解哪些 PDE，并用 L1 试他们的 PDE。
- 遗传算法可以加，**不替换 L1**。用户说「这个思路待会再补充」，之后没有补充，这轮不接。
- 调研开源 GitHub 辅助实验；作废的实验重做。
- 全部正确后再写报告，再做 PPT。报告按科研论文写。本地有 LaTeX。下载了 Nature 与 CCF-A 写作 skill。
- 交付是报告 + PPT，不做前端。

---

## 4. 当前总进度

| 类别 | 通过 | 未通过 / 带条件 |
| --- | --- | --- |
| 正问题 5 | 热方程、粘性 Burgers、波动方程、Chafee–Infante **从零选对** | KdV 行波支撑对，但用了真解附近起点（0.9 倍真系数），从零出发相对残差 0.36 |
| 反问题 5 | 热方程（双模态）、波动方程（双模态）、平流扩散（重做后） | **粘性 Burgers 两条波**、**KdV 双孤子**：真项系数能读对，支撑稳定多一列 \(u_x\) |

**最大问题：** 反问题 Burgers 和 KdV 双孤子，L1 稳定多留一列很小的 \(u_x\)。稳定性选择去不掉。残差门槛 \(10^{-8}\) 整条 \(\lambda\) 路径都过不去。用户要求都正确才写论文，所以 LaTeX 和 PPT 停着。

`batch5.py` 里第一次平流扩散和 KdV **作废 / 被取代**，不要引用其中的通过与否。平流扩散作废原因：宽度写成常数，真方程相对残差 0.3758，数据和方程对不上。重做用 \(\sigma^2(t)=\sigma_0^2+2\nu t\)。

---

## 5. 文件对照

全部在 `D:\03_项目代码\PDE`。Python 脚本和对应 JSON 成对出现。

### 正问题

| 脚本 | 结果 | 方程 / 真解 | 结论 |
| --- | --- | --- | --- |
| `tier1_heat.py` | `tier1_heat_result.json` | \(u_t=u_{xx}\)，初值 \(\sin(\pi x)\)，14 列 | 从零通过。49/50 个 \(\lambda\) 只剩真项。重拟合系数 1。相对残差 \(1.835\times10^{-16}\) |
| `tier2_burgers.py` | `tier2_burgers_result.json` | \(u_t+uu_x=0.1 u_{xx}\)，行波速度 0.5 | 从零通过。39/40。重拟合系数 1。真解残差 \(7.42\times10^{-17}\)。波速是预先离散成三列，不是搜出来的 |
| `tier3_kdv.py` | `tier3_kdv_result.json` | \(u_t+6uu_x+u_{xxx}=0\)，真解 \(2\mathrm{sech}^2(x-4t)\) | 支撑对，**不是盲搜**。从零停在相对残差 0.36、10 列全非零。第二起点 = 真项系数的 0.9 倍后 40/40 只剩波速 4 |
| `chafee_infante.py` | `chafee_infante_result.json` | \(u_t=u_{xx}+u-u^3\)，稳态 \(\tanh(x/\sqrt{2})\) | 从零通过。39/40。论文 PDF 未打开，系数用标准形式，不是原文抄的 |
| `batch5.py`（正问题部分） | `batch5_result.json` | 波动 \(u_{tt}=u_{xx}\)，真解 \(\sin(\pi x)\cos(\pi t)\) | 从零通过。39/40。重拟合 1 |

### 反问题

| 脚本 | 结果 | 方程 | 结论 |
| --- | --- | --- | --- |
| `inverse_burgers.py` | `inverse_burgers_result.json` | \(u_t=-uu_x+0.1 u_{xx}\)，两条行波 | **未通过。** \(uu_x\) 49/50，\(u_{xx}\) 41/50，\(u_x\) 46/50。L1 物理系数约 \(-6.5\times10^{-6}\)。重拟合后 \(u_x\) 约 \(10^{-16}\)，真项 \(-1\) 和 \(0.1\)。单条行波只留下 \(u_x=-0.5\) |
| `batch5.py`（反问题部分） | `batch5_result.json` | 热方程、波动：两个模态 | **热、波动通过**，只留 \(u_{xx}\)、系数 1。同文件里的平流扩散和 KdV **不要引用** |
| `redo_two.py` | `redo_two_result.json` | 平流扩散；KdV 双孤子 | **平流扩散通过。** \(\sigma_0=0.5,0.9\)，\(c=0.4\)，\(\nu=0.05\)，残差 \(1.35\times10^{-16}\)。\(u_x\) 39/40，\(u_{xx}\) 31/40，重拟合 \(-0.4\)、\(0.05\)。**KdV 双孤子未通过。** 解析导数残差 \(2.21\times10^{-13}\)。\(uu_x\) 39/40，\(u_{xxx}\) 26/40，\(u_x\) 39/40。重拟合真项 \(-6\)、\(-1\) |
| `optimize_ux.py` | `optimize_ux_result.json` | 同上两条未通过的 | 新规则：残差 \(\le 10^{-8}\) 的 \(\lambda\) 里取 L1 目标最小，再丢掉归一化 \(\lvert\xi\rvert<10^{-3}\) 的列。**两条都没有任何 \(\lambda\) 达到 \(10^{-8}\)，按规则停、不报支撑。** Burgers 最好 L1 残差 \(2.1\times10^{-6}\)，末端 \(u_x\) 归一化系数 \(7.2\times10^{-4}\)。KdV 最好 L1 残差 \(1.6\times10^{-5}\)，末端 \(u_x\) 归一化系数 \(7.2\times10^{-3}\)（丢掉规则也盖不住） |

### 写作 skill（本地已克隆，**不要推进 GitHub**）

- Nature：`skills/nature-skills`，写作用 `skills/nature-skills/skills/nature-writing/SKILL.md`。仓库 https://github.com/Yuan1z0825/nature-skills
- CCF-A：`skills/ccfa-skills`，写作用 `skills/ccfa-skills/ccf-paper-writer/SKILL.md`。仓库 https://github.com/mikubaka88/ccfa-skills
- 同目录还有 nature 的 citation、figure、polishing、paper2ppt，以及 ccfa 的 humanization、common、reviewer、submission-checker。

---

## 6. 求解器约定（现有脚本共性）

- 列归一化后再做加权 L1（坐标下降 + 软阈值）。
- 自适应权重来自岭回归：\(w_j = 1/(\lvert\beta_j^{\mathrm{ridge}}\rvert+\varepsilon)^\gamma\)。
- \(\lambda\) 从 \(\lambda_{\max}\) 几何降到 \(\lambda_{\max}\times 10^{-6}\)，Burgers 反问题 50 个，其他常见 40 个。
- 支撑阈值 \(10^{-5}\)（作用在**归一化系数**上，不是物理系数）。这就是 Burgers 上物理 \(u_x\) 只有 \(6.5\times10^{-6}\)、归一化后仍可能过阈值的原因。
- 稳定性选择：整条 \(\lambda\) 路径上 \(\lvert\beta\rvert>10^{-5}\) 的次数过半才进支撑。分母是全部 \(\lambda\)，不是残差达标的那些。
- 重拟合：在选出的支撑上对未归一化设计阵做最小二乘。

Burgers 反问题里岭回归给 \(u_x\) 的物理系数约 \(-0.29\)，自适应权重因此偏小，压不掉这一列。普通 L1（权重全 1）同样多出 \(u_x\)。

---

## 7. 文献（调研过，方向多数是「数据 → PDE」）

文献主流是从数据反推 PDE，不能当「已知 PDE、反推解」的直接证据。同向但无 L1 的 SPINN、ASPINN、FEX、PINN、Deep Ritz **不进第一刀**。

- SINDy，PNAS 2016，arXiv:1509.03580
- PDE-FIND，Sci. Adv. 2017，doi:10.1126/sciadv.1602614，arXiv:1609.06401
- PDE-Net，Long、Lu、Ma、Dong，ICML 2018，PMLR 80:3208–3216，arXiv:1710.09668
- DeepMoD，arXiv:1904.09406v3
- Sparsistent，arXiv:2106.11936
- Koch，arXiv:2101.02122
- SPINN，arXiv:2102.13037
- ASPINN，arXiv:2207.00377
- FEX，arXiv:2206.10121
- Deep Ritz，arXiv:1710.00211
- Schaeffer，Proc. R. Soc. A 473:20160446，2017（L1 未从原文核对）
- DLGA-PDE，Xu、Chang、Zhang，arXiv:2001.07305（陈云天 / 张东晓线：遗传算法从数据反推 PDE）
- SGA-PDE，Chen、Luo、Liu、Xu、Zhang，arXiv:2106.11927

陈、张测过 Burgers、KdV、波动、Chafee–Infante。我们已用 L1 做过对应正问题；Chafee–Infante 按标准形式 \(u_t=u_{xx}+u-u^3\)。

开源对照：PySINDy 公开 KdV 例子里，L1 / SR3 也会多留 `x0_1`（即 \(u_x\)）。  
https://pysindy.readthedocs.io/en/latest/examples/10_PDEFIND_examples/example.html  
这支持「多一列不是本实验独有」，但不算我们自己的通过。

---

## 8. 已尝试、失败的修法（不要重复）

1. 稳定性选择（频率过半）：\(u_x\) 入选 46/50（Burgers）、39/40（KdV），去不掉。
2. 残差 \(\le 10^{-8}\) 再取最短 / 最小 L1 目标：整条路径最好残差是 \(2\times10^{-6}\)（Burgers）和 \(1.6\times10^{-5}\)（KdV），门槛过不去，按规则不报支撑。
3. 丢掉归一化 \(\lvert\xi\rvert<10^{-3}\)：Burgers 末端 \(u_x\) 是 \(7.2\times10^{-4}\) 可以盖住，但门槛没过不能算通过；KdV 末端是 \(7.2\times10^{-3}\)，规则盖不住。
4. 单条行波 / 单孤子：存在更短等价方程 \(u_t+c u_x=0\)，L1 只会留 \(u_x\)。这是预期失败，不是 bug。

允许继续试、但还没做的方向（仍必须是 L1，不许放宽 \(10^{-5}\) 当删列手段）：

- 迭代重加权 L1（Candes IRL1）：权重来自上一轮 **L1 系数**，不要用岭回归系数（岭回归把能量分给了相关的 \(u_x\)）。
- 权重按**物理系数**而不是归一化系数来更新。
- 不把整条路径都拿来计频率（极小 \(\lambda\) 接近最小二乘，相关列全非零）。

---

## 9. 怎么跑

```bat
D:\Miniconda3\envs\py312\python.exe tier1_heat.py
D:\Miniconda3\envs\py312\python.exe tier2_burgers.py
D:\Miniconda3\envs\py312\python.exe tier3_kdv.py
D:\Miniconda3\envs\py312\python.exe chafee_infante.py
D:\Miniconda3\envs\py312\python.exe inverse_burgers.py
D:\Miniconda3\envs\py312\python.exe batch5.py
D:\Miniconda3\envs\py312\python.exe redo_two.py
D:\Miniconda3\envs\py312\python.exe optimize_ux.py
```

每条脚本把 JSON 写到同目录。重跑会覆盖对应 JSON。已通过的不要为了「再看一眼」去改脚本。

---

## 10. 群与角色

群名：`PDE解析解讨论`  
成员：用户、研究员、审核员、辅助决策员、实验员、产品经理、信息员、总经理助理、Chief（admin）。

实验由 **实验员** 写脚本并跑。Chief 对 JSON、定通过与否、决定下一刀。论文 skill 已在本地，等两条反问题通过后再写 LaTeX。

---

## 11. 下一步（等用户拍板）

两条路，不要自己替用户选：

A. 继续改 Burgers / KdV 反问题（新脚本，L1，不放宽阈值）。  
B. 带着这两条失败写论文和 PPT，把「L1 在共线库上稳定多留 \(u_x\)」写成结果，并引用 PySINDy 同类现象。

遗传算法怎么接到 L1 前面，等用户补充思路。
