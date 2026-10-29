# PDE L1 解析解 / 反问题：交接文档

最后更新：2026-10-28  
工作区：`D:\03_项目代码\PDE`  
Python：`D:\Miniconda3\envs\py312\python.exe`（NumPy 2.2.5）  
不要用 `D:\Miniconda3\python.exe`，那个环境没有 NumPy。

本文档给下一任接手人。论文和 PPT **还没写**，按用户条件：实验结果都正确后再写。

> **2026-10-28 更新：10 条全部通过（本机 py312 复核）。** 用户选了 A。新脚本 `inverse_irl1.py`（反问题 Burgers / KdV）、`kdv_forward_blind.py`（KdV 正问题盲搜），公共求解器 `lasso_exact.py`。详见第 4、5、8 节。
>
> **同日追加：通用库正问题** `forward_generic.py`：三个方程共用一个 tanh 幂次库，波参数连续搜索，全部选对；两个反例被门槛正确拦下。Burgers 多起点收敛慢（种子 3/5），已诊断为平移近似可被吸收、LM 步数不够，不是错盆地。**10-29 已加延长迭代整体重跑（`forward_generic_v2.py`）：三个正例种子全部 5/5，两个反例都被拦下。** 这些在分支 `exp/generic-library`。下一步见第 11 节。

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
| 正问题 5 | 热方程、粘性 Burgers、波动方程、Chafee–Infante **从零选对**；KdV 行波 **盲起点通过**（`kdv_forward_blind.py`） | — |
| 反问题 5 | 热方程（双模态）、波动方程（双模态）、平流扩散（重做后）；粘性 Burgers、KdV 双孤子 **IRL1 通过**（`inverse_irl1.py`） | — |

**原最大问题已解决：** 反问题 Burgers 和 KdV 双孤子多留 \(u_x\)，原因是 \(u_x\) 列**不满足不可表示条件**（KdV 2.77，Burgers 1.39 / 1.52，需 < 1），不是求解器没收敛。迭代重加权 L1 两轮后支撑 = 真项。

**仍有的限制（论文要写）：**
- IRL1 第 0 轮 \(\lambda\) 比例要足够小，真项才会被选上。敏感性：KdV 12/20，Burgers 两组各 16/20；失败都在比例 0.1（KdV 还有 0.01），与 \(\varepsilon\) 无关。现用 \(10^{-3}\) 是探索中选的，论文需要给选取规则。
- KdV 正问题波速仍是离散列，没有连续搜索。
- 旧 Burgers 两行波数据设计阵秩只有 8/10（库里还有两个更长的精确方程），所以加了满秩的新数据（Cole–Hopf 三激波合并）作主结果。

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

### 2026-10-28 新脚本（本机 py312 / NumPy 2.2.5 跑过，结果与云端 Python 3.13 / NumPy 2.5.3 一致）

| 脚本 | 结果 | 内容 | 结论 |
| --- | --- | --- | --- |
| `lasso_exact.py` | — | 公共求解器：LARS 同伦路径精确 L1，带 KKT 检查 | 被下面两条 import |
| `inverse_irl1.py` | `inverse_irl1_result.json` | 反问题 KdV 双孤子（旧数据）、Burgers Cole–Hopf 三激波合并（新数据，满秩 10/10，FD 校验 \(7.95\times10^{-8}\)）、Burgers 两行波（旧数据，秩 8/10）。先算不可表示条件，再普通 L1，再 IRL1 \(w=1/(\lvert\beta\rvert+\varepsilon)\)，权重来自上一轮 L1 系数，无岭回归，阈值仍 \(10^{-5}\) | **三组全通过。** 普通 L1 KKT 误差 \(10^{-9}\)–\(10^{-10}\)，仍留 \(u_x\)（39/40）。IRL1 两轮后 \(u_x\) 0/40。重拟合 KdV \(-6,-1\)，Burgers \(-1, 0.1\)，误差约 \(10^{-15}\)。敏感性 12/20、16/20、16/20 |
| `kdv_forward_blind.py` | `kdv_forward_blind_result.json` | KdV 正问题，30 个随机起点（只用列尺度，不用真解位置）+ IRL1 + \(\lambda\) 路径 | **通过。** 7/30 起点落到正确盆地；支撑只剩 \(2\mathrm{sech}^2(x-4t)\)（39/40），重拟合 1，残差 \(7.54\times10^{-16}\)。10 个种子 10/10；\(\lambda\) 比例 \(10^{-2},10^{-3},10^{-4}\) 都过 |

### 2026-10-28 通用库正问题（回答「库里本来就放了答案」）

`forward_generic.py` → `forward_generic_result.json`，分支 `exp/generic-library`，本机 py312 跑约 31 分钟。

- 三个方程**共用同一个库**：行波变量 \(\theta=k(x-x_0-ct)\) 上的 \(1,\tanh,\tanh^2,\tanh^3,\tanh^4\)（tanh 展开法），加 \(\mathrm{sech}\,\theta\)、\(e^{-\theta^2}\)、\(\theta\) 三列干扰。库里没有真解那一列，真解要由几列拼出来。
- \((k,c,x_0)\) 是**连续未知数**，搜索范围 \(k\in[0.1,10]\)、\(c\in[-8,8]\)、\(x_0\in\) 计算域，不预先离散。
- 步骤：A 多起点 LM（40 个，无正则）定 \(p^*\)，相对残差 \(\le10^{-8}\) 才继续；B \(p^*\) 固定，用 `kdv_forward_blind.py` 的 IRL1 + λ 路径定支撑（阈值 \(10^{-5}\)）；最后只在支撑上联合重拟合系数和 \(p\)。

| 情形 | 结果 |
| --- | --- |
| KdV | 支撑 {1, tanh²}，系数 2、−2，\(k=1,c=4,x_0=0\) 误差 \(\le10^{-15}\)。多起点命中 13/40；种子 5/5；λ 比例 3/3 |
| Burgers | 支撑 {1, tanh}，系数 0.5、−0.5，\(k=2.5,c=0.5,x_0=0.3\) 精确。多起点命中 **1/40**；种子 **3/5**（失败的两个最好残差 \(1.9\times10^{-8}\)、\(2.5\times10^{-8}\)，刚好卡在门槛上，诊断见下）；λ 比例 3/3 |
| Chafee–Infante | 支撑 {tanh}，系数 1，\(k=1/\sqrt2,c=0\)。命中 10/40；种子 5/5；λ 比例 3/3 |
| 反例：KdV 删 tanh² | 最好残差 0.178，门槛拦下，不报支撑（正确） |
| 反例：KdV 双孤子 | 最好残差 0.306，门槛拦下，不报支撑（正确） |

限制：参数搜索是这个设置下的难点（Burgers 命中率低）；只覆盖单一行波，双孤子这种多波解需要另设库。

**Burgers 种子 3、4 诊断（2026-10-28，临时代码经标准输入跑，没有存成脚本，结果没写 JSON）：**

| 种子 | 多起点最好结果 | 同一点继续 LM 3000 步 |
| --- | --- | --- |
| 3 | \(k=2.49999,c=0.5,x_0=0.2895\)，残差 \(1.9\times10^{-8}\) | 残差 \(4.2\times10^{-11}\)，\(x_0=0.2977\) |
| 4 | \(k=2.49999,c=0.5,x_0=0.3112\)，残差 \(2.5\times10^{-8}\) | 残差 \(4.3\times10^{-11}\)，\(x_0=0.3023\) |
| 0（对照，通过） | \(x_0=0.2970\)，残差 \(1.4\times10^{-10}\) | 基本不动 |

- **不是错盆地。** 三个种子前 5 名起点的 \(k,c\) 全对，只有 \(x_0\) 偏约 0.01。
- **原因：tanh 幂次库能近似吸收平移。** \(\tanh(\theta+\varepsilon)\approx\tanh\theta+\varepsilon(1-\tanh^2\theta)\)，所以 \(x_0\) 偏一点可以由常数项和 tanh² 项补偿（种子 3：\(x_0\) 偏 −0.0105 → \(\varepsilon=-0.026\)，常数 0.513、tanh² −0.013，与公式吻合）。残差沿 \(x_0\) 几乎是平的，LM 走得慢，300 步不够。
- **可写进论文的观察：** 只用最小二乘时 \(x_0\) 停在 0.297 / 0.2977 / 0.3023；L1 选出 {1, tanh} 后在支撑上重拟合，\(x_0\) 精确回到 0.3。即稀疏性帮助唯一确定位置参数。v2 里 5 个种子都是这样。

### 2026-10-29 v2：加延长迭代后整体重跑（**正问题主结果用这个**）

`forward_generic_v2.py` → `forward_generic_v2_result.json`，本机 py312 约 36 分钟。只改一处：多起点 LM 后，对残差最小的那个结果原地继续 LM 最多 3000 步。门槛、IRL1、λ 路径、阈值、重拟合与 v1 完全相同（v2 直接 import v1 的函数）。

| 情形 | v1 | v2 |
| --- | --- | --- |
| KdV | 通过，种子 5/5，λ 比例 3/3 | 同左（延长迭代后残差不变，\(7\times10^{-11}\)） |
| Burgers | 通过，种子 **3/5**，λ 比例 3/3 | 通过，种子 **5/5**，λ 比例 3/3 |
| Chafee–Infante | 通过，种子 5/5，λ 比例 3/3 | 同左 |
| 反例 KdV 删 tanh² | 拦下（0.178） | 拦下（0.178） |
| 反例 KdV 双孤子 | 拦下（0.306） | 拦下（0.306） |

Burgers 延长迭代前后的最好残差：种子 1 \(3.4\times10^{-9}\to3.7\times10^{-11}\)，种子 3 \(1.9\times10^{-8}\to4.2\times10^{-11}\)，种子 4 \(2.5\times10^{-8}\to4.3\times10^{-11}\)，其余基本不变。5 个种子的支撑都是 {1, tanh}。

仍有的限制：多起点命中率仍低（Burgers 原始命中 1–2/40，靠最好的那一个延长迭代过门槛）；只覆盖单一行波，双孤子需要另设库；运行慢。

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

2026-10-28 已做（见 `inverse_irl1.py`）：

- 精确 L1（LARS）重跑：结果同旧脚本，排除「求解器没收敛」。无噪声数据上残差和 \(u_x\) 系数都与 \(\lambda\) 成正比，所以 \(\lambda\) 只降到 \(10^{-6}\lambda_{\max}\) 时残差只有 \(10^{-6}\) 量级，旧的 \(10^{-8}\) 门槛本来就到不了。
- 不可表示条件：\(u_x\) 列 > 1，其他错列都 < 1，解释了为什么普通 L1 只会多留 \(u_x\)。
- 迭代重加权 L1（权重来自上一轮 L1 系数）：**通过**。

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
D:\Miniconda3\envs\py312\python.exe inverse_irl1.py
D:\Miniconda3\envs\py312\python.exe kdv_forward_blind.py
D:\Miniconda3\envs\py312\python.exe forward_generic.py
D:\Miniconda3\envs\py312\python.exe forward_generic_v2.py
```

每条脚本把 JSON 写到同目录。重跑会覆盖对应 JSON。已通过的不要为了「再看一眼」去改脚本。

---

## 10. 群与角色

群名：`PDE解析解讨论`  
成员：用户、研究员、审核员、辅助决策员、实验员、产品经理、信息员、总经理助理、Chief（admin）。

实验由 **实验员** 写脚本并跑。Chief 对 JSON、定通过与否、决定下一刀。论文 skill 已在本地，等两条反问题通过后再写 LaTeX。

---

## 11. 下一步

用户选了 A，已完成，10 条全部通过。之后用户问「徐刚老师是否满意」，结论是要补「库里不含真解」的实验，已做（`forward_generic.py`，见第 5 节），只剩 Burgers 参数搜索的收敛问题。

2026-10-29 已完成延长迭代的整体重跑（`forward_generic_v2.py`），三个正例种子全部 5/5，两个反例都被拦下，见第 5 节。

**论文初稿已写（2026-10-29）：** `paper/main.tex` + `paper/refs.bib`，英文，NeurIPS 2026 模板 preprint 模式（用户选 CCF-A 会议式，未指定会议，按 ccf-paper-writer 默认用 NeurIPS），8 页，编译无报错无溢出，PDF 在 `paper/main.pdf`。编译：在 `paper/` 下 `latexmk -pdf -outdir=build main.tex`。用户已定：λ 比例写「固定 1e-3 + 敏感性表」；Burgers 两行波秩亏数据放附录 A。
待补：作者姓名和单位（现为占位）；目标会议与页数限制；是否加图（目前只有表）；导师审阅意见。所有数字取自各结果 JSON，没有新跑实验。

**以下几点已按用户选择写进论文：**

- IRL1 第 0 轮 \(\lambda\) 比例的选取规则怎么写。
- 主结果用 Burgers Cole–Hopf 新数据，旧两行波数据（秩亏）放附录还是正文。
- 正问题主线用通用库结果（`forward_generic`），旧的离散列结果（tier2 / tier3 / kdv_forward_blind）作为对照。

**运行环境备注：** Claude 桌面端这次可以直接在本机用 PowerShell 调 `D:\Miniconda3\envs\py312\python.exe` 跑。长任务放后台。Claude 自己的临时目录对 PowerShell 不可见，临时代码用标准输入传给 Python（`$code | python -`）。

遗传算法怎么接到 L1 前面，等用户补充思路。
