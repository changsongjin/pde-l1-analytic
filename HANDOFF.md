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

### 2026-10-29 噪声误差分析 + 遗传算法对比（导师意见：加高斯噪声、和遗传算法比）

**论文还没改，等用户确认结果。** 噪声 = σ·std(u)·N(0,1)。图在 `figures_noise/`，由 `noise_figures.py` 只读 JSON 生成。

**反问题** `noise_inverse.py` → `noise_inverse_result.json`（5 个方程 × 9 个噪声水平 × 20 个种子，约 5 分钟）：

- 强形式（Savitzky–Golay 求导）不行：1% 噪声时 \(u_{xx}\) 误差 45%。改用**弱形式**（weak SINDy，测试函数 \((1-s^2)^6\)，库列为 \(\partial_x^k(u^m)\)，不对噪声数据求导）。
- 判据：BIC 在无噪声时 RSS→0 会多选项，改用 MSTLS 判据 \(L(S)=\lVert A(w_S-w_{LS})\rVert/\lVert Aw_{LS}\rVert+\lvert S\rvert/p\)。所有方法共用同一判据，只比「怎么产生候选支撑」：L1（LARS 路径）、IRL1、STRidge、自写 GA（种群 40、60 代）、穷举（≤6 项，判据上界）。
- 结果（选对率）：IRL1 在 5 个方程上到 5% 噪声全部 1.00；7% 时 Burgers 0.75、KdV 0.75（GA 0.90、穷举 0.90/1.00）；10% 时 heat/Burgers/KdV 所有方法（含穷举）都失败，是数据本身不够。普通 L1 在 Burgers（有噪声）和 KdV（任何噪声包括无噪声）总多留 \(u_x\)，与不可表示条件一致。STRidge 在 Burgers 2–3% 就开始掉。
- 系数相对误差在 2–3% 以内随噪声线性增长（斜率 1），之后超线性；KdV 有约 \(3\times10^{-4}\) 的求积误差下限。
- 代价：IRL1 中位数约 15 次判据评估、0.003 s；自写 GA 约 450 次、0.09 s；穷举 4096 次。GA 在 KdV 上有 1–2/20 个种子卡在 \(\{u_x\}\)（搜索失败，穷举是对的）。

**正问题** `noise_forward.py` → `noise_forward_result.json`（初边值加噪声，σ = 0.1%、1%、5%、10%，5 个种子，约 100 分钟）：

- 门槛改为差异原理：3 × 噪声本身的相对残差（\(3\sqrt2\,\sigma\,\mathrm{std}(u_{ic})/\lVert\text{数据}\rVert\)），假定 σ 已知。支撑用与反问题同型的判据在 λ 路径上的支撑里挑。
- 三个正例（KdV、Burgers、Chafee–Infante）在所有噪声下 5/5 选对，系数和波参数误差都随 σ 线性（KdV 10% 时系数误差 0.5%）。论文原来的「过半」规则在噪声下会掉（Burgers 5% 时 0/5），所以噪声下要用判据选支撑。
- 正例的残差 / 噪声水平稳定在 0.88–1.10。反例：删 tanh² 在 σ ≥ 5% 被放过（比值 2.5–2.9 和 1.5–1.8，门槛是 3），双孤子在 10% 被放过（比值约 2.5）。**门槛倍数 3 太松**；事后按已存残差算，倍数 1.5 时只剩删 tanh² 在 10% 放过 2/5。改不改倍数要用户定（图 `figures_noise/forward_gate.png`）。

**SGA-PDE 对比** `sga_compare.py` → `sga_compare_result.json`：代码克隆在 `external/SGA-PDE/`（已加入 `.gitignore`，代码未改；只在运行器里补了两处新版 NumPy 兼容）。数据：SGA 自带的 Burgers、KdV、Chafee–Infante + 我们的 Burgers、KdV；噪声 0、1%、5%，种子 0–2。SGA 用论文默认参数（种群 20、深度 4、宽度 5、100 代），单进程单线程。第一次跑每次限 1 小时，9 次里 8 次超时（无噪声 KdV 跑完 100 代用了 41 分钟），改为每代写一次当前最优方程、每次限 30 分钟、超时取最后一代并记录代数（跑完约 2 小时，35 次里多数超时，到达代数中位数 47、最少 24）。SGA 选出的项用余弦相似度（≥ 0.9）归到常见候选项，集合等于真项才算对。

结果（图 `figures_noise/sga_compare.png`）：

| 数据 | 噪声 | SGA-PDE | IRL1 | 自写 GA | 穷举 |
| --- | --- | --- | --- | --- | --- |
| 无噪声，5 组数据 | 0 | 3/5（Burgers ×2、C–I 对；两组 KdV 都错） | 5/5 | 5/5 | 5/5 |
| 1% 噪声，5 组 × 3 种子 | 1% | 0/15 | 15/15 | 15/15 | 15/15 |
| 5% 噪声，5 组 × 3 种子 | 5% | 0/15 | 6/15 | 7/15 | 7/15 |

- 5% 时我们失败的是 C–I（SGA 数据）、我们的 Burgers 和 KdV：这三组网格较小（为了让 SGA 跑得动），弱形式测试函数少；穷举也错，说明是数据不够，不是 IRL1 的问题。在大网格（`noise_inverse.py`）上 5% 全对。
- SGA 在有噪声时选出的多是含 \(x\) 的项或单项（如 \(2.8u\)、\(x u_x\)），原因是有限差分在噪声上求导误差大，与我们强形式失败的原因相同。
- **补跑 SGA-PDE 论文另两个方程**（`sga_compare_extra.py` → `sga_compare_extra_result.json`，网格 100×251，测试函数半宽 (10, 20)，同样 0 / 1% / 5% 噪声、30 分钟预算）。本文方法跑两版库：基础库（13 列）和加 4 个含 x 项（x u、u/x、x u_x、u_x/x）的扩展库（17 列）。

  | 方程 | 噪声 | SGA-PDE | L1 | IRL1 | 自写 GA | 穷举 |
  | --- | --- | --- | --- | --- | --- | --- |
  | PDE_compound \(u_t=(uu_x)_x\)（两版库结果相同） | 0 | 0/1（\(0.469(u^2)_{xx}-0.085u_x^2\)，结构对但多一项） | 1/1 | 1/1 | 1/1 | 1/1 |
  | | 1% | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 |
  | | 5% | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
  | PDE_divide \(u_t=-u_x/x+0.25u_{xx}\)，基础库 | 全部 | — | 0 | 0 | 0 | 0（真项不在库里，预期） |
  | PDE_divide，扩展库 | 0 | 0/1（跑满 100 代，\(-2.86u+\dots\)） | 0/1 | 1/1 | 1/1 | 1/1 |
  | | 1% | 0/3 | 0/3 | **0/3** | **3/3** | 3/3 |
  | | 5% | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |

  PDE_divide 1% 是**第一个 GA 明显好于 IRL1 的情形**：扩展库里含 x 的列彼此高度相关（u/x 与 u、x u 等），L1 的不可表示条件更难满足。5% 时所有方法都错，网格小、数据不够。
- **公平性注意**：SGA-PDE 论文处理噪声时用了神经网络去噪 / 元数据（`use_metadata`），默认关闭，我们没开；SGA 时间预算 30 分钟（多数没跑满 100 代）；KdV 在 SGA 论文中报告能找到，我们在 30 分钟预算内没有找到。写论文时要如实说明这三点。

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
5. 噪声反问题用强形式（Savitzky–Golay 求导）：1% 噪声 \(u_{xx}\) 误差 45%，改弱形式。
6. 噪声反问题用 BIC 选支撑：无噪声时 RSS→0，log 项主导，多选项，改 MSTLS 判据。

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
D:\Miniconda3\envs\py312\python.exe noise_inverse.py
D:\Miniconda3\envs\py312\python.exe noise_forward.py
D:\Miniconda3\envs\py312\python.exe sga_compare.py
D:\Miniconda3\envs\py312\python.exe sga_compare_extra.py
D:\Miniconda3\envs\py312\python.exe noise_figures.py
```

`sga_compare.py` 需要先 `git clone https://github.com/YuntianChen/SGA-PDE external/SGA-PDE`，跑 35 次 SGA，8 进程并行约 2 小时。

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
待补：作者姓名和单位（现为占位）；目标会议与页数限制；导师审阅意见。所有数字取自各结果 JSON，没有新跑实验。
已加 4 张图（10 页）：引言后的背景示意图 `paper/figures/overview.tex`、方法里的算法流程图 `paper/figures/flowchart.tex`（都是 TikZ，随论文编译），以及 `paper/make_figures.py` 生成的 `inverse_path.pdf`、`gate.pdf`。

**PPT（2026-10-29）：** `slides/PDE_L1_进展汇报.pptx`，中文，16 页（含目录、相关工作、实验设置），带演讲者备注；插图用中文版，由 `slides/make_figures_zh.py` 生成（TikZ 图从论文英文源文件替换文字后用 xelatex+ctex 编译，论文图本身不改）。生成：`cd slides; npm install; $env:PPTX_SKILL=<pptx skill 目录>; $env:NODE_PATH="$PWD\node_modules"; node build_deck.js`。图片在 `slides/img/`（TikZ 图用 `slides/fig_standalone.tex` 单独编译后转 PNG）。本机没有 LibreOffice，用 PowerPoint COM 导出 PDF 检查版面；skill 自带的 validate.py 在本机加载不了自己的 schema 文件（每个部件都报同一个错，包括没动过的），所以没用它的结果。

**以下几点已按用户选择写进论文：**

- IRL1 第 0 轮 \(\lambda\) 比例的选取规则怎么写。
- 主结果用 Burgers Cole–Hopf 新数据，旧两行波数据（秩亏）放附录还是正文。
- 正问题主线用通用库结果（`forward_generic`），旧的离散列结果（tier2 / tier3 / kdv_forward_blind）作为对照。

**运行环境备注：** Claude 桌面端这次可以直接在本机用 PowerShell 调 `D:\Miniconda3\envs\py312\python.exe` 跑。长任务放后台。Claude 自己的临时目录对 PowerShell 不可见，临时代码用标准输入传给 Python（`$code | python -`）。

遗传算法怎么接到 L1 前面，等用户补充思路。

---

## 12. 2026-11-03 新指示

用户原话，按这个标准做后续实验，不要再把「高噪声下选错」本身当成失败：

- 噪声大到一定程度后，方程选错是正常现象。没有一份数据是完全无误差、完全符合某个式子的。牛顿力学也只是近似。
- 要比较的是哪个算法更耐噪声。
- 进一步的问题：能否对付 outlier。还没做实验。
- 终极问题是从数据里找近似因果关系。要优化的是结构简单和数据一致性的折中，不是残差必须到机器精度。

方法不限定 PDE。候选项残差对系数线性时，常微分方程和代数关系也可以用同一套库加 L1。开普勒三定律不是微分方程；写成 \(F=ma\)、力按 \(1/r^2\) 之后才是常微分方程，那是反问题。

## 13. 2026-11-05 最大库与方法对照实验

新增实验文件：`gml1_maxlib_compare.py`、`gml1_spec.md`、`dlga_compare.py`、`benchmark_compare.py`，结果见 `gml1_*_result.json`、`dlga_*_budgeted_result.json` 和 `benchmark_compare_result.json`。旧脚本没有修改。

### 13.1 GML-L1 最大库

新方法是弱形式语法闭包库 + KKT 列生成。默认核心库包含 113 个生成项：(u^m)（(m\le8)）及 (x^a u^m) 的有限阶空间导数（(a\le2)），最高空间导数阶 6。每个固定 \(\lambda\) 先在活动列上求精确 L1，再加入违反全库 KKT 条件的候选列；KKT 扫描结束后才认定是全库 L1 解。IRL1 仍然只求加权 L1。`--nonpoly` 的 (sin,cos,exp) 和有理函数只作扩展实验。

### 13.2 五个已通过反问题的噪声曲线

使用 heat、wave、advection-diffusion、Burgers、KdV，噪声为 0、1%、3%、5%、7%、10%，每个非零水平 3 个种子。主要结果：

- 无噪声时 GML1-IRL1 五个方程都选对；普通 GML1 在 Burgers/KdV 中保留 (u_x)，与原先不可表示条件一致。
- advection-diffusion 的 IRL1 在 1%–5% 为 1.00，7% 为 0.67，10% 为 0.33。
- Burgers 的 IRL1 在 0%–3% 为 1.00，5% 以上开始失败。
- KdV 的 IRL1 在 0%–5% 为 1.00，7% 为 0.33，10% 为 0。
- heat 在当前最大库下从 1% 开始出现 (u_{xxxx})、(u_{xxxxxx}) 等混淆；wave 在 1% 仍有部分正确，3% 后显著下降。

这说明大库下的失败有两类：算法没有压掉近相关项，以及数据本身不能区分高阶项。高噪声按用户新指示只比较耐噪声曲线，不把所有错误都归为算法错误。

### 13.3 对照方法

- `PDE-FIND_STRidge`：在相同弱形式矩阵上运行 STRidge，作为 PDE-FIND 的稀疏回归对照。经典 PDE-FIND 不是 L1，而是岭回归加硬阈值。
- SGA-PDE：使用已有 `sga_compare_result.json` 和 `sga_compare_extra_result.json`。已有结果中，SGA 在 Burgers/KdV 的 1%、5% 噪声多数未选对；原实验有 30 分钟单次预算和超时记录。
- DLGA-PDE：通过 KD 的神经 surrogate + 遗传搜索做了 30 代、40 个体、500 个 surrogate epoch、4 倍网格下采样的预算对照。Burgers 在 0%/5% 选对，advection-diffusion 只在 0% 选对；heat、wave、KdV 在该预算下未选对。此结果是记录预算的可复现实验，不与弱形式 L1 直接比较单次运行时间。

方法定义和论文链接见 `method_comparison.md`。总表以 `benchmark_compare_result.json` 为准。

### 13.4 同一项目数据的直接比较

新增 `same_data_compare.py` → `same_data_compare_result.json`：直接读取 `sga_compare.py` 的 SGA-PDE 原始数据、同一噪声生成函数和同一测试函数窗口，重跑 GML1、IRL1、PDE-FIND/STRidge，并读取同一次 SGA 运行结果。

新增 `dlga_same_data_compare.py` → `dlga_same_data_result.json`：在同一批原始场数据上运行 DLGA，使用 4 倍网格下采样、30 代、40 个体、500 个 surrogate epoch。

同数据结果显示：SGA Burgers 无噪声选对，但 1% 和 5% 失败；SGA KdV 在三档噪声都没有稳定选对；Chafee–Infante 只在无噪声选对。GML1-IRL1 在 SGA Burgers 的 1% 噪声为 1.00、5% 为 0.67；在 SGA KdV 的 1% 为 1.00、5% 为 0。Chafee–Infante 的最大多项式库在当前单轨迹数据上即使无噪声也不可辨识，不能把这个结果写成单纯的算法失败。DLGA 的同数据预算对照中，`ours_burgers` 0%/5% 选对，SGA 原始三组和 KdV 同数据预算下均未选对。

## 14. 2026-11-06 库效应审计与 outlier 第一轮

按最终技术路线新增实验，旧脚本和旧结果没有修改。

### 14.1 L13/L43/L113 库效应

`library_effect_audit.py` → `library_effect_audit_result.json`：在同一弱形式数据链上记录 L13、历史 L43、当前 113 项多项式库和二者并集 L121 的设计矩阵秩、条件数、列相关性及真支撑残差。历史 L43 含 `sin/cos`，当前 L113 不含，因此二者不是严格嵌套；报告额外保留 L121 并集，避免混淆库效应和命名差异。

`library_effect_selection_heat_wave_result.json`、`library_effect_selection_adv_burgers_kdv_result.json`：五个基准反问题在 0% 和 5% 噪声下跑 GML1、GML1-IRL1、PDE-FIND/STRidge。无噪声时各层通常能选真支撑；扩大库后 heat/wave 在 5% 更早出现高阶项，Burgers/KdV 的普通 L1 在无噪声也会受 `u_x` 不可表示条件影响。L113 的相对秩（阈值 (10^{-6})）为 heat 87/113、wave 105/114、advection-diffusion 84/113、Burgers 67/113、KdV 77/113，最大列相关普遍接近 1。

### 14.2 弱方程行 outlier

`outlier_l1_compare.py` → `outlier_l1_compare_result.json`：实现

\[
\min_{\beta,e}\tfrac12\lVert A\beta+e-b\rVert_2^2
+\lambda\lVert\beta\rVert_1+\gamma\lVert e\rVert_1,
\]

并提供联合普通 L1 与联合 IRL1。异常直接注入弱方程行，比例 0/1/5/10%，幅度 5/10 倍行 RMS，L13/L113、3 个种子；当前主跑固定 `gamma=2`。L13 上五个方程在这组行异常中联合 IRL1 都保留真 PDE 支撑并找回异常行；L113 在 heat 的 10% 和 KdV 的 5%/10% 开始下降，符合大库近相关性的限制。`outlier_gamma_sensitivity.py` → `outlier_gamma_sensitivity_result.json` 显示小规模扫描中 KdV 需要 `gamma>=2` 才能恢复真支撑，当前结果已记录这个敏感性。

### 14.3 原始场点 outlier 的边界

`point_outlier_probe.py` → `point_outlier_probe_result.json`：把异常注入原始 `u(x,t)` 网格，再构造弱形式矩阵。一个原始点会污染多个弱测试行并同时改变 A、b；L13 的 0.01% 点异常已使 Burgers 支撑失败，0.1% 时 heat/Burgers/KdV 基本都失败。因此弱行模型的成功不能写成传感器点异常已经解决。若继续做 raw-field outlier，需要单独建模设计矩阵污染或使用鲁棒测试函数/行权重。

### 14.4 稳定性与复杂度—一致性汇总

`stability_pareto_summary.py` → `stability_pareto_summary_result.json`：从上述选择结果汇总跨种子支撑稳定性、选对率、平均复杂度，并对联合 outlier 模型给出“重拟合残差—结构复杂度”非支配点。复杂度只用于汇报候选结构，不改 L1/IRL1 产生的支撑。

## 15. 2026-11-06 上传前复核与结论更正

本节优先于第 13、14 节中尚未审计的指标和推断。当前上传为探索性检查点，详见 `CHECKPOINT_STATUS.md`；新技术方案由用户确认后再上传，下一轮实验暂不启动。

`publication_checkpoint_audit.py` → `publication_checkpoint_audit_result.json`：检查了相关脚本语法和 JSON，可读性检查通过，但三个 GML1 JSON 含非标准 `-Infinity`。旧文件按约定保留，后续新文件须使用有限标准 JSON。

已确认两项指标问题：`G.support_eval` 将归一化系数与物理真系数比较，相关系数误差不能用作论文证据；已存 DLGA 支撑混用 `uxx`、`u_xx` 等名称，且 `dlga_compare.field` 只使用第一条轨迹，故旧总表尚不支持严格同输入的算法排名。

第 14 节列出的 87/113 等是相对阈值 `1e-6` 下的有效秩，条件数也经过截断。advection-diffusion 与 KdV 在机器精度阈值下均为 113/113；因此不能仅凭这些有效秩把失败判为“数据不可辨识”。求解器 KKT、支撑竞争、验证轨迹和选模规则仍需区分。

现有 `|S|/p` 复杂度惩罚会随候选库大小变化，库效应与判据效应尚有混淆。现有 Pareto 汇总跨噪声种子合并最终模型，也不是同一验证数据上整条候选路径的前沿。

outlier 代码实际沿固定相对权重 `rho=gamma/lambda` 搜索；`gamma_weight=2` 是探索性取值。弱方程行异常与原始场点异常必须分开报告，不将初步支撑恢复推断为普适 outlier 能力。投稿前先完成这些审计与公平评价，再正式重跑；不靠提高支撑阈值修补结果。

## 16. 2026-11-06 用户补充 L1 方法可按具体问题改进

用户明确：以普通 L1、IRL1 为起点，若效果不好，可以根据具体情况继续优化 L1 norm 方法，不限于使用现成的两种形式。研究路线增加“基线复现 → 失效诊断 → 有针对性的 L1 改进 → 消融与独立验证 → 正式噪声对照”。

具体改法根据诊断决定，可研究有限正权重、IRL1 迭代稳定性、噪声加权及后续鲁棒损失加 L1。保持 L1 稀疏化核心、`1e-5` 支撑阈值和旧脚本不改等约束。原始 L1、IRL1 必须保留为对照，不能利用测试真支撑调权重或通过事后最小二乘删项。

已将此要求纳入 `RESEARCH_PLAN_DRAFT.md`。用户随后明确要求先上传 GitHub，本次将方案修订及交接记录同步到已有研究分支；新的方法实验尚未启动。

## 17. 2026-11-06 用户授权按阶段执行

用户已明确要求依次执行各阶段并及时反馈问题，因此第 15、16 节中“暂不启动”的状态已结束。当前执行状态和边界见 `EXECUTION_STATUS.md`，结果以各新脚本的对应 JSON 及其具体检查范围为准。

阶段 0 的 `stage0_gate_result.json` 已通过；阶段 1 已完整枚举 1158 个有限 jet 多项式并完成基本重建检查。全部旧脚本保留。新数据包含六类 PDE（新增 `u_t=u_xx+u_x^2` 的 logheat 案例），每例 4 条训练、2 条验证轨迹；最终测试数据未生成。

目前发现的关键前置问题是高阶重建噪声放大和弱积分离散误差。分部积分降阶的符号身份已核对，但第一版弱降阶实现的形式检查通过不等于高阶数值精度充分；待新积分核检查后再开始矩阵审计和真实 L1/IRL1 开发基线。此处不更新旧论文/PPT，也不将开发检查称为正式耐噪声结论。

阶段更新：`stage1_weak_reduction_v2_result.json` 完成离散矩校正及无噪声精度检查；`stage2_review_result.json` 独立验收全部 72 对开发矩阵/目标，已进入阶段 2a。第一版求积和差分检查的失败记录保留，不能只读早期 `all_checks_passed` 判断当前研究门槛。

用户随后选择“每批不超过 2 小时，保存后续跑”。heat 开发实验暴露弱残差评分偏好不适定高阶项的问题，因此将动态预测和适定性诊断提前到方法冻结前。120 秒内未完成的普通 L1 路径按计算未完成处理；后续补算保留原超时和全部额外成本。详见最新执行记录，不提前生成正式论文结果。

本批收尾：72 组开发试验全部执行，合并补算后 71 组路径数值完整；剩余 advection-diffusion、5%、J1158、L1 的两个 λ 比例 `1.823348e-6` 与 `1e-6` 相对对偶间隙约 `1.13e-9/3.04e-9`，尚未达到原 `1e-9` 门槛。最新候选与失败详情在 `stage2a_retry_remaining_v3_result.json`，不得放松支撑阈值或将其算作统计耐噪失败。

下一位智能体先读 `stage2a_batch_checkpoint_result.json`、`stage2a_final_diagnosis_result.json` 和 `EXECUTION_STATUS.md`。最终基线来源为 `stage2a_heat_pilot_result.json`、`stage2a_heat_reconciled_result.json`、`stage2a_other_cases_reconciled_result.json`；不要将原始/修复后的同一实验重复计入汇总。下一批先处理数值尾项，再进入阶段 2b。统计 L1 变体尚未测试，最终测试数据未生成；本批新结果保存于工作区，尚未同步到 GitHub。

## 18. 2026-11-06 第二批续跑与收尾

本节优先于第 17 节的第一批收尾状态。本批约从新加坡时间 18:49 开始，按用户约束最迟在 20:48 前保存收尾。`stage2a_retry_remaining_v4_result.json` 已解决两个尾部点：β 与支撑不变，仅改善合法对偶下界，对保存的双精度 A、b、β、q 做精确有理数全列可行性与间隙验证。旧残差缩放证书仍 false；新证书相对间隙 `6.75e-11/7.57e-11`，原精度门槛未变。

当前阶段 2a 统一来源是 `stage2a_reconcile_complete_result.json`，检查点为 `stage2a_batch_checkpoint_v2_result.json`，诊断为 `stage2a_complete_diagnosis_result.json`。72/72 组、1728 个 λ 完整；48 组选中真支撑、6 组评分折中、18 组路径未产生真支撑。历史所有求解与补算成本保留，不将纯数值证书修复作为统计方法贡献。

阶段 2b 协议已在任何真实 PDE 新变体结果之前冻结于 `stage2b_protocol_result.json`。唯一统计改动是 IRL1 的 `epsilon=max(1e-8,0.01*max(abs(beta)))`，其余原设置保留。先检查 heat 1%、Burgers 5%、logheat 5%，J18/J1158 两库；保留 L1、原 IRL1 对照。新开发复验使用轨迹 12..17、seed 1，训练 12..15、验证 16..17；最终测试轨迹 6..11 与 seeds>=1000 均未生成。新数据与矩阵检查见 `stage2b_data_result.json`，变体合成数值检查见 `stage2b_smoothed_irl1_result.json`。

全频谱热方程评价器 `stage2b_heat_full_spectrum.py` 保留全部 255 个网格可解析模式，提供原始初值及统一冻结平滑两个方案，使用对数域避免溢出。它仅支持常系数线性偶阶项，并明确采用 Dirichlet Laplacian 的多项式谱边界实现；不支持的结构不能被悄悄过滤或记为失败。此时对比运行仍在进行，最终结果须以新增 JSON 与最新执行记录为准。当前新增结果仍在本地，未同步 GitHub。

第二批后续更新：阶段 2b 的 36 条对照记录已全部数值完整，统一来源为 `stage2b_reconcile_complete_result.json`；`stage2b_complete_diagnosis_result.json` 表明相对 ε 变体在全部 12 个 cohort×case×library 设置中，与原 IRL1 的主选支撑相同。不得把权重比例受到约束或合成检查通过当成实际改善。两个新开发普通 L1 大库路径原本各有 2 个尾部 λ 超时，已在 `stage2b_numerical_retry_result.json` 用原数值门槛补齐，所有原失败与补算成本保留。

`stage2b_heat_dynamic_compare_v2_result.json` 对完整 heat 模型做 36 个全频谱/初值方案诊断；原 IRL1 与平滑 IRL1 的相同主选支撑具有相同 LS 系数和动态结果，仍有高频增长。`stage2b_heat_noise_mechanism_result.json` 解释纯线性模型弱残差的噪声增益差异：这是冻结已选算子后独立重抽噪声的期望，分母用 clean target，仅作诊断，不是给定原选择事件的条件期望，也未引入修正选模规则。

用户已明确选择后续优先“先补齐最大候选库，再做系统耐噪实验”，因此暂不继续增加统计 L1 变体。已补实现原计划 54 个非多项式弱列（`stage1_nonpolynomial.py`），通过 affine 和非线性曲率场的独立链式导数验算（`stage1_nonpolynomial_curved_check_result.json`）。全部 18 组旧开发数据已追加 54 列并核对原多项式区块与 b 位值不变，见 `stage2c_extended_matrices_result.json`。扩展仅对原声明有限语法完整，不等于无边界“全部可能方程”。新最大库 L1/原 IRL1 pilot 正在准备，最终测试尚未生成。

最大库后续完成：`stage2c_reconcile_complete_result.json` 为 36/36 条完整开发路径的唯一合并来源，`stage2c_review_v2_result.json` 已独立验收 864 个 λ 及外层、数据、支撑和评分。原 4 个 5% 普通 L1 路径共有 9 个数值尾项，已补齐；adv 的最后 2 点与 logheat 最后 1 点用合法保存 q 的精确对偶证书，其他使用原证书。原失败和补算成本均保留，求解累计约 1040.06 秒。不得重复计入原 pilot 与合并报告。

`stage2c_paired_complete_diagnosis_result.json` 给出相同原始观测下 J1158→J1212 的成对诊断：本轮 17 组选真、3 组评分折中、16 组路径缺真支撑，13 个主选模型带非多项式干扰项。这不是正式恢复概率或算法排名，且所有真方程都已属于多项式库，不能声称本轮验证了真非多项式项恢复。

下一步阶段 3 的 `stage3_protocol_draft.json` 仍未冻结。`stage3_data.py` 与更严格绑定来源的 `stage3_data_v2.py` 已做合成预检，未生成正式噪声种子或测试轨迹。运行器 `stage3_campaign.py` 仍在独立检查中。正式计划为 606 个数据组、3636 个方法任务，每个非零噪声水平 20 个配对种子，按照用户两小时上限分批。每组必须先完成三个库×两方法的全部数值认证并封存选模，才能开放该组 test IDs6..11；不得用测试数据决定 λ/α/ε 或后续统计方法。最终状态以本批收尾时的执行记录为准。

第二批收尾更新（优先于上一段的准备状态）：`stage3_campaign_preflight_result.json` 已通过 16 项合成契约检查；`stage3_freeze.py` 完成当前接口、已有数值门槛与历史保全的 68 项检查，生成不可覆盖的 `stage3_protocol_result.json` 和 `stage3_freeze_result.json`，冻结 43 个本地脚本。协议 SHA256 为 `1c5b11961e327a2f1b92605ad5e3bdea7e5663bf6cd6ce8a7a0c11ab15b85feb`。旧草案和第一版预检较早的草案哈希均保留，当前配置重新验证通过。

阶段 3 已完成配置冻结，正式任务仍为 0：没有正式 seed 1000..1019 原始场，也没有 test IDs6..11。本批接近两小时截止，因此不启动新计算。下一批直接使用 `EXECUTION_STATUS.md` 中的有界命令；它按已存索引续跑，并通过独立 Windows 进程树监督器限制时长。不要重新执行会覆盖历史报告的旧预检，也不要重跑只允许一次的冻结脚本。

正式协议明确：暂停后可能缺少早先未落盘的原始数据生成耗时，不能据此作端到端速度优势结论。支撑、数据一致性、噪声比较和数值门槛未改。后续统计绘图代码必须服从已经冻结的统计定义，不能用正式结果修改选模规则。第二批整体保全、语法与有限 JSON 审计另存 `stage_execution_audit_v2_result.json`；具体通过状态以该文件为准。本批新增代码/结果仍在本地，未再次上传 GitHub。

## 19. 2026-11-06 第三批：首批正式实验

本节优先于第 18 节的“正式任务为 0”。本批从 UTC 12:52:22（新加坡 20:52:22）开始，保守收尾截止 UTC 14:51。冻结协议及 43 个源脚本均未改。实际执行了 1500 秒和 3600 秒调度段，之后单独数值补算，再用 900 秒、450 秒有界段合并并评价；各段实际状态均为正常检查点保存，单数值工作进程。

统一索引：`stage3_campaign_result.json`。完成数据组为 seed 1000、六类 PDE 的 0/1/3/5%，另有 heat 7%，合计 **25/606 组**；共 **150 条完整方法路径**。每组均完成三库×两方法封存，才开放该组 test IDs6..11，不按测试结果重选。wave 7% 仅启动了第一条 L1 路径，另外 5 条尚未运行；其外部补算请求已认证完整，但未合并，所以索引仍有 1 条待续跑记录，wave 7% 无测试数据。seed 1001..1019 和其余 7/10% 任务均待后续批次。

数值补算统一报告为 `stage3_numeric_continuation_20261106T140942378192Z_result.json`：20 条原不完整路径、32 个尾点全部认证，32 次尝试均成功；累计补算求解墙钟约 609.22 秒，适配器总计约 620.78 秒。原始路径、旧超时和之前认证点未修改。运行器已合并其中 19 条，剩下 wave 7% 的 1 条请求由下一批运行器自动合并。不要把未合并候选重复计入正式路径数。

注意补算版本：初版 `stage3_numeric_continuation.py` 多一次 C-order 复制改变了列范数的末位舍入，输入哈希守卫在求解前中止。失败脚本和报告保留。使用 **`stage3_numeric_continuation_v2.py`**；其预检对全部 20 个任务验证原归一化与 solver A/b 哈希逐位相同。它调用早已冻结的固定权重救援函数，外部续算格式已在原运行器中预先定义；方法、λ、正权重、阈值和原认证门槛均未变。新增适配器来源哈希写入每次尝试和续算证据。先运行正式运行器合并已有请求，之后才能对仍不完整路径安排新补算；同一用户批次每点最多 2 次 120 秒尝试，批 token 应标记本批开始时间。

`stage3_statistics.py/_result.json` 按已冻结规则完成统计函数及合成验算，不读取正式结果，不产生正式区间。没有完整 20 个配对种子的噪声档，不做正式恢复率区间或算法排名。零噪声是每例一次，不算 20 次重复。

复核使用 `stage3_checkpoint_review_v2.py`：封存不可变索引快照，再调用原只读数值/评分/留出误差复核。首段 `stage3_checkpoint_review_20261106T131219Z_result.json` 的历史索引已按其原哈希精确归档于 `stage3_review_snapshots/`，后续索引更新不会使证据丢失。最终复核及整体门槛见最新 v2 复核结果和 `stage_execution_audit_v3_result.json`；其通过只表示对应范围，不代表整个研究完成。

下一批续跑命令在 `EXECUTION_STATUS.md`。示例调度预算改用 6000 秒，为批末审计预留时间；还需按实际剩余时间下调，不能因接到“继续”就重置同一批两小时计时。不要重跑会覆盖结果的旧预检/审计，也不要改冻结脚本。所有本批新增文件仍在本地，未再次上传 GitHub。

第三批收尾：新加坡时间约 22:30，累计约 98 分钟。最终复核为 `stage3_checkpoint_review_v2_20261106T142615204003Z_result.json`，对应完整数值复核为 `stage3_checkpoint_review_20261106T142657Z_result.json`；150 条完整路径及部分路径共 3621 个已认证最终 λ 点重新核验通过。`stage_execution_audit_v3_result.json` 整体检查通过、无错误：26 组已启动、25 组完成、151 条方法记录中 150 条完整，剩余 1 条有待合并的续算请求。冻结 43 源文件哈希仍一致。没有后台计算任务继续运行。

## 20. 2026-11-07 GitHub 上传检查点

用户明确要求推送 GitHub，沿用 `codex/research-checkpoint-20261106` 分支。代码、冻结方案、主要报告、任务索引及逐任务 JSON 进入 Git；原始场、矩阵与超过 2 MB 的报告连同全部相关文件打包到同仓库的 `research-checkpoint-20261107-batch3` Release。详见 `UPLOAD_CHECKPOINT.md`，文件清单为 `GITHUB_CHECKPOINT_MANIFEST.json`，压缩包 SHA256 为 `github_checkpoint_result.json`。Git 属性保持 Python/JSON 字节，冻结脚本和历史实验结果没有为上传而改写。

恢复时先检出该研究分支，再核对并解压 Release 的完整快照，避免为了补齐缓存重新运行会覆盖证据的旧脚本。上述“未上传”是各计算批次结束时的历史状态；本次发布成功状态以 GitHub 远端分支、Release 与 `_github_upload/upload_receipt_result.json` 的实际核验为准。本次只做上传与打包，正式实验进度仍为 25/606，未启动新的计算。
