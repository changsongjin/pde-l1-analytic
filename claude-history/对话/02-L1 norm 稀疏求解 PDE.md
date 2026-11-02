# L1 norm 稀疏求解 PDE

- uuid: 5670bfbc-5ebf-45ed-9ac7-26a550422cdd
- 创建: 2026-10-28T10:25:50.900036Z
- 更新: 2026-10-28T10:37:30.574082Z

## 你 · 2026-10-28 10:25:51

请你先阅读交接文档，然后这个是导师给我出的题目，“现在有一个想法。

首先，我在面对一个新问题的时候，尤其是只知道PDE，我想知道它的解析解方程是什么。但是这些方程是由很多项和系数构成的，我并不清楚这些项是多少，系数又是几。

所以你看能不能用 L1 norm 的方法？先把这些项和系数全部设出来，然后 L1 norm 会让大部分的系数为 0。那这样的话，我们看看能不能求解具体的问题？”

有什么不确定的可以问我

## Claude · 2026-10-28 10:26:06

我先找一下你提到的交接文档。

> 调用工具 Glob：*

> 工具 Glob 返回：No files found

我这边没有找到交接文档：对话里没有附件，你的电脑也还没有连接任何文件夹。你可以直接把文档拖进对话框；如果它在电脑上，也可以在 Claude 桌面端点消息框旁边的“+”，选“添加文件夹”，我再去读取。

在等文档的同时，先说说我对这个题目的理解，方便你判断我们的方向是否一致。

导师描述的思路，在文献里已经有比较成熟的框架，叫**稀疏回归识别方程**。最有代表性的是 SINDy（Brunton 等，2016，用于 ODE）和 PDE-FIND（Rudy 等，2017，用于 PDE）。基本做法如下：

1. 从数据 u(x,t) 出发，用数值方法算出 u_t、u_x、u_xx、u·u_x 等量。
2. 构造一个“候选项库” Θ(u) = [1, u, u², u_x, u_xx, u·u_x, …]。
3. 求解 u_t = Θ(u)·ξ，并在目标函数里加入 L1 正则（LASSO）：min ‖u_t − Θξ‖² + λ‖ξ‖₁。L1 会让大部分系数变为 0，剩下的非零项就是方程的结构，对应的数值就是系数。
4. 实际中纯 LASSO 在候选项高度相关时容易选错项，所以 PDE-FIND 改用 STRidge，也就是带阈值的迭代岭回归。

不过我注意到导师的原话里有一点需要先弄清楚，它决定了整个题目的走向：**导师说“只知道 PDE，想知道它的解析解”**。这和上面的方法其实是两个不同的方向。

- **方向 A：已知数据，未知方程。** 用 L1 稀疏回归把 PDE 本身识别出来，这就是 SINDy/PDE-FIND。
- **方向 B：已知方程，未知解。** 把解写成一组候选函数的线性组合，例如 sin、cos、exp、多项式、tanh 及其乘积，代入 PDE 和边界条件，让残差最小，同时用 L1 让大多数系数为 0，从而得到一个**简洁的闭式解析解**。这更接近“稀疏符号回归求解析解”，也和 PINN 加稀疏化的一些工作有关。

我需要你确认几件事：

1. 导师的意思是 A 还是 B？从“只知道 PDE，想知道解析解”来看我倾向于 B，但“项和系数”这种说法又很像 A。
2. 交接文档里是否已经有前人做到一半的代码或具体的测试方程（比如 Burgers、KdV、热方程）？
3. 你希望使用什么工具？Python（numpy/scikit-learn/PySINDy）还是 MATLAB？

你把文档发过来、回答一下这几个问题后，我就可以挑一个具体的 PDE，把完整流程跑一遍：先生成数据或构造候选库，再做 L1 稀疏求解，最后检验它能否恢复出正确的方程或解。

## 你 · 2026-10-28 10:27:03

"D:\03_项目代码\PDE\HANDOFF.md"
这个不就是交接文档吗

## Claude · 2026-10-28 10:27:21

> 调用工具 ToolSearch：select:mcp__remote-devices__device_request_folder_access,mcp__remote-devices__device_bash,mcp__remote-devices__device_list_dir,mcp__remote-devices__get_device_info

> 工具 ToolSearch 返回：（无文本）

> 调用工具 mcp__remote-devices__device_request_folder_access：paths, reason

> 工具 mcp__remote-devices__device_request_folder_access 返回：{"granted":["D:\\03_项目代码\\PDE"]}

> 调用工具 mcp__remote-devices__device_bash：ls $HOME/mnt/; cd $HOME/mnt/PDE && ls -la && cat HANDOFF.md

> 工具 mcp__remote-devices__device_bash 返回（失败）：Workspace unavailable. The isolated Linux environment on this device failed to start. Use device_stage_files / device_commit_files instead.

> 调用工具 ToolSearch：device_stage_files device_commit_files

> 工具 ToolSearch 返回：（无文本）

> 调用工具 ToolSearch：select:mcp__remote-devices__device_stage_files,mcp__remote-devices__device_commit_files

> 工具 ToolSearch 返回：（无文本）

> 调用工具 mcp__remote-devices__device_list_dir：D:\03_项目代码\PDE

> 工具 mcp__remote-devices__device_list_dir 返回：{"entries":[{"name":"__pycache__","type":"dir"},{"name":".git","type":"dir"},{"name":".gitignore","type":"file","size":102,"mtimeMs":1793181454560},{"name":"batch5_result.json","type":"file","size":7367,"mtimeMs":1792745729049},{"name":"batch5.py","type":"file","size":15862,"mtimeMs":1792745709049},{"name":"chafee_infante_result.json","type":"file","size":2672,"mtimeMs":1792743997922},{"name":"chafee_infante.py","type":"file","size":12411,"mtimeMs":1792743993931},{"name":"HANDOFF.md","type":"fil

…（已截断，原文 1612 字）

> 调用工具 mcp__remote-devices__device_bash：cat $HOME/mnt/PDE/HANDOFF.md

> 工具 mcp__remote-devices__device_bash 返回（失败）：Workspace unavailable. The isolated Linux environment on this device failed to start. Use device_stage_files / device_commit_files instead.

> 调用工具 enable__mcp__remote-devices__computer：task

> 工具 enable__mcp__remote-devices__computer 返回（失败）：Task underway: this call was accepted and you will carry it out as you continue the task right after this turn — nothing failed; this result is marked as an error only to end this turn's tool use. Do not retry this call or call other tools this turn, and add no further text: your turn ends here, and the user sees your progress as the work continues.

> 调用工具 ToolSearch：select:mcp__remote-devices__device_stage_files,mcp__remote-devices__device_list_dir

> 工具 tool 返回：（无文本）

> 调用工具 mcp__remote-devices__device_stage_files：paths

> 工具 tool 返回：{"staged":[{"devicePath":"D:\\03_项目代码\\PDE\\HANDOFF.md","stagedPath":"/mnt/user-data/uploads/PDE/HANDOFF.md","mtimeMs":1793181454608,"bytes":11724,"ok":true,"dispatched":true},{"devicePath":"D:\\03_项目代码\\PDE\\README.md","stagedPath":"/mnt/user-data/uploads/PDE/README.md","mtimeMs":1793181466633,"bytes":947,"ok":true,"dispatched":true}]}

> 调用工具 Read：/mnt/user-data/uploads/PDE/HANDOFF.md

> 工具 tool 返回：1	# PDE L1 解析解 / 反问题：交接文档
2	
3	最后更新：2026-10-28  
4	工作区：`D:\03_项目代码\PDE`  
5	Python：`D:\Miniconda3\envs\py312\python.exe`（NumPy 2.2.5）  
6	不要用 `D:\Miniconda3\python.exe`，那个环境没有 NumPy。
7	
8	本文档给下一任接手人。论文和 PPT **还没写**，按用户条件：实验结果都正确后再写。
9	
10	---
11	
12	## 1. 课题一句话
13	
14	已知 PDE，把很多候选函数列和系数都先设出来，用 **L1 范数** 把大部分系数压成 0，从而确定解析解由哪些项、哪些系数构成。
15	
16	用户原话：「先把这些项和系数全部设出来，然后 L1 norm 会让大部分的系数为 0。」
17	
18	方向后来扩成两边：
19	
20	| 方向 | 已知 | 未知 | 通过标准 |
21	| --- | --- | --- | --- |
22	| 正问题 | PDE + 初边值 | 解析解的项和系数 | L

…（已截断，原文 7701 字）

> 调用工具 Read：/mnt/user-data/uploads/PDE/README.md

> 工具 tool 返回：1	# PDE L1：已知方程反推解析解 / 已知解反推方程
2	
3	用手写函数列加 **L1 范数** 做稀疏选择。
4	
5	- 正问题：已知 PDE，不给解的内部采样，选出解析解的项和系数。
6	- 反问题：已知精确解采样，选出方程的项和系数。
7	
8	详细进度、约束、未通过原因和下一步见 **[HANDOFF.md](HANDOFF.md)**。
9	
10	## 状态（2026-10-28）
11	
12	正问题 5 个里 4 个从零选对；KdV 正问题支撑对，但用了真解附近起点。  
13	反问题 5 个里热方程、波动方程、平流扩散通过；**粘性 Burgers 和 KdV 双孤子稳定多一列 \(u_x\)**。  
14	论文和 PPT 按「全部正确再写」停着。
15	
16	## 怎么跑
17	
18	```bat
19	D:\Miniconda3\envs\py312\python.exe tier1_heat.py
20	```
21	
22	其余脚本同目录，成对写出 `*_result.json`。已有脚本不要改，重做写新文件。
23	
24	## 不包含
2

…（已截断，原文 558 字）

我读完了 `HANDOFF.md` 和 `README.md`。上次我问的“导师指的是正问题还是反问题”，文档里已经回答了：两个方向都在做，各 5 个方程。我先说现状，再说需要你定的事。

## 现状

- **正问题**（已知 PDE，选出解析解的项和系数）：热方程、粘性 Burgers、波动方程、Chafee–Infante 从零出发都选对了。KdV 选出的项是对的，但起点用的是真系数的 0.9 倍；从零出发会卡在相对残差 0.36，所以不算盲搜通过。
- **反问题**（已知解的采样，选出方程）：热方程、波动方程、平流扩散通过了。**粘性 Burgers 和 KdV 双孤子都会稳定地多留一列很小的 u_x。** 能试的修法基本都试过了，见文档第 8 节。
- 你定的规则是“全部正确再写论文”，所以论文和 PPT 都停着。文档最后把选择留给你：
  - **A**：继续修这两条反问题，要求仍用 L1、写新脚本、不靠放宽阈值删列。
  - **B**：带着这两条失败直接写论文，把“L1 在高度相关的候选库上会多留 u_x”作为一个结果写进去，并引用 PySINDy 里的同类现象。

## 关于 A，我看到一个之前没查过的点

`optimize_ux` 里，L1 在整条 λ 路径上最好的残差只有 2×10⁻⁶ 和 1.6×10⁻⁵。可是真方程本身的残差是 10⁻¹³ 到 10⁻¹⁶ 量级。这有两种可能：

1. **求解器没收敛。** 坐标下降在高度相关的列上收敛很慢，而且 λ 最小只降到 λ_max×10⁻⁶。如果是这个原因，多出的 u_x 是数值问题，不是 L1 本身的结论。
2. **L1 理论上就会选中 u_x。** LASSO 有一个可以直接算出来的判据：把真项的列和 u_x 列代入 KKT 条件，看 u_x 那一列是否满足（这叫不可表示条件）。如果不满足，任何 λ 下普通 L1 都会留下 u_x。这样就成了可以写进论文的定量解释，而不只是“和 PySINDy 现象一样”。

先算这个判据，再用高精度求解器（比如 LARS 同伦路径）重跑 L1，就能分清是哪一种。如果是第 2 种，就接着试文档里列的迭代重加权 L1：权重来自上一轮的 L1 系数。u_x 的 L1 系数只有约 6.5×10⁻⁶，权重会变得很大，下一轮很可能直接压成 0。整个过程都符合你的约束：仍然是 L1，不改旧脚本，不放宽 10⁻⁵ 的阈值。

## 需要你定

1. **走 A 还是 B？** 我建议先做 A 里的上述检查。工作量不大，而且无论结果如何，都能直接写进论文。
2. **在哪里跑？** 你电脑上的 Linux 运行环境这次没启动，我没法直接调用你的 `py312`。我可以把脚本复制到云端环境里跑，新脚本和结果 JSON 再写回 `D:\03_项目代码\PDE`，旧文件不动。这样可以吗？
3. KdV 正问题从零出发没通过，这次要不要一起处理？还是先只管两条反问题？

## 你 · 2026-10-28 10:29:46

1先做 A 里的上述检查
2在本机跑实验
3一起处理

## Claude · 2026-10-28 10:37:30

我先读现有的脚本，确认数据怎么生成、候选列怎么排，然后在新脚本里依次做三件事：算 u_x 那一列的 KKT 判据，用高精度 L1 重跑，以及做迭代重加权 L1。KdV 正问题也写新脚本处理。

> 调用工具 TaskCreate：inverse_burgers.py, redo_two.py, optimize_ux.py, tier3_kdv.py 及 JSON

> 工具 tool 返回：Task #1 created successfully: 读现有反问题与 KdV 正问题脚本

> 调用工具 TaskCreate：新脚本，Burgers 两条波与 KdV 双孤子

> 工具 tool 返回：Task #2 created successfully: 反问题诊断：不可表示条件 + 高精度 L1 + IRL1

> 调用工具 TaskCreate：新脚本，L1，不用真解附近起点

> 工具 tool 返回：Task #3 created successfully: KdV 正问题从零出发重做

> 调用工具 TaskCreate：本机运行，结果写回 D:\03_项目代码\PDE

> 工具 tool 返回：Task #4 created successfully: 在用户本机 py312 跑新脚本并核对 JSON

> 调用工具 mcp__remote-devices__device_stage_files：paths

> 工具 tool 返回：{"staged":[{"devicePath":"D:\\03_项目代码\\PDE\\inverse_burgers.py","stagedPath":"/mnt/user-data/uploads/PDE/inverse_burgers.py","mtimeMs":1792744843904,"bytes":10907,"ok":true,"dispatched":true},{"devicePath":"D:\\03_项目代码\\PDE\\redo_two.py","stagedPath":"/mnt/user-data/uploads/PDE/redo_two.py","mtimeMs":1792746402082,"bytes":8445,"ok":true,"dispatched":true},{"devicePath":"D:\\03_项目代码\\PDE\\optimize_ux.py","stagedPath":"/mnt/user-data/uploads/PDE/optimize_ux.py","mtimeMs":1792747828915,"bytes":6561

…（已截断，原文 1635 字）

> 调用工具 Read：/mnt/user-data/uploads/PDE/inverse_burgers.py

> 工具 tool 返回：1	# 反问题：粘性 Burgers。已知解的网格值，未知方程的项。
2	# 真方程 u_t = -u u_x + nu u_xx，nu=0.1。
3	# 一条行波还精确满足 u_t + c u_x = 0，比真方程更稀疏。
4	# 所以主实验叠两条不同波速的精确解。单条只作对照，写进结果。
5	# 导数用解析式，不是数值差分。求解是自适应 L1，不是序贯阈值。
6	
7	import json
8	from pathlib import Path
9	
10	import numpy as np
11	
12	NU = 0.1
13	DEFAULTS = {
14	    "pde": "u_t = -u u_x + nu u_xx",
15	    "nu": NU,
16	    "speeds": [0.5, 1.0],
17	    "front_x0": 0.0,
18	    "domain_x": [-1.0, 2.0],
19	    "domain_t": [0.0, 1.0],
20	    "n_x": 121,
21	    "n_t": 41,
22

…（已截断，原文 11277 字）

> 调用工具 Read：/mnt/user-data/uploads/PDE/redo_two.py

> 工具 tool 返回：1	# 重做两条反问题。batch5.py 不改。
2	# 平流扩散：宽度按 σ²(t)=σ0²+2νt 增长。
3	# KdV 双孤子：导数用 τ 的幂级数，不用差分。
4	# 真方程相对残差大于 1e-8 就停，不报支撑。
5	
6	import json
7	from pathlib import Path
8	
9	import numpy as np
10	
11	SUPPORT_TOL = 1e-5
12	C = 0.4
13	NU = 0.05
14	SIGMAS = (0.5, 0.9)
15	
16	
17	def soft(rho, thr):
18	    if rho > thr:
19	        return rho - thr
20	    if rho < -thr:
21	        return rho + thr
22	    return 0.0
23	
24	
25	def weighted_lasso(a, b, lam, w, beta0=None):
26	    n, p = a.shape
27	    beta = np.z

…（已截断，原文 9055 字）

> 调用工具 Read：/mnt/user-data/uploads/PDE/optimize_ux.py

> 工具 tool 返回：1	# 只重做两条反问题：粘性 Burgers 两条波、KdV 双孤子。
2	# inverse_burgers.py 和 redo_two.py 不改，数据和导数从那边调用。
3	# λ 路径仍是 40 个，从 lambda_max 几何降到 1e-6 倍。阈值 1e-5 只用来计数。
4	# 支撑：相对残差 <= 1e-8 的 λ 里，取 L1 目标最小的那个；
5	# 归一化系数绝对值 < 1e-3 的列丢掉。
6	
7	import json
8	from pathlib import Path
9	
10	import numpy as np
11	
12	import inverse_burgers
13	import redo_two
14	
15	RESID_GATE = 1e-8
16	DROP_TOL = 1e-3
17	COUNT_TOL = 1e-5
18	N_LAMBDA = 40
19	
20	
21	def select(a_raw, b, names):
22	    p = a_raw.shape[1]
23	    col = np.maximum(

…（已截断，原文 6789 字）

> 调用工具 Bash：mkdir -p /home/claude/pde && cp /mnt/user-data/uploads/PDE/*.py /home/claude/pde/ && cd /home/claude/pde && python -c "import numpy;print(numpy.__version__)"; python -c "import json;d=json.load(open('/mnt/user-data/uploads/PDE/optimize_ux_result.json'));print(json.dumps(d,ensure_ascii=False)[:2500])

…（已截断，原文 301 字）

> 工具 tool 返回：2.5.3
[{"library": ["u u_x  [真项, 系数 -1]", "u_xx  [真项, 系数 0.1]", "1", "u", "u^2", "u^3", "u_x", "u_xxx", "u^2 u_x", "u u_xx"], "n_rows": 9922, "lambda_max": 4206.69397480994, "n_lambda": 40, "n_lambda_residual_leq_1e-8": 0, "best_relative_residual_on_path": 2.0602875341754217e-06, "frequency_at_1e-5": {"u u_x  [真项, 系数 -1]": 39, "u_xx  [真项, 系数 0.1]": 33, "1": 0, "u": 0, "u^2": 0, "u^3": 0, "u_x": 37, "u_xxx": 0, "u^2 u_x": 0, "u u_xx": 0}, "rule": "相对残差 <= 1e-8 的 λ 里取 L1 目标最小；归一化 |系数| < 1e-3 的列丢掉。

…（已截断，原文 2243 字）

> 调用工具 Bash：cat > /tmp/claude-0/diag.py <<'EOF'
import numpy as np, sys
sys.path.insert(0,'/home/claude/pde')
import inverse_burgers as ib, redo_two as rt
def kdv():
    x=np.linspace(-10,12,90);t=np.linspace(0,1.2,16)
    xx,tt=np.meshgrid(x,t,indexing='xy')
    u,ut,ux,uxx,uxxx=rt.two_soliton_fields(xx.ravel(

…（已截断，原文 1251 字）

> 工具 tool 返回：burgers cond full 6438834336975447.0 cond S 1.5426554797268077
 irrep plain |A_j^T A_S G^-1 s|: [1.     1.     0.5981 0.2642 0.0975 0.0347 1.5217 0.8132 0.6305 0.2585]
 adaptive ratio |..|/w_j: [1.0000e+00 1.0000e+00 1.7200e-02 3.7000e-03 4.0000e-03 1.0000e-03
 1.1485e+00 1.3610e-01 5.4250e-01 5.5000e-03]
 ridge [-43.3984  40.7714  -1.2231   0.615    1.9953  -1.766  -32.1529   7.1325
 -38.2633  -0.7864]
 u_x dist to span(S) 0.16412025350399284
kdv cond full 36.02832302360792 cond S 5.08240615702

…（已截断，原文 867 字）

> 调用工具 Bash：cat >> /tmp/claude-0/diag.py <<'EOF'
print('--- null space')
names_b=ib.library_spec()
for name,(A0,b) in [('burgers',ib.design([0.5,1.0])),('kdv',kdv())]:
    col=np.linalg.norm(A0,axis=0);A=A0/col
    U,sv,Vt=np.linalg.svd(A,full_matrices=False)
    print(name,'sing vals',sv)
    v=Vt[-1]/col; v=v

…（已截断，原文 441 字）

> 工具 tool 返回：--- null space
burgers sing vals [2.21408665e+00 1.51928494e+00 1.27748651e+00 8.44893798e-01
 6.20617818e-01 2.08543594e-01 1.16418570e-01 3.95732707e-02
 6.06208326e-16 3.43864516e-16]
 smallest right sing vec (physical): [ 0.98286  -0.098857  0.        0.057132 -0.085698  0.028566 -0.316194
 -0.006667 -1.        0.2     ]
kdv sing vals [1.91563625 1.61934728 1.01887305 0.56744797 0.5616622  0.13506321
 0.10686014 0.05317029]
 smallest right sing vec (physical): [-1.00000e+00 -1.11324e-01  1.0

…（已截断，原文 575 字）

> 调用工具 Bash：python -c "import scipy;print(scipy.__version__)"; sed -n 1,80p tier3_kdv.py

> 工具 tool 返回：1.18.1
# 第三档：KdV 行波，无内部解采样，自适应 L1。
# 波速只放成离散列。真解是 c=4 的那一列，系数应为 1。
# 残差对系数是二次的。每一步线性化，子问题仍是加权 L1。
# 不是序贯阈值，也不是普通 Lasso。

import json
from pathlib import Path

import numpy as np

DEFAULTS = {
    "pde": "u_t + 6 u u_x + u_xxx = 0",
    "true_speed": 4.0,
    "domain_x": [-8.0, 12.0],
    "domain_t": [0.0, 1.0],
    "exact": "u = (c/2) sech^2(sqrt(c)/2 * (x - c t)), c=4，即 2 sech^2(x-4t)",
    "n_x_pde": 49,
    "n_t": 17,
    "n_x_ic": 97,
    "block_scale": "每个残差块除以 sqrt(行数)，PDE / 边界 / 初值三块总权重相同

…（已截断，原文 2564 字）

> 调用工具 Write：/home/claude/pde/lasso_exact.py

> 工具 tool 返回：File created successfully at: /home/claude/pde/lasso_exact.py (file state is current in your context — no need to Read it back)

> 调用工具 Bash：cat > /tmp/claude-0/t1.py <<'EOF'
import numpy as np, sys
sys.path.insert(0,'/home/claude/pde')
import redo_two as rt, inverse_burgers as ib
from lasso_exact import *
x=np.linspace(-10,12,90);t=np.linspace(0,1.2,16)
xx,tt=np.meshgrid(x,t,indexing='xy')
u,ut,ux,uxx,uxxx=rt.two_soliton_fields(xx.ravel

…（已截断，原文 1011 字）

> 工具 tool 返回：plain knots 5
  ratio 0.1 kkt 4.0e-15 res 1.64e-01 supp [0 4] ux -5.57e+01
  ratio 0.01 kkt 4.8e-14 res 1.30e-01 supp [0 4] ux -5.87e+01
  ratio 0.001 kkt 1.2e-12 res 1.40e-02 supp [0 1 4 5] ux -6.34e+00
  ratio 0.0001 kkt 1.4e-11 res 1.40e-03 supp [0 1 4 5] ux -6.34e-01
  ratio 1e-06 kkt 1.3e-09 res 1.40e-05 supp [0 1 4] ux -6.34e-03
  ratio 1e-09 kkt 1.4e-06 res 1.40e-08 supp [0 1] ux -6.34e-06
ridge knots 4
  ratio 0.1 kkt 2.0e-16 res 1.65e-01 supp [0 4] ux -5.20e+01
  ratio 0.01 kkt 1.9e-15

…（已截断，原文 1175 字）

> 调用工具 Bash：cat > /tmp/claude-0/t2.py <<'EOF'
import numpy as np, sys
sys.path.insert(0,'/home/claude/pde')
from lasso_exact import *
NU=0.1
def ch(x,t,ks,x0s):
    ks=np.array(ks)[:,None];x0s=np.array(x0s)[:,None]
    th=ks*(x[None]-x0s)+NU*ks**2*t[None]
    th-=th.max(0);e=np.exp(th);p=e/e.sum(0)
    m=[(p*ks

…（已截断，原文 2003 字）

> 工具 tool 返回：res 1.3790355093433986e-16
fd ut err 1.3299814582978797e-09
fd ux err 7.243095012654521e-10
0 [1.98 1.94 1.82 1.56 1.26 1.07 0.95 0.79 0.5  0.22 0.08 0.02 0.01 0.
 0.   0.   0.  ]
0.5 [2.   2.   1.99 1.98 1.94 1.82 1.55 1.2  0.87 0.52 0.23 0.08 0.02 0.01
 0.   0.   0.  ]
1 [2.   2.   2.   2.   2.   1.99 1.98 1.94 1.79 1.39 0.73 0.26 0.08 0.02
 0.01 0.   0.  ]
sv [2.36031998e+00 1.35955348e+00 1.22323039e+00 8.30115307e-01
 4.78660041e-01 3.10880847e-01 2.59686877e-01 4.10224389e-02
 1.54094300e-

…（已截断，原文 1070 字）

> 调用工具 Bash：cat > /tmp/claude-0/t3.py <<'EOF'
import numpy as np, sys
exec(open('/tmp/claude-0/t2.py').read().split("ks=[0,-5,-10]")[0])
def run(ks,x0s,x,t):
    xx,tt=np.meshgrid(x,t,indexing='xy');xx=xx.ravel();tt=tt.ravel()
    u,ut,ux,uxx,uxxx=ch(xx,tt,ks,x0s)
    A0=np.column_stack([u*ux,uxx,np.ones_like(u

…（已截断，原文 916 字）

> 工具 tool 返回：[0, -5, -10, -15] minsv 1.10e-03 cond 2.0e+03 irrep [1.   1.   0.6  0.33 0.16 0.08 1.39 0.65 0.69 0.48] res 1.4e-16
[0, -2.5, -5, -7.5, -10] minsv 1.40e-03 cond 1.8e+03 irrep [1.   1.   1.05 0.55 0.23 0.08 1.42 0.68 0.65 0.53] res 1.4e-16
[0, -4, -7, -10, -12] minsv 2.75e-03 cond 9.0e+02 irrep [1.   1.   1.16 0.62 0.37 0.24 1.39 0.49 0.73 0.3 ] res 1.4e-16

> 调用工具 Bash：cat > /tmp/claude-0/t4.py <<'EOF'
import numpy as np, sys
sys.path.insert(0,'/home/claude/pde')
exec(open('/tmp/claude-0/t3.py').read().split("x=np.linspace(-1,3,161)")[0])
import redo_two as rt
from lasso_exact import *
def kdvA():
    x=np.linspace(-10,12,90);t=np.linspace(0,1.2,16)
    xx,tt=np.m

…（已截断，原文 1415 字）

> 工具 tool 返回：burgers_ch
 plain freq [39 31  0  0  0  0 39  0  0  0]
 ols   freq [39 31  0  0  0  0  0  0  0  0]
  irl1 it 0 [0 1 6] [-223.992    54.6673    0.        0.        0.        0.       -2.3773
    0.        0.        0.    ]
  irl1 it 1 [0 1] [-226.231   54.766    0.       0.       0.       0.       0.       0.
    0.       0.   ]
  irl1 it 2 [0 1] [-226.2284   54.7582    0.        0.        0.        0.        0.
    0.        0.        0.    ]
  irl1 it 3 [0 1] [-226.2283   54.7581    0.        0

…（已截断，原文 2025 字）

> 调用工具 Write：/home/claude/pde/inverse_irl1.py

> 工具 tool 返回：File created successfully at: /home/claude/pde/inverse_irl1.py (file state is current in your context — no need to Read it back)

> 调用工具 Bash：time python inverse_irl1.py

> 工具 tool 返回：python 3.13.18 numpy 2.5.3
Cole-Hopf FD check 7.96e-08
== 反问题 KdV 双孤子（旧数据，不改）
  exact residual 2.25e-13  rank 8/8  min sv 5.32e-02
  irrepresentable (plain): {'1': 0.02, 'u': 0.001, 'u_x': 2.771, 'u_xx': 0.076, 'u^2': 0.036}
  plain L1 freq: {'u': 0, 'u_xxx': 26, '1': 0, 'u_x': 39, 'u_xx': 21, 'u^2': 0} KKT 1.3e-09
  IRL1 iters: [['u u_x  [真项, 系数 -6]', 'u_xxx  [真项, 系数 -1]', 'u_x', 'u_xx'], ['u u_x  [真项, 系数 -6]', 'u_xxx  [真项, 系数 -1]'], ['u u_x  [真项, 系数 -6]', 'u_xxx  [真项, 系数 -1]']]
  IRL1 freq: {'

…（已截断，原文 2261 字）

> 调用工具 Bash：python -c "
import json;d=json.load(open('inverse_irl1_result.json'))
for c in d['cases']:
  print(c['name'])
  for s in c['irl1_sensitivity']['settings']: print('  ',s['ratio'],s['eps'],s['n_iter'],s['passed'],[n.split('  ')[0] for n in s['support_names']])
"

> 工具 tool 返回：反问题 KdV 双孤子（旧数据，不改）
   0.1 1e-08 3 False ['u_x']
   0.1 1e-06 3 False ['u_x']
   0.1 0.0001 3 False ['u_x']
   0.1 0.01 3 False ['u_x']
   0.01 1e-08 3 False ['u_x']
   0.01 1e-06 3 False ['u_x']
   0.01 0.0001 3 False ['u_x']
   0.01 0.01 3 False ['u_x']
   0.001 1e-08 3 True ['u u_x', 'u_xxx']
   0.001 1e-06 3 True ['u u_x', 'u_xxx']
   0.001 0.0001 3 True ['u u_x', 'u_xxx']
   0.001 0.01 3 True ['u u_x', 'u_xxx']
   0.0001 1e-08 3 True ['u u_x', 'u_xxx']
   0.0001 1e-06 3 True ['u u_x', 'u_xx

…（已截断，原文 2361 字）

> 调用工具 Bash：python - <<'EOF'
import re
s=open('inverse_irl1.py').read()
s=s.replace("k.split()[0]: round(v, 3)","k.split('  ')[0]: round(v, 3)").replace("k.split()[0]: v","k.split('  ')[0]: v")
s=s.replace('("对照：粘性 Burgers 两条行波（旧数据，秩亏）"','("粘性 Burgers 两条行波（旧数据，不改；设计阵秩亏 2）"')
s=s.replace("# Burgers 旧数据（两条 tanh 行

…（已截断，原文 576 字）

> 工具 tool 返回：== 反问题 KdV 双孤子（旧数据，不改）
  plain L1 freq: {'u u_x': 35, 'u_xxx': 26, '1': 0, 'u': 0, 'u_x': 39, 'u_xx': 21, 'u^2': 0, 'u^2 u_x': 0} KKT 1.3e-09
  IRL1 freq: {'u u_x': 39, 'u_xxx': 31, '1': 0, 'u': 0, 'u_x': 0, 'u_xx': 0, 'u^2': 0, 'u^2 u_x': 0}
  support: ['u u_x  [真项, 系数 -6]', 'u_xxx  [真项, 系数 -1]'] PASSED
  sensitivity 12/20
== 反问题 粘性 Burgers Cole–Hopf 激波合并（新数据）
  plain L1 freq: {'u u_x': 39, 'u_xx': 31, '1': 0, 'u': 0, 'u^2': 0, 'u^3': 0, 'u_x': 39, 'u_xxx': 0, 'u^2 u_x': 0, 'u u_xx': 0} KKT 4.0

…（已截断，原文 11658 字）

> 调用工具 Bash：cat > /tmp/claude-0/k1.py <<'EOF'
import numpy as np, sys, time
sys.path.insert(0,'/home/claude/pde')
import tier3_kdv as K
K.DEFAULTS['cd_max_iter']=3000
names=K.library_spec();raw=K.pack();scale=K.column_scales(raw);data=K.apply_scale(raw,scale);p=len(names)
r0,_=K.residual_and_jacobian(data,np.ze

…（已截断，原文 1294 字）

> 工具 tool 返回：true normalized coef 3.4950782917704646
ICBC lstsq res 3.984857888966171e-10 [ 1.  0. -0. -0. -0.  0.  0.  0. -0.  0.]
 GN from IC 3.984857888966171e-10 [ 1.  0. -0. -0. -0.  0.  0.  0. -0.  0.]
 GN from 0 0.36055371653183793 [ 0.003 -0.678  1.174  0.     0.221  0.076 -0.009 -0.002  0.013 -0.046]
random 30 starts [0.     0.     0.     0.     0.     0.     0.     0.3606 0.3606 0.3606] 0.13353586196899414

> 调用工具 Bash：cat > /tmp/claude-0/k2.py <<'EOF'
import numpy as np, sys, time
sys.path.insert(0,'/home/claude/pde')
import tier3_kdv as K
from lasso_exact import lars_lasso_path, lasso_at
names=K.library_spec();raw=K.pack();scale=K.column_scales(raw);data=K.apply_scale(raw,scale);p=len(names)
r0,j0=K.residual_and

…（已截断，原文 2065 字）

> 工具 tool 返回：multistart best 1.5290482298793326e-14 n hit<1e-8 7 [ 1. -0. -0. -0. -0.  0.  0. -0. -0. -0.] 0.20134520530700684
1.0e+00 res 1.00e+00 pick 0 supp []
7.0e-01 res 7.43e-01 pick 0 supp [1]
4.9e-01 res 6.54e-01 pick 0 supp [1]
3.5e-01 res 6.26e-01 pick 0 supp [1]
2.4e-01 res 6.14e-01 pick 0 supp [1]
1.7e-01 res 5.86e-01 pick 0 supp [1 2 9]
1.2e-01 res 4.44e-01 pick 0 supp [1 2 9]
8.4e-02 res 4.21e-01 pick 0 supp [1 2 9]
5.9e-02 res 4.17e-01 pick 0 supp [2 5 9]
4.1e-02 res 4.11e-01 pick 0 supp [2 4

…（已截断，原文 1745 字）

> 调用工具 Bash：cat >> /tmp/claude-0/k2.py <<'EOF'
print('--- IRL1')
w=np.ones(p);prev=None
for it in range(10):
    lm=float(np.max(np.abs(j0.T@(-r0))/w))
    z=gn_l1(xs,lm*1e-3,w);sup=list(np.where(np.abs(z)>1e-5)[0]);print(it,sup,np.round(z/scale,4))
    w=1/(np.abs(z)+1e-8)
    if sup==prev:break
    prev=sup
l

…（已截断，原文 665 字）

> 工具 tool 返回：--- IRL1
0 [np.int64(0), np.int64(1), np.int64(9)] [9.982e-01 1.000e-04 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00
 0.000e+00 0.000e+00 2.300e-03]
1 [np.int64(0)] [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
2 [np.int64(0)] [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
[39  0  0  0  0  0  0  0  0  0] final res 1.4698642136004068e-07 0.808821439743042

> 调用工具 Write：/home/claude/pde/kdv_forward_blind.py

> 工具 tool 返回：File created successfully at: /home/claude/pde/kdv_forward_blind.py (file state is current in your context — no need to Read it back)

> 调用工具 Bash：time python kdv_forward_blind.py

> 工具 tool 返回：python 3.13.18 numpy 2.5.3
exact residual 7.12e-16  multistart hits 7/30
IRL1: [['2 sech^2(x-4t)  c=4  [真项]', '(c/2) sech^2  c=1', 'exp(-((x-4t)/2)^2)'], ['2 sech^2(x-4t)  c=4  [真项]'], ['2 sech^2(x-4t)  c=4  [真项]']]
frequency: {'2 sech^2(x-4t)': 39, '(c/2) sech^2': 0, '2 sech^2(x-2t)': 0, 'sech^2(x)': 0, '1': 0, 'x': 0, 'tanh(x-4t)': 0, 'exp(-((x-4t)/2)^2)': 0}
support: ['2 sech^2(x-4t)  c=4  [真项]'] PASSED
refit: {'2 sech^2(x-4t)  c=4  [真项]': 1.0} res 7.12e-16
seeds passed 10/10
ratio check: [(0

…（已截断，原文 584 字）

> 调用工具 TaskUpdate：status, taskId

> 工具 tool 返回：Updated task #1 status

> 调用工具 TaskUpdate：status, taskId

> 工具 tool 返回：Updated task #2 status

> 调用工具 TaskUpdate：status, taskId

> 工具 tool 返回：Updated task #3 status

> 调用工具 TaskUpdate：status, taskId

> 工具 tool 返回：Updated task #4 status

> 调用工具 Bash：mkdir -p /mnt/user-data/outputs && cp /home/claude/pde/{lasso_exact.py,inverse_irl1.py,kdv_forward_blind.py} /mnt/user-data/outputs/ && ls /mnt/user-data/outputs

> 工具 tool 返回：inverse_irl1.py
kdv_forward_blind.py
lasso_exact.py

> 调用工具 ToolSearch：select:mcp__remote-devices__device_commit_files,mcp__remote-devices__computer_request_access,mcp__remote-devices__computer_screenshot,mcp__remote-devices__computer_open_application,mcp__remote-devices__computer_type,mcp__remote-devices__computer_key,mcp__remote-devices__computer_left_click,mcp__remo

…（已截断，原文 325 字）

> 工具 tool 返回：（无文本）

> 调用工具 mcp__remote-devices__device_commit_files：files

> 工具 tool 返回：{"written":["D:\\03_项目代码\\PDE\\lasso_exact.py","D:\\03_项目代码\\PDE\\inverse_irl1.py","D:\\03_项目代码\\PDE\\kdv_forward_blind.py"],"rejected":[]}

Three new scripts are now in the project folder. Next I'll open a terminal on your computer and run them with `py312`.

> 调用工具 ToolSearch：select:mcp__remote-devices__computer_resolve_access

> 工具 tool 返回：（无文本）

> 调用工具 mcp__remote-devices__computer_resolve_access：apps

> 工具 tool 返回（失败）：Computer use is available on this device but not turned on yet. Call computer_request_access with just `device` and a short `reason` (no `apps`) to show the user an in-chat prompt to turn it on; don't call other computer_* tools until that succeeds.

> 调用工具 mcp__remote-devices__computer_request_access：reason

> 工具 tool 返回（失败）：The user doesn't want to proceed with this tool use. The tool use was rejected (eg. if it was a file edit, the new_string was NOT written to the file). To tell you how to proceed, the user said:
Computer use is now turned on on this device. Call computer_resolve_access with the app names you need, then computer_request_access with its returned `apps` entries.

> 调用工具 mcp__remote-devices__computer_resolve_access：apps

> 工具 tool 返回：{
  "apps": [],
  "alreadyGranted": [],
  "notInstalled": [
    {
      "requestedName": "Command Prompt",
      "didYouMean": [
        "Node.js command prompt",
        "Anaconda PowerShell Prompt"
      ]
    },
    {
      "requestedName": "Windows Terminal",
      "didYouMean": [
        "Windows 工具",
        "Windows 备份",
        "Windows Software Development Kit"
      ]
    }
  ],
  "screenshotFiltering": "mask",
  "guidance": "Some requested names are not installed (see notInstalled). R

…（已截断，原文 564 字）

> 调用工具 mcp__remote-devices__computer_resolve_access：apps

> 工具 tool 返回：{
  "apps": [
    {
      "bundleId": "c:\\windows\\system32\\windowspowershell\\v1.0\\powershell.exe",
      "displayName": "Anaconda PowerShell Prompt",
      "tier": "click",
      "isSentinel": true
    }
  ],
  "alreadyGranted": [],
  "willHide": [
    {
      "bundleId": "c:\\windows\\systemapps\\microsoftwindows.client.cbs_cw5n1h2txyewy\\textinputhost.exe",
      "displayName": "Textinputhost"
    },
    {
      "bundleId": "c:\\windows\\systemapps\\microsoft.lockapp_cw5n1h2txyewy\\lockap

…（已截断，原文 1744 字）

I can't run commands on your computer from here. The terminal can only be operated by clicking, not by typing commands. And the Linux shell that would let me run commands on your computer failed to start earlier.

So the new scripts are already in `D:\03_项目代码\PDE`, but **they haven't been run with your `py312` yet**. The results below are from a test run in the cloud environment (Python 3.13, NumPy 2.5.3), so treat them as preliminary. Please run these two lines in Anaconda PowerShell Prompt:

```bat
cd D:\03_项目代码\PDE
D:\Miniconda3\envs\py312\python.exe inverse_irl1.py
D:\Miniconda3\envs\py312\python.exe kdv_forward_blind.py
```

Each takes a few seconds and writes `inverse_irl1_result.json` and `kdv_forward_blind_result.json`. Once they're done, tell me and I'll read both JSONs and check them against the cloud results. None of the old scripts were changed.

## Preliminary results (cloud run)

**1. The extra u_x isn't caused by the solver failing to converge.** I re-ran plain L1 with an exact solver (LARS homotopy path). It satisfies the optimality conditions to 10⁻⁹ and gives the same result as before. With noise-free data, the residual and the u_x coefficient both shrink in proportion to λ. That's why, with λ going down only to 10⁻⁶ λ_max, the residual reaches only about 10⁻⁶. The old gate of 10⁻⁸ was never reachable on that path.

**2. The real cause is that the irrepresentable condition fails.** For the u_x column the value is 2.77 for KdV and 1.39–1.52 for Burgers; a column that should be zero needs this below 1. Plain L1 then picks u_x at every small enough λ, whatever solver is used. The other wrong columns are all below 1. This can go into the paper as a quantitative explanation.

**3. Something the handoff doesn't record.** In the old two-traveling-wave Burgers data, the candidate columns have rank only 8 out of 10. So the library contains two other exact equations besides the true one, both with more terms. To rule out a pass that only happens because of this degeneracy, I added a new full-rank dataset: an exact Cole–Hopf solution with 3 shocks catching up and merging.

**4. With iteratively reweighted L1, all three cases select the correct terms.**
- Round 0 is plain L1. After that each weight is w = 1/(|β|+ε), taken from the previous round's L1 coefficients. No ridge regression is used, and the 10⁻⁵ threshold is unchanged.
- KdV two-soliton (old data), Burgers Cole–Hopf (new data) and Burgers two waves (old data) all end with support equal to the true terms after 2 rounds. Refitted coefficients are −6/−1 and −1/0.1, accurate to about 10⁻¹⁵.
- **Limitation:** in round 0, λ must be small enough that the true terms are selected. In the sensitivity scan, KdV failed at λ ratios of 0.1 and 0.01, and Burgers failed at 0.1, regardless of ε; every smaller ratio passed. Overall that's 12/20 settings passing for KdV and 16/20 for each Burgers case. The ratio I used, 10⁻³, was chosen during exploration, so the paper needs to give a rule for choosing it.

**5. KdV forward problem: now passes from a blind start.** I replaced the starting point at 0.9 times the true coefficients with 30 random starts. These use only the column scales, not where the true solution is; 7 of the 30 fell into the right basin. Then iteratively reweighted L1 plus the λ path gives support of only the true term (39/40) with refitted coefficient 1. Ten different random seeds all passed, and λ ratios of 10⁻², 10⁻³ and 10⁻⁴ all passed. The old limitation remains: wave speeds are still set as discrete columns, not searched continuously.

So on the cloud run, all 5 forward problems and all 5 inverse problems pass. If your local run gives the same results, I'll update `HANDOFF.md` and then we can talk about starting the paper.

附件：lasso_exact.py、inverse_irl1.py、kdv_forward_blind.py
