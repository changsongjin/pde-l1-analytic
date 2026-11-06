# PDE 相关历史：Claude 截止状态

本文件夹是按用户要求整理的 PDE 相关历史快照。

## 截止点

当前进度**截断在** `claude-history/对话/02-L1 norm 稀疏求解 PDE.md` 的最后一条 Claude 消息：

- 时间：2026-10-28 10:37:30；
- Claude 已写出 `lasso_exact.py`、`inverse_irl1.py`、`kdv_forward_blind.py`；
- 云端预跑有初步结果；
- 但这三个新脚本还没有用本机 `D:\Miniconda3\envs\py312\python.exe` 复跑，因此不能把云端结果当成最终通过结果。

2026-10-29 之后的通用库、噪声、遗传算法、SGA-PDE、大库、论文和 PPT 内容都暂时不纳入这个截止状态。它们仍保留在原工作区，之后如需继续可以另开时间线。

## 文件

- `TIMELINE.md`：只写到上述截止点的时间线；
- `STATUS_AT_CLAUDE_HANDOFF.md`：截止点的实验状态、已确认内容和待办事项。

## 资料范围

纳入：

- `claude-history/对话/10-PDE解析解的L1稀疏方法.md`；
- `claude-history/对话/02-L1 norm 稀疏求解 PDE.md`；
- 截止点以前的 `README.md`、`HANDOFF.md`、脚本和结果 JSON。

排除：

- `claude-history/对话/03-讲义修改与编译.md`，它是另一个数学讲义项目；
- 截止点之后的所有实验和写作产物。

原始文件不改。本文件夹只保存整理后的阅读材料。
