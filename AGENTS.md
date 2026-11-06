# 给后续智能体

这是 PDE 的 L1 稀疏选择目录。Claude 项目名是「PDE L1 norm」。

进度以 HANDOFF.md 为准，课题说明看 README.md。Claude 对话在 claude-history/对话/，算法说明在 claude-history/文档/。两者和 HANDOFF 冲突时，以 HANDOFF 和脚本的结果 JSON 为准。

## 不要改的约束

- 求解用 L1。序贯阈值不能当默认求解器，也不能用来把某项压成 0。
- 是否通过只看支撑，不看 L1 之后最小二乘读出的系数。
- 不放宽阈值去删掉反问题里很小的 \(u_x\)。现用阈值 \(10^{-5}\)。
- 已有脚本不改。重做写新脚本，并写出对应的 *_result.json。
- Python 用 D:\Miniconda3\envs\py312\python.exe。D:\Miniconda3\python.exe 没有 NumPy。
- 论文和 PPT 等实验结果按 HANDOFF 的标准过了再写。
