# 2026-11-07 GitHub 研究检查点

对应研究分支：`codex/research-checkpoint-20261106`。

代码、冻结方案、主要诊断报告、正式运行索引和逐任务 JSON 保存在 Git 中。完整字节快照（含原始场、矩阵缓存、全部路径及大体积合并报告）放在同一仓库的 [Release 附件](https://github.com/changsongjin/pde-l1-analytic/releases/tag/research-checkpoint-20261107-batch3)。这是未完成研究的预发布检查点。

本次状态为 25/606 个正式数据组完成留出评价，150 条完整路径已合并认证。另有 wave 7%、seed 1000 的一条续算请求待合并，其他五条方法路径待执行。没有完整 20 种子噪声曲线或算法排名；后续进度以 `EXECUTION_STATUS.md`、`HANDOFF.md` 和结果 JSON 为准。

Git 中不保存 `.npz` 和单文件超过 2,000,000 字节的数值报告；这些文件完整保留在 Release ZIP 中。`GITHUB_CHECKPOINT_MANIFEST.json` 列出快照中各文件的 SHA256，`github_checkpoint_result.json` 记录 ZIP 的大小、SHA256 及检查结果。未修改原实验脚本与已有结果；原失败记录一并保存。无关的 `REVIEW_UNRELATED/` 和两份未归属本轮的 library-effect JSON 不在本次上传范围内。

恢复完整检查点时，先克隆上述研究分支，再下载 Release 中的 `pde-research-checkpoint-20261107.zip`。核对其 SHA256 与 `github_checkpoint_result.json` 一致，然后在仓库目录解压覆盖：

```powershell
Get-FileHash .\pde-research-checkpoint-20261107.zip -Algorithm SHA256
Expand-Archive -LiteralPath .\pde-research-checkpoint-20261107.zip -DestinationPath . -Force
```

ZIP 保留原始文件字节，能恢复哈希绑定的输入、模型和历史证据。不要重新运行会覆盖旧结果的生成器或审计脚本来替代恢复。只有 fit/test 网关确认某组模型已封存后，才允许生成该组尚未存在的测试数据。

实验运行环境为 Windows、Python 3.12.10。主要依赖见 `requirements-experiment.txt`，Clarabel 0.11.1 按 `requirements-research.txt` 安装到 `.research_deps/`。本机继续使用 `D:\Miniconda3\envs\py312\python.exe`。从现有运行索引继续的命令见 `EXECUTION_STATUS.md`，每批按用户要求不超过两小时。
