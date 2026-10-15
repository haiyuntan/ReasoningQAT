# REAL-Prover × ReasoningQAT 实验准备

当前处于准备阶段。审计结论和阻塞见 [legacy_selection.md](reports/legacy_selection.md)，机器可读证据见 [preparation_audit.json](manifests/preparation_audit.json)。

已实现只读旧资产审计、环境元数据盘点、候选测试集及两题 smoke 的复制与哈希、P0 候选 TOML、通用训练数据契约和官方 BF16 模型 revision/下载脚本。脚本不会修改 FedProver 或安装依赖。

在工作区根目录执行：

```bash
source research/scripts/env.sh
python research/scripts/audit_preparation.py
python research/scripts/prepare_local_assets.py
env/bootstrap/bin/python research/scripts/download_realprover.py --metadata-only
env/bootstrap/bin/python research/scripts/download_realprover.py
```

下载默认使用 `https://hf-mirror.com`，仍锁定官方仓库的同一 revision；可用 `--endpoint https://huggingface.co` 切回官方端点。镜像 API 已返回与官方一致的 SHA。后台下载进程记录及日志位于 `runs/prepare_realprover_download/`。只有模型 manifest 标记 `DOWNLOADED_REQUIRES_LOAD_CHECK` 才表示下载及本地哈希完成。

下载方式为单文件串行 HTTP Range 续传，固定保存到 `<文件名>.part`；同一时刻只有一个文件传输。重启同一命令会从该文件长度继续，完整文件检查大小及 LFS SHA256 后跳过。旧 Hub 临时文件按固定 revision 的 LFS hash 匹配并验证远程前缀后复用，每个文件最终仍需完整 SHA256 校验。当前日志为 `runs/prepare_realprover_download/serial_hf_mirror.log`。本地续传集成检查：`python research/tests/check_serial_resume.py`。

前两个脚本仅依赖 Python 标准库（审计需要 Python 3.11+ 的 tomllib）。下载脚本使用已有 bootstrap 环境的 huggingface_hub，固定首次解析到的官方 revision，下载后记录文件 SHA256。模型下载完成仍需检查 tokenizer、权重完整加载和架构。

`configs/p0_candidate.toml` 和 smoke 版本尚不可视为正式评测协议，不能直接启动旧运行时。完整执行顺序为：恢复并验证 Lean/检索 → BF16 smoke → 完整 P0 → GPTQ/AWQ → P3 block QAT → P4 Teacher-RR+KL → P5 EfficientQAT / P6 CE 消融。

训练仅使用 OpenThoughts/FineWeb；PTQ 使用固定 NuminaMath 128 条；Lean 文件仅用于评测。P4 与 P6 必须共用同一 P3 初始化。阶段状态和结果空值见 `configs/experiment_matrix.json`。
