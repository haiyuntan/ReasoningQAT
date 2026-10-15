# 准备阶段状态（2026-10-15）

- REAL-Prover 已固定官方 revision `fe76f68d9a88f342cb7b546307c20292fea9cced`。根据用户建议改用 HF Mirror；镜像 revision API 返回相同 SHA。下载完成仍需执行完整加载和 tokenizer 检查。
- 普通与沙箱外重试官方 GitHub 源码克隆均出现 `Proxy CONNECT aborted`；尚未核对官方评测数据及代码 SHA。
- 旧 Lean/Jixia/Interactive 路径失效，检索服务未验证。没有运行 BF16 Lean smoke 或正式训练。
- 已验证 Python AST、shell 语法、配置 JSON/TOML 及候选评测数据题数、唯一 ID、哈希。
- 旧 Qwen3 手册发现一处疑似明文 GitHub 凭证，未复制其内容；建议撤销或轮换。

其余审计结论见 legacy_selection.md。下载日志与进程记录见工作区 `runs/prepare_realprover_download/`，下载状态见 `research/manifests/model_revision.json`。
