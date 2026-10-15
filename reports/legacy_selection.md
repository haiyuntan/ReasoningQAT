# REAL-Prover 迁移准备审计（2026-10-15）

状态：PREPARING，尚未通过 Gate A/B；没有新增 BF16 或量化实验分数。

本次以 `REALProver_ReasoningQAT.md` 为迁移执行依据，并阅读旧 Qwen3 手册的实验矩阵、数据规范与迁移 Gate，以及 ReasoningQAT README 和 repro_notes/README.md。旧 Qwen3 的结果不能代替此次 REAL-Prover 基线。

旧工程位于同级 `FedProver`，只读审计。Git HEAD 为 `c5dd5b75d5ec1e4a3a8b0fb2d1d2ca5246627703`；工作树状态见 `../manifests/preparation_audit.json`。旧代码尚未选为冻结评测器，也未复制到 eval_runtime。

候选入口为 `FedProver/experiment/run.py`，可接受 TOML、`--resume-from`、`--result-dir`，实际调用 BatchMainService 和 ProofParseManage。审计发现 223 份历史 TOML，清单记录各自模型路径及 result.txt/generated 是否存在。存在结果文件不代表完整评测或正确的模型身份。

例如 2026-08-05 的 `formal_stage2_bf16_baseline_minif2f_pass64_resume-minif2f_test-64-1024-0225/config.toml` 指向 `output/formal_4gpu/stage2_bf16_baseline/merged`；早期日志指向 `model/bubble_merged`。这些是历史派生模型，不能作为官方 REAL-Prover BF16 真值。目前没有证据支持选定一份日志作为官方权重的完整 P0。

候选测试数据已复制到 `datasets/eval/legacy_candidate/`，并检查 JSON 字段、ID 唯一性和 SHA256：miniF2F-test 244 题，ProofNet-test 186 题，FATE-M 141 题。miniF2F 前两题单独作为 smoke 输入，未按历史成功率挑题。尚需与固定 SHA 的官方数据逐字节比对，因此未命名为正式冻结数据。

硬件实测为一张 NVIDIA A800 80GB，审计时空闲。PROVER 和 REQAT_PROVER 的安装元数据均为 Python 3.10.20、torch 2.5.1+cu124、vLLM 0.6.6、transformers 4.57.6、accelerate 1.14.0；未发现 trl/flash-attn。这里只验证包元数据读取，尚未验证 torch/vLLM 实际导入与运行。没有安装或升级这两个环境。

已确认的阻塞：

- `models/REAL-Prover-v1-bf16` 尚无完整模型，不能以已有 Qwen2.5-Math-7B 替代。
- 旧工程 Lean/Jixia/Interactive 符号链接失效；conf/config.py 中的 `/HOME/.../project/` 依赖路径也不存在。本次有限深度检查未定位到可复用构建，不能据此声称整台服务器都不存在这些资产。
- Interactive 的加载逻辑可能调用 `lake build` 写依赖目录；恢复时需在新工作区构建，不能直接调用旧工程触发修改。
- 旧配置检索端点为 localhost:8080/retrieve_premises；服务尚未验证，索引约 4.1GB 在旧 model 目录，暂只引用路径。
- `ReasoningQAT/main_e2e_distill.py:553` 有 Qwen3ForCausalLM 专用导入；尚未通过 Qwen2 全层量化、scale 梯度、保存重载与导出 parity 检查。

已有 OpenThoughts 原始 parquet 120 个、FineWeb-Edu 一个 shard，以及 NuminaMath 和来源 manifest，可优先校验后复用。文件数量不代表内容哈希已重新验证。Qwen3 token cache 必须重新用 REAL-Prover tokenizer 生成。

下一步先固定官方源码和模型 revision，恢复 Lean 4.16.0/Jixia/Interactive 与检索服务，审计历史配置和官方数据差异；再制作最小 eval_runtime 副本，执行 BF16 两题 smoke。通过后冻结协议并跑 miniF2F 244 题完整 P0，之后再进入 PTQ 和 QAT。
