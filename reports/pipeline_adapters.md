# P0–P6 流水线适配（2026-10-15）

方法运行适配已实现，已通过小型 Qwen2 检查。**真实 REAL-Prover P0–P6 冒烟尚未通过，正式实验保持关闭。**
本文取代 pipeline_preparation.md 中最初「尚未绑定命令」的状态描述。

| 实验 | 实现 | 同模式依赖 |
|---|---|---|
| P0 BF16 | 完整加载/重载与原生 Lean | 模型、Lean、检索 |
| P1 GPTQ W2g128 | GPTQModel Torch 后端，NuminaMath128；反量化导出 | P0 |
| P2 AWQ W2g128 | llm-awq scale search+clipping+伪量化，同 NuminaMath128 | P0 |
| P3 block-only | 原 ReasoningQAT 量化器、参数分组、block MSE | P0 |
| P4 RR+KL | 同 P3，RR alpha=.2、top20 KL beta=1、scale only | P0/P3 |
| P5 EfficientQAT | 独立量化器/初始化/Block-AP；原生 packing 与 scale-only CE E2E-QP | P0，不复用 P3 |
| P6 CE-QAT | 同 P3、同 Stage2 token/mask；关闭 Teacher/RR/KD | P0/P3 |

## 执行

在 Reasoning 工作区根目录使用新的运行目录：

```bash
env/bootstrap/bin/python research/scripts/inspect_pipeline_readiness.py
env/bootstrap/bin/python research/scripts/experiment_pipeline.py plan --mode smoke
env/bootstrap/bin/python research/scripts/experiment_pipeline.py campaign --mode smoke --results runs/realprover_campaign_v1
env/bootstrap/bin/python research/scripts/experiment_pipeline.py gate --results runs/realprover_campaign_v1
env/bootstrap/bin/python research/scripts/experiment_pipeline.py campaign --mode formal --results runs/realprover_campaign_v1
```

可用 run --mode smoke --run P0 单独执行。campaign 按 P0–P6 顺序运行，验证后跳过仍有效的完整回执。
失败的部分目录不会覆盖，修复后使用新的 --results。单 GPU 文件锁阻止本流水线并发。
解释器在 configs/runtime.json；允许 REALPROVER_{TRAIN,GPTQ,EVAL}_PYTHON 显式覆盖。
未安装/升级已有环境的包，未修改旧 FedProver 或原 ReasoningQAT 训练仓库。
上游算子小型隔离快照与原始/适配 SHA 位于 vendor/；REALPROVER/ 已建立独立 Git 基线。

## 门禁与产物

正式批次必须通过**全部 P0–P6** 冒烟，AWQ 没有自动豁免。
正式 P1–P6 另要求正式 P0，正式 P4/P6 要求正式 P3。
源代码/配置/小评测集/manifest、原始 parquet 元数据、核心 Python 包版本、
Lean 路径或二进制变化均纳入指纹。大 parquet 不在每次门禁全量哈希；
实际所选 token/mask SHA 与有效 token 数写入训练证据，方法之间检查共享预算。
检索服务的模型/索引版本仍需恢复服务时补充冻结清单，端口不能识别服务版本。

顺序：Lean/检索前检 → 原始模型 revision+完整哈希 → 训练/PTQ → 原生 checkpoint 重载 →
16提示 logits/greedy generation parity → 标准 HF 权重导出/重载 → 原生 Lean → 固定分母汇总 → PASS 回执。
训练检查有限 loss/gradient、实际更新、可训练参数；P4/P6 校验同一 P3。
W2 反量化为 BF16/FP16 仅用于精度比较，不声称 INT2 性能收益。
没有真实证据不写 PASS；失败检查会保留日志并返回非零退出码。

冒烟：Stage1 10 train/5 val×256、1epoch，Stage2 2样本/2更新；PTQ同128×512。
Lean固定2题、统一4 candidates/8 calls/1 retry/16 nodes/depth8；不作为正式分数。
正式：4096/64×2048，OpenThoughts80%/FineWeb20%；Stage2 32768 OpenThoughts、global shuffle seed2。
训练完全禁止 Lean/state–tactic。最后 assistant completion 监督、右截断后零监督样本明确失败，
不会静默丢弃样本。正式大规模 tokenization 尚需实际验证。

评测增加 vLLM seed、检索成功/失败遥测、完整证明 verifier 标记。
REPL 非零退出、空/异常响应不能算验证成功。新汇总固定题集分母，缺题保留失败，重试去重；
子进程非零退出仍尝试保存汇总，并阻止 PASS。

## 已执行检查

- 3项门禁测试；2项配置/固定分母/重复重试测试。
- 1项 verifier 契约测试，含有效响应、异常响应、非零退出、sorry、错误消息。
- 4项 Qwen2/损失测试：与固定源码 RR/top20 KL 数值和梯度一致；CE mask；
  7投影层、scale更新、原生 FP32 scale 重载保真、物化与重载；独立 packing。
- GPU随机单层 Qwen2：P3重建/validation、P4 RR+KL、P6 CE、P5独立 Block-AP/E2E-QP，均有更新。
- GPU随机单层 Qwen2：实际 GPTQ W2g128 量化/保存/重载/Torch forward；实际 AWQ W2g128 搜索/clipping/量化，均通过。

上述是组件检查，**不等于完整模型冒烟**。脚本位于 tests/。
旧 P3 驱动忽略 validation 且错误重复 attention mask；新驱动逐样本传 mask/rotary、实际计算 validation，
保留原量化器/MSE/参数分组/cosine，并统一可重载 fake-quant forward。
Stage2 显式 torch 循环替代 TRL，优化器/warmup 是记录在训练证据中的迁移工程选择。

## 环境恢复更新

- 模型已下载并完成完整 BF16 加载、生成及重载一致性检查。
- Lean4.16 工作区、Mathlib、Header、REPL、Jixia、Interactive 已恢复并编译。
  已执行有效证明/sorry、Jixia 声明解析、Interactive 错误 tactic/目标关闭测试。
- 检索服务已恢复，本地基座+LoRA、252444 条前提；启动校验资产哈希，评测核对服务身份。
  最终准备验证以 runs/realprover_preparation_v4/preparation.json 为准（按影响范围重测，保留原失败报告）。
  恢复过程和可重复命令见 [environment_restoration.md](environment_restoration.md)。

## 正式实验前仍需完成

- 协议已于 2026-10-15 审查并冻结：与固定官方 default.toml 语义一致（仅描述/本地路径不同），miniF2F 244 题逐字节一致且 ID 唯一；固定检索资产和 Lean 清单。证据见 manifests/evaluation_protocol.json，正式入口逐项校验冻结文件哈希。当前代码和调度配置仍须完整 P0–P6 冒烟通过。
- 完整模型的 PTQ/QAT、导出 parity 与原生 Lean 端到端尚未执行。

本地 Git 提交不代表远程推送成功；之前记录的 GitHub 网络问题仍是独立阻塞。

## nohup 自动衔接

使用 scripts/run_smoke_then_formal.py --results <新目录>，以 nohup 启动。控制器依次运行 smoke campaign、全七项回执门禁、formal campaign；任一步非零停止。控制状态在 <新目录>/control/status.json，分阶段日志 smoke.log/gate.log/formal.log。已有 partial 结果不会覆盖。控制器全局锁和每项 GPU 锁避免重复并发。协议及源码变化后重新冒烟，不修改运行中回执。

自动衔接测试 3 项通过：正常阶段顺序、smoke/gate/formal 各阶段失败时停止、协议未冻结时不启动任何阶段。另通过门禁 4 项、固定分母 2 项和 verifier 契约检查。协议冻结不是宣称复现论文分数；完整正式运行仍待所有冒烟实际通过。
