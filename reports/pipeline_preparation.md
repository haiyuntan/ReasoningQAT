# P0–P6 冒烟与正式实验准备

当前仅完成实验编排与门禁，**没有完成运行适配，不可启动正式实验**。
模型下载不是编写这些代码的前置条件。GPU/Lean 冒烟需要完整模型及恢复后的依赖。

## 使用

在工作区根目录运行：

```bash
python research/scripts/experiment_pipeline.py plan --mode smoke
python research/scripts/experiment_pipeline.py gate
python research/scripts/experiment_pipeline.py run --mode smoke --run P0
python research/scripts/experiment_pipeline.py run --mode formal --run P0
```

尚未绑定命令时，run 明确报 BLOCKED_ADAPTER_NOT_BOUND；plan 不执行任务。
默认结果位置 runs/realprover_pipeline/{smoke,formal}/P*/。
正式运行先验证全部 P0–P6 冒烟，再验证本方法的正式前置实验。
P4/P6 共用同一模式下的 P3；P5 只依赖 P0，独立训练，禁止从 P3 冒充 EfficientQAT。
每次重跑使用新的 --results 目录，避免覆盖记录。

## 适配顺序及验收

1. P0：恢复新可写 Lean 4.16 工作区、Jixia、Interactive 和检索；核对官方题集；审计固定分母及失败题保存；冻结统一协议。完整加载 BF16，验证两题原生搜索/Lean 校验。
2. 公共导出：Qwen2 tokenizer、模块数、W2g128 元数据检查；fake-quant 与 BF16 反量化导出的 logits parity；保存重载；禁止将其声称为 INT2 性能测试。
3. P1/P2：检查本地固定版本的 Qwen2/W2 支持，复用同一 NuminaMath128；P2 必须确实执行 AWQ 搜索和 clipping。W2 不支持记 BLOCKED，按用户要求阻止正式批次，不能静默豁免。
4. P3：适配 block QAT 的 Qwen2 模块及 REAL-Prover tokenizer；重建通用混合数据 token 缓存。正式4096/64条、2048 tokens，OpenThoughts80%/FineWeb20%。
5. P4：适配 Qwen2 E2E；验证 Teacher-RR、top20 KL、监督 mask、仅 scale 可训练与有限梯度。正式32768 OpenThoughts，global shuffle seed2，alpha=.2/beta=1。
6. P5：审查独立 EfficientQAT Block-AP/E2E-QP 实现，匹配有效 token 预算；不复用主方法 block checkpoint。
7. P6：与 P4 同 P3、同数据与 token 预算；普通 CE、关闭教师与 RR/KD；验证只有 scale 更新。

训练冒烟目标为两次优化更新，同时检查损失、梯度、参数更新、保存/重载、导出 parity 和两题 Lean 评测。
冒烟数据和预算缩小仅供连通性检查，不报告正式分数。所有方法正式评测固定244题、同 prompt/检索/搜索预算。
任何 state–tactic、Lean proof 或测试题均禁止用于训练。

## 运行适配器契约

在 configs/pipeline.json 的 commands.smoke/formal 中绑定经过审查的 argv 数组（不使用 shell）。
适配器读取 REALPROVER_RUN、REALPROVER_MODE、REALPROVER_FINGERPRINT、REALPROVER_OUTPUT。
运行成功后写 output/receipt.json，包含 run、mode、fingerprint、status=PASS、checks、
expected_problems、completed_problems、evidence。checks 必须逐项完成 plan 中的检查，不能仅根据退出码置 true。
evidence 为 {path: 绝对路径, sha256: 内容哈希} 数组，需包含完整每题记录和方法检查报告。
当前门禁核验回执、检查项和证据哈希；各科学检查的实际计算仍由待实现的适配器负责。

门禁将本地训练/评测代码、配置和 manifest 纳入指纹；变更后旧冒烟失效。
下载进度 manifest 不纳入指纹，模型身份由固定 revision 与适配器完整文件哈希检查保证。
环境版本、外部 Lean build、检索索引/服务版本也需要适配器写入证据并在运行时核对；当前尚未实现这部分验证。

## 验证边界

CPU 单元测试只验证门禁拒绝失效指纹、缺失实验、检查失败、分母不全和证据变更，不能代表 P0–P6 冒烟通过。
现有 Qwen3 的 repro_scripts/pipeline.py 不直接复用，避免错误模型、文字 benchmark 和论文分数阈值进入迁移实验。
