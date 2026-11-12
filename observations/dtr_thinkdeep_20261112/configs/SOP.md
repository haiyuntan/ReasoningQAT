# REAL-Prover W2G128 × Deep-Thinking Tokens（DTR）迁移观察实验 SOP（交给 Code Agent）

> **版本**：`v1.0 · 2026-11-12`　**任务性质**：独立、只读模型的机制观察、Lean 验证和可选测试时排序实验；**不训练模型、不改线上证明搜索、不声称已复现 Think@n**。  
> **论文**：Wei-Lin Chen 等，**Think Deep, Not Just Long: Measuring LLM Reasoning Effort via Deep-Thinking Tokens**，ICML 2026, PMLR 306, pp. 16342–16360，<https://proceedings.mlr.press/v306/chen26di.html>，arXiv v2 <https://arxiv.org/pdf/2602.13517>。重点读 **§2 Eq.(1)–(6)、Algorithm 1、§3 Eq.(7)–(10)、§3.2、§4、Appendix A/B**；仅以论文原文为准。  
> **Code Agent 的首要目标**：在真实的 REAL-Prover BF16 与 W2G128 模型上正确计算**层间预测 JSD、settling depth、Deep-Thinking Ratio（DTR）**，检验它是否比 token 数、logprob、熵、自信度更能预测 **Lean tactic 有效性 / proof-step 成功**，以及 QAT 是否改变了该信号；所有主张必须经独立验证、严格统计与来源审计。  
> **执行边界**：本文件独立于前述 ReQAT 熵机制实验和 ABCD 观察；禁止它们共享可写目录、Python 环境、进程、样本选择结果；只读复用 P0–P6 checkpoint、v4 manifests、官方 REAL-Prover 实现、Lean/LeanSearch 资源。Lean **只用于评价、验证、收集评价状态**，绝不进入 QAT loss/reward/训练。  
> **重要的适用性限制**：REAL-Prover-v1 通常生成的是单步 **Lean tactic**，而不是论文中的数千 token CoT。本文中的「DTR」在本任务首先应被严谨地称作 **tactic-generation depth-revision statistic**；是否代表“推理努力”必须由证据检验，不能事先假定。

---

## 0. 总执行契约——先读、后审计、再运行

### 0.1 本轮明确禁止

1. 不修改/训练 P0–P6，禁止 optimizer、LoRA、量化参数更新、蒸馏、M0/M10/M20、量化部署 benchmark；不启动另一个同名生产服务。
2. 禁止在旧 `observations/abcd_20261111`、`observations/abcd_confirm_v4_20261111`、`observations/realprover_20261104`、`observation_reqat_entropy_lean_20261112` 或正式 `runs/` 下写结果、环境、cache、Lean 文件；checkpoint 只读。
3. 禁止修改或停止正式 miniF2F-test/FATE-M/LeanSearch 检索服务、已有 GPU0–2 评估及任何不属于本实验的进程；如果资源冲突，只暂停/终止**本实验子进程**并写 `WAITING_RESOURCE` receipt。
4. 禁止把 `valid` theorem 的效果用于新 QAT 更新；禁止读取 miniF2F-test 的成败结果进行选题、调阈值、挑 checkpoint。其 244 道只用于最终冻结后别的工作，不属于本任务。
5. 禁止写出“DTR 大=证明正确”“Deep Thinking Token=Lean 验证有效 token”“W2 DTR 下降=推理能力下降”“DTR 变化=因果伤害”“Think@n 节省 50% 成本”等未在本任务验证的句子。
6. 不得把**论文的 first-crossing settling depth**悄悄换成“之后所有层都已稳定”，不得用 top-k/词表截断 JSD 伪装论文 Eq.(2)，不得漏掉最终 Qwen RMSNorm 的影响。
7. 不得默认 Agent 拥有官方论文代码。未核验官方代码链接时，按**论文 Eq.(1)–(6) 独立实现**；第三方非官方实现只可作参考，代码版本/偏差须记录。

### 0.2 预注册研究问题和负结果条件

- **D1 / Task Validity**：同一 REAL-Prover checkpoint、同一 Lean proof state 下，模型生成的 tactic 的 DTR 是否与 `VALID_TRANSITION` 和 `GOALS_CLOSED` 正相关？**主研究模型 P4**。
- **D2 / Beyond Confidence**：DTR 是否比论文基线（length、mean logprob、negative PPL、negative entropy、Self-Certainty）更好地区分**同一 theorem/state 的成功与失败 tactic**？需检验**成对预测效用差**，而不只是单个相关系数显著。
- **D3 / Quantization Shift**：在**完全相同的 prompt + tactic token IDs**下，P4 W2 与 P0 BF16 的各层预测收敛/JSD/DTR 是否出现稳定而与 Lean 行为相关的变化？区分“数值扰动引起 DTR 增大”和“高 DTR 真正代表更好动作”。
- **D4 / Search Utility（仅当 D1/D2 通过）**：对于同一固定候选池，按 DTR 排序是否优于长度/置信度/随机排序，提高前 K 次 Lean 验证命中有效 tactic 或闭合目标的概率？**不是原论文 CoT majority voting 的严格复现**。
- **D5 / Proof Search（可选）**：只有前述效应稳健且官方搜索适配可行时，测试仅改变排序策略的封闭 budget ablation，评价 theorem-level success；保留正常 REAL-Prover 评估不受影响。

允许输出：`SUPPORTED`、`NOT_SUPPORTED`、`INCONCLUSIVE_LOW_POWER`、`BLOCKED_NUMERIC`、`BLOCKED_LEAN`、`BLOCKED_RESOURCE`。不能因结果不显著扩充假说或重选阈值。

### 0.3 首要预测及可能的反例

| 观察对象 | 如果 DTR 有任务价值 | 如果结果相反/不稳 | 论文潜在结论边界 |
|---|---|---|---|
| P4 同状态正确 vs 错误 tactic | 有效动作的 DTR（或排序胜率）较高 | DTR 无效或更高反而更差 | 最多是 tactic-level association |
| P4 DTR vs Self-Certainty | DTR 同状态 concordance 优于自信度 | DTR 不优于基线 | 无法主张 Think@n 在 Lean 有效 |
| P0 vs P4 同一已冻结 token 前缀 | 层间修订分布不同且与行为损伤有关 | 单纯差异却无 outcome 关系 | 差异本身不代表能力损害 |
| 长度很短 / DTR 离散退化 | 清楚报告覆盖与 tie | `DTR ∈{0,1}` 大量集中、缺乏方差 | 应判定不适用短 tactic，不强行挖显著性 |
| 真实搜索路径 | 每 theorem outcome 关联稳健 | 自选状态偏差、样本不足 | 禁止用根状态结果概括整条搜索链 |

---

## 1. 固定论文依据：什么是 DTR、什么不是

### 1.1 论文主定义（Eq. 1–6，严格保持）

设模型有 `L` 层，生成时第 `t` 个**下一 token 决策**对应每层隐状态 `h_{t,l}`。原文定义中间层 logit lens 和分布：

\[
 z_{t,l}=W_U h_{t,l},\qquad p_{t,l}=\operatorname{softmax}(z_{t,l}),\qquad l=1,\ldots,L.
\]

对**同一个模型、同一个位置**，计算每层与最终层分布的 Jensen–Shannon divergence（不是 token 熵）：

\[
 D_{t,l}=\operatorname{JSD}(p_{t,L}\Vert p_{t,l})
 =H\left(\frac{p_{t,L}+p_{t,l}}2\right)-\frac12H(p_{t,L})-\frac12H(p_{t,l}).
\]

\[
 \bar D_{t,l}=\min_{j\le l}D_{t,j},\qquad
 c_t=\min\{l\in\{1,\dots,L\}:\bar D_{t,l}\le g\}.
\]

\[
 \mathrm{Deep}_t=\mathbf1\big[c_t\ge\lceil\rho L\rceil\big],\qquad
 \operatorname{DTR}(S)=\frac1{|S|}\sum_{t\in S}\mathrm{Deep}_t.
\]

**论文默认** `g=0.5`、`ρ=0.85`。REAL-Prover / Qwen2.5-7B 有 28 decoder layers，因此深层门槛为 **`ceil(0.85 × 28)=24`，即第 24–28 层**。特殊 token / 换行 / EOS 是否进入 `S` 需在开发阶段冻结；主结果统计“实际生成的 tactic token”，并单独给出包含/剔除 EOS、空白、停止标志的敏感性分析。

**关键实现陷阱：**论文 Eq.(3) 用的是层前缀 `cummin`，因此 Eq.(4) 测量的是**第一次跨越**阈值的位置，不要求后续层永远在阈值下；如果出现先跨越后反弹，原公式仍按首次 crossing 计。必须实现原公式为主，另实现“尾部持续稳定”(suffix-max) 为**明确标注的次要诊断**；不得混淆定义。最后一层必须有 `D_{t,L}=0` 且 `c_t≤L`。

### 1.2 与前述 ReQAT 熵分析的根本不同

- ReQAT 的 token entropy：**同一层的 next-token 分布不确定性** `H(p_{t,L})`。
- DTR：**模型不同深度层的预测分布如何接近最终预测**（JSD、跨层）。
- DTR 不是 `H`, `−log p(gold)`, `NMSE`, `t-SNE`, “思考 token 个数”或“CoT 长度”。
- 同一个 token 完全可能**低熵、但 DTR=1**，或**高熵、DTR=0**。必须用二维分箱/偏相关而不是混为一类。

### 1.3 论文对照指标（§3）

对 tactic 生成序列 `S=(y_1,...,y_T)`，仅在真实**生成 token**之前的 prefix 上计算：

- `length = T`，`negative_length = -T`；
- `mean_logprob = (1/T) ∑ log p_{t,L}(y_t)`，**原始模型预测分布**，不与 top-p 之后的采样分布混写；
- `PPL=exp(-mean_logprob)`，论文比较 `-PPL`；
- `negative_entropy = -(1/T) ∑ H(p_{t,L})`；
- **Self-Certainty** 依原文 Eq.(10) 取 `mean_t KL(u||p_{t,L})`，其中 `u` 为**完整词表均匀分布**；**注意是 KL(u||p)，不是 KL(p||u)**。FP32 stable logsoftmax + 全词表求和；记录有限值检查。
- 必要时加 `min/median settling depth`、`late JSD area`、`head agreement depth` 作为探索性指标，不算论文 DTR 原公式。

参考论文测试的是 math/science **长 CoT**：每问题多次采样，比较 DTR 与答案正确率的相关性；`Think@n` 用短前缀选择较高 DTR 的候选继续生成。**在 Lean proof state 上，不能默认多数投票字符串有数学意义，也不能把验证器反馈偷偷放入无验证 reranker。**

### 1.4 论文实现中的开放问题：必须对齐，而非猜测

`reports/paper_alignment.md` 至少逐项记录：`paper version (v2, 2026-07-06)`、Eq.1–6、`g`、`rho`、JSD log 底、投影前 RMSNorm、full-vocab / top-k、special tokens、prefix length 与 paper Think@n 的原始 `n=48, η=50%, prefix=50`、本任务偏差和影响。

**必须显式核验两个非同义的“原论文移植口径”：**

1. **Qwen architecture-aware lens（本任务主测量）**：对于 block `l` 的 residual hidden state `h_l`，施加该 checkpoint 既有的最终 `model.norm`，然后用同一个 `lm_head` 投影；保证 `l=28` 的 logits 与 `AutoModelForCausalLM` 的**真实最终 logits**在允许误差内一致。层间预测此时被解释为“应用相同输出头的早退出近似”。
2. **Raw unembedding lens（论文公式的文字口径/敏感性分析）**：对每层（**包括 28 层**）直接用 `lm_head(h_l)` 投影，不加最终 RMSNorm；第 28 层此时未必等于原模型真实 next-token logits，必须说明这是一条 raw logit-lens 诊断曲线。**禁止**早层 raw logits 与末层 normalized logits 直接拼成一条声称同口径的曲线。

论文 Eq.(1) 写作 `W_U h`，但没有在式中明确 Qwen-style 最终 RMSNorm 的适配；不能称本任务 normalized 方案与作者代码“逐行相同”。如果能核实作者官方代码，则报告确切实现和 commit；第三方 GitHub 非官方库不得作为“作者原实现”的证据。原论文未明确 JSD 使用 base-2 或自然对数时：**主版固定 natural-log JSD（nats）**，次要并列给出 base-2 JSD（bits）及是否用相同字面 `g=0.5`，不能不标单位；对基底差异产生的门槛行为做 numerical sensitivity。自然 log JSD 的理论上界 `ln 2≈0.693`，因此在 nats 下扫描 `g=0.75` 会退化为全层最早 crossing，**不得把该退化误解释为“全部不需要深层思考”**。

### 1.5 前向获取 hidden states：必须保证下一 token 对齐

- 若 `input_ids=[prompt_ids, action_ids]`，对于生成的第 `t` 个 action token 使用 hidden-state **位置 `len(prompt_ids)+t−1`** 的层输出预测 `action_ids[t]`，不允许从该 token 已输入后的状态读 logits（label leakage）。
- 生成时在线的 next-token logits 和离线 teacher-forced replay 的 logits，在**完全相同 token IDs、position IDs、attention mask、cache 设置、dtype**下，必须逐位置验证数值一致；特别检查 KV-cache、rotary/position 编码、重复读取 `last_hidden_state`、batch/padding 敏感性。
- EOS/stop、空格、换行的生成 token 是否计入分母必须保存原始 token IDs 和清洗规则；**不可将 `tokenizer.encode(decoded_text)` 重新编码的结果直接当作生成 token ID**，因为空格和特殊符号可能改变边界。
- 每个模型各自用**自己的最终层分布**当该模型的 JSD 终点。跨 P0/P4 比较要先冻结共同的 prompt + token IDs，不能以 P0 末层分布直接代替 P4 末层分布定义 P4 DTR。
- 全词表（Qwen 大词表）FP32/logsumexp 分块计算；不得只用 top-20 KD 截断词表来近似 JSD 主指标。

---

## 2. 项目资产与现有实验：只读复用的锚点

### 2.1 固定历史资料

- 用户仓库：<https://github.com/haiyuntan/ReasoningQAT>。在服务器根目录**重新确认当前 git SHAs、branch 和 checkout**，不得因为参考历史分支而覆盖生产 checkout。
- P0–P6 的主要路径/sha 参考 `observations/abcd_confirm_v4_20261111/manifests/models.json`（v4 参考分支：`reports/realprover-abcd-confirm-v4-20261112`，提交 `090b34a53e445e499d7806263d3ba2cd406d4982`）。P5 为独立 EfficientQAT 初始化/训练路线；不能将 P4/P5 差异归因于同一个 Stage 2。
- v4 冻结 `manifests/frozen_cases.jsonl`、`selected_theorems.jsonl`、`src/engine.py`, `src/collect.py`, `manifests/lean_runtime.json`, `manifests/retrieval_identity.json`、`metrics/*_native_parity.json`；可以借鉴**代码思想**，新实验必须拷贝/重写到隔离目录并记录来源 SHA。
- 老 Observation 纠正熵：`reports/realprover-observations-20261104` 的 `all_layers_corrected`；只读参考 BF16/W2 数值稳定与 sampler audit，不能直接把 entropy 当作 DTR。
- ABCD v4 的结论：第 28 层 P4 功能误差优势未在 128 题确认，B 未证实额外 α-renaming 破坏，C prompt 删除响应对照显著但语义未证明，D 28 层方向结构描述性。**此背景不构成本任务 DTR 有效性的先验结论。**
- 同一仓库下另有 ReQAT 熵 SOP `REALProver_ReQAT_Entropy_Lean_Observation_SOP_v1_20261112.md`：**独立研究**，共享只读模型/公有 Lean 资源，但不可共享可写 cache、env 或偷用对方确认题预测结果。

### 2.2 Checkpoint 角色

| ID | 方法 | 本轮优先级 |
|---|---|---|
| `P0` | REAL-Prover-v1 BF16 | 主教师 / 同序列参考，必需 |
| `P4` | ReasoningQAT Stage1+Stage2 W2G128 | 主研究对象，必需 |
| `P3` | ReasoningQAT block-only W2G128 | 预注册次要，对照 Stage2 作用 |
| `P6` | ReasoningQAT CE-only Stage2 W2G128 | 预注册次要，对照 KD 作用 |
| `P5` | EfficientQAT Block-AP+E2E-QP W2G128 | 可用则独立次要基线 |
| `P1/P2` | GPTQ/AWQ W2G128 | 可选 PTQ 外部基线，不阻塞主要实验 |

**严禁**把 W2 的 BF16-dense 反量化前向当作 packed INT2 硬件部署；所测仅是权重量化数值对生成行为、层间概率结构和 Lean 成败的影响。模型必须保持 `eval()`、`requires_grad_(False)`，只启用 `torch.inference_mode()`。

### 2.3 服务器工作路径只作为查找线索

之前材料出现两种等价/非等价的绝对路径写法：

```
/HOME/sysu_qling/sysu_qling_3/HDD_POOL/tan/Reasoning
/XYFS01/HDD_POOL/sysu_qling/sysu_qling_3/tan/Reasoning
```

**先 `realpath` 验证真正 ROOT**。未找到时搜索挂载点，不创建指向生产工作区的可写替身软链；严禁对生产环境运行 `pip install`、`conda install`、`git checkout`、`kill` 等改变状态的命令。

### 2.4 真实系统中的生成/验证协议

- REAL-Prover `best_first.py`、`generator.py`、`prompt_manage.py`、`interactive.py` 作为**行为标准**。冻结 chat template、retrieval order、top-k 可见性、max tokens、stop 条件、temperature/top_p、生成数、搜索 budget。
- `v4` static candidates `rfl/simp/aesop/...` 只是**静态 loglikelihood 评分**，不等于模型实际生成的 tactic。**本轮 DTR 主要在 model-sampled tactic 上计算**；旧 candidates 可以用于无标签实现校验/诊断，但不作为 D1 主要样本。
- 如果拟分析完整 search tree，必须单独在 `miniF2F-valid` 独立评估环境中采集，不能从正式 test 搜索结果里抽成败案例；不改变 production test 的负载与计划。
- Lean verdict 分为 `INVALID_TACTIC`、`VALID_TRANSITION_GOALS_REMAIN`、`GOALS_CLOSED`、`UNKNOWN_TIMEOUT/TRANSPORT/ENV_ERROR`；只有真正 kernel-verified 的完整 proof 才能标 `PROOF_VERIFIED`。`run_tactic` 后 goals=0 与完整 kernel proof-check 必须分开记录。

---

## 3. 严格隔离的文件系统、环境和资源守护

### 3.1 唯一允许写入的工作区

```
$ROOT/observation_dtr_thinkdeep_20261112/
├── README.md                         # scope、进度、非结论
├── configs/
│   ├── protocol.json                  # 预注册参数，一次冻结
│   ├── split_registry.json             # 与 ReQAT 共存的样本排他锁
│   └── environment.lock
├── src/
│   ├── audit_assets.py
│   ├── build_cases.py
│   ├── lens.py
│   ├── jsd_dtr.py
│   ├── decode.py
│   ├── replay_identical.py
│   ├── lean_verify.py
│   ├── frozen_search_states.py
│   ├── analyze_task_validity.py
│   ├── analyze_quant_shift.py
│   ├── analyze_ranker.py
│   ├── statistics.py
│   ├── plots.py
│   ├── pipeline.py
│   ├── resource_guard.py
│   └── verify_results.py
├── tests/
│   ├── test_jsd_dtr.py
│   ├── test_next_token_alignment.py
│   ├── test_lens_parity.py
│   ├── test_sampling_replay.py
│   ├── test_lean_contract.py
│   ├── test_splits_and_identity.py
│   └── test_pipeline_safety.py
├── manifests/                       # immutable source SHA, frozen IDs, receipts
├── metrics/
│   ├── per_token/                     # 小而精；切勿全模型 logits 原样存储
│   ├── per_tactic/
│   ├── per_state/
│   ├── per_theorem/
│   └── summaries/
├── plots/
├── reports/
├── caches/                         # 全实验自有，中间数据不发布
├── envs/                           # 全实验自有的环境副本
├── runtime/lean/                   # 仅自有 Lean 生成文件
├── tmp/
└── logs/
```

与 `observation_reqat_entropy_lean_20261112` **不同根目录**，严禁互写，二者独立 `envs`, `caches`, `tmp`, `logs`, `runtime/lean`。源 checkpoint 和 retrieval 的索引只读；引入源文件先只读复制并记录 SHA，禁止热修生产目录。

### 3.2 独立环境要求

- 推荐完整复制**已隔离并经审计的** `ABCD v4` forward conda/prefix 到新 `envs/dtr_forward`，随后在该**独立**副本中做 `sys.prefix` / `torch` / `transformers` / `tokenizer` / `Lean` smoke。禁止只把基础 conda environment 的 `bin/python` 指向旧生产 env 来伪装隔离。
- 不要在 `PROVER` / `REQAT_PROVER` / 原 ReasoningQAT 或 ReQAT 熵 env 中安装依赖。若采用 `conda create --clone`/文件级复制，审计前缀内的 shebang、动态链接以及是否有指向生产可写缓存；复制时务必等待完全完成并校验 hash。
- 单独设置 `HF_HOME`, `TRANSFORMERS_CACHE`（若版本支持）、`XDG_CACHE_HOME`, `TORCH_HOME`, `MPLCONFIGDIR`, `TMPDIR`, `PYTHONPYCACHEPREFIX`, `TOKENIZERS_PARALLELISM`, `WANDB_DISABLED`。离线加载 `local_files_only=True`；不因缺包而自动修改生产。
- 独立 `runtime/lean` 放置只属于本次的文件；Mathlib 等依赖可以在确认不会写入后通过只读 mount/共享 assets 使用，禁止重新编译或覆盖生产 Lean cache；执行前对 `lake env lean`、interactive/jixia、test imports 做 smoke。

### 3.3 GPU/检索安全门槛

- 已有 v4 只使用**GPU3 特定 UUID**，现状必须重新以 `nvidia-smi --query-gpu=uuid,index,memory.total,memory.free` + `ps` + 现有 service health 核验；不能仅靠“GPU index 3”推断同一块卡。
- 严格限 1 个正在占用 GPU 的本实验模型进程；主实验按 `P0→P4→P3→P6→(P5)→(P1/P2)` 串行。GPU owner 可能变化，锁 `tmp/gpu.lock` 仅避免本实验互抢，不等于保护全部外部程序。
- v4 曾设置 GPU allocator cap 56GiB、GPU free reserve 8GiB、host available 64GiB、低速 retrieval health checks。可作为初始资源配置线索，**必须以当下服务负载实测调整**。不要因为 DTR 各层投影慢就提高占用到威胁 retrieval。
- 只对开发时冻结 proof state 进行少量 retrieval snapshots；生成/打分阶段读本实验自己的冻结快照，**不在每 token forward 调用检索**。按 v4 低频策略（上限 ≤0.2 查询/s）并记录 service identity、index SHA、latency、health。若正式 retrieval 影响明显立即停止本实验，保持线上运行。
- `CUDA_VISIBLE_DEVICES` 通过核实的 UUID/设备映射设置；`nvidia-smi` 监控显存、GPU 使用、内存、进程、OOM；发生 GPU OOM 可降低**单次 microbatch 或 token chunk**但不可更改科学样本，写新的工程 receipt。
- 严禁自动 kill 不属于本实验 PGID 的 PID；遇到其他实验同卡/同服务竞争，优先写 `WAITING_RESOURCE` 并给用户可恢复状态。

---

## 4. 数据协议——特别防止与 ReQAT SOP 撞“独立确认集”

### 4.1 现有 `miniF2F-valid` 的占用事实

官方 valid 为 244 道 theorem：旧 ABCD pilot 用过 32 道，ABCD v4 又独立使用 128 道，即**历史已研究 160 道，表面上剩余 84 道**。但是 ReQAT 熵 SOP 建议从其中预留 64 道做独立确认。**两份 SOP 不能各自把同一 64 道都称为未经使用的独立确认样本。**

G0 必须创建不可变的 `configs/split_registry.json`，读取 ReQAT 研究当前已经冻结的 theorem ID 与 state/seed 使用情况（若未运行则记 `NOT_FROZEN`），由 user-owned 全局登记或本实验提供双方数据锁定建议：

- `D-DEV`: 旧 160 道 miniF2F-valid，**用于开发工具与挑参数**；如果此前已被其他 Observation 研究，不再叫 independent holdout；不同生成 seed 不构成新 theorem。
- `D-CONFIRM`: **只从尚未被 DTR/ReQAT 任何开发阶段依据 outcome 选择过的 IDs** 中预注册抽取，且不与 ReQAT 已冻结确认 IDs 冲突；优先固定 20–40 道，不得为了凑 64 擅自挪用其他 SOP 的 64 道。
- 如 ReQAT 已冻结 64/84，DTR 只余最多 20 道有效独立 theorem：**明确标注低统计功效**，可以另找外部 Lean 题库/自建可 kernel 验证的新规则集进行确认；未完成外部确认前只出 `INCONCLUSIVE_LOW_POWER`，不可假装同分布 64 题。
- 若两个 SOP **尚未冻结**确认题，可建议一次性冻结明确互斥的 `42/42` 或 `48/36` 分配，但**改变 ReQAT 已下达的确认样本预算属于协议修订，不能由本 Agent 偷改**；必须记录批准的版本和签名。未获批准则默认以旧 ReQAT 优先占用为保守规则。
- `D-EXTERNAL`（可选）：独立 provenance 的 Lean 演绎题、规则生成题或可靠开源 formal theorem 子集，必须 theorem/source hash 去重、Lean 编译和数据许可证审计，不能把简易模板题冒充 miniF2F 难度，也不能声称 BF16 SFT 未见过语义相似题。
- `T-FINAL`: `miniF2F-test`，绝不参与这份观察 SOP 的样本选择、阈值调整、candidate 选择、Early rerank 参数调优。

以上每个 theorem 记录 `theorem_id, source_sha, canonical_statement_sha, historical_pilot_presence, reqat_reservation, DTR_dev_or_confirm, selection_seed, frozen_before_scoring, semantic_overlap_status`。禁止依据 P0/P4 证明结果选择困难题或有利题做 confirm。

### 4.2 proof-state 采集层次（不能只看根状态）

必须先定义清楚三类：

1. **ROOT**：同 miniF2F-valid theorem 的初始 Lean 证明状态（无需模型搜索即可稳定创建），可提供最小可复现评估。
2. **FROZEN-ROLLOUT**：在 `D-DEV` 上，使用预先指定的 P0 或均衡 P0/P4 搜索策略产生中间状态，先经过 Lean 验证并冻结 state/history/retrieval；注明**state distribution 与状态来源模型相关**，至少提供 ROOT_ONLY 主对照。
3. **RERANK-POOL**：对于相同 frozen state 和相同 retrieved-premise prompt，按照固定种子生成多条 tactic，冻结 pool 后再计算 DTR、Lean outcome、比较排序策略。不得为每个模型因成功率不同而挑不同状态做所谓“公平比较”。

**禁止复用 v4 的 8 个 static tactic 分数直接计算 DTR 当作在线探索样本**，因为那只含候选字符串且未真实从模型采样；静态候选允许作 lens/数值 smoke 或附录对照。

### 4.3 主 pilot 和确认的建议预算（资源不足时可按 Gate 缩小而不篡改结论）

| 阶段 | 状态与候选 | 模型 | 目的 |
|---|---|---|---|
| `S0` 数值 smoke | 4 旧开发 proof states，短序列及规则 token | P0/P4 | 校正 JSD、RMSNorm、KV parity、batch shape |
| `S1` 开发 pilot | **24–32 DEV theorem** × 至少 1 root + 1–3 中间状态（有则采）× 每状态 8 个 sample，固定 3 seeds 的嵌套策略或 8 个明确 seed | P0/P4 | 算 DTR 动态范围、Lean outcome、有效配对数、资源开销 |
| `S2` 开发比较 | 保留已冻结 states 与候选生成协议，增加样本/检查类别 | P3/P6/P5（若资源许可） | QAT 特异性与跨方法差异 |
| `S3` 独立确认 | 冻结的 `D-CONFIRM` 所有 eligible theorem；目标每题 ≥2 个状态/各状态 ≥8 个 P4 sample，真实数记录 | P0/P4 主；P3/P6 次 | 预注册主效应的复制检验 |
| `S4` 可选排序/搜索 | 只使用 frozen confirmation pools，不再生成新候选挑优 | P4 | Think@n-adapted 评价与真实成本核算 |

**最重要的统计 sample size 不是 token 数，而是有效的、含正负标签候选的独立 theorem/state cluster 数。** 某 state 全部 candidate 有效/全部无效时，不能用于同状态 concordance 的分子，但必须列在覆盖表中。若有效 theorem cluster 不足，优先写低功效报告，不可仅通过多生成同一题的 token 假装独立样本变多。

### 4.4 生成参数、候选身份与停止条件

- 真实生成协议以 REAL-Prover 官方及现有 formal eval config 为准。此前观察出现 `temperature=1.5, top_p=0.9`，**只能当线索**，在 G0 从当前正式配置取原值并冻结 `T, top_p, max_new_tokens, stop_sequences, sample seeds, context_length, use_retrieval`；任何替换作为**单独**次要方案而非混为主结果。
- 主观测不增加模型训练时没有使用过的“think harder”指令，也不主动要求模型写长 CoT；否则任务已变。
- 保存 `prompt_ids`, 原始 `generated_ids`, `stop_reason`, `used_token_ids`, decoding sampler parameters, per-step original logits hash / next-token logprob, output UTF-8 text, normalized tactic candidate text, sample_seed, search_state_origin, retrieval_sha`。
- 若 8 条样本全同/大多 `rfl`、或者无效样本比例太低，必须报告生成多样性和统计功效不足；不得悄悄加温度、扩展预算或只挑有分歧的 state。
- `GOALS_CLOSED` 是当前 state 的闭合，不等价于完整 theorem 的 kernel 通过；若 root state 并完整构建 theorem 且 Lean kernel 验证通过，单独标注 proof verified。

---

## 5. G0–G2：资产、数值与数学实现验收（任何科学分析之前）

### 5.1 G0：只读来源与版本锁

执行：

```text
realpath(ROOT)
git -C ROOT rev-parse HEAD / status --porcelain
nvidia-smi --query-gpu=index,uuid,memory.total,memory.free --format=csv
ps / GPU process inventory / retrieval health snapshot
model manifest + tokenizer/config + quantization adapter + native parity receipts
miniF2F-valid SHA + ReQAT reserved IDs + old v4 IDs + Lean/Mathlib SHA
```

写 `manifests/assets.json`, `manifests/production_processes_READONLY.json`, `manifests/model_identity.json`, `manifests/split_registry_receipt.json`, `reports/00_scope.md`；若模型路径、group size、tokenizer identity 或保留样本冲突未解决，阻塞对应阶段，不改旧资产。

### 5.2 G1：独立环境与单元测试（必须先通过）

`tests/test_jsd_dtr.py` 包含以下**人工可手算**的例子：

- `JSD(p,p)=0`、`JSD(p,q)=JSD(q,p)≥0`、finite；自然对数上界 `ln2`，比特上界 `1`。
- `L=28`, `g=.5`, `rho=.85`, cutoff=`24`；人工 `D[:,1:23]>.5, D24<.5` 应 `c=24, deep=1`；第 23 层首次小于 g 应 `deep=0`。
- early crossing + later re-divergence：论文 first crossing `c` **保持第一次**，suffix-stable 辅助指标则不同；测试防止误换定义。
- 末层 `D_L=0`、对 `T=0` 严格报 `INVALID_EMPTY_SEQUENCE`，不可 0/0；DTR 为 [0,1]，T<阈值时有整数离散性。
- 主 lens `norm(h_l)→lm_head` 第 28 层全词表 logits 与原 forward 原始 logits 在声明容差内一致；raw lens 同一投影口径全层一致但不声称与真实 final logits 具有完全相同语义。
- 全词表 JSD 与 CPU float64 参照逐位置比对；用随机 logits / 极端 one-hot / 均匀 softmax / 近零概率；决不在 BF16 上计算 `p*logp` 直接累加。
- 额外单测 `offset = len(prompt)+t−1`，强制在被预测 token 之前读隐藏状态，模型重放与同前缀 `forward` parity。
- `Self-Certainty = KL(uniform || p)` 方向测试；与 `KL(p||uniform)` 构造异例必须不同。
- full-vocab vs top-k 截断的 JSD 必须能构造明显不同案例；主计算不能依赖 top-k 近似。

必要的数值诊断：`min/max/quantiles JSD`, 层间 JSD 曲线图，所有方法的 deep token 覆盖率，`NaN/Inf`, mean step time, peak VRAM, 是否 g 太宽造成 `c≤3` 集中。`g=.5` 出现 DTR 大量零时应继续**如实报告**，并将 `g=.25` / `.75` 和 `rho=.80/.90/.95` 作为预注册次要敏感性，不得用挑出来的“最佳 g”宣称原论文已复现。

### 5.3 G2：在线生成 / 离线重放一致性

同一个 frozen state，先以 P4 生成真实 token IDs（固定 sampler/seed），然后在同一个模型上用 full-sequence teacher-forcing 复算所有 token 的：

`p_final(y_t)`, `H_final`, `Self-Certainty`, `JSD_{1..28}`, `c_t`, `Deep_t`。

必须报告 online stepwise vs batched teacher-forced difference 的 FP32 max/mean 和 deep-label disagreement；差异来自 KV cache、batch shape、attention mask、RMSNorm 或 dtype 时，修复**计算路径**并生成新的 protocol receipt，不得按结果选择更好看的路径。可从最安全的 batch1 / no padding / no KV 实现开始，再证明 cache 等价后启用。

**主输出必须保留每步全层 JSD 的小型 `[T,28]` 数组**（float32/float64 可压缩）和 token ID / classification，不用保存 `[T,28,|V|]` logits。只在设备上分块进行 full-vocab JSD，再流式回收，禁止 OOM 后悄悄改成 top-k。

`G2` 验收 Gate：核心数学测试全通过、P0/P4 最终 logits parity 通过、生成/replay 对齐通过、P0/P4 原模型 native probe 对齐、检索和 Lean 服务没有被影响。否则 `BLOCKED_NUMERIC`，**不进入科学统计**。

---

## 6. T1：DTR vs Lean Proof-Action Validity（主要研究实验）

### 6.1 单个冻结 state 的标准操作

对每个 `state_id`：

1. 在**评价用的 Lean** 打开已冻结 formal theorem/state，取得 `state_repr`；写完整 hash。
2. 调用受限的真实 retrieval 一次（或读取同一冻结检索 snapshot），使用**生产同款 prompt builder** 生成 prompt，保存编码 token IDS/sha。
3. 按固定 seed 用 P4（以及 P0）独立采样 `n=8` 个 tactic；禁止根据结果提前终止或挑出“好样本”。
4. 每个 generation 保存**原生 token ID**、stop reason、T、sampler，计算 architecture-aware DTR 与原文基线。
5. 对每条完整生成的 tactic 调用独立 Lean Interactive；严格分开 `INVALID_TACTIC / VALID_TRANSITION_GOALS_REMAIN / GOALS_CLOSED / UNKNOWN_TIMEOUT`。如果 `GOALS_CLOSED`，必要时进一步跑 full theorem kernel verification，不得将 `goals=0` 自动标成 kernel-verified。
6. 记录失败原因 (`parser/type/tactic/timeout/context/retrieval/...`) 及是否使用了检索 lemma；未知归未知，不能把 timeout 记为模型语法错误。
7. 该 theorem 的所有 states/samples 全部写入，而不是按成功情况筛选。

### 6.2 主分析单位及预注册估计量

主比较 **P4-only、同一个 frozen state 内**，避免 theorem 难度和跨模型不同分数尺度导致虚假相关。

对有一条 `VALID_TRANSITION` 与一条 `INVALID_TACTIC` 的 state，令 `S+` / `S−` 为正负候选，定义 predictor `m` 的成对辨别率：

\[
 C_s(m)=\frac{1}{|S^+||S^-|}\sum_{a\in S^+,\,b\in S^-}
 \left[\mathbf1(m_a>m_b)+\tfrac12\mathbf1(m_a=m_b)\right].
\]

再每个 theorem **等权平均**其具备正负候选的 states 的 `C_s`，最后跨 theorem 平均得到 `C_theorem(m)`；论文主预测分数**DTR**，主竞争 baseline 为 **Self-Certainty**，其余 logprob/entropy/length 次要。

**唯一预注册主检验**：

\[
 \Delta_{\mathrm{main}}=C_{\mathrm{theorem}}(\mathrm{DTR})-C_{\mathrm{theorem}}(\mathrm{SelfCertainty}).
\]

零假设 `Δ≤0`，主报告双侧 bootstrap 95% CI（如额外单侧需预先声明）。theorem-cluster bootstrap **10,000** 次，所有同 theorem states/samples 整组抽样；统计有效 theorem 数、有效 states、无正负对照 groups 的数量和 DTR ties 比例。若 bootstrap theorem 有效数 `<20` 或 POS/NEG 覆盖不足，则主效应标 `INCONCLUSIVE_LOW_POWER`，不声称无差异或已支持。

**主终点** `VALID_TRANSITION` 与 `INVALID_TACTIC`；`GOALS_CLOSED`、`kernel proof verified`、`best-of-n success` 为次要，不允许不同终点之中挑唯一显著者。主自变量使用 paper `(g=.5, rho=.85)` 的 architecture-aware full-vocab JSD，并单独报告 raw lens、bits、persistent settling、`g/rho` 探索性敏感性。

### 6.3 统计报告必须覆盖的重要对照

- 按 `theorem` 分层、`ROOT` vs `FROZEN-ROLLOUT` 分层；报告 P0/P4 的“样本来自哪个模型”的分布条件，不能混同成同一真实生成轨迹。
- 控制 `length`（短 tactic DTR 离散特别严重）、`mean logprob`、`entropy`、`Self-Certainty`、`tactic-first keyword`、`retrieval history/prompt len`；可以使用条件 logistic/固定效应模型为**补充**，不能在小样本中报告复杂模型的虚假显著性。
- `DTR=0/1` 高集中、EOS/missing、`<4` token 的比例单独列出；T<4 的短样本一律保留且列为单独 strata，不能在看到结果后只保留长 tactic。
- `score = larger-better` 定义一致，negative perplexity、negative entropy 的方向严格校验。
- `UNKNOWN_TIMEOUT/TRANSPORT` 主对照中排除标签但计入 coverage/失访与可能的不均衡性，须报告 worst/best-case sensitivity。
- 不得把 1000 token 的 1 个 theorem 变成 1000 个独立统计样本；token 热图是**描述性**，结论仍按 theorem cluster。

---

## 7. T2：P0 vs W2 的 Deep-Token Shift（区分噪声和思考）

### 7.1 必须固定相同序列做比较

对每个 frozen state 的实际生成序列，构建 `source_model = P0 / P4` 两个来源的 token pool；然后**相同的 prompt_ids + generated_ids** 分别送给 P0 和 P4（及 P3/P6）。得到：

- `DTR_P0(S | same tokens)`、`DTR_P4(S | same tokens)`；
- 每层 `JSD_P0(t,l)`、`JSD_P4(t,l)`，`ΔDTR = DTR_P4-DTR_P0`；
- `Δ settling_depth = c_t^P4 - c_t^P0`，`deep_label_flip`（P0 vs P4）；
- 每个模型自身的 `final_entropy`, `mean_logprob`, `token_top1_agreement`, `final logit TV/JSD`；
- 与**冻结 token 序列实际 Lean verdict**对齐：同一字符串在同一 Lean state 的验证结果是相同的，不能把 P0 重放它叫“P0 独立生成的成功”。

核心区分：`generation-origin effect`（两个模型采样了哪些 tactic）与 `model-evaluation effect`（同一 tactic 在不同权重下被赋予的深层收敛行为）。不能用两个不同序列的 `DTR_P4−DTR_P0` 宣称纯量化影响。

### 7.2 主要对照与诊断

- P4 相对 P0 的 `ΔDTR` 分布是否有稳定方向，按 theorem cluster、generated-origin、tactic token type 和 length 分层。
- P3/P4/P6 的差异是否与 Stage 2 对齐，P5 只做外部模型比较（不同训练 lineage）。
- **相同 token、不同模型：**检查深层 JSD 大是否仅由 quantization-induced prediction instability 导致，而非实际模型进行了更多有效“思考”。关键做法：报告 `ΔDTR` 与 `Lean valid` 的关系，同时报告 final-logit mismatch、prefix entropy、tactic 长度，不能单用整体均值解释。
- 若 W2 DTR 高于 BF16，但 tactic 有效率更低，**这将是“DTR 不可跨精度直接视为能力”重要反例**，应完整记录，无须为了复现论文正相关而消除。
- 28 层各层曲线只能按模型/合法 outcome 描述；不要由某一层交叉的图案推断因果层或直接发明功能子空间 loss。

### 7.3 自然生成 vs 共同序列必须分别做图

1. `Natural`: 模型自己生成的 `DTR` 分布与 Lean outcome（描述不同输出+不同权重）。
2. `Paired replay`: 对完全相同 token IDs 的 P0/P4 `ΔDTR` 和 `ΔJSD`（估计模型本身贡献）。
3. `Stratified`: 长度、tactic 功能类别、prefix entropy、root/rollout、生成来源的分层结果。

禁止将 natural 与 replay 的结果合并成一张单独的“证明能力随 DTR 提升”曲线。

---

## 8. T3：Think@n → Lean Tactic Ranking 的安全观察性迁移

> **只有 G2 + T1 pilot 完成、具有足够正负候选变化才实施**。本研究的任务形式与论文不同；本节叫 `Think@n-inspired tactic reranking`，不是论文 48 条长 CoT 的 majority vote 复现。

### 8.1 同一候选池的离线排序对照（不修改生产搜索）

在 frozen P4-generated `n=8`（或开发预注册独立 `n=16`）tactic pool 上，不使用任何 Lean label 作排序输入，分别用以下分数排序：

1. `DTR`（论文主定义）；
2. `Self-Certainty`；
3. `Mean logprob`；
4. `-Length`，以及 `+Length`；
5. `-Mean entropy`；
6. `Random`（固定 100 个 permutation seeds，不借助 lean verdict）；
7. 可选 `FINAL_LOGIT_SCORE`（如原 REAL-Prover candidate ranking 实际使用的 score），前提是已固定协议；
8. `Lean oracle` 只作为**后验可达上界**，绝不参与模型排序或阈值选择。

**指标：** `Top1_VALID`, `Top1_GOALS_CLOSED`, `AnyValid@K`, `AnyClosed@K`，K=`1,2,4`；如果模拟“按排名逐条执行”，必须使用**冻结候选与对应 Lean verdict**、相同验证调用预算，不得选择不同候选集合；去重策略预注册。`AnyValid@K` 不能冒充 theorem `pass@K`。

**“成本降低”不能照搬论文**：

- 离线排序需额外计算 28 层 full-vocab lens，成本包含原生成 token、DTR 投影 FLOPs/墙钟、Lean verification 调用数和 I/O；旧候选全部生成后再排序，本身**不能节省生成 token**。
- 仅若实际实现 prefix-based early rejection 的完全在线 pipeline，且按相同 seed、KV state、生成预算严格对照，才允许报告真实 measured inference cost difference；不允许按纸面 `n×prefix_tokens` 直接宣称节省百分比。
- 论文 `prefix=50` 在短 tactic 中常晚于 EOS，因此先报告 `P(T>=50)`，若覆盖不足本任务**不运行 50-token early reject 主结果**。可预注册 `prefix=4/8` 的 task-adapted 探索性设置，报告 eligible% 和早期 DTR 与 full DTR 的一致性，不能称作者原协议复现。

### 8.2 通过标准

只有 `DTR` 在固定候选池的至少一个**预注册主要 K** 优于随机及自信度，且 theorem-cluster CI 可靠并未因短序列 ties 饱和，才提出后续研究 `proof-search candidate ranking` 的可行性。即使 rerank 改善 `Top1_VALID`，也不得称提升了完整 miniF2F pass@64。

---

## 9. 可选 T4：隔离 Lean proof-search 的固定预算验证

只有 Rerank 明确有效且资源允许才做。另建 `runtime/search_valid/` 和完全独立的 search trace logger：

- 仅使用 miniF2F-valid 中 **冻结、未用于选算法参数** 的开发/确认子集（如果确认题已用于选 Rerank 则不称严格独立）；必要时使用新来源的独立 formal benchmark。
- 同 checkpoint P4、同 prompt、同 retrieval snapshot、同搜索预算、同 seed、同 sample pool / generation budget；仅改变选择候选的排序方式。
- 真正执行 `best_first` 或相同搜索协议中的 Lean tactic，按 theorem 列出 `PROOF_VERIFIED`, search steps, verifier calls, invalid tactics, timeout, tokens, projection FLOPs/latency；完整 kernel proof check 才计 theorem success。
- 任何原始正式搜索进程/评估调度都不能重启或覆盖；优先单卡资源守护。如果需要修改生产搜索源码才能运行，复制到新工作区并带 `source_diff`，不得写回 production。
- 没做写 `NOT_RUN`, 不能用 Rerank offline proxy 代替该章节数据。

---

## 10. 完整预注册、bootstrap、多重检验与避免选择偏差

### 10.1 冻结前必须落盘的参数

`configs/protocol.json` 至少包含：

```json
{
  "version": "DTR-REALPROVER-v1-20261112",
  "training": false,
  "models_primary": ["P0", "P4"],
  "models_secondary": ["P3", "P6", "P5"],
  "num_decoder_layers": 28,
  "primary_lens": "final_rmsnorm_then_unembedding_at_every_layer",
  "sensitivity_lens": "raw_unembedding_at_every_layer",
  "primary_jsd_log_base": "e_nats",
  "threshold_g": 0.5,
  "depth_fraction_rho": 0.85,
  "deep_layer_cutoff_1based": 24,
  "replay_full_vocab": true,
  "primary_outcome": "Lean_VALID_TRANSITION_vs_INVALID_TACTIC",
  "primary_estimand": "theorem_macro_within_state_concordance_DTR_minus_SelfCertainty_P4",
  "bootstrap_unit": "theorem_id",
  "bootstrap_replicates": 10000,
  "primary_alpha": 0.05,
  "tactic_sample_pool_size": 8,
  "secondary_k": [1, 2, 4],
  "pilot_dev_only": true,
  "allow_training": false,
  "allow_miniF2F_test_outcome_read": false,
  "source_reqat_sop": "readonly_reference_not_shared_runtime"
}
```

实际 sampler 和具体 theorem/seed ID 清单、停机规则必须在 `G0/G1` 从**当前实测配置和互斥登记**填入，再计算 config sha 与 frozen `protocol.sha256`。后续更改参数只允许新版本、新工作区或新子目录，不允许覆盖旧冻结协议。

### 10.2 主检验与统计严谨性

- 唯一主检验为 **P4 within-state theorem-macro** `Δ=C(DTR)−C(SelfCertainty)`。报告 point estimate、10k theorem-cluster paired bootstrap CI、有效 theorem/state 数及状态覆盖。`Δ>0` 且 CI 下界 `>0` 才标 `SUPPORTED`。如果 CI跨零，报告统计功效/置信区间及最小有意义效应，不将“不显著”说成“无作用”。
- P0、P3、P6、P5、raw lens、bits、persistent depth、`g/rho` 扫描、`Top1Closed` 与 `Top1Valid`、生成来源的所有结果都作为**预注册次要/探索性**，必须标多重检验和后验选定风险。
- 预先固定必要的非空分类群计数门槛，例如有效 theorem `≥20`，每 state 至少一正一负，`DTR` tie 比例 `<80%` 才讨论排序潜能；不满足门槛直接报告 `LOW_IDENTIFIABILITY`，不要为了达标删掉短 tactic。
- 不在中途看确认分数来改变 g、rho、lens、prompt、模型、候选生成预算或 baseline。
- 对 D1 若 Lean outcome 存在严重类别不平衡，先做只读描述；必要时换 `GOALS_CLOSED` 分析，**标为次要**而非主端点改名。
- 若同个 theorem 的多个状态来自 P0 rollout，必须标记 state selection bias；不能称真实 i.i.d. model distribution。
- DTR 依赖模型本身权重及输出长度，**跨模型 DTR 均值不直接代表两个模型能力排名**；比较模型须优先使用相同生成序列的 paired replay、各模型内部效果或明确校准比较。

### 10.3 数据文件最小 schema（作为代码契约）

`per_tactic.jsonl` 每条：

```json
{
  "theorem_id": "...",
  "state_id": "...",
  "state_origin": "ROOT|P0_ROLLOUT|P4_ROLLOUT",
  "model_id": "P4",
  "checkpoint_sha": "...",
  "prompt_sha": "...",
  "retrieval_sha": "...",
  "sample_seed": 2,
  "generation_origin": "P4",
  "generated_token_ids_sha": "...",
  "generated_token_count": 0,
  "lens": "ARCH_NORM",
  "jsd_unit": "nats",
  "g": 0.5,
  "rho": 0.85,
  "dtr": null,
  "mean_settling_depth": null,
  "mean_logprob": null,
  "negative_entropy": null,
  "self_certainty": null,
  "lean_status": "UNKNOWN",
  "goals_remaining": null,
  "proof_verified": null,
  "numeric_parity_status": "UNVERIFIED",
  "observation_status": "NOT_RUN"
}
```

初始 JSON 是 **schema 示例**，`0/null/UNKNOWN` 不是实验结果；必须在执行后填写真实值。`per_token` 保存 `token_id, t, entropy, sampled_logprob, per_layer_jsd[28], first_crossing_depth, persistent_depth_secondary, deep_flag, stop_class` 及输出 hash，不保存整段 full-vocab logits。`per_state` 保存同状态下全部 sample IDs、Lean 错误与验证时间；`per_theorem` 宏观估计，禁止只发布经过筛选的成功案例。

### 10.4 完整审计与安全用例（最低测试清单）

1. JSD math/self-certainty 方向、base、`g`、`ρ`、off-by-one、empty EOS、crossing反弹。
2. final norm + lm_head logits parity，28层 hook identity，P0/P4 `native_probe` parity。
3. online generation ↔ offline teacher-force token IDs/JSD parity；模型 own final reference 不得错用 teacher final。
4. payload 中相同 generator output 不因 tokenizer decode/re-encode 而偷换 token IDs；验证提示词 template 和 frozen retrieval SHA。
5. 不能访问 `miniF2F-test` 结果；valid split 必须与其他 ReQAT SOP confirmation ID 排他。
6. 模型文件哈希只读，`git status` source 不变化；整个实验输出 `realpath` 必须落在隔离 WORKDIR 内。
7. generator/Lean `TIMEOUT` 显式与 INVALID 分开，`goals=0` 不会自动变 `kernel_verified=true`。
8. 每条 score/replay results 有 protocol sha、checkpoint sha、prompt sha、token sha，断点重跑只跳过**签名匹配**的完成 receipt。
9. GPU/检索守护条件失败只终止自有 PGID，无未知 PID `kill`；并记录正式服务的恢复情况。
10. 所有模型用同一冻结 state/candidate pool 比较，只读正式 `runs/`。
11. 主要 bootstrap theorem cluster，不把 token/trajectory 作为独立 observation；zero POS/NEG state 分母明确。
12. P5 的训练来源独立，P0/P4 指标的模型间可比性严格标注。

---

## 11. G0–G9 分阶段执行清单（没有条件就交付部分结果，绝不虚构）

| Gate | 任务 | PASS 验收 | BLOCKED / FAIL CLOSED |
|---|---|---|---|
| `G0` | sources、权重、模型身份、正式服务、ReQAT reservation 审计 | models SHA + split registry + runtime owner + paper alignment 初版 | provenance 不清则先出 audit 报告 |
| `G1` | 新隔离环境、独立 Lean scratch、GPU守护、单测 | env sys.prefix 独立 + 全部安全/数学测试 | 环境指回生产或 GPU 被占用立即阻塞 |
| `G2` | JSD/DTR 核心实现、RMSNorm/raw lens、在线/离线 parity | 28层 final identity、same-token、full vocab、无数值错误 | 数值不一致不做科学报告 |
| `G3` | 24–32 DEV theorem pilot，少量真实 sample + Lean verdict | 每 state 完整候选/分母，报告 DTR 分布/ties/资源 | 多样性/label 覆盖不足报告 `LOW_IDENTIFIABILITY` |
| `G4` | DTR 与论文 confidence baselines 的开发比较、P0/P4 pair replay | 估计量、有效 theorem count、preliminary CI，协议锁定 | 不能据 pilot 显著选 confirm 有利题 |
| `G5` | 冻结与 ReQAT 不冲突的 `D-CONFIRM` IDs/seed/protocol | `frozen_cases`, `split_registry`, hashes，确认开始前锁定 | holdout 不足则明确低功效或找外部数据 |
| `G6` | P0/P4 独立确认 DTR × Lean + matched replay | 主统计、完备 per-token/tactic/theorem、失败案例 | 不改参数直到确认结束 |
| `G7` | P3/P6/P5 次要对照 + 可选固定 pool 排序 | 同状态同 seed、同模型/同输入身份 | 没资源就 `NOT_RUN`，不伪造基线 |
| `G8` | 仅有充分证据时 optional proof-search | 同等预算、kernel verified theorem success | 没做只能 `NOT_RUN`，不得推断 pass@64 |
| `G9` | 统计、图、校验、报告、独立分支推送 | SHA256SUMS + verify_results PASS + README/decision memo | 负结果原样交付，不得删掉失败组 |

严格执行顺序 `G0→G1→G2→G3→G4→G5→G6→G7→G8→G9`。`G7/G8` 可 skip，但是必须留阶段 receipt 和理由。**若当前没有足够确认 theorem 或 GPU 资源，只可完成到合规 Gate，不能回头改论文主张。**

---

## 12. 输出报告和图：让研究者无需登录服务器也能审查

### 12.1 强制交付文件

| 路径 | 必须回答的问题 |
|---|---|
| `reports/00_EXECUTION_STATUS.md` | 哪些 Gate done/blocked、运行命令、UTC/本地时间、失败阶段、资源影响 |
| `reports/01_PAPER_ALIGNMENT.md` | Eq1–6、Algorithm1、g/rho、RMSNorm、JSD log base、raw lens vs main 的精确偏差 |
| `reports/02_NUMERIC_PARITY.md` | full logits + KV + replay + P0/P4/模型SHA + JSD FP32/FP64 审计 |
| `reports/03_DATA_SPLITS_LEAKAGE.md` | 旧 160、剩余 84、ReQAT 64 保留冲突检查、训练语义泄漏未知 |
| `reports/04_DTR_TACTIC_VALIDITY.md` | DTR vs Lean outcome，主主比较/CI/分母/负例、短 tactic 适用性 |
| `reports/05_QUANTIZATION_SHIFT.md` | same-token paired P0/P4 全层 JSD 与 ΔDTR，生成来源区分 |
| `reports/06_TACTIC_RERANKING.md` | DTR/SC/length/logprob/random、同 pool、TopK Lean rate 和真实计算量 |
| `reports/07_PROOF_SEARCH.md` | 若未做为 `NOT_RUN`; 若做真实 success 与证明校验/预算 |
| `reports/08_RESULTS.zh.md` | 各主次指标和图，包括负结果、方差退化、超时、结论边界 |
| `reports/09_DECISION_MEMO.md` | DTR 有无独立预测力？值得加入下一步 QAT 方法/搜索策略吗？ |
| `manifests/experiment_manifest.json` | checkpoint/tokenizer/split/protocol/src SHA、种子、GPU/Lean/检索身份 |
| `metrics/summaries/primary.json` | theorem macro concordance、CI、分母、stats 程序签名 |
| `metrics/summaries/secondary.json` | 额外模型/参数/effort 分析，全部标 EXPLORATORY |
| `README.md`, `REPRODUCE.md`, `verify_results.py`, `SHA256SUMS.json` | 读者无需 GPU，可独立复算摘要、CI、source/split/protocol 哈希 |

### 12.2 必须画的图

1. **Fig-D1：`token_index × layer` full-vocab JSD 热图**。分别展示相同输入上 P0/P4 的正常/失败真实 tactic；附原文 first crossing、g=.5、late layer24–28 与 token boundaries。极少样例只做 illustrations，不用于 claim。
2. **Fig-D2：DTR / mean settling depth 的 ECDF 和 short-tactic ties**（P0/P4，同状态分类、EOS 规则）；如果都是0或1就明确写上。
3. **Fig-D3：DTR / Self-Certainty / length / logprob 的同状态 concordance**（theorem-cluster 95% CI，主比较 DTR−SC 高亮但不使用颜色美化显著性）。
4. **Fig-D4：5-bin DTR × Lean 有效率**，另画每 bin theorem 支持数；如几乎全为0导致无法5分箱则标 `BIN_COLLAPSE` 不硬画。
5. **Fig-D5：same-token P0/P4 的逐层 JSD 及 settling depth 差异**，按合法/非法 sample 分层。
6. **Fig-D6：可选 fixed-pool Top1/TopK 有效率 vs DTR/SC/logprob/random**，附 DTR 计算 overhead；不把 `AnyValid@K` 标为 theorem pass@K。
7. **Fig-D7：`DTR × final entropy` 二维关系**，按 tactic token 类别/position、模型生成来源分层；防止把 DTR 当作高/低熵的一种。

每张图必须能从 `metrics` 的机器文件直接生成，不能绘图时选择有利子样本或改变颜色/坐标营造机制成立假象。

### 12.3 上报仓库的安全 allowlist

建议新报告分支（若不存在）`reports/realprover-dtr-thinkdeep-20261112`；用户授权后仅推送：

`README/REPRODUCE/RESULTS/SOP`, `configs/protocol.json`, `src`, `tests`, `reports`, `manifests` 的可发布摘要、`metrics/summaries`、小型 token-level JSONL 或 parquet（需许可证审计）、`plots`, `verify_results.py`, `SHA256SUMS.json`。

禁止推送：模型权重、HF cache、envs、完整 full-vocab logits、生产日志、Lean 二进制、服务凭据、含隐私/机器特定密钥的 manifest。原始公开 theorem text 可引用许可证但别将服务器上未授权资料直接上传；可发布 hash 与归档定位，读者在授权环境复算。

---

## 13. 可直接交给 Code Agent 的最终指令和 Go/No-Go 判断

**Agent 必须先作出 `DTR_METHOD_COMPATIBLE` / `DTR_MEASUREMENT_DEGENERATE` / `BLOCKED_NUMERIC` 三选一的计量评估**：如果 Qwen 短 tactic + raw/normalized lens 导致大量 first-crossing 在第1–3层或所有 DTR=0，必须如实报告模型/任务适配限制，而不是改 g 追求图形上类似原文。

| 结果 | 可以支持的结论 | 下一步（**不属于本次训练任务**） |
|---|---|---|
| DTR 在**独立状态**上比 SC 更准确预测 Lean tactic validity，且短 tactic 适配 robust | 深度预测修订统计具有额外任务效用 | 可研究 proof-search reranking 或 QAT 中的深层分布保持，须另设同预算训练对照 |
| DTR 只在 P0 成立、P4 不成立 | W2 可能损坏 DTR 与 correctness 的关联 | 先分析 same-token shift 和误差混杂，不直接声称恢复 DTR 会改善证明 |
| P4 的 DTR 大但 tactic 更无效 | DTR 可被量化噪声抬高，不可跨精度当作“更多思考” | 暂不把提升 DTR 当作训练目标 |
| DTR 与 SC/length 没显著优势或大量饱和 | 原文长 CoT 指标在短 Lean tactic 上不一定有可移植预测性 | 停止该单步 DTR 方向；必要时探索真实多步 search path，而非挖特定层 |
| DTR 与 search outcome 相关，fixed-pool 排序也有效 | 有发展 Think@n-inspired proof search 的基础 | 新任务另行设计真实预算搜索实验 |
| 低功效、确认撞样本、数值 parity 未通过 | 不作机制定论 | 交付可复验失败记录、等待独立数据/资源，不偷换 protocol |

### 交付顺序

**真正运行实验**，不是只写计划或生成合成 `RESULTS`：

`G0 sources/splits → G1 isolated env/tests → G2 full-vocab JSD + RMSNorm parity → G3 DEV tactic sample + Lean → G4 pilot stats/paired replay → G5 confirm freeze → G6 independent confirmation → G7 cross-method/reranking → G8 optional formal search → G9 reports, artifact verification, safe publication`。

- 每个任务都要给出 `COMMAND / EXIT_CODE / INPUT_SHA / OUTPUT_SHA / STATUS` receipt。
- 哪个步骤没做就写 `NOT_RUN`，样本不足写 `INCONCLUSIVE_LOW_POWER`，失败要附最短复现命令；任何假说不支持都应原样报告。
- **不得将论文发表结果、以前 ABCD 的统计、先前 ReQAT 熵数据作为本轮 DTR 的实际发现**。
- **最终研究者需要的是**：一个能否在 REAL-Prover W2 上成立的可靠 DTR 测量及 Lean outcome 证据，而不是预先确定的创新点。

---

## 14. 参考文献 / 现场检查入口

1. Chen et al., *Think Deep, Not Just Long: Measuring LLM Reasoning Effort via Deep-Thinking Tokens*, **ICML 2026 / PMLR 306**：<https://proceedings.mlr.press/v306/chen26di.html>。全文 v2：<https://arxiv.org/pdf/2602.13517>；重点 §§2–4，Appendix A/B。论文 Eq.(3) 的 `cummin` / first-crossing 必须按原式执行。
2. 用户仓库：<https://github.com/haiyuntan/ReasoningQAT>，v4 固定依据 `reports/realprover-abcd-confirm-v4-20261112`、目录 `observations/abcd_confirm_v4_20261111`、提交 `090b34a53e445e499d7806263d3ba2cd406d4982`。`manifests/models.json`, `manifests/frozen_cases.jsonl`, `src/engine.py`, `src/collect.py`, `src/replay.py` 只读参考。
3. ReQAT 熵观察 SOP：`REALProver_ReQAT_Entropy_Lean_Observation_SOP_v1_20261112.md`（相互隔离，不覆盖其独立确认集）；已修正熵报告和 ABCD 旧结果只提供工程数值经验，不是 DTR 结果。
4. REAL-Prover 官方代码与数据：<https://github.com/frenzymath/REAL-Prover>，`FrenzyMath/state_tactic_pairs` 原始 SFT 分布已见数据不能声称 OOD；miniF2F-valid 是开发集，miniF2F-test 为最终集。
5. 第三方参考（**非作者官方复现，不能作为正确性依据**）：<https://github.com/mostafaelaraby/deep-thinking-tokens>。如查阅，明确记录它的 `apply_final_norm` 等与 Eq.(1) 的偏差，不要直接拿它输出替代数学/数值 Gate。

> **最关键的科学提醒**：长 CoT 的“Deep-Thinking Ratio”被移植到 Lean tactic 时，**有效性尚未建立**。本观察既要检验它是否与真实 Lean 成败有独立关联，也要证明它不只是 QAT 数值噪声/短序列离散性/最终 RMSNorm 的副产物。若结果为负，同样是完整、合理的观察实验结论；不要为了 CCF-A 论文故事将负结果改写成算法有效。
