# REAL-Prover × ReasoningQAT：Lean-aware Calibration 与 A/B/C/D Observation 数据协议修订（v3）

> **Code Agent 执行任务书 / 研究协议修订**  
> 日期：2026-11-11  
> 参考：`REALProver_NextStep_Motivation_Observation_v2_20261102(1).md`、`REALProver_ABCD_Observation_SOP_20261111.md`。本文件优先覆盖以上两份文件中关于**数据来源、开发/测试划分、检索服务和 Lean 交互、校准混合实验**的相冲突表述；两份文件原有的环境隔离、单卡 Gate、模型只读审计、日志回执和统计规范继续有效。  
> **状态**：计划文件，不表示已训练 Lean-mixed QAT，也不表示已运行新的 miniF2F-valid 或获得额外证明成绩。

## 0. 研究定位与必须守住的两条边界

### 路线 M：ReasoningQAT Stage 1 Lean-aware calibration（第一个待验证的改进）

- 已有 P3/P4 基线的 Stage 1 使用 OpenThoughts3 / FineWeb-Edu = 80% / 20%，4096 train + 64 validation，序列长度 2048，2 epochs。P4 的 Stage 2 使用固定 OpenThoughts 数据、Teacher-RR + top-20 KD，仅更新 scale。
- **M 尚未执行**：提出在同样 Stage 1 总样本数、序列长度、训练轮数和量化配置下，使用少量 `FrenzyMath/state_tactic_pairs` 替换原通用校准样本，检验形式化证明能力是否改善；Stage 2 固定原协议，不增加额外 Lean 监督。
- 方法研究动机：混合领域校准已经是 ReasoningQAT 的基本设计，直接加入 Lean 数据首先只是**域适配消融**，不得预先宣称是充分独立的算法创新。A/B/C/D Observation 用来寻找进一步创新依据。
- **Lean 不参与 M 的优化闭环**：M 的训练数据是预先下载、整理、冻结的离线文本；绝不在 optimizer.step()、loss 计算、采样/重训过程中请求 Lean、在线检索或读入 miniF2F 验证结果。

### 路线 O：A/B/C/D 机制 Observation（不训练模型）

- **A**：Proof-Functional Subspace。功能方向误差是否优于 total NMSE 解释推理行为漂移。
- **B**：Quantization-Induced Symmetry Breaking。量化是否增加了 Lean 可验证的结构变换下的行为不稳定。
- **C**：Counterfactual Premise Influence Drift。量化是否改变模型对固定检索前提的响应结构。
- **D**：Cross-Layer Functional Error Dynamics。跨层新增量化误差与传播误差是否发生功能相关的放大/抵消。
- 首轮只读分析 P0/P3/P4/P6，扩展 P5，必要时 P1/P2。**M 的新权重尚不存在，不能在观察表格中预填数值。**
- 若随后训练 M，新权重可作为预先登记的 *external check*，检查 A/B/C/D 的诊断是否随 proof success 共同变化；但不能利用 test 重新设计机制。

### 严格隔离的三个过程

1. **离线训练**：Stage 1 M 使用 `state_tactic_pairs/train` 的选定文本 + 通用文本；Stage 2 原样不变。训练期间不调用 Lean/verifier/retrieval。
2. **机制开发和验证**：允许在**独立 evaluation-only 环境**中对 miniF2F-valid / 合成案例进行 Lean tactic 执行、证明检查、预先冻结的检索查询；所得标签只能用于评估和开发集决策，不进入 M 的训练数据或 QAT loss。
3. **最终证明评估**：miniF2F-test 244 题、冻结搜索预算和检索配置；其日志仅允许 protocol 冻结后的 post-hoc 解释，严禁作为 M 的 ratio 搜索、A 的子空间拟合、B 的规则筛选、C 的 premise 挑选或 D 的层选择依据。

## 1. 实际核对过的事实；Code Agent 须在本机重新验证

| 事项 | 当前证据 / 含义 | 不得误称 |
|---|---|---|
| `FrenzyMath/state_tactic_pairs` | Hugging Face dataset card：56,407 条 expert-iteration 数据，只有 `train` split；既有项目 audit 发现实际 Parquet 通常为 `input,tactic`，`input` 可能是含检索和证明历史的完整模板 | 不能说它提供官方独立 validation；也不能把整个 `input` 当作纯 Lean goal |
| BF16 REAL-Prover | 已经经过上游形式证明 SFT；公开 56k 数据属于其训练来源的一部分 | 不能把此数据的内部留出子集称为对 BF16 严格未见的数据 |
| P3/P4/P5/P6 | 项目 `adapters/training.py` 明确 Stage 1 通用校准、Stage 2 OpenThoughts；已运行的 QAT 没有加入 Lean state–tactic 数据 | 新 M 只是拟议实验，不是已完成基线 |
| Stage 1 loader | `vendor/reasoningqat/datautils_block.py` 对 OpenThoughts 只接受 >=2048-token 样本，并截取最前 2048 token；block reconstruction 输入为固定长度 | 不可直接把大量短 Lean prompt 填入 loader，静默变成 padding / 复制 / 截断损坏的样本 |
| miniF2F | REAL-Prover 官方仓库有 `Realprover/data/minif2f_valid.jsonl` 和 `minif2f_test.jsonl`，但它们主要是待证明 `id,formal_statement`，不是带 gold tactic 的 state–tactic pairs | 不能把 valid theorem statement 直接当完整 proof-state/action 数据 |
| 当前正式评估 | 旧报告运行的是 **miniF2F-test**、244 题；v10→v12 可续跑；具体完成状态以本机 `receipt.json` / `status.json` 为准 | 不能因“已评估”就称这些 case 是 valid |
| REAL-Prover runtime | `manager/thirdparty/generator.py` 的 `get_lean_tactics(...use_retrieval=True)` 先调用 LeanSearch-PS，随后模型生成；`best_first.py` 再用 `Interactive.run_tactic()` 与 Lean 交互 | 检索和 Lean 是两个不同组件；在线交互不是 QAT Stage 1 的一部分 |
| 先前 Observation | 已有 P0/P3/P4/P5/P6 的 all-28-layer 统计及 entropy 诊断 | 它们来自训练分布或既定采样，不能替代新 miniF2F-valid 的外部验证 |

需先锁定相关来源的 SHA、实际 prompt builder、当前本地 config、tokenizer 和 retrieval index 的身份；不得把 GitHub 默认分支当作 REAL-Prover 实验分支。

## 2. 正确的数据分层：四种互不冒充的样本集

| 标识 | 来源 | 用途 | 是否允许 M Stage 1 训练 | 证明“独立验证”的程度 |
|---|---|---|---|---|
| `S-CAL-TRAIN` | `state_tactic_pairs/train` 的冻结、按 state/source group 去重与防重合后的校准部分 | Lean-mixed calibration 的训练输入 | **允许**（仅离线文本） | 训练来源，不独立 |
| `S-OBS-DISC` | 同一官方数据集的另一个不重合 source/state 分组 | A/C/D 早期机制发现；B 的源模板初筛 | **禁止**（相对 M 留出） | 对新 M 校准训练留出，但 BF16 上游 SFT 可能见过；必须如实标记 `BF16_TRAIN_DISTRIBUTION` |
| `V-REAL` | **miniF2F-valid** 的固定 theorem、真实 Lean states、固定检索文本、冻结候选 tactic | A/B/C/D 的主要外部验证；M 的 ratio 开发选择 | **禁止** | 相对当前 QAT 训练独立；仍需排查 BF16 原 SFT 同源/语义重叠 |
| `V-RULE` | 与训练定理无关、独立 rule-based theorem families，Lean 验证变换 | B 主结果，A/C 的受控辅证 | **禁止** | 合成受控验证，不等同真实搜索分布 |
| `T-FINAL` | miniF2F-test 244 题和已完成正式搜索回执 | 锁定后的最终 theorem-level metric / post-hoc | **绝对禁止** | 任何 ratio/规则/子空间调整都不得使用它 |

**核心修订**：原 ABCD SOP 中 `DEV-CONFIRM` 若仍是从 56k 官方 train 内切分，必须统一改称 `S-OBS-DISC-internal-confirm`；不能称为独立 theorem 验证。真正独立的主要结果必须来自 `V-REAL` / `V-RULE`。

**泄漏与污染控制**：按 theorem ID、statement canonical form、exact state、normalized tactic 以及可验证的 source theorem group 做去重；无法判断语义同源时写 `SEMANTIC_OVERLAP_UNKNOWN`。对 `S-CAL-TRAIN` / `S-OBS-DISC` 采用 group-disjoint 切分，但不谎称可与 BF16 上游 SFT 去重。`V-REAL` 定理选择必须在读取任何模型成功/失败标签前完成，固定 ID manifest、seed 和 hash。

## 3. 关键工程：从 state–tactic pairs 制备合法的 Stage 1 Lean 校准样本

**这是 M 实验的 Gating Item。不能通过就不能启动 M 训练。**

### 3.1 先做数据分布审计（CPU only）

针对 pin revision 的每条 `input,tactic`：记录 `row_id/source_group`, 完整输入是否含历史 proof/in-context、retrieval 区、当前 STATE、原 gold tactic，BF16 tokenizer 的各部分 token 长度及是否可安全定位 span。审计该条是否与 `V-REAL` / `T-FINAL` theorem 显式重叠。已有 pilot 的模板分支 `native_incontext` 与 `deepseek_wrapped` 必须做一次规范映射，不可重复 ChatML 包装或破坏原 prompt 格式。

### 3.2 两种候选校准内容，不得混淆

- **`LEAN_PROMPT_ONLY`（建议主设置）**：使用评估兼容的真实 `proof history + current state + retrieved premises + TACTIC:` 的输入前缀，**不拼 gold tactic**。优点是尽量匹配真实推理时看到的上下文；代价是样本长度可能短。
- **`LEAN_PROMPT_PLUS_TACTIC`（受控消融）**：加入样本已有 gold tactic 后的离线 token 序列。它仍然是 Stage 1 block reconstruction 输入，而不是额外的 CE SFT；但会让量化重构过程直接观察老师行为文本，与 prompt-only 不同，必须另立对照。

任何方式都不得引入 miniF2F-valid/test 的 theorem/proof/Lean 输出到 Stage 1 训练输入。`input` 已含检索文本时一般不应再次请求检索服务重新插入新版检索结果。

### 3.3 2048-token 兼容策略（先验证再定稿）

当前 loader 只接受长度 >=2048 且截取**前**2048 tokens 的 OpenThoughts。直接迁移到 Lean 往往会产生：完全没有合格样本、只见历史上下文而截掉当前 STATE、重复同一短输入、pad token 被当作真实激活、或不合适的跨 theorem 拼接。

Agent 必须先报告 Lean 原始输入长度分位数、`state_span_retention`、`retrieval_span_retention`、`gold_tactic_span_retention`、短文本占比、可无损形成2048有效 token 的样本比例。然后**选定且锁定一个数据表示协议**：

1. 仅用真实且足够长的 Lean prompt 及明确的末尾窗口，记录是否覆盖 `STATE/TACTIC`；此方案会改变样本分布，报告筛选偏差。
2. 或者实现正确的 mask-aware variable-length block capture / reconstruction，对 attention mask、rotary/position_ids、pad 的损失聚合和二阶段 activation 缓存进行完整 parity 测试；Stage 1 所有方法需使用可比的有效 token budget。
3. 或者使用显式有边界的多记录 packing，但必须验证跨记录注意力不会污染校准语义；不能在普通 causal mask 下把相邻 theorem 随便拼接当作独立样本。若不可保证 segment isolation，则仅列为实验性对照，不作主设置。

**不要擅自改已有 P3/P4 基线加载逻辑**。所有新实现放独立源码分支、独立实验目录；如新协议改变了 loss token mask/归一化，要新增对应 zero-Lean 复现对照，否则与旧 P3 的差不只是 Lean 比例。

### 3.4 冻结预算、比例与因果对照

总 `4096` Stage1 train windows、`64` baseline validation windows、长度 `2048`、epochs=2、W2A16/g128、same optimizer/lr、同一 BF16 teacher、同量化模块集和随机种子。**校准比例按有效 token 数为主统计口径**；只有当所有 window 均有 2048 个有效 token 时，才能把 window 占比直接称为 token 占比。

建议第一轮配方（纯 proposal，不表示已运行）：

| 名称 | Lean share | 保持 OpenThoughts:FineWeb = 80:20 的剩余 share | 目的 |
|---|---:|---:|---|
| `M0` | 0% | 80%:20% | 必须有**同新版数据管道**重跑的零 Lean anchor；另外报告与历史 P3 的 parity |
| `M10` | 10% | 72%:18% | 首个主要检验 |
| `M20` | 20% | 64%:16% | 额外剂量敏感性；若资源不足可延后 |

在有限资源下先做 **CPU data integrity → 单卡 128/256-window smoke（非论文结果）→ 完整 M0 vs M10 的 Stage1 严格同预算比较**。只有开发集和机制结果值得继续时再考虑 M20。Stage 2 先不变；M 的 Stage 1 对比应与历史 P3（Stage 1）同级，若要研究最终 ReasoningQAT 必须对 M0/M10 **都**按同预算运行一模一样的 Stage 2，再比较 P4 风格结果。不能拿 `M10 Stage1-only` 去比较 `P4 Stage2` 然后归因于数据比例。

至少准备这些反混杂实验（可按资源分阶段）：

- `M10-RAW/TEMPLATE`：Lean 语法完整 prompt vs 经过等 token/模板匹配的非 Lean 数据，区分 Lean 信息和模板/长度效果。
- `M10-PROMPT` vs `M10-PROMPT+TACTIC`：区分推理输入分布校准与 gold tactic 文本暴露。
- 固定 Stage 2 是否使用 Lean 数据 = **否**。无在线 verifier/retrieval 参与训练。

报告 `total_train_effective_tokens`, `effective_token_count_by_source`, `window_count_by_source`, `duplicate_rate`, `source_group_count`, `state_retention_rate`, `prompt_template`, `method_hash`, `checkpoint_sha`。不能只报 `ratio=10%`。

## 4. miniF2F-valid 如何变成可做 A/B/C/D 的 state / action / premise 数据？

### 4.1 先确定已有有效日志

**只读**检查服务器上 `runs/`：各运行的 `data_path`, TOML, split, theorem_id、proof state、history、prompt、retrieve list / FAISS identity、candidates、tactic score、Lean transition/verification 是否真正落盘。如果只存在 `miniF2F-test` 日志，不得用它代替 valid，也不得从已完成 test 里挑 cases 开发。

### 4.2 若没有 V-REAL 日志，使用独立 evaluation-only 链路生成

1. 从官方 `minif2f_valid.jsonl` 固定例如 32 个 theorem（上限扩到64），**按 ID/hash 固定，先于任何成功率读取**。选择时只能使用静态 theorem 文字或复杂度分层、seed；不能按 P0/P4 解题成功优先选。
2. 在独立 Lean/Jixia（匹配原 kernel / Mathlib revision）、独立 frozen retrieval index 上，先获得每个 theorem 的 root proof state。可用**单一固定 P0 proposer**在预先规定预算内扩展少量合法后继 state（含未成功完整证明的状态），记录全部尝试及有效/无效过渡；仅抽取完全 frozen 的 state 集。注意这种 state 集对 P0 proposer 有选择偏差，明确标记 `P0_ROLLOUT_STATES`，同时保留 `ROOT_ONLY` 子集作不依赖模型的对照。
3. 对每个 state 查询固定 retrieval top-k（建议与评估配置一致，且必须保存检索返回内容的顺序、text、ID、index SHA、prompt builder SHA）。**LeanSearch-PS 是检索服务；Lean/Jixia 是验证执行器**。二者可能占用不同资源。
4. 为 state 构建固定候选 tactics：来自 frozen P0 proposer、已有被 Lean 验证的候选或事先静态规定的 tactic 模板。对每条 candidate 保存 `origin` / `is_lean_valid` / `log_prob_source`；不要伪称 miniF2F-valid 给出了 gold tactic。**所有 P0/P3/P4/P6 只在相同 state、retrieval、candidate action set、tokenizer 与 prompt 下 forward。**
5. 若要完整 proof-search metric，必须在独立评估环境对模型运行受控搜索，**只在 miniF2F-valid** 做开发；最初 A/B/C/D 仅需固定状态/动作和少量 Lean 验证，尽量不启动大规模搜索。
6. 所有 Lean 生成的 valid 结果只用于 validation/observation 与模型/ratio 开发选择，**不得回流到 M 的 Stage1/Stage2 训练、校准文本或 QAT 梯度**。这保留“Lean 只在评估中使用”的研究定位。

### 4.3 现有生产服务的资源与身份隔离

用户当前有可用检索和 Lean 服务，**不能据此假定可无影响地复用**：先审计其端口、GPU UUID、检索 index SHA、Lean toolchain、进程所有者、是否由 v12 正式评估占用。优先独立 Lean eval 进程 + 只读冻结检索缓存；不能无授权停止、重启、重配置或并发压测正式检索。若确实获准复用服务，使用极小批量、独立 request IDs、明确的限速/空闲窗口与缓存快照，并在运行前验证对正式评估无干扰。若无法安全复用，置 `WAITING_RESOURCE` 而不是强行占用 GPU。

## 5. 四项 Observation 的正确主要数据与可信证据

| Idea | Discovery（允许） | **主要独立验证（必须）** | 主要指标 | 特别防范 |
|---|---|---|---|---|
| **A 功能子空间** | `S-OBS-DISC` 的 teacher-forced reference tactic 梯度，8→32 case pilot | `V-REAL` frozen state/candidate 分数变化；`V-RULE` 辅助 | functional error vs total NMSE 对 action drift 的**held-out**解释力、random/PCA 对照 | 不能在同一 valid states 拟合 U 又报告 held-out 性能；不能用 gold tactic 幻觉 |
| **B 对称性** | `V-RULE` 预先定义且 Lean 验证的 α-renaming、受限括号等；`S-OBS-DISC` 只供模板调试 | 全新 `V-RULE` 留出 rule instances + 合法转化的 `V-REAL` states | Quantization-excess symmetry error = transform sensitivity(Q) − transform sensitivity(P0)；rule pass rate | 捕获规避、依赖假设顺序、prompt 历史/retrieval 的标识符不同步、tokenization 长度混杂 |
| **C 检索影响** | `S-OBS-DISC` 中可核实的既有 retrieval 字段 | `V-REAL` 的固定真实 LeanSearch results 与前提干预 | per-premise per-action influence drift + length/order/distractor controls | 删除前提 ≠ 语义等价；真实检索相关性不能由模型 self-report 推断；未保存 R 就不得虚构 |
| **D 误差动力学** | `S-OBS-DISC` 16–32 fixed prefixes | `V-REAL` 的 root/rollout fixed prefixes | exact propagated/injected error、cosine、functional projection、replay residual | 非线性 block 与 residual/position/mask 需 exact parity；不得用简单线性公式假称 exact |

### A 的特别要求

- 在 S-OBS-DISC 上拟合教师 Jacobian 子空间 U，并冻结 rank/层集合；V-REAL 上只做投影/预测指标，不重新估计 U；与 PCA、随机子空间和 total NMSE 做**相同输出与相同聚合尺度**比较。
- Valid 无官方 gold tactic 时，使用固定、可审计的 action candidate scoring；可另按 Lean 验证有效/无效 tactic 分层，但不能据有效性反向重新选择子空间。

### B 的 rule-based 建议（先少而精）

- **R1 主规则：捕获规避的局部变量 / 假设 α-renaming**。转换 state 与引用该名字的 tactic；使用 AST/范围分析，不得 `str.replace('x','y')`。例如：

```lean
theorem rule_original (a b : Nat) (h : a = b) : b = a := by
  exact h.symm

theorem rule_renamed (x y : Nat) (heq : x = y) : y = x := by
  exact heq.symm
```

- **R2 保守结构控制**：在经解析器确认的同一命题中插入/移除冗余括号，例如 `⊢ a = b` 与 `⊢ (a = b)`；Lean 编译验证，注意它可能是 trivial tokenizer perturbation，不能作为“深层语义对称性”的唯一证据。
- **R3 探索性**：无依赖上下文声明交换、经 Lean 验证的 definitional rewriting、等价 tactic 写法；这些不自动等价于 α-renaming，逐族独立统计，任何未验证转换 `UNKNOWN` 不进入 primary。
- **禁忌**：将 retrieval 删除/顺序置换当成 proof state 的语义等价；在完整 `input` 中用全局 regex 无差别替换变量名；只保留转换成功且输出有利的案例。
- 正式规则必须记录转换前后 theorem `Lean elaboration/kernel` 结果、proof/tactic 运行状态、工具版本 hash。**“两个 theorem 都可证明”本身并不证明命题等价**；等价关系需由转换定义（如 α-renaming）或额外形式化证明支持。

### C 的反事实检索要求

- 固定 `state + proof history + retrieval top-k`，先用 full retrieval，随后对某个 premise 做 remove/replace，添加 same-length distractor/order control；全部使用冻结的同一个 prompt builder 和相同的其他候选动作。
- 主要量是 `I_m(a,r)=score_m(a|s,R) - score_m(a|s,R_without_r)`；差值 `I_Q-I_P0`。将位置/长度变更作为单独对照，避免误归因于“知识因果依赖”。
- 可以另外做 `R=∅`（无检索）对照，但必须使用原版 builder 的准确模板；不允许混淆检索服务未返回、内容空列表、`None` 这三种情况。
- 引用是否真实有效，可让 Lean evaluation-only 执行候选 tactics；**Lean verdict 是测试/开发标签，不是 M 的训练 reward**。

### D 的局部误差 exact decomposition

对同一 token、attention mask、position ids、residual convention，在 block `l` 的完整映射 `f_T, f_Q` 下定义：

```
propagation = f_T(h_l^Q) - f_T(h_l^T)
injection   = f_Q(h_l^Q) - f_T(h_l^Q)
actual      = f_Q(h_l^Q) - f_T(h_l^T)
assert_close(actual, propagation + injection)
```

- 这是局部**代数精确**分解（在 hook/parity 和数值精度下），不同于一阶 `J_l e_l + η_l` 近似。
- 同时记录功能投影内外的方向/规模；只有 propagation 和 injection 为可比的同一表示空间，才计算 cosine 与 cancellation coefficient。
- 单 GPU 分时完成 BF16 teacher forward、P3/P4/P6 student forward、BF16 block replay；只缓存少量 layer/position tensors，避免全量 logits/activations；失败保留真实 `REPLAY_PARITY_BLOCKED`。

## 6. 评价指标与统计设计

### M 的主要结果（用于选择 0%/10% 等比例）

1. **Primary**：miniF2F-valid 的 frozen native proof search success rate（逐 theorem 统计，明确搜索预算和同一 retrieval 快照）。资源不足时，可用 valid frozen state 的 frozen candidate action likelihood/ranking/Lean-valid coverage 做**代理指标**，但不得用代理替代真实 proof success。
2. **Secondary**：V-REAL 中 Lean-valid candidate preference、Lean-valid rule consistency、retrieval influence stability、A 的 functional error、General/Math/Code 的代表性未见数据上的质量退化。
3. 每个 M 变体使用相同 Stage 1/Stage 2 token budgets，报告实际消耗与独立 checkpoint hash；同一 valid theorem 的结果做 paired comparisons，CI 按 theorem 聚类，避免把 64 次 search 的内部搜索步骤当成 64 个独立样本。
4. 多个 ratio/特征均使用 valid 作开发，**最终 test 只能在 ratio/method 冻结后运行**；若已有 test 曾对历史 P0–P6 使用，不应新试验反复利用这些分数挑方案。

### O 的主要结果

- A：held-out regression/R2 或 paired rank correlation；对 low-rank 和 total-NMSE 做相同复杂度/样本聚合对照。
- B：每组变换的 W2-excess instability / P0 instability 与 rule validity，提供 base theorem/state cluster bootstrap 95% CI。
- C：influence matrix 的 pairwise drift；固定 relevant/distractor groups、prompt length/position contrasts。
- D：exact replay residual 必须先 PASS，再讨论误差抵消系数与功能方向行为；无复核则不能作物理机制解释。
- 一个原始 state 的多个 tactic/token/规则/检索前提不是独立样本，所有主统计需按 theorem 或 base state 聚类；预先选定主要层和统计量，多层图作为探索性并标注 multiple comparisons。

## 7. GPU / 服务 / 环境资源规约

- 工作区和 conda 新建并独立：`$ROOT/observation_calibration_v3_20261111/`，**不能在旧 `research/`、`REALPROVER/`、`runs/`、生产 `PROVER/REQAT_PROVER` 里写入**；可继续独立使用 ABCD SOP 的 observations 工作区，但 M 必须单独新实验目录与权重目录。
- 当前生产 miniF2F 评估若还在运行，优先 CPU 审计与规则生成；有独占授权的 1 张 GPU 才启动模型前向或新校准。`CUDA_VISIBLE_DEVICES` 只暴露一个物理 UUID；单卡不足内存时先减 seq/sample，并在试验状态标记为 pilot，不擅自分片成多卡。
- 对现有 checkpoint、官方测试输入、`v10/v12` logs、检索索引与 Lean 原有源码始终只读。任何已有评估服务不允许重启/修改；必要时只读核验/复制已完整提交的证据。
- Lean-valid 交互是 evaluation-only：可复用独立服务，也可安全地另起 Lean eval 工程；绝不自动 fork 当前生产检索 GPU 服务。服务无法安全访问就输出 `WAITING_RETRIEVAL_RESOURCE`，先做不依赖检索的 A/B/D。
- 所有来源保存 code SHA, model SHA, dataset revision, prompt builder SHA, retrieval index SHA, Lean toolchain SHA, seed, input_ids SHA, split, GPU UUID, VRAM peak, timestamp, run status。

## 8. Code Agent 应交付的目录和文件

```text
observation_calibration_v3_20261111/
  reports/
    00_scope_and_source_audit.md
    01_data_split_and_leakage.md
    02_calibration_loader_2048_feasibility.md
    03_valid_search_log_inventory.md
    04_lean_retrieval_toolchain_and_snapshot.md
    05_M_stage1_mixture_pilot.md
    06_ABCD_external_validation.md
    07_decision_memo.md
  manifests/
    datasets_and_splits.json
    S_cal_train.jsonl
    S_obs_disc.jsonl
    V_real_theorems.jsonl
    V_real_states.jsonl
    V_rule_pairs.jsonl
    T_final_readonly_index.json
    retriever_snapshot.json
    lean_runtime_receipts.jsonl
    frozen_protocol.sha256
  configs/
    data_protocol_v3.yaml
    calibration_mix_M0.yaml
    calibration_mix_M10.yaml
    calibration_mix_M20.yaml
  src/
    audit_stage1_loader.py
    build_calibration_mix.py
    audit_search_log_splits.py
    collect_valid_states.py
    freeze_retrieval_prompts.py
    generate_and_validate_rules.py
    score_frozen_candidates.py
    evaluate_observation_abcd.py
  tests/
    test_no_test_leakage.py
    test_group_disjoint.py
    test_prompt_builder_parity.py
    test_window_or_mask_correctness.py
    test_mixture_token_ratios.py
    test_no_lean_retrieval_in_training.py
    test_same_candidate_across_models.py
    test_retrieval_content_identity.py
    test_rule_elaboration_receipts.py
    test_block_error_replay_exact.py
    test_source_readonly.py
  metrics/   # compact per-state/per-rule/per-premise data, with same IDs across P0/P3/P4/P6
  plots/     # observed data only, no fictitious paper figures
  logs/      # separate from live runs
  caches/    # pinned dataset/prompt snapshots; gitignore large assets
  envs/      # new conda prefixes; do not modify production envs
  checkpoints/  # only NEW M runs, never P0–P6
```

数据契约中的 JSONL 字段允许补充，但不得略去：`split`, `theorem_id/source_group`, `state_id/hash`, `input_sha`, `reference_or_candidate_action`, `candidate_origin`, `rule_id`, `retrieval_id`, `retrieval_sha`, `lean_status`, `proof_verified`, `model_id`, `checkpoint_sha`, `protocol_sha`, `status`。未提供的字段写 `null/UNKNOWN`，不可补造。

## 9. 分阶段 Gate 与优先执行顺序

| Gate | CPU/GPU | 完成条件 | 失败处理 |
|---|---|---|---|
| G0：Code / env / logs | CPU | 旧 P0–P6、v12 当前状态、minif2f split 与 prompt builder 精确核对 | `SOURCE_OR_LOG_BLOCKED` |
| **G1：Data contract** | CPU | 56k Parquet schema、state/group 去重、长度/模板审核、S-CAL/S-OBS/V-REAL/T-FINAL 四源分离 | `DATA_BLOCKED` |
| **G2：Stage1 loader** | CPU | 2048-token 有效样本、STATE retention、可实施 mask/packing 的严格测试，M0 新管道 parity 方案 | `CALIBRATION_FORMAT_BLOCKED`；**不能启动 M** |
| **G3：Valid Lean** | CPU/必要时服务 | valid root states、独立 Lean verifier / retrieval runtime + frozen prompt receipts | `WAITING_RETRIEVAL_RESOURCE`，可先做 synthetic B |
| G4：O pilot | 独占单卡 | P0/P3/P4/P6 共同样本/候选、B/C/D pairing parity；A rank small | 逐项 `PARTIAL` |
| G5：O external | 独占单卡 | A/B/C/D 在 V-REAL/V-RULE 的 CI + 负面控制，输出可复核 CSV | `INSUFFICIENT_EVIDENCE` |
| **G6：M small smoke** | 独占单卡 | M0/M10 新加载路径所有 parity、batch/seed/token budget pass；128/256 window only for工程验证 | `M_NOT_STARTED`（不得当论文结果） |
| G7：M full controlled | 独占单卡 | Stage1 M0/M10 同预算完成；若资源允许，再做 same Stage2；保存 checkpoint 与训练 receipt | `WAITING_RESOURCE` / `TRAINING_PARTIAL` |
| G8：Development decision | CPU + valid eval | miniF2F-valid 同预算 + O 指标；冻结选定 ratio/method SHA | `INCONCLUSIVE` |
| G9：Final | 现有正式协议允许后 | 单次独立 miniF2F-test 冻结评测；不能以测试表现继续 ratio 搜索 | `WAITING_FINAL_EVAL` |

**资源优先级建议**：G0→G1→G2→G3（纯审计/低算力）→ G4/G5（机制 Observation）和 G6（校准能否实施的 smoke）→ 根据证据再花单卡成本执行 G7。用户可允许两路线分别推进，但它们不能共用写入目录、不能互相使用 test 数据挑超参数。

## 10. 必须给研究者的最终结论

至少输出一页结构化 decision memo，逐项回答：

1. Lean calibration 是否能与原 Stage 1 在 **有效 token、2048 截断、prompt 内容保留、step 数量、评估输入** 上做到可比？若不能，最小可行修正及它引入的混杂是什么？
2. miniF2F-valid 是否已有真实日志？其 `root/rollout`, retrieval, tactic candidates 和 Lean verdict 能否完整恢复？若不能，应生成哪些新日志？
3. A/B/C/D 各自在哪个 split 上 fit、在哪个 split 上检验、在何种因果/任务解释下成立？负结果和控制实验有哪些？
4. M0/M10/M20 的实际完成状态、valid 指标与对非 Lean 域的影响；**没有执行不得填数值或声称提高成功率**。
5. 第一篇方法论文的真正可区分新意是什么：是 Lean calibration 域适配、功能子空间对齐、对称性保护、检索依赖保护，还是跨层功能误差协同？仅凭一条域适配曲线不预支高水平会议 novelty。

## 11. 主要来源（执行时重新锁 SHA）

- 用户已上传的两份 SOP 与观察目录：`observations/realprover_20261104/all_layers_corrected/`，仓库 <https://github.com/haiyuntan/ReasoningQAT>（非默认 Qwen3 分支）。
- REAL-Prover 官方：<https://github.com/frenzymath/REAL-Prover>，`Realprover/manager/thirdparty/generator.py`、`Realprover/manager/search/best_first.py`、`Realprover/manager/manage/prompt_manage.py`；`Realprover/data/minif2f_valid.jsonl`、`minif2f_test.jsonl`。
- `FrenzyMath/state_tactic_pairs` 官方：<https://huggingface.co/datasets/FrenzyMath/state_tactic_pairs>（只有 train split）。
- ReasoningQAT ICLR 2026：<https://proceedings.iclr.cc/paper_files/paper/2026/hash/209423f076b6479ab3a4f45886e30306-Abstract-Conference.html>。

**最终执行指令**：请 Code Agent 按 G0–G9 逐级验证，优先执行 G0–G3 的真实本机审计。任何正在运行的正式 miniF2F-test 与生产检索服务不允许被中断、抢占、污染或使用其 test success 来决定数据、ratio、规则或模型。所有未能执行的工作写清楚原因和缺口；不生成虚假实验结果。
