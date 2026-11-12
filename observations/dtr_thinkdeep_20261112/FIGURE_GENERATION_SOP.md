# REAL-Prover × Deep-Thinking Tokens 图表生成任务书（Code Agent）

**文件名**：`REALProver_DTR_Figure_Generation_SOP_v1_20261112.md`  
**用途**：为 `REALProver_DeepThinkingTokens_DTR_Transfer_Observation_SOP_v1_20261112.md` 的观察实验生成论文级图表。  
**定位**：本任务书**只负责出图**，不重新定义 DTR 主实验协议；若缺少必要数据，可补跑最小分析脚本，但**不得改变原实验设计、样本划分和主要统计口径**。  
**原则**：图表服务于“解释与呈现”，而不是事后挑选结论。所有主图都必须能由仓库中的原始结果或冻结数据可重现生成。

---

## 0. 目标

本任务书要求 Code Agent 生成一组与论文 *Think Deep, Not Just Long: Measuring LLM Reasoning Effort via Deep-Thinking Tokens* 的 Figure 1 / Figure 2 对齐、但适配 REAL-Prover 场景的图表。

本轮重点图表分为两类：

1. **Figure 1 风格图**：比较 **token length** 与 **DTR** 对 Lean tactic 质量的解释能力。  
2. **Figure 2 风格图**：显示某条 tactic 序列中，不同层的预测分布如何逐渐收敛到最终层，并比较 **P0 BF16** 与 **P4 ReasoningQAT W2G128** 的差异。

此外，增加一组**补充图**，用于论文主文或附录：

- DTR vs Self-Certainty / Avg logprob / Length 的同状态比较；
- P0/P4 在同一 token 序列上的 DTR 差值分布；
- 按 tactic outcome（`VALID_TRANSITION` / `GOALS_CLOSED` / `INVALID`）分组的 DTR 分布；
- 案例可视化图。

---

## 1. 输出要求

所有图输出到独立目录：

```bash
$ROOT/observation_dtr_thinkdeep_20261112/plots_final/
```

并产出以下文件：

```text
plots_final/
  main/
    Fig1_length_vs_dtr_validity.png
    Fig1_length_vs_dtr_validity.pdf
    Fig2_jsd_heatmaps_p0_p4.png
    Fig2_jsd_heatmaps_p0_p4.pdf
    Fig3_same_state_metric_comparison.png
    Fig3_same_state_metric_comparison.pdf
    Fig4_dtr_shift_p0_vs_p4.png
    Fig4_dtr_shift_p0_vs_p4.pdf
  appendix/
    FigA1_dtr_by_outcome.png
    FigA2_case_panels.png
    FigA3_per_model_binned_curves.png
    FigA4_token_type_breakdown.png
  tables/
    figure_source_manifest.json
    figure_statistics_summary.json
  reports/
    FIGURE_README.md
    figure_decision_log.md
```

其中：

- `png` 用于快速查看；
- `pdf` 用于论文排版；
- `figure_source_manifest.json` 记录每张图使用的数据文件、样本数量、模型、生成脚本哈希；
- `figure_statistics_summary.json` 记录图中出现的所有数值来源；
- `FIGURE_README.md` 说明每张图的含义和复现实验入口；
- `figure_decision_log.md` 记录遇到的替代方案、失败尝试和最终选型理由。

---

## 2. 环境与隔离

### 2.1 工作区

必须在独立工作区执行：

```bash
$ROOT/observation_dtr_thinkdeep_20261112/
```

禁止把出图脚本、缓存或中间结果写入：

- 旧的 `observations/abcd_*`
- 旧的 `observations/realprover_20261104/*`
- 正在运行的正式评测目录

### 2.2 环境

建议复用上一份 DTR 观察实验的独立环境；若不存在，则新建：

```bash
conda create -n RP_DTRFIG python=3.10 -y
conda activate RP_DTRFIG
pip install matplotlib pandas numpy scipy pillow
```

如果图中需要更高质量排版，可增加：

```bash
pip install jinja2 pyyaml
```

**禁止**依赖 notebook 手工改图。所有图必须由脚本自动生成。

---

## 3. 输入数据来源

优先使用 DTR 观察实验已生成的数据；若该实验尚未完全跑完，则允许基于其冻结样本补跑最小分析，以生成画图所需 CSV/JSON。

### 3.1 首选数据资产

来自：

```text
$ROOT/observation_dtr_thinkdeep_20261112/
```

优先寻找或生成：

```text
metrics/
  per_tactic_metrics.csv
  per_token_metrics.csv
  per_layer_jsd.jsonl
  same_state_pairwise_metrics.csv
  model_comparison_summary.json
cases/
  selected_case_manifest.json
  case_<id>_p0.json
  case_<id>_p4.json
```

字段要求如下。

### 3.2 `per_tactic_metrics.csv` 最低字段

每一行对应一个“proof state × candidate tactic × model”的观测：

- `theorem_id`
- `state_id`
- `candidate_id`
- `model` （至少含 `P0`, `P4`，若已有也可含 `P3/P5/P6`）
- `split` （开发 / 确认）
- `tactic_text`
- `token_length`
- `dtr`
- `self_certainty`
- `avg_logprob`
- `lean_status` （`VALID_TRANSITION`, `GOALS_CLOSED`, `INVALID`, `PARSER_ERROR`, 等）
- `is_valid_transition` （0/1）
- `is_goal_closed` （0/1）
- `token_count_used_in_dtr`
- `deep_token_count`

### 3.3 `per_token_metrics.csv` 最低字段

每一行对应一个“proof state × tactic token × model”的观测：

- `theorem_id`
- `state_id`
- `candidate_id`
- `model`
- `token_pos`
- `token_str`
- `token_id`
- `jsd_deep`
- `is_deep_thinking_token`
- `final_logprob`
- `token_type`（若可得：keyword / lemma / variable / punctuation / numeral / other）

### 3.4 `per_layer_jsd.jsonl` 最低字段

每一条记录对应一个样本的完整 token × layer JSD 矩阵：

- `theorem_id`
- `state_id`
- `candidate_id`
- `model`
- `token_strings`
- `layer_ids`
- `jsd_matrix` （二维数组，shape = [num_layers, num_tokens]）
- `dtr`
- `lean_status`
- `tactic_text`

### 3.5 允许补跑，但必须冻结

如果没有上述数据文件，可允许 Code Agent 基于 DTR 观察实验 SOP 中冻结的 theorem、proof state 和 candidate tactics 补跑以下最小分析：

- P0 / P4 的 teacher-forced 全层 JSD；
- tactic-level 的 DTR、length、avg logprob、self-certainty；
- Lean 执行结果。

但必须在 `reports/FIGURE_README.md` 中明确写明：

1. 补跑使用的样本来源；
2. 是否与主观察实验完全一致；
3. 是否只为出图补充，不改变任何主统计。

---

## 4. 主图一：Figure 1 风格图

### 4.1 图的目的

回答：

> 在 REAL-Prover 上，DTR 是否比 token length 更能解释 Lean tactic 的有效性？

### 4.2 图的结构

输出文件：

- `main/Fig1_length_vs_dtr_validity.png`
- `main/Fig1_length_vs_dtr_validity.pdf`

建议为 **左右两个 panel**：

- **Panel (a)**：`Token Length` vs `Lean tactic validity`
- **Panel (b)**：`DTR` vs `Lean tactic validity`

其中 `Lean tactic validity` 的默认纵轴为：


a. `VALID_TRANSITION` 比例；  
b. 可选附加版本：`GOALS_CLOSED` 比例。

若版面允许，可在每个 panel 中再用颜色区分：

- `P0`（主参考）
- `P4`（主量化模型）

P3/P5/P6 放在附录图 `FigA3_per_model_binned_curves.png`。

### 4.3 统计与分箱

#### Length panel

- 对 `token_length` 做 5 个等频分箱（quintile）或 6 个预注册固定区间；
- 每个箱内计算：
  - `mean validity`
  - `95% CI`（按 theorem 聚类 bootstrap）
  - `sample count`

#### DTR panel

- 对 `dtr` 做 5 个等频分箱（quintile）或 6 个固定区间：
  `[0,0.2), [0.2,0.4), ... [0.8,1.0]`
- 同样计算 `mean validity`, `95% CI`, `sample count`。

### 4.4 必须同时提供的数值比较

在图内或图注中给出：

- `Length` 与 `validity` 的相关性（建议 Spearman）；
- `DTR` 与 `validity` 的相关性；
- 同状态比较时，DTR 是否优于 Length（这个结论以 Fig3 为主，这里只简要注释）。

### 4.5 重要限制

- 不允许只选有利于 DTR 的样本；
- 不允许只展示 P0 而省略 P4；
- 不允许把 `GOALS_CLOSED` 当作唯一标签而忽略 `VALID_TRANSITION`，除非样本过少并已说明；
- 若 DTR 曲线基本平坦，必须如实保留，不得通过改变分箱边界制造趋势。

---

## 5. 主图二：Figure 2 风格图

### 5.1 图的目的

回答：

> 对于同一条 Lean tactic 序列，P0 BF16 与 P4 W2G128 的各层预测分布如何收敛？量化是否改变了“深思 token”分布？

### 5.2 图的结构

输出文件：

- `main/Fig2_jsd_heatmaps_p0_p4.png`
- `main/Fig2_jsd_heatmaps_p0_p4.pdf`

建议为 **2 × 2 布局**：

|                | P0 BF16 | P4 W2G128 |
|----------------|---------|-----------|
| Lean-valid tactic   | heatmap | heatmap |
| Lean-invalid tactic | heatmap | heatmap |

每个 heatmap：

- 横轴：token position（并在底部标出 token string）
- 纵轴：layer id（1..28 或 0..27）
- 颜色：`JSD(layer_dist || final_layer_dist)`

并在图中标出：

- Deep-thinking 区间的层边界（例如在第 24–28 层加浅色框或虚线）
- 该样本的 overall DTR
- Lean verdict

### 5.3 案例选择规则

禁止人工挑选“最漂亮”的样例。

必须按预注册规则自动选 4 个 case：

1. **Case V0**：P0 与 P4 都是 `VALID_TRANSITION`，从确认集随机选一个；
2. **Case V1**：P0 `VALID_TRANSITION` 且 P4 `INVALID`，若存在，从确认集随机选一个；
3. **Case V2**：P0 `INVALID` 且 P4 `VALID_TRANSITION`，若存在，从确认集随机选一个；
4. **Case V3**：P0 与 P4 都 `INVALID`，从确认集随机选一个。

如果某类不存在：

- 必须在 `selected_case_manifest.json` 和 `FIGURE_README.md` 中说明；
- 使用实际存在的最接近替代类，但保留缺失信息。

### 5.4 视觉规范

- 同一张图中四个 heatmap 必须共享相同颜色范围；
- token 标签太长时允许旋转 45° 或截断，但需要在附录 `FigA2_case_panels.png` 给完整文本；
- 如果序列很长，可只显示前 32 个 token，并在图注说明；
- 不允许对不同模型使用不同的颜色范围。

---

## 6. 主图三：同状态指标比较

### 6.1 图的目的

回答：

> 在同一个 proof state 上，DTR 是否比 Length / Self-Certainty / Avg logprob 更能区分较好的 tactic？

### 6.2 输出

- `main/Fig3_same_state_metric_comparison.png`
- `main/Fig3_same_state_metric_comparison.pdf`

### 6.3 图形建议

推荐使用 **paired win-rate bar chart** 或 **C-index / concordance score bar chart**。

比较对象：

- `DTR`
- `Self-Certainty`
- `Avg logprob`
- `Length`

标签：

- 任务 A：区分 `VALID_TRANSITION` vs 非 valid
- 任务 B：区分 `GOALS_CLOSED` vs 其他

按模型至少报告：

- P0
- P4

可附加：P3/P5/P6。

### 6.4 统计规则

对于同一 `state_id` 中的候选 tactic：

- 若一个指标更高的 tactic 更可能是 `VALID_TRANSITION`，记为一致；
- 统计 concordance / pairwise accuracy；
- 给出按 theorem 聚类的 95% CI。

若样本不足以稳定评估 `GOALS_CLOSED`，则该部分移入附录。

---

## 7. 主图四：P0 与 P4 的 DTR 偏移

### 7.1 图的目的

回答：

> 在完全相同的 token 序列上，W2 量化是否系统性改变了 DTR？

### 7.2 输出

- `main/Fig4_dtr_shift_p0_vs_p4.png`
- `main/Fig4_dtr_shift_p0_vs_p4.pdf`

### 7.3 图形建议

推荐使用 **violin + box + paired points** 或 **raincloud plot**。

横轴可按 outcome 分组：

- `VALID_TRANSITION`
- `GOALS_CLOSED`
- `INVALID`

纵轴：


a. `DTR(P4) - DTR(P0)`；  
b. 附加子图：`Self-Certainty(P4) - Self-Certainty(P0)`（若有必要）。

### 7.4 解释边界

- `DTR` 更高不自动表示“思考更深”或“能力更强”；
- 它也可能仅表示量化使预测分布更晚收敛；
- 因此图注必须写明：本图是“层间预测收敛结构变化”的描述性证据。

---

## 8. 附录图

### 8.1 `FigA1_dtr_by_outcome.png`

展示 DTR 在不同 Lean outcome 下的分布：

- `GOALS_CLOSED`
- `VALID_TRANSITION`
- `INVALID`
- 其他错误类型（可合并）

模型至少展示 P0 与 P4。

### 8.2 `FigA2_case_panels.png`

以更大版面呈现一到两个代表案例：

- 原 theorem / state 简述；
- tactic 文本；
- Lean verdict；
- P0 与 P4 的 token-level `is_deep_thinking_token` 标注；
- 可配合小型热力图或 token 高亮。

### 8.3 `FigA3_per_model_binned_curves.png`

将 Fig1 的分箱曲线扩展到：

- P0
- P3
- P4
- P5
- P6

用于比较不同 QAT 方法是否改变 DTR–validity 关系。

### 8.4 `FigA4_token_type_breakdown.png`

如果 `token_type` 可用，则展示：

- 不同 token 类型的平均 DTR；
- 不同 token 类型在 P0/P4 间的 DTR 差值；
- 或其 `VALID_TRANSITION` 条件概率。

---

## 9. 数据处理与统计细则

### 9.1 统一口径

所有主图默认以 **确认集** 为主；开发集只能用于：

- 选图样式；
- 检查代码；
- 发现绘图 bug。

最终主图的数值一律来自确认集，除非在图标题里明确写“development only”。

### 9.2 置信区间

统一要求：

- 以 theorem 为聚类单位做 bootstrap；
- `B = 10,000` 次；
- 固定随机种子；
- 在 `figure_statistics_summary.json` 记录种子、重采样次数、统计量定义。

### 9.3 显著性

出图任务本身不要求做新的假设检验，但若图注中写显著性，必须引用主分析结果；
禁止在出图阶段重新发明新的主显著性指标。

### 9.4 缺失值处理

若某模型在某项图中样本不足：

- 允许不画该模型；
- 但必须在 `figure_decision_log.md` 中解释原因。

---

## 10. 脚本要求

建议在：

```text
$ROOT/observation_dtr_thinkdeep_20261112/plots_final/scripts/
```

下提供：

```text
build_fig1.py
build_fig2.py
build_fig3.py
build_fig4.py
build_appendix.py
common_plot_utils.py
run_all_figures.sh
```

每个脚本都必须支持：

```bash
python build_fig1.py --input metrics/per_tactic_metrics.csv --outdir plots_final/main
```

且不得依赖手工编辑。

---

## 11. 验收标准

### G0 数据完整性

- [ ] `per_tactic_metrics.csv`、`per_layer_jsd.jsonl` 等源文件存在；
- [ ] 行数与 `figure_source_manifest.json` 一致；
- [ ] P0/P4 数据可对齐到同一 `theorem_id/state_id/candidate_id`。

### G1 图文件生成

- [ ] 四张主图均成功生成 PNG + PDF；
- [ ] 四张附录图均成功生成 PNG；
- [ ] `FIGURE_README.md` 完整。

### G2 可复现

- [ ] 删除 `plots_final/` 后重新运行 `run_all_figures.sh` 可再次生成；
- [ ] 图中数值与 `figure_statistics_summary.json` 一致；
- [ ] 所有脚本在独立环境中可运行。

### G3 质量检查

- [ ] 轴标题、图例、颜色范围清晰；
- [ ] 热力图共用色标；
- [ ] token 标签不重叠到不可读；
- [ ] 图注中没有夸大结论。

---

## 12. FIGURE_README.md 必须包含的内容

至少包含：

1. 每张图的生成脚本命令；
2. 每张图的输入数据文件；
3. 每张图的统计口径；
4. 图中所展示的模型；
5. 图的解释边界；
6. 与原论文 Figure 1 / Figure 2 的对应关系；
7. 若有替代设计，为什么最终未采用。

---

## 13. figure_decision_log.md 必须记录的事项

至少记录：

- 是否尝试过不同分箱策略；
- 是否尝试过散点图、箱线图、柱状图等多种形式；
- 为什么最终采用当前布局；
- 某些 case 类别不存在时，替代方案是什么；
- 是否有某张图因为结果平坦而仍然保留；
- 是否发生过脚本错误、字段缺失、数值不一致。

---

## 14. 推荐图注模板（可直接复用）

### Fig1 图注模板

> **Figure 1.** Relationship between tactic length / deep-thinking ratio and Lean tactic validity on the REAL-Prover validation set. Tactics are binned into equal-frequency bins; error bars denote theorem-clustered 95% bootstrap confidence intervals. Panel (a) uses raw token length, while panel (b) uses the deep-thinking ratio (DTR). The plot is descriptive and does not by itself establish theorem-level proof success.

### Fig2 图注模板

> **Figure 2.** Layer-wise convergence of token prediction distributions for representative Lean tactics. Each heatmap shows the Jensen–Shannon divergence between an intermediate layer and the final layer across tactic tokens. P0 (BF16) and P4 (ReasoningQAT W2G128) are evaluated on the same prompt and the same target tactic sequence. Higher DTR indicates that more token positions remain far from the final prediction until deeper layers, but does not necessarily imply better reasoning quality.

---

## 15. 最终交付

Code Agent 最终需返回：

1. `plots_final/` 下全部图片；
2. `tables/figure_source_manifest.json`；
3. `tables/figure_statistics_summary.json`；
4. `reports/FIGURE_README.md`；
5. `reports/figure_decision_log.md`；
6. 一段简短总结：
   - 哪些图最适合主文；
   - 哪些图适合附录；
   - 哪些图显示 DTR 在 REAL-Prover 上有价值；
   - 哪些图说明 DTR 的迁移边界。

如果主图结果与预期不符（例如 DTR 不优于 length，或 BF16/W2 热力图差异很弱），仍然必须如实交付，不得因为“图不好看”而省略。

---

## 16. 一句话执行摘要

> 在独立工作区中，基于冻结的 REAL-Prover DTR 观察实验数据，自动生成与原论文 Figure 1 / Figure 2 对齐的论文级图表：比较 Length vs DTR 对 Lean tactic 质量的解释能力，并展示 P0 与 P4 在相同 tactic 序列上的层间 JSD 收敛热力图，同时补充同状态指标比较、DTR 偏移分布和 outcome 分组图，确保所有图都有可追溯的数据来源、统一统计口径与清晰解释边界。

