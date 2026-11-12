# DTR 论文风格图

主对照图：Fig2_DTR_top_REALProver_P0_bottom.png（另有 PDF）。上半幅是原论文 Fig2 的原始 PDF 裁图，下半幅是 REALProver P0 对同一 P4 生成 tactic 的真实逐层测量；不是独立 P0 自然生成。另提供 P4 自然生成测量版本。

DTR_source_Fig1.png/pdf 是原论文 Fig1 原图；DTR_source_Fig2.png/pdf 是原论文 Fig2 原图。来源：https://arxiv.org/pdf/2602.13517v2（2026-07-06），仅裁剪图像区域，未重绘或修改作者数值。

下方热图对齐原图的 YlOrBr 配色、顶部原生 token 标签、浅层在下/深层在上、竖直色条及单幅布局。所有 28 层、连续完整 tactic 的全部 token 均保留，空格 token 保留空列，不插值、不重排、不合并。为避免不完整 UTF-8 BPE 字节误标，选择无替换字符的样本。具体确定性筛选规则、全部 token、原始矩阵和输入哈希见 source_and_validation.json；筛选不依据 JSD、DTR 或热图相似度。

彩条 0–1 的版本明确使用 JSD bits，即原始 nats 除以 ln(2)，不是按本图最大最小值拉伸。原论文图色条虽为 0–1，但仅凭 PDF 不能确定作者使用的对数底数，因此不能声称两图阈值/单位已完全复现。另提供 Fig2_REALProver_P0_nats 与 P4_nats，保留 0–ln(2) 的原始单位。原实验 g=0.5 nats 与 DTR 统计不变；若在 bits 上解释同一阈值，应为 0.5/ln(2)，不是 0.5 bits。

Fig1_REALProver_paper_style 使用确认集 20 个证明状态、160 条 tactic，按指标五分位分箱（相同值不拆分），计算分箱均值之间的 Pearson r，供对齐论文 Fig1 的呈现方式。纵轴明确为 tactic 有效性，不是完整证明 pass@1。P0/P4 使用相同候选及 Lean 标签，所以长度曲线重叠；不能当作独立两模型准确率比较，亦不能通过调整样本获得预期正相关。原先统计和图不覆盖。

复现：使用同时有 PyMuPDF、NumPy、Matplotlib 的 Python 运行 src/figures/build_paper_comparison.py。需要 paper_sources/2602.13517v2.pdf 与已有冻结 metrics。全流程 CPU，无新增模型推理，不干扰正式评估。

## P0/P1 四子图

DTR_four_panel_source_Fig1_Fig2_P0_P1.png/pdf：上排仍为原论文 Fig1/Fig2；下排为 P0 BF16 与 P1 GPTQ W2G128。同一条 34-token tactic，P4 原始生成、P0/P1 均进行相同前缀和 token 序列的重评分；两图各自参考各自最终层分布。两图均显示 JSD bits，0–1 固定色标；不按每张图单独缩放。P1 使用已验证的 BF16 反量化数值导出；原实验指标和统计未修改。P1 只新增此单一展示案例，不是新确认实验，不能由此推断准确率优劣。

P1 原始逐 token 测量位于 metrics/P1_figure_replay.json，权重身份位于 manifests/models_P1_figure.json。34 个 token 的最终层投影与实际 logits 一致；首 token 的 FP32/FP64 JSD 检查通过。

复现命令：`python src/figures/build_four_panel.py --quantized-model P1`；不带参数仍生成原 P0/P4 四子图。两个版本均保留。
