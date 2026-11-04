# REAL-Prover 观察实验：结果与复现材料

本目录发布已完成的观察结果，**不是正式 miniF2F 最终成绩**。权重为 REAL-Prover BF16 或 W2A16/g128 数值反量化后的 BF16 dense；不作 INT2 内核性能声明。

## 当前有效版本

**[all_layers_corrected](all_layers_corrected/)**：全28层，General / Math / Code / Lean 每域256个独立输入组，共1024个输入；P0/P3/P4/P5/P6 均完成。熵分析使用原507个完整 state→tactic 前缀，使用 FP64 稳定统计重新计算。

- [代表层：Block 3 t-SNE](all_layers_corrected/plots/figure2_right/realprover_bf16_layer3_tsne.png)
- [Block 14 对照](all_layers_corrected/plots/figure2_right/realprover_bf16_layer14_tsne.png)
- [全部层的域分离统计图](all_layers_corrected/plots/figure2_right/all_layer_domain_metrics.png)
- [全部层的量化表征误差图](all_layers_corrected/plots/figure2_right/all_layer_quantization_nmse.png)
- [28层数值表](all_layers_corrected/metrics/all_layer_domain_statistics.csv)
- [代表层选择记录](all_layers_corrected/manifests/representative_layer.json)
- [修正后的低/高熵效应及置信区间](all_layers_corrected/metrics/entropy_bootstrap.csv)
- [数值精度审计](all_layers_corrected/metrics/tail_numeric_audit.csv)
- [epsilon 敏感性](all_layers_corrected/metrics/epsilon_sensitivity.csv)

## 结果摘要与边界

1. **按预先固定的指标选择了第3层，而非按2D图外观挑选。** 选择指标为内部 discovery-half 上、去掉聊天模板后的高维余弦 silhouette；另一半仅用于确认，不再选层。Block 3 的 discovery/confirmation silhouette 为0.2386/0.2233。该选择是探索性结果，不是独立语料复现，也不能证明知识模块独立储存。
2. **低熵尾部相对膨胀在修正后仍为正。** Low−High 的 state-level 平均无 epsilon log-tail-ratio 差值：P3 3.482 [3.073,3.907]；P4 3.760 [3.397,4.155]；P5 3.651 [3.307,3.980]；P6 4.020 [3.591,4.444]。274个同时含Low/High位置的state用于主比较，2000次cluster bootstrap。不能把该排序解释成证明成绩排序。
3. **相对概率变化与实际采样风险必须区分。** 修正后低熵组1684个位置的BF16尾部无数值零；同一批新logits重建旧算法时1629个变为零。BF16参考分布上，FP32稳定算法相对FP64的最大log-tail误差约2.8e-6。实际 temperature=1.5/top_p=.9 下，P4低熵组平均非teacher-top1采样质量增量为0，而高熵组约0.171；因此不能仅凭相对膨胀宣称低熵的绝对采样危害更大。详见数值审计表。
4. Lean 是 BF16 训练分布。尚未完成独立来源 Lean 验证、全部混杂控制、原生量化尺度梯度诊断、P1/P2 原生参考校验或正式证明结果关联。P5独立Block-AP；P4/P6从P3初始化，差异不能全部归因于Stage2 loss。

## 历史版本

- [pilot128](pilot128/)：每域128个样本的表征 pilot。旧熵表仅作为有明确警告的历史记录，已被修正版替代。
- [scale256](scale256/)：每域256个样本、5个指定层的扩展实验。全部1024个输入与全层版本一致。
- [provenance.json](provenance.json)：各次运行真实完成状态、源码SHA、协议SHA、输入SHA。
- [artifact_index.json](artifact_index.json) 和 [SHA256SUMS](SHA256SUMS)：来源与文件内容校验。

较大的逐样本CSV和输入JSONL使用gzip压缩。图、聚合表、源码、精确输入token清单和环境锁均已发布；没有上传模型、隐藏状态张量、全量logits、Conda环境、下载缓存或认证信息。

## 复现

先阅读 [REPRODUCE.md](REPRODUCE.md)。可仅用已发布坐标与CSV在CPU重画图；重新提取激活需要固定版本模型和独立观察环境。源运行目录的绝对路径保留为证据，不代表其他服务器可以直接使用。
