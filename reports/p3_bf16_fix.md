# P3 非有限 logits 修复（2026-10-16）

autorun_v3 的 P0/P1/P2 通过；P3 在 28 层重建结束后，完整模型探针报 Nonfinite export probe logits，流水线退出且未启动正式实验。原日志保留。

## 修复范围

原实现使用 BF16 autocast 做块重建，但加载基础模型以及将训练层存回时使用 FP16，最后的完整模型探针没有 autocast，因此执行了 FP16 前向。该路径与 REAL-Prover 的 BF16 权重和评测配置不一致，也有更小的数值范围。

P3 的基础模型、训练层存回以及后续 P4/P6 计算权重统一采用 BF16。新增 cast_qat_dtype，在 dtype 转换时保留 scale、zero_point 的 FP32 学习参数以及 RoPE inv_freq 的 FP32 缓冲；checkpoint 重载也为 scale 和 zero_point 保持 FP32，避免舍入已学习参数。量化器定义、W2g128、初始化 .7、损失、学习率、样本预算与搜索预算不变。不跳过非有限值检查，不放宽导出阈值。

## 清理

对完成批次中 SHA256 相同的 safetensors 权重，先重新读取校验，再原子替换为同一份不可变文件的硬链接。所有回执路径和文件内容保持不变，实际释放约 29.4 GiB。清单见 runs/project_cleanup_20261016/dedup_passed_checkpoints.json。失败日志和原始数据保留。本次独立组件验证生成的临时模型权重，在验证结束后可删除，数值报告与日志保留。

## 验证与边界

- 新增精度回归：BF16 转换和保存重载后，量化 scale/zero_point 与 RoPE 频率保持原值，模型输出有限且重载输出一致；原损失和独立算子检查继续通过。
- 真实 7B P3：28 层重建、280 次有限梯度更新，以及训练后 16 个提示的 logits/贪心生成探针已通过。
- checkpoint、普通 HF 权重导出和重载的最终数值见 p3_bf16_fix_validation.json，原始组件日志位于 runs/realprover_p3_bf16_validation_v1/。

这些组件检查不代表 P3 的 Lean 评测已经通过。新代码指纹下全部 P0–P6 冒烟必须重新执行；仍由 nohup 控制器在七项全部通过后自动启动正式实验。训练、量化和导出期间检索服务保持关闭。
