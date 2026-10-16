# GPTQ 导出与单卡检索生命周期修复（2026-10-16）

原 `realprover_autorun_v1` 的 P0 通过；P1 在量化、打包及原生重载后因导出 logits 检查失败而停止，正式实验没有启动。失败日志保留。

## GPTQ 导出检查

真实 checkpoint 单层诊断显示：TorchQuantLinear 先 matmul 再独立加 bias；普通 Linear 可能融合 bias。相同 BF16 权重和输入下，直接 Linear 输出最大差 0.25，而使用原 matmul + add 路径误差为 0。因此，原生 packed 与 dense BF16 的逐点 logits 不一致不能直接证明权重导出错误。

转换检查改为：

1. 在改变 dtype 之前保存原生 BF16 反量化结果的精确值，不用更高精度重新计算 scales。
2. 使用 FP64 数学参照对比 packed 与 dense 模型，逐项核对全部 196 个投影的权重和 bias。
3. 仍使用原 `atol=rtol=1e-4` 检查 16 个提示的 logits，并检查 4 token 贪心生成。
4. 将 dense 模型转换为部署用 BF16，再独立比较保存前与重载后的 logits 和生成。保留原 FP32 缓冲，避免 RoPE inv_freq 在 dtype 往返时被错误舍入为 BF16。

这明确区分权重转换的数学正确性与低精度内核舍入；不宣称原生 packed BF16 与 dense BF16 内核逐点等价。正式评测仍统一使用 BF16 dense 权重，不声称 INT2 推理速度。

实际完整模型结果见 `runs/realprover_gptq_export_fix_v3/eval_model/export.json`：196 个投影通过精确权重检查；数学参照 logits 最大误差 0；BF16 重载 logits 最大误差 0；16 个提示的生成及 tokenizer 检查均通过。先前 FP32 参照失败、RoPE 缓冲未保留导致的重载失败均保留在 fix_v1/v2，不计为通过。fix_v3 是独立导出组件验证，不能替代新 P1 完整冒烟回执。

## 单 GPU 检索策略

`retrieval_service.py` 仅管理工作区 process.json 记录且命令行确实匹配本工作区 server.py 的进程；拒绝终止无关进程。量化、训练、导出前停止检索并等待进程退出。Lean 评测前启动检索、等待 health、核对 LoRA/源版本/资产清单哈希；在同一服务生命周期中执行 preflight 和实际评测，finally 关闭服务。服务异常或停止失败会阻止后续阶段。

每项回执的证据新增 `resource_lifecycle.json`，记录量化训练前关闭检索和评测后关闭检索。检索资产、搜索预算、统一 vLLM 配置保持已冻结版本。原常驻 PID 140010 已实际停止并释放约 14 GB。

生命周期安全测试 2 项、自动衔接测试 3 项、门禁测试 4 项通过。真实启动/检索请求/Lean 前检/停止的集成结果位于 `runs/realprover_retrieval_lifecycle_v1/`。修复后以新结果目录运行全部冒烟，旧回执不迁移；全部七项当前指纹冒烟通过后，nohup 控制器才自动启动正式实验。
