# 项目清理与 Parquet 加载修复（2026-10-16）

## 清理

仅清理本项目中可再生成的重复产物，合计约 91.4 GiB：

- 旧中断 GPTQ 批次和诊断导出的权重分片，约 35.7 GiB。
- 从原始 OpenThoughts Parquet 展开的重复 Arrow 缓存，约 55.7 GiB。

逐文件清单和精确字节数见 `runs/project_cleanup_20261016/completed.json` 与 `arrow_cache_cleanup.json`。缓存来源元数据移至同目录 `expanded_dataset_info.json`。清理后旧诊断目录保留日志、配置和数值结果，不再包含可加载的完整临时权重。

原始 BF16 模型、120 份 OpenThoughts Parquet、FineWeb、Lean 环境和代码均保留。旧 autorun_v2 的 P0/P1/P2 PASS 目录全部保留，并再次逐个验证回执证据 SHA256。它们只证明旧指纹的通过，不能替代新代码冒烟。

## 原因与修复

P3 失败发生在 HF datasets 将全量 OpenThoughts Parquet 再展开为约 60 GB Arrow 数据时，报 `Disk quota exceeded`。此前子进程未显式固定 HF 缓存路径，可能使用用户主目录默认缓存。

新的 ParquetRows 只从页脚建立全局行号与 row group 的索引。使用与 Dataset.shuffle 相同的 NumPy 全局 permutation；不是 streaming buffer shuffle。select 根据 row group 批量读取所选行，再恢复要求的全局顺序，保留重复选择语义；仅所选记录驻留内存，不生成第二份全量 Arrow 数据。Stage1 根据实际预算批量预取；Stage2 保持原 32768 条全局 shuffle 后的选择方式。仅更改数据访问，不更改样本选择、tokenizer、混合比例或损失。

HF_HOME、HF_DATASETS_CACHE、HF_HUB_CACHE 显式绑定项目内 `cache/realprover/huggingface/`，避免后续 fallback 写入 HOME。原始数据和固定测试题集不变。新适配源码的 SHA256 已更新到 vendor/sources.json。

## 验证

- 跨文件/row group 顺序、多种 seed、重复选择、嵌套对话内容与 HF Dataset 对照通过；没有生成 Arrow 副本。
- 使用现有真实 Arrow 缓存作只读参考，全部 1200000 个 shuffle 索引完全一致，前 16 条真实记录完全一致；对照完成后才删除重复缓存。
- 真实 Stage1 冒烟：10 train、5 val、长度 256，固定 seed=2、80/20 混合；token SHA 在验证 JSON 中。
- 真实 Stage2 冒烟：2 样本、196 个有效 causal completion tokens，监督 mask 检查通过。
- 原始验证结果位于 `runs/realprover_parquet_fix_v1/validation.json`，摘要见 `parquet_cache_fix_validation.json`。

新代码必须重新运行全部 P0–P6 冒烟；原 nohup 控制器的失败即停、全部回执门禁和自动正式实验策略保持不变。完整正式训练尚未运行，不能把本次数据加载验证当作全流程成功。
