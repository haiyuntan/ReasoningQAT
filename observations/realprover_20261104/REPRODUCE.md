# 复现说明

## 1. 校验发布包

在本目录执行 `sha256sum -c SHA256SUMS`。`artifact_index.json` 同时记录原文件SHA和压缩后SHA；gzip解压后的SHA应等于source_sha256。

## 2. CPU重画可视化

使用独立Python环境，按 `environment/data-pip-freeze.txt` 选择依赖；本脚本只需 numpy、pandas、matplotlib。执行：

```bash
python replot.py --output /tmp/realprover-observation-replots
```

该命令根据发布的t-SNE坐标重画Block 3和14图，并根据全部层数值表重画分离度曲线。它不重新fit t-SNE、不需要GPU或权重。原始fit参数、随机种子、源码及两seed坐标均保留在各run目录中。

## 3. 重新运行模型前向与全部统计

模型与数据下载、QAT检查点生成参考仓库原有 [REPRODUCTION.md](../../reports/REPRODUCTION.md)。需要REAL-Prover BF16模型和P3/P4/P5/P6的已完成检查点及prepared/evidence链；本包不提供权重。官方state_tactic_pairs revision为91bb339e61ff045d3ebf97cd94d1df1d755dd745。

1. 在已准备好的Reasoning项目根目录下创建一个全新的兄弟观察目录，将 `all_layers_corrected/src`、`tests`、`configs`、`manifests` 复制进去。不要覆盖运行中的research/REALPROVER工作树或任何正式结果。
2. 解压 `domain_inputs.jsonl.gz`、`entropy_inputs.jsonl.gz` 到新观察目录的manifests，保留固定input_ids、分组及阈值/分析split。不要用test success选择样本或层。
3. 按配置和environment中的URL+SHA新建独立data/forward环境及缓存；准备固定SHA的upstream源码。旧manifest里的绝对路径、已通过标记和校验和不可盲信，必须针对本机真实prepared/evidence重建只读checkpoint inventory，验证全部权重、tokenizer、模块数和16前缀native参考parity。现有跨服务器路径映射不自动实现。
4. 新目录的 `common.ROOT=OBS.parent` 必须指向真正的Reasoning项目根；原报告中的source_path只是来源证据。不要直接在本仓库嵌套的发布目录中运行GPU提取入口。
5. 运行数值和隔离测试；在确认GPU3对应的UUID、localhost:8080检索身份和正式状态后，使用 `run_corrected_observations.py`。共享GPU保护策略绑定本次服务器，需要在新机器按实际部署审查；不要自动停止生产服务。

GPU源码入口按P0 domains→全层统计选图→P0 entropy冻结阈值→P3/P4/P6/P5 domains+entropy→全层误差与修正熵分析执行。单任务结果、协议、源码和输入hash要绑定新run；不能复用本包的ready状态冒充本机验证通过。

## 4. 数值定义

BF16 logits在token chunk内转换FP64；max-shift后排除top1做logsumexp，normalizer使用log1p。主指标是log_tail_Q−log_tail_T，不加epsilon。保留1e-12/1e-10/1e-8敏感性和旧float32重建作为对照。full-vocab熵、gold-prefix teacher forcing、completion-only mask、温度与top-p采样质量分别报告。

Layer 28 hook为最后decoder block输出，位于最终RMSNorm之前；没有将embedding或最终norm错误计作一个block。全部层统计与探索性选择规则见协议JSON。
