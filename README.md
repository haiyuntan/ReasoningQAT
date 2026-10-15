# REAL-Prover 原生评测代码副本

从同级只读 `FedProver` 整理，服务于 REAL-Prover × ReasoningQAT 迁移实验。当前状态是 **代码整理完成，Lean 端到端运行尚未验证**。

目录内容：

- `experiment/run.py`：TOML 驱动的批量评测入口，支持 `--result-dir` 和 `--resume-from`。
- `manager/`：生成、检索客户端、Lean Interactive、证明校验、搜索及结果整理；保留导入依赖闭包，包括可选搜索分支。
- `util/`、`conf/`：工具与本地配置。
- `data/`：miniF2F-test 244 题、ProofNet-test 186 题、FATE-M 141 题；全部仅用于评测。来源为旧工程，尚待与固定官方源码比对。
- `configs/`：候选 BF16 全集及两题 smoke 配置，尚未冻结为正式协议。
- `dependencies/`：新 Lean 运行依赖预留位置，当前为空；不会复制大型模型、索引或旧 Lean build。
- `provenance/`：逐文件来源/目标 SHA256、来源 Git SHA、全部代码改动 diff。
- `scripts/preflight.py`：只读 CPU 预检，不加载模型、不运行 Lean、不触发构建。

最初整理时调整了三处集成代码：`conf/config.py` 将模型、Lean/Jixia/Interactive 路径改为环境变量及新工作区默认路径；`verifier.py` 共用同一个 Lean 路径；`claude.py` 延迟导入可选 SDK，避免本地评测必须安装 Anthropic。搜索、prompt、采样和评分逻辑未改动。

没有复制联邦训练、state–tactic 训练数据、历史 checkpoint、旧输出日志或大检索索引。没有使用旧 `requirements.txt` 安装包：其中训练依赖和旧版本约束不适合直接用于现有评测环境。

## 预检与使用

从此目录执行，用既有 PROVER/REQAT_PROVER 的 Python：

```bash
python scripts/preflight.py
```

Lean 依赖恢复到新的可写工作目录后，按 `dependencies/README.md` 设置环境变量。确认模型下载完成、检索服务有效、预检通过后，可执行两题 smoke：

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiment.run configs/p0_smoke2_candidate.toml \
  --result-dir ../runs/P0_smoke2
```

当前不能直接报告正式分数：Lean 依赖尚未恢复、模型正在下载、retrieval 未验证，候选配置也未完成历史/官方协议审查。旧评分器仍需核对失败样本和完整固定分母后才能用于正式结果。

该副本不是官方干净 checkout：来源是带有本地修改的 FedProver 工作树。`provenance/source_manifest.json` 的逐文件 hash 才是本次代码快照的精确标识，Git SHA 仅标识来源仓库 HEAD。

## 流水线适配更新

评测现已增加显式 vLLM seed、检索成功/失败遥测、完整证明校验标记，以及对 REPL 非零退出和异常响应的拒绝。统一流水线使用 research/adapters/evaluation.py 按固定题集汇总，保留缺失题并对重试去重；旧 get_stats 输出不作为正式指标。运行入口和验证边界见 ../research/reports/pipeline_adapters.md。当前仍未完成真实模型 Lean 冒烟。
