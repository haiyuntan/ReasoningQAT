# 外部依赖

2026-10-15：已恢复到此持久目录，源码压缩包、SHA256 和固定提交见 `source_restore.json`；
各默认路径现在指向 `source_cache/` 内的真实版本目录。旧 `~/project` 不再是运行依赖。
Lean4.16、Mathlib、Jixia、Interactive、REPL 的恢复与检索服务说明见
`../../research/reports/environment_restoration.md`。

默认目录位于此处；也可通过环境变量显式设置已核实的新依赖路径：

```bash
export REALPROVER_LEAN_WORKSPACE=/absolute/new/lean_test_v4160
export REALPROVER_JIXIA_BIN=/absolute/new/jixia/.lake/build/bin/jixia
export REALPROVER_INTERACTIVE_ROOT=/absolute/new/interactive
export REALPROVER_MODEL_PATH=/absolute/Reasoning/models/REAL-Prover-v1-bf16
export LEANSEARCH_PORT=8080
```

Lean 工具链要求 v4.16.0；工作区需要 Header.lean、可运行的 REPL 和匹配版本的 Interactive/Jixia。评测会写临时 TestOne 文件，因此 Lean 工作区必须可写，不能直接使用只读旧工程作为工作目录。Interactive 当前 `IS_BUILD=False`，不会自动恢复缺失构建。

检索客户端默认连接 `http://localhost:8080/retrieve_premises`，服务和索引未复制。已有资产候选：`FedProver/model/LeanSearch-PS`、`FedProver/model/LeanSearch-PS-faiss.index`；只读引用，需先验证来源和服务。所有模型评测必须使用同一检索协议。

Python 核心依赖：torch、vllm、transformers、requests、heapdict、matplotlib、networkx。Anthropic SDK 只在显式启用旧远端分支时需要；本实验默认使用本地模型。优先审计现有 PROVER/REQAT_PROVER，不能直接升级它们。
