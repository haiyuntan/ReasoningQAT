# REAL-Prover 评测环境恢复（2026-10-15）

旧 `~/project` 已删除，但 Lean 4.16.0 工具链和检索模型、索引仍在。
恢复目标为持久工作区 `REALPROVER/dependencies/`，不修改旧 FedProver 或既有 Python 包。

## 来源与版本

| 组件 | 官方来源 | 提交 |
|---|---|---|
| Lean 工作区 | frenzymath/lean_test_v4160 | faf496a4bc025db8f13170ab85b8bdd5a0737a36 |
| Mathlib | leanprover-community/mathlib4 | a6276f4c6097675b1cf5ebd49b1146b735f38c02 |
| Jixia | frenzymath/jixia，v4.16.0 | 157c55dd5754db3795838e7c05104b91535ee5fc |
| Interactive | frenzymath/interactive | 9e7417953d80de8e4daff17c550ad872fde7f9bb |
| REPL | leanprover-community/repl，v4.16.0 | eed8e6ff51e2c168e13d89e75015719e676bd9c9 |
| 检索服务和官方评测数据参照 | frenzymath/REAL-Prover | 3e5987dc1009a7addc935bcd7b4ae777738a0848 |

所有传递依赖按原 lake-manifest 固定提交恢复，原始压缩包及 SHA256 位于
`REALPROVER/dependencies/source_restore.json`。源码先通过官方 codeload 下载，随后补全真实 Git 对象。
Git 原配置指向失效代理；单次命令使用当前 HTTPS_PROXY 后访问成功，不修改全局 Git 配置。
已验证两个源码压缩包镜像可访问；ModelScope 的对应 Lean 工作区模型 API 为 404。

Mathlib 6014 个官方预编译缓存全部下载、解压成功。
原始 Mathlib 源码和 lockfile 不变；工作区 lakefile 增加 lockfile 已有的 REPL 显式依赖，
使 `lake exe repl` 可用。最初的临时 path lockfile 已恢复为原始 Git lockfile。

`minif2f_test.jsonl`、`proofnet_test.jsonl`、`fate_m.jsonl` 与上述官方提交逐字节一致。
数据一致性不等于完整评测协议已冻结，正式协议仍保留门禁。

## 检索服务

官方 inference 源码整理到 `REALPROVER/retrieval/`，保留 MIT 许可证；导入基线仅规范化空白。
适配仅涉及本地路径、显式加载本地 e5 基座和 LeanSearch-PS LoRA、eval 模式、
索引/答案数量及 ID 检查、资产哈希校验和健康端点。
保留原任务指令、last-token pooling、L2 归一化、FAISS 搜索和返回格式。

只读复用旧 `FedProver/model/` 的 e5、LoRA 和索引，以及本地 answers.json。
资产哈希位于 `research/manifests/retrieval_assets.json`；这是本地内容固定，未冒称独立核验过上游来源。
服务启动时核对全部资产，`/health` 返回启动时固定的 manifest 哈希、源码版本、LoRA 状态和前提数量。
评测 preflight 必须与该身份一致，不能仅凭端口存在就通过。

默认监听 `127.0.0.1:8080`，日志和 PID 在 `runs/realprover_retrieval/`。
单张 A800 上常驻约 14 GB；流水线串行执行方法，避免并发模型占用。
仅在这些子进程中设置 USE_TF=0、USE_FLAX=0，保持 PyTorch 后端且不升级环境。
第三方包入口改为延迟导入，Lean/检索检查不再间接导入整个 vLLM 模型注册表。

## 恢复与验证命令

在 Reasoning 根目录：

```bash
env/bootstrap/bin/python research/scripts/restore_lean_sources.py
env/bootstrap/bin/python research/scripts/build_lean_environment.py --skip-download --output runs/lean_restore_new
env/bootstrap/bin/python research/scripts/pin_retrieval_assets.py
env/bootstrap/bin/python research/scripts/start_retrieval.py
env/bootstrap/bin/python research/scripts/validate_preparation.py --gpu --full-model --native-runtime --output runs/preparation_new
```

服务启动可能需读取约 18 GB 资产并校验，等待 `/health` 就绪后再运行验证。
最终准备结论为 **READY_FOR_SMOKE**，证据汇总见 `environment_validation.json` 和
`runs/realprover_preparation_v4/preparation.json`。12 组检查通过：未修改的训练/PTQ/完整模型/Lean
组件使用 v3 的成功记录，generator 修正后在 v4 重新验证候选过滤和实际 vLLM 生成。
v3 总体失败及更早诊断均保留，不改写为成功。
修复保留原 sorry 过滤策略，同时过滤对应分数，避免候选和排序分数错位；
无 incontext 时也正确传递 qwen 模板参数。vLLM 探针只生成 8 tokens，检查运行链路，不作为有效证明或正式指标。
准备测试不是 P0–P6 冒烟回执。必须另行启动完整冒烟 campaign，全部通过后才能执行正式实验。

```bash
env/bootstrap/bin/python research/scripts/experiment_pipeline.py campaign --mode smoke --results runs/realprover_campaign_v1
env/bootstrap/bin/python research/scripts/experiment_pipeline.py gate --results runs/realprover_campaign_v1
```

正式实验还要求协议冻结。代码/配置/检索资产固定清单或实际 Lean 二进制改变后，旧冒烟失效。
门禁对大 checkpoint 分块计算 SHA256；构建/cache 输出不混入源码指纹。
