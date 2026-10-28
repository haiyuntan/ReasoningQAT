# REAL-Prover × ReasoningQAT 跨服务器复现手册

审计日期：2026-10-26；2026-10-28更新：检索资产已迁移至工作区models/retrieval，FedProver不再是当前训练/评估的运行依赖。历史审计/首次抽取脚本仍保留来源路径，不需要重新执行。本文描述当前已经运行的实现；早期 README 中“准备阶段”等状态是历史记录。本文及附件只改变文档，不改变正在运行的实验指纹。

## 1. 仓库状态和复现范围

远端：<https://github.com/haiyuntan/ReasoningQAT>。**两个实验分支分别是独立目录仓库，不能仅 clone 默认分支，也不能将两个分支依次 checkout 到同一目录。** 审计时两个工作树干净，GitHub `ls-remote` 与本地 HEAD 完全一致：

| 目标目录 | 分支 | 本手册之前的代码提交 |
|---|---|---|
| research | feat/realprover-reasoningqat-w2 | b8802631145eee7de9f0cb96196e5f31f10944b0 |
| REALPROVER | feat/realprover-evaluation-adapter | 141655e860f92eefbb5fbdb7574d0bebf13b5a62 |

手册之后的 research 提交仅增加复现文档/附件。完整实验还依赖未随上述分支保存的大型资产和 PTQ 源码；下文逐项恢复。凭证不属于复现依赖，不复制旧 Markdown 中的 PAT。

```bash
mkdir -p Reasoning
cd Reasoning
git clone --branch feat/realprover-reasoningqat-w2 --single-branch https://github.com/haiyuntan/ReasoningQAT.git research
git clone --branch feat/realprover-evaluation-adapter --single-branch https://github.com/haiyuntan/ReasoningQAT.git REALPROVER
git -C REALPROVER checkout 141655e860f92eefbb5fbdb7574d0bebf13b5a62
export ROOT="$PWD"
mkdir -p models datasets env runs migration ptq qat_baselines
```

两条路线：**从头复现**需要全部训练数据、PTQ 与训练环境；**搬运已训练模型继续评估**不必重新下载训练语料，但必须搬运相应模型和审计证据，并处理绝对路径绑定。跨服务器不会自动获得旧 smoke PASS，旧回执不能直接改路径/指纹后冒充新回执。

## 2. 实验含义和固定配置

| 实验 | 方法/初始化 |
|---|---|
| P0 | 官方 REAL-Prover BF16 |
| P1 | 同一基座 GPTQ W2g128，NuminaMath128 |
| P2 | 同一基座 AWQ W2g128，实际 scale search + clipping，同一校准集 |
| P3 | ReasoningQAT 混合通用数据 block reconstruction |
| P4 | 同一 P3 原生 checkpoint + Teacher-RR α=0.2、top20 KL β=1，仅更新 scale |
| P5 | 独立 EfficientQAT Block-AP + E2E-QP；不能从 P3 继续冒充独立基线 |
| P6 | 同一 P3 checkpoint + 普通 CE，仅更新 scale，与 P4 同 Stage2 token 预算 |

共同基座是官方已经训练好的 REAL-Prover（Qwen2，7B），不是从 Qwen 裸基座训练。评估加载的是量化权重数值导出的 BF16 dense 模型；W2 指权重数值精度，激活为16位，不代表原生 INT2 推理性能或相应显存节省。

正式评估：miniF2F-test 固定244题；pass@64 是每题最多64轮搜索、每个搜索步骤64候选，并非一次生成64篇完整证明。temperature=1.5、top_p=0.9、max_tokens=256、seed=2、best-first α=0.5、depth=128、calls/nodes=1024，beam/MCTS 关闭，检索开启。以 `REALPROVER/configs/p0_candidate.toml` 为准。

vLLM 固定 `gpu_memory_utilization=0.85`、`max_model_len=8192`、`max_num_seqs=64`、`max_num_batched_tokens=8192`。训练 attention 使用 eager；评估由 vLLM 后端管理，不要将训练安装了 flash-attn 等同于训练实际启用了它。

Stage1：OpenThoughts/FineWeb=80%/20%，4096训练+64验证，长度2048，2 epochs，batch2，weight LR=2e-5、quant LR=1e-4。Stage2：OpenThoughts 全量 shuffle(seed2) 后取32768，1 epoch、长度2048、microbatch1、累积64、LR=1e-4、cosine、warmup0.2、AdamW β=(0.9,0.95)、eps=1e-6、decay0、clip0.5。超长 prompt 使用 completion-preserving window；原训练选中152条需要该处理，不丢弃/替换样本。具体执行以 adapters/training.py 为准。

## 3. 硬件、空间与目录

已验证环境是 Linux x86_64、4×A800 80GB、主机内存约480GiB。其他卡型必须重新做容量验证，0.85 不能保证任何GPU都可用。建议使用本地SSD放 Lean、token/activation缓存；四卡 block 训练会产生数百GiB临时激活，按至少1TiB可用空间规划并监控，实际占用受并发影响。

```text
Reasoning/
  research/                    # 实验控制/适配器/私有算子快照
  REALPROVER/                  # Lean 搜索、检索服务、244题
  ReasoningQAT/                # 固定旧上游源码；影响fingerprint
  qat_baselines/EfficientQAT/  # 固定旧上游源码；执行算子主要在research/vendor
  ptq/GPTQModel/               # P1实际依赖
  ptq/llm-awq/                 # P2实际依赖，必须应用附件补丁
  models/REAL-Prover-v1-bf16/
  assets/retrieval/{e5,adapter,index}/
  datasets/{openthoughts3,fineweb-edu,numinamath_128,eval}/
  env/{bootstrap,train,gptq,prover}/
  runs/                       # 不随Git推送
```

## 4. 解包固定小资产和当前环境清单

附件 [reproduction_assets.md](reproduction_assets.md) 保存当前128条校准数据、评估输入、原数据revision清单、AWQ工作树补丁以及**从实际解释器采集**的完整包版本。旧 `env/*-pip-freeze.txt` 并不等于当前运行环境，尤其不能用旧 eval freeze 替代 PROVER 环境。

在工作区根目录执行一次（已有文件内容不同会停止，不覆盖）：

```bash
python3 - <<'PY'
import base64,gzip,hashlib,json,pathlib,re
root=pathlib.Path.cwd()
doc=(root/'research/reports/reproduction_assets.md').read_text()
packed=base64.b64decode(re.search(r'```base64-gzip-json\n(.*?)\n```',doc,re.S).group(1))
assert hashlib.sha256(packed).hexdigest()=='6fc979988ff9c5a6e1ef102d371ce6ef6ae1fb3cf3ff3a52ecde307185ef238f'
for name,value in json.loads(gzip.decompress(packed))['files'].items():
    relative=pathlib.PurePosixPath(name)
    assert not relative.is_absolute() and '..' not in relative.parts
    p=root/name; raw=base64.b64decode(value)
    if p.exists(): assert p.read_bytes()==raw, f'Existing file differs: {p}'
    else: p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(raw)
print('Assets verified and restored')
PY
```

恢复外部源码（先保持未修改版本，再应用当前P2补丁）：

```bash
git clone https://github.com/haiyuntan/ReasoningQAT.git ReasoningQAT
git -C ReasoningQAT checkout 75d2a38c743c22dcaefd0740fdd4ff79f25b2eb7
git clone https://github.com/OpenGVLab/EfficientQAT.git qat_baselines/EfficientQAT
git -C qat_baselines/EfficientQAT checkout 39f37f3b6053681c9b1cd4c9dcaf692d8999459e
git clone https://github.com/ModelCloud/GPTQModel.git ptq/GPTQModel
git -C ptq/GPTQModel checkout ca9d634db6a933cfae2c2d8e8be3fe78f76b802d
git clone https://github.com/mit-han-lab/llm-awq.git ptq/llm-awq
git -C ptq/llm-awq checkout d6e797a42b9ef7778de8ee2352116e0f48a78d61
git -C ptq/llm-awq apply --check "$ROOT/migration/llm-awq.patch"
git -C ptq/llm-awq apply "$ROOT/migration/llm-awq.patch"
```

补丁涉及六个AWQ文件，包括Qwen2适配、本地Numina校准及导入兼容性。不能用未打补丁的干净AWQ仓库报告当前P2复现。

## 5. Python环境

| 角色 | Python | 关键实际版本 |
|---|---|---|
| bootstrap/控制器 | 3.11+ | 标准库tomllib；下载另需huggingface_hub |
| train（也运行AWQ） | 3.11.16 | torch2.2.2、transformers4.52.4、datasets3.6.0、accelerate1.6.0、peft0.15.2、numpy1.26.4 |
| gptq | 3.11.16 | torch2.2.2、transformers4.40.1、datasets2.18.0、accelerate0.28.0、gptqmodel2.2.0（上述源码） |
| prover/eval/检索 | 3.10.20 | torch2.5.1+cu124、vllm0.6.6、transformers4.57.6、peft0.19.1、faiss-gpu1.7.2、Flask3.1.0、HeapDict1.0.1、numpy1.26.4 |

最稳妥的迁移是分别用 conda-pack 打包当前实际 train、gptq、PROVER 环境，到目标机解包后运行 conda-unpack；普通venv直接rsync不能保证旧shebang和基础Python路径有效。当前 train/gptq 解释器实际路径可由 `readlink -f env/train/bin/python` 等查询。CUDA驱动由目标服务器提供，需兼容对应torch构建。

也可用 conda/mamba 创建相同Python版本的独立环境，安装附件 `migration/*-requirements.txt` 的包版本。**这些是实测inventory，不是已在空白服务器验证过的通用pip lock**：torch CUDA轮子、faiss-gpu、flash-attn和本地GPTQModel需要单独处理，不能无差别一次pip安装后宣布环境通过。GPU FAISS优先恢复原conda环境。P1从固定源码安装GPTQModel，避免pip偷偷替换torch/transformers版本；不需要安装旧evalscope、TRL外部trainer来替换现有适配器。

配置新机路径；流水线命令仍要求 `env/bootstrap/bin/python` 存在：

```bash
export REALPROVER_TRAIN_PYTHON="$ROOT/env/train/bin/python"
export REALPROVER_GPTQ_PYTHON="$ROOT/env/gptq/bin/python"
export REALPROVER_EVAL_PYTHON="$ROOT/env/prover/bin/python"
export USE_TF=0 USE_FLAX=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
```

将这些变量保存到**新机**环境脚本，所有控制器/子进程继承同一组值。runtime.json 中 eval 默认是旧机器绝对路径，必须覆盖。环境重建后重新运行预检和smoke；本手册在原服务器做的是文档/资产验证，没有宣称目标服务器已验证。

## 6. 需要下载哪些模型

| 资产 | 来源 | 固定身份/大小（约） | 用途 |
|---|---|---|---|
| REAL-Prover BF16 | FrenzyMath/REAL-Prover | fe76f68d9a88f342cb7b546307c20292fea9cced；15.23GB权重 | 所有方法基座及P4教师，同一份即可 |
| 检索基座 | intfloat/e5-mistral-7b-instruct | 当前两片权重共14.22GB；按retrieval_assets.json逐文件SHA核验 | 生成检索embedding |
| 检索LoRA | FrenzyMath/LeanSearch-PS | adapter_model约42MB；按SHA核验 | 不是独立完整7B模型 |
| FAISS及answers | FrenzyMath/LeanSearch-PS-faiss | index约4.14GB，answers约232MB | 252444条前提与其索引，必须配套 |

REAL-Prover使用仓库脚本，默认hf-mirror，**单文件串行HTTP Range续传**，固定 `.part`，完整文件做大小/SHA校验：

```bash
env/bootstrap/bin/python research/scripts/download_realprover.py
# 官方端点可改为：--endpoint https://huggingface.co
```

其他资产可用HF CLI/Hub snapshot_download或ModelScope搬运，但必须保持表中同一上游内容。检索资产目前没有独立核实的上游revision，**本实验身份以research/manifests/retrieval_assets.json的内容SHA为准**；最新HEAD下载结果不匹配时停止，优先从本服务器复制原文件，不能重新pin变化内容后声称完全一致。

HF下载示例（先安装huggingface_hub到bootstrap；新下载的检索模型必须再核验）：

```python
from huggingface_hub import snapshot_download
endpoint='https://hf-mirror.com'
snapshot_download('intfloat/e5-mistral-7b-instruct', local_dir='assets/retrieval/e5', endpoint=endpoint, max_workers=1)
snapshot_download('FrenzyMath/LeanSearch-PS', local_dir='assets/retrieval/adapter', endpoint=endpoint, max_workers=1)
snapshot_download('FrenzyMath/LeanSearch-PS-faiss', local_dir='assets/retrieval/index', endpoint=endpoint, max_workers=1)
```

查看下载的索引/answers实际文件名（上游可能为answer.json），设置：

```bash
export REALPROVER_RETRIEVAL_BASE="$ROOT/assets/retrieval/e5"
export REALPROVER_RETRIEVAL_ADAPTER="$ROOT/assets/retrieval/adapter"
export REALPROVER_RETRIEVAL_INDEX="$ROOT/assets/retrieval/index/LeanSearch-PS-faiss.index"
export REALPROVER_RETRIEVAL_ANSWERS="$ROOT/assets/retrieval/index/answers.json"
export LEANSEARCH_PORT=8080
```

新机必须把 retrieval_assets.json 中旧绝对路径改为对应的新绝对路径，**保留每条size/sha256不变，并逐文件重新计算验证**；answers改名不影响内容SHA。pin_retrieval_assets.py现默认使用models/retrieval，并支持上述环境变量；重新pin仅记录当前内容身份，不能代替与原清单逐项比对。路径重绑改变manifest哈希，健康检查必须使用新manifest，旧服务不能混用。此操作在启动新机实验之前完成并commit。

## 7. 需要下载哪些数据

| 数据 | revision | 下载范围 |
|---|---|---|
| open-thoughts/OpenThoughts3-1.2M | 61bcf9d4eb38b30295efc2021227a63cc5bb34c8 | 全部120个data/train-*.parquet，约28.19GB；全量shuffle取32768，不能只取首片 |
| HuggingFaceFW/fineweb-edu | 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 | **只需sample/10BT/000_00000.parquet**，不下载完整数TB库 |
| AI-MO/NuminaMath-CoT | 9d8d210c9f6a36c8f3cd84045668c9b7800ef517 | 当前实验直接使用附件固定datasets/numinamath_128/calib.jsonl；无需重抽样或下载全库 |
| miniF2F-test | 官方REAL-Prover源码3e5987dc1009a7addc935bcd7b4ae777738a0848对应文件 | REALPROVER/data/minif2f_test.jsonl已经随代码保存；244题 |

OpenThoughts/FineWeb用于训练；Numina仅PTQ；miniF2F及其他Lean题仅评估。无需下载Qwen3-1.7B、WikiText或旧state–tactic训练集。ProofNet/FATE-M不属于本次正式主评估。

```python
from huggingface_hub import snapshot_download
snapshot_download('open-thoughts/OpenThoughts3-1.2M', repo_type='dataset',
    revision='61bcf9d4eb38b30295efc2021227a63cc5bb34c8',
    allow_patterns=['data/*.parquet'], local_dir='datasets/openthoughts3',
    endpoint='https://hf-mirror.com', max_workers=1)
snapshot_download('HuggingFaceFW/fineweb-edu', repo_type='dataset',
    revision='87f09149ef4734204d70ed1d046ddc9ca3f2b8f9',
    allow_patterns=['sample/10BT/000_00000.parquet'], local_dir='datasets/fineweb-edu',
    endpoint='https://hf-mirror.com', max_workers=1)
```

上面的Hub下载以单文件并发数1限制，不等同于REAL-Prover专用脚本的Range续传实现。训练实际强制离线读取本地parquet，下载必须提前完成。保持文件路径与排序；不要复用Qwen3的token缓存。可在训练环境执行 `research/scripts/prepare_stage2_cache.py --output <新目录>` 提前生成Stage2缓存。

## 8. Lean 4.16与本地存储

需Lean **v4.16.0**，不能直接换新版。主要固定提交：lean_test_v4160=faf496a4bc025db8f13170ab85b8bdd5a0737a36；mathlib=a6276f4c6097675b1cf5ebd49b1146b735f38c02；jixia=157c55dd5754db3795838e7c05104b91535ee5fc；interactive=9e7417953d80de8e4daff17c550ad872fde7f9bb；repl=eed8e6ff51e2c168e13d89e75015719e676bd9c9。其余传递依赖由原lake-manifest固定。

安装elan后安装 `leanprover/lean4:v4.16.0`；若GitHub不可达，可下载工具链压缩包上传。然后：

```bash
env/bootstrap/bin/python research/scripts/restore_lean_sources.py
# build脚本使用历史工具链目录名，标准elan目录可加兼容软链接：
ln -s "$HOME/.elan/toolchains/leanprover--lean4---v4.16.0" "$HOME/.elan/toolchains/leanprover-lean-v4.16.0"
env/bootstrap/bin/python research/scripts/build_lean_environment.py --skip-download --output runs/lean_build
env/bootstrap/bin/python research/scripts/stage_lean_local.py --destination /tmp/realprover_lean_new
export REALPROVER_LEAN_WORKSPACE=/tmp/realprover_lean_new/lean_test_v4160
export REALPROVER_INTERACTIVE_ROOT=/tmp/realprover_lean_new/interactive
export REALPROVER_JIXIA_BIN=/tmp/realprover_lean_new/jixia/.lake/build/bin/jixia
export REALPROVER_LEAN_BIN=/tmp/realprover_lean_new/toolchain/bin
export PATH="$REALPROVER_LEAN_BIN:$PATH"
```

兼容链接仅在该目标不存在时创建。恢复脚本尝试官方codeload及GitHub镜像，ModelScope没有已经确认可用的Lean工作区包。Mathlib还需要匹配的预编译cache；若在线恢复不可用，可搬运本机完整已构建 dependencies 和工具链。**保留Lake依赖的.git元数据**，否则lake可能删除目录并重新克隆。新机重新做源码/二进制哈希及live验证；/tmp重启会丢失，需重新stage。不要让多个独立服务器共享同一可写Lean临时目录。

## 9. 预检、smoke与从头正式实验

完成所有路径修改后先commit，运行中不要修改被fingerprint覆盖的代码、JSON、TOML、数据文件或环境。新机fingerprint会因为绝对路径/解释器/数据mtime不同而变化，这是预期行为。

```bash
CUDA_VISIBLE_DEVICES=3 env/bootstrap/bin/python research/scripts/start_retrieval.py
CUDA_VISIBLE_DEVICES=0 env/bootstrap/bin/python research/scripts/validate_preparation.py --gpu --full-model --native-runtime --output runs/new_preflight
```

等待 `/health` ready，核对asset_manifest_sha256、adapter_loaded=true、252444条前提；检查真实Lean可证明样例与拒绝sorry、生成和导出parity。再停止由该工作区管理的检索服务（单卡smoke训练期间不能常驻检索）：

```bash
PYTHONPATH="$ROOT/research/scripts:$ROOT/research/adapters" env/bootstrap/bin/python -c 'import retrieval_service; retrieval_service.stop()'
mkdir -p runs
CUDA_VISIBLE_DEVICES=0 nohup env/bootstrap/bin/python -u research/scripts/experiment_pipeline.py campaign --mode smoke --results runs/reproduce_v1 > runs/reproduce_v1.smoke.nohup.log 2>&1 < /dev/null &
# 完整smoke退出后执行，必须所有P0-P6都PASS：
env/bootstrap/bin/python research/scripts/experiment_pipeline.py gate --results runs/reproduce_v1
# gate通过后：
CUDA_VISIBLE_DEVICES=0 nohup env/bootstrap/bin/python -u research/scripts/experiment_pipeline.py campaign --mode formal --results runs/reproduce_v1 > runs/reproduce_v1.formal.nohup.log 2>&1 < /dev/null &
```

上述是现有通用入口支持的**从头串行复现**，会包含P0。smoke是两题、缩小训练/搜索预算，只验证正确执行，不能报告正式分数。泛用campaign是依赖顺序执行，失败会退出该campaign；它不提供四卡任务失败隔离。

**四卡独立失败隔离只在 resume_four_gpu.py 的Adaptive控制器中实现。** 老run_four_gpu.py遇到一个任务失败会终止整批，不应直接当作新的四卡续跑入口。Adaptive入口要求已经审计导入P1/P2/P3 prepared，不能当零资产从头训练入口。不要为了绕过这些条件伪造prepared/receipt/validation PASS。

## 10. 搬运现有训练结果、四卡继续评估

当前本机批次是 `runs/realprover_formal_four_gpu_v10`，全部P1–P6训练已完成。若目的是加速剩余评估，优先搬运对应**完整方法目录**及其证据，而不是重新训练：每个方法的 `prepared.json`、模型权重/tokenizer/export检查、方法身份/native checkpoint、训练报告、reuse_manifest；P4/P6还需P3共同初始化证据，共享stage1/stage2摘要和stage2缓存来源也要保留。`prepared.json.evidence` 是确切文件清单，不能只靠目录名挑文件。只评估时可将训练语料下载留待之后，但现有通用导入/指纹审计仍可能要求完整source资产布局。

复制大文件可用 `rsync -a --partial --append-verify`，先对静态已完成模型复制；**不要把正在写入的generated目录当一致性快照**。可用“只搬运模型、在另一服务器独立评估”的方式继续推进本机实验；要接续某个正在运行的方法，应先停止该方法的写入并固定快照，再迁移对应原始attempt JSON。不能让两台机器共同写同一个结果目录。

现有launcher的适用边界：

| 入口 | 已实现用途 | 跨服务器额外工作 |
|---|---|---|
| launch_continuation.py | 审计复用P1/P2/P3；继续P4/P5/P6训练，训练优先或mixed | source/status和prepared证据包含旧绝对路径，需先做显式路径映射与内容审计 |
| launch_evaluation_resume.py | 停止的旧批次全6模型 + partial attempts导入新批次 | 要求source STOPPED_BY_USER、固定验证报告、实际Lean路径可用、干净Git；不是任意机器开箱迁移器 |
| resume_four_gpu.py | 已准备资产的独立失败隔离调度 | 需要当前指纹prepared和明确绑定的smoke门禁/授权文件 |

**跨机器自动路径映射尚未实现。不能把下面同机命令直接当成跨机成功保证。** 原机的停止批次可用：

```bash
env/bootstrap/bin/python research/scripts/launch_evaluation_resume.py --source runs/STOPPED_SOURCE --results runs/NEW_BATCH
```

新机需先完成迁移适配：保存不可变原始source清单；逐条将旧根路径映射到新路径并比对内容SHA；保留原prepared/receipt及其哈希作为lineage；生成绑定当前fingerprint的新prepared/continuation证据；重新执行目标机验证和smoke，或记录用户明确允许的本次续跑豁免。不能直接批量替换JSON里的指纹并保留PASS。完成这些适配后再运行Adaptive调度，GPU3独占共享检索，GPU0/1/2每卡一个评估任务，结束后领取P4/P5/P6队列。单任务FAILED只影响自身；全局指纹变化、共享基础设施故障和控制器退出仍可能影响整批。

attempt恢复规则：`native/generated/<id>/<id>_<0..63>.json`，成功必须有完整Lean验证；成功题跳过，失败题只补缺失槽，历史超时/Lean错误占用原尝试并算失败，不追加到64之外。JSON原子排他写入，禁止覆盖。重启后的随机数状态不等于不中断进程，故保证预算与证据连续，不保证bitwise一致。旧v9发生的内存压力超时会作为历史失败保留；这种续跑与修复后全新评估应分别标识。

## 11. 必须保留的修复、监控和结果

- vLLM0.6.6的n>1采样旧引用会累积host内存；`REALPROVER/manager/thirdparty/generation_memory.py` 在同步generate完成后清除已消费group。两条生成路径均使用该包装，不修改site-packages，不改变返回结果。旧机修复前评估进程合计约471GiB；修复后每个约8.3GiB稳定。新机仍需持续监控，不能仅凭短测保证永不泄漏。
- Lean timeout/异常结束当前attempt并继续下一次；保留超时记录。stderr不再使用无人读取PIPE。Lean放本地SSD可减少共享文件系统stall。
- P5不保存逐block checkpoint；E2E仅rolling恢复文件，最终导出核验后移除，不能删除仍被训练进程使用的缓存。
- 正式分数以固定244为分母；中途“成功/已完成”只作阶段统计。P3生成有效策略、搜索更深，单attempt比P1/P2慢不等于卡死。

查看位置：

```text
runs/<batch>/control/status.json                  调度状态/GPU/PID/当前日志路径
runs/<batch>/formal/P*/prepare.<timestamp>.log     训练/量化日志
runs/<batch>/formal/P*/evaluate.<timestamp>.log    当前评估日志（以status内log为准）
runs/<batch>/formal/P*/eval/native/generated/<id>/<id>_<attempt>.json
                                                  实际候选/节点/证明/Lean验证
runs/<batch>/formal/P*/eval/evaluation_resume.json  继承attempt及历史证据
runs/<batch>/formal/P*/receipt.json                仅全量完成并通过检查后生成
runs/realprover_retrieval/server.log               检索日志
```

原始输出 `collect_results[*].calls` 中每次调用为 `[state,candidate_tactics,scores,prompt]`；可按同题同attempt比较方法。最终聚合采用research/adapters/evaluation.py，不使用旧get_stats或仅扫描success字段。定期查看 `nvidia-smi`、主机MemAvailable和各vLLM worker的VmRSS/RssAnon；GPU利用率为0可能是在等待Lean，要结合新attempt文件、CPU/磁盘活动和超时日志判断。

## 12. Git与复现记录

新机记录：两个分支HEAD、外部四个源码HEAD与AWQ补丁SHA、模型/索引/校准数据SHA、Python包清单、GPU/驱动、Lean二进制SHA、环境变量路径、完整启动argv和批次指纹。下载完成、路径重绑、代码适配后先检查diff并commit，再smoke/正式运行。运行中不要改会进入fingerprint的文件；新改动使用新批次，保留旧失败/中断证据。不要将权重、原始大语料、cache、运行日志或访问令牌提交Git。

配套历史审计：[环境恢复](environment_restoration.md)、[四卡设置](four_gpu_formal.md)、[内存修复](host_memory_leak_20261026.md)、[评估续跑](evaluation_resume_20261026.md)。
