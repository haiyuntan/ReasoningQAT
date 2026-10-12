# R8 EfficientQAT W2g128

Status: PREPARING / adapter BLOCKED_PENDING_IMPLEMENTATION.

Official source and original paper are downloaded by `prepare_qat_baselines.py`. Model and dataset files are shared with the ReasoningQAT project. Planned independent stages: Block-AP, then scale-only CE E2E-QP for three epochs. R3 reuse requires a source/initialization/operator equivalence audit. Original packed checkpoint format and `.scales` differ from the current ReasoningQAT fake-quant export path.

Table 3 references: MATH-500 24.8, LiveCodeBench 0.6, MMLU-Redux 29.1, GPQA-Diamond 12.1, IFEval 25.1, average 18.3 percent. These are paper values, not measured results.

Required next evidence: Qwen3 block capture and reload parity, scale-only updates, no teacher forward, exact effective token budget vs R4, author optimizer configuration, minimum smoke and unified five-task evaluation.
