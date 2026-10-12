# R8 EfficientQAT W2g128

Status: CPU_ADAPTER_PASS / FULL_MODEL_SMOKE_PENDING / FORMAL_NOT_COMPARABLE.

Official source and original paper are downloaded by `prepare_qat_baselines.py`. Model and dataset files are shared with the ReasoningQAT project. Planned independent stages: Block-AP, then scale-only CE E2E-QP for three epochs. R3 reuse requires a source/initialization/operator equivalence audit. Original packed checkpoint format and `.scales` differ from the current ReasoningQAT fake-quant export path.

Table 3 references: MATH-500 24.8, LiveCodeBench 0.6, MMLU-Redux 29.1, GPQA-Diamond 12.1, IFEval 25.1, average 18.3 percent. These are paper values, not measured results.

CPU integration checks independent Block-AP, two scale-only CE updates without teacher, strict packed reload and materialized logits parity (difference 0). Production collator tensors match R4's actual collator (204 masked smoke tokens). Full-model GPU smoke and five-task evaluation remain pending. Author formal optimizer/token configuration remains unverified; the executable adapter supports smoke only.
