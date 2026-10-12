# R9 BitDistiller W2g128

Status: CPU_ADAPTER_PASS / FULL_MODEL_SMOKE_PENDING / FORMAL_NOT_COMPARABLE.

Official source and original paper are downloaded by `prepare_qat_baselines.py`. Initialization must independently use mixed-domain asymmetric clipping and native INT2 quantization. Do not initialize from R3. Stage 2 must preserve upstream CAKLD directions and normalization, freeze the shared BF16 teacher, and audit confidence estimation and effective token counts for three epochs.

Table 3 references: MATH-500 12.2, LiveCodeBench 1.0, MMLU-Redux 22.4, GPQA-Diamond 21.7, IFEval 19.0, average 15.2 percent. These are paper values, not measured results.

CPU integration checks independent clipping, original QLinear INT2 STE and CAKLD updates with frozen teacher, strict native reload and materialized logits parity (difference 0). Confidence averages actual diagnostic batches rather than the original buggy distributed loop. Collator tensors match R4 (204 masked smoke tokens). Full-model GPU smoke and five-task evaluation remain pending. Formal clipping count, confidence estimator and trajectory/optimizer settings remain unverified; the executable adapter supports smoke only.
