# R9 BitDistiller W2g128

Status: PREPARING / adapter BLOCKED_PENDING_IMPLEMENTATION.

Official source and original paper are downloaded by `prepare_qat_baselines.py`. Initialization must independently use mixed-domain asymmetric clipping and native INT2 quantization. Do not initialize from R3. Stage 2 must preserve upstream CAKLD directions and normalization, freeze the shared BF16 teacher, and audit confidence estimation and effective token counts for three epochs.

Table 3 references: MATH-500 12.2, LiveCodeBench 1.0, MMLU-Redux 22.4, GPQA-Diamond 21.7, IFEval 19.0, average 15.2 percent. These are paper values, not measured results.

Required next evidence: Qwen3 clipping hooks, clipping range checkpoint reload, native STE quantizer updates, CPU CAKLD value/gradient/mask checks, single-process confidence estimator, audited teacher trajectory choice, author clipping budget, minimum smoke and export parity.
