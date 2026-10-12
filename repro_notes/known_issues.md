# Preparation findings

No GPU experiment has been run. The pre-existing GPU process is preserved.

## Source fixes

- `PolicyGKDTrainer` overrides the GKDConfig beta with its own default 0.5. The paper-aligned path now explicitly passes beta=1.0. The actual top-k computation gathers full-vocabulary log probabilities without renormalizing; it is a truncated KL contribution, not a standard normalized top-k KL.
- Stage 2 now explicitly uses cosine, seed=2 and data_seed=2. Scale-only selection targets `.weight_quantizer.scale`. Non-finite loss and scale gradients fail rather than silently replacing NaNs.
- Local BF16 paths in Stage 1 were incorrectly treated as quantized checkpoints. The initial BF16 snapshot is now loaded normally. Cached validation data are restored rather than discarded.
- Local parquet loaders require the complete 1.2M OpenThoughts dataset. Shuffle/tokenization semantics stay unchanged. Incomplete block calibration raises an error.
- The upstream conversion script loads without restoring QuantLinear modules. Use `repro_scripts/export_qat.py`, which restores modules strictly, materializes quantized weights and verifies logits before/after reload.

## Environment and downloads

- Upstream requires accelerate<0.34 but TRL 0.17 requires accelerate>=0.34. The isolated training environment uses accelerate=1.6.0; this deviation must be documented and checked.
- Official PyPI's large torch wheel stalled. A private copy of torch=2.2.2/triton=2.2.0 and CUDA libraries from the existing efficientqat environment is being used in the new environment. No package in the old environment is modified.
- HF large files time out. After testing parallel bounded ranges, the user requested single-file resume. The mirror downloader now uses one continuous streaming connection per file, resumes on disconnect and checks official LFS SHA256. A `.partial` file is never considered a completed asset.
- Current Arrow 26 wheels require NumPy 2 at import time, even though pip check passes. Training pins Arrow 19.0.1, pandas 2.2.3, scipy 1.15.3, scikit-learn 1.6.1 and PEFT 0.15.2 with NumPy 1.26.4 / transformers 4.52.4.
- GPTQModel v2.2.0 supports Qwen3 and W2 TORCH inference. Its wheel command accesses optional `torch.xpu` on torch 2.2. CPU-only Python wheel construction uses `GPTQMODEL_FORCE_BUILD=1` with CUDA hidden, avoiding both that branch and CUDA extension compilation. GPU quantization/reload remains unvalidated until user release.

## Strict comparison blockers

The paper does not publish exact NuminaMath-128 indices/template, all GPTQ/AWQ hyperparameters, Stage 2 context length, EvalScope version, benchmark revisions/date slices, thinking/stop configuration or per-model 32K/8K limits. Formal comparison remains blocked until these are resolved or explicitly reported as unavailable. A candidate run under an implementation assumption must never be labelled strict reproduction.

R6 is especially ambiguous: Table 6 reports W2/W3 while its prose describes W3 single-epoch experiments. W2 three epochs is an inference from Section 4.1, not an independently confirmed Table 6 setting.
