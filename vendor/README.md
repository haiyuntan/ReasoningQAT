# Isolated training operators

These small source snapshots come from the existing pinned local ReasoningQAT
and EfficientQAT checkouts. `sources.json` records each original path/hash and
the adapted hash; upstream commit identities are in
`../manifests/local_source_commits.json`. The original checkouts and FedProver
are not modified.

Runtime imports use the ReasoningQAT quantizer and an independent private
EfficientQAT package. No Qwen3 experiment pipeline is launched. Adaptations:

- Qwen2 RMSNorm instead of Qwen3 RMSNorm; eager attention available.
- Native QAT loader respects checkpoint dtype and preserves FP32 scale values.
- Independent operator loader resolves this vendored EfficientQAT copy.
- Training helpers use the migration workspace and no longer require TRL at import.
- Legacy block driver snapshot retains its provenance; executable block training
  uses `adapters/training.py` with per-sample masks/rotary data and actual validation.

P3 keeps the ReasoningQAT affine quantizer, `.7` initialization range multiplier,
block MSE, original weight/quant parameter grouping and cosine schedules. P5
uses EfficientQAT's separate quantizer, native packing layout and independent
Block-AP initialization. Its differentiable PyTorch unpacking is a kernel port;
it supports accuracy evaluation, not an INT2 performance claim.

Stage2 uses an explicit torch loop, tested against pinned RR/top20 KL math.
It preserves shifted, final-assistant-only labels, global shuffle seed2 and
matched effective token budgets. Optimizer/warmup implementation is an explicit
engineering adaptation, recorded in training evidence. This is a migration,
not a claim of byte-identical execution of the original TRL trainer.
