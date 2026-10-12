# Table 3 baseline preparation and equivalence audit

R8 and R9 are required. R6/R7 remain optional. Reference accuracies are stored separately from observations; neither new baseline has measured benchmark results yet.

| Item | R8 EfficientQAT | R9 BitDistiller |
| --- | --- | --- |
| Official source | OpenGVLab/EfficientQAT, `39f37f3b6053681c9b1cd4c9dcaf692d8999459e` | OpenBitSys/BitDistiller, `9faaab822614a167fdf3d4421d0a054c55c56f9c` |
| Original dependency issue | Transformers 4.40.1 does not support Qwen3 | Transformers 4.31.0 does not support Qwen3; example uses DeepSpeed/distributed reduction |
| Prepared runtime | Independent entry venv using the pinned Qwen3-compatible runtime read-only | Independent entry venv using the pinned Qwen3-compatible runtime read-only |
| Initialization | Block-AP, 4096 mixed sequences of length 2048; independently audit before any R3 reuse | Independent asymmetric clipping with original `auto_2clip_layer`; author calibration count remains unconfirmed |
| Quantizer | Original native EfficientQAT packed quantization and trainable `.scales` require port/parity checks | Original `int2-asym` STE quantizer, group 128; train quantized weights, not ReasoningQAT scale-only |
| Training | Original scale-only E2E-QP with CE; no teacher forward | CAKLD self-distillation with frozen BF16 teacher and audited confidence coefficient |
| Data | Same frozen mixed calibration and OpenThoughts selection as R4 | Same domains and Stage 2 sample/token budget; no Pile/Alpaca substitution |
| Missing checks | Qwen3 block capture, quantized reload, R3 equivalence, optimizer and masked-token accounting | Qwen3 clipping hooks, independent clip reload, coefficient estimator, Trainer API, masked-token accounting and teacher trajectory source |

Upstream CAKLD computes `beta * KL(student || teacher) + (1-beta) * KL(teacher || student)`, applies `labels != -100`, sums across sequence positions and averages across examples. It does not use ReasoningQAT's token-normalized top-20 KL. Upstream teacher logits come from a `no_grad` forward. The coefficient estimator uses distributed operations unconditionally in the original entry, which must be adapted and numerically checked for a single process.

The wrappers now execute **smoke-only independent adapters**. Formal execution explicitly reports NOT_COMPARABLE until author settings and a full-token training implementation are verified. R8/R9 must pass full-model GPU smoke before `adapter_verified` becomes true.

`baseline_ops.py` imports original operators under private namespaces. R8 uses original fake quantization and native pack, with differentiable PyTorch bit unpack replacing the Triton dequantization kernel. Its independent one-block MSE loop optimizes W/s/z with AdamW and cosine minimum LR/20; parameters are FP32 with BF16 autocast. Zero points are clamped to INT2 range consistently with the fake quantizer. E2E-QP freezes packed codes and only trains FP32 `.scales` using CE, without a teacher.

R9 uses the original asymmetric clipping function (grid20, shrink.5, q/k exclusion, original unmodified-output propagation). Only the short-sequence zero-stride bug is guarded. `clip_ranges.pt` stores upper/lower group bounds, permitting initialization reconstruction from the pinned base without another full checkpoint. Original QLinear INT2-asym/STE and all original student parameters are used. CAKLD retains the exact upstream unshifted mask, KL directions and sequence-sum/batch-mean reduction. The diagnostic confidence coefficient averages actual teacher batches including prompts, correcting the original off-by-one/fixed-denominator/distributed-only bugs; formal author equivalence remains unresolved.

CPU integration on real tiny Qwen3 checks both stages, finite gradients and actual optimizer updates, strict native state reload and materialization parity (logits max difference 0). R8 Block-AP matches upstream name-based parameter groups, including input/post-attention RMSNorm and q/k normalization weights; gradient norms are measured without clipping during Block-AP. Actual R4 collator functions are extracted and compared tensor-for-tensor: the frozen two-example smoke inputs contain 204 effective tokens. `export_baseline.py` applies separate native reload/export gates for these formats. Full Qwen3-1.7B GPU training/export and five-task evaluation remain required. Two smoke steps and diagnostic matrix LR do not constitute verified three-epoch paper baselines. BF16 materialized exports do not imply packed INT2 inference speedup.

Runtime sharing saves repeated CUDA/PyTorch library copies. Changes needed only by a baseline must be installed in its own venv and recorded; modifying the shared training environment requires revalidating all users of that environment.
