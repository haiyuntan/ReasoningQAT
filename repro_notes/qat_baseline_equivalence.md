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

The preparation wrappers support `--plan-only`; execution currently reports `BLOCKED` until the independent adapters and their export/runtime verification are implemented. They are interface contracts, not runnable training implementations. Do not relabel R5 as R8 or switch R4's loss flag and call it R9.

Runtime sharing saves repeated CUDA/PyTorch library copies. Changes needed only by a baseline must be installed in its own venv and recorded; modifying the shared training environment requires revalidating all users of that environment.
