# Fig. 2(b) reconstruction v2: calibration-aligned inputs

Status: DONE (exit0, 56 fixed-grid panels plus4 point-count controls). A completed forward is not a confirmed reproduction of the paper's geometric claim.

The original paper explicitly specifies full-precision Qwen3-1.7B, Transformer block14, and Pre-training/Code/Math/Science. Its exact plotting inputs, activation location within the block, pooling, normalization, sampling and t-SNE configuration are not released. Official calibration code is therefore evidence for an input construction hypothesis, not proof of the figure's original procedure.

## Evidence-based corrections from v1

- v1 used only human questions, 128 to512 tokens. Official `get_openthoughts_shuffled` includes all human/assistant turns, applies the native chat template, requires at least2048 tokens, and truncates from the beginning. v2's primary input follows that construction.
- v1 selected the first eligible reasoning questions in shard order. v2 uses seed2 random shard order and random candidate rows, caps each reasoning domain at32 per shard, and collects128 eligible groups/domain. This is not the official whole-dataset HF shuffle. All three views share the same source groups.
- FineWeb follows the official BOS+document+EOS buffering, exactly2048 tokens and discard-after-cut behavior. It is sequential, as in upstream; source-order bias remains. The same FineWeb representations are a reference across views.
- Primary panel uses sampled individual token activations, not only sequence means. Mean and last-position controls remain available. Sixteen token positions/source are sampled uniformly from positions32..2047 with a source-hash-derived seed. Tokens in one source are correlated. No token identity filtering or domain-specific position selection is performed.
- Block14 is explicitly one-based: zero-based block13. A full28-block forward checks the block input against `hidden_states[13]` and block output against `hidden_states[14]`. Execution is then shortened to14 blocks and all captured locations must match the full forward exactly on the probe. The hook remains before the final model norm.

## Frozen comparison grid

Three views: official calibration-style chat; raw full conversation without template; response-only without template. Non-chat views are controls, not claims about upstream.

Three activation locations: block input, block output, and post-attention layer-normalized MLP input. Three summaries:16 sampled tokens, sequence mean, and last token. Two preprocessing choices: raw and L2 normalized. All use joint PCA50 and joint t-SNE(perplexity30,1500iterations). Fixed seed2 for all panels; the declared primary panel additionally has seeds3/42 as sensitivity checks. No selected layer/seed winner.

Declared primary: `plots/calibration_chat_block_out_token_none_seed2.png`.

High-dimensional within-domain mean squared distance to centroid and reasoning/pre-training dispersion ratios are retained for every configuration. Domain silhouette is a supplementary measure, not the paper's dispersion claim. A visually similar t-SNE plot alone does not confirm the claim; a ratio above1 in one selected control does not establish robustness.

## Isolation and output

GPU3 only, allocator cap12GiB, batch1, approximately50% duty cycle. Existing guard monitors retrieval health and actual queries, memory reserves, active production fingerprint, and new retrieval failures. It can stop only the observation child. Running production source/configuration and v1 outputs are untouched. Shared hardware has no strict QoS guarantee.

Local high-dimensional features and tokenized inputs are retained for analysis but are not intended for Git. Publish PNGs, 2D coordinates, protocol, selection metadata, hook validation, statistics and scripts. Original and v1 figures remain separately labeled.

Sources:

- https://proceedings.iclr.cc/paper_files/paper/2026/file/209423f076b6479ab3a4f45886e30306-Paper-Conference.pdf
- https://github.com/yasu0001/ReasoningQAT/blob/main/datautils_block.py
