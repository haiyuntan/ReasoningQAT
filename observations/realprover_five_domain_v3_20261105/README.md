# REAL-Prover five-domain representation observation

Status: inputs frozen; GPU forward not yet complete.

## Clean two-panel Qwen3 comparison requested by the user

[PNG](fig2b_original_vs_v3_nopca_clean.png) · [PDF](fig2b_original_vs_v3_nopca_clean.pdf)

Left: the original Fig.2(b), including its existing header, axes and colorbar. Right: the v3 Qwen3-1.7B no-PCA candidate, using historical-default-style random init, LR200, perplexity30,1000iterations,seed42. No additional titles, panel letters, captions, arrows or footnotes. The two source images are pasted losslessly, without resizing. Existing chart labels are retained. This is a Qwen3 comparison, not a REAL-Prover result. Display transforms of the original v3 candidate remain documented in its report.

## What data is used for t-SNE?

Math here is OpenThoughts3 mathematics training conversations. Code and Science also come from OpenThoughts3. Pre-training comes from FineWeb-Edu. MATH-500 is an evaluation benchmark, not the source of the Math activation points. Test data is not repurposed as calibration here.

For REAL-Prover, the added domain is **Formal Proof (Lean State–Tactic)**: actual current Lean proof states and reference next tactics from the official FrenzyMath training dataset. It should not be described as a newly proven knowledge module; a t-SNE cluster can arise from syntax or format as well as semantic features.

## Frozen protocol and why there are two input views

The same512 original four-domain source groups from Qwen v2/v3 are recovered from the corpora and retokenized with REAL-Prover's own tokenizer. Qwen3 token IDs are not reused. Formal Proof contributes128 unique state groups, selected with seed2 from the already exact-overlap-screened training corpus. No evaluation outcome is consulted. Source-theorem IDs and semantic overlap remain unknown.

Raw states are much shorter than the original2048-token reasoning conversations. Median raw-state length in the earlier corpus audit was75 tokens, versus914 tokens for full prompted inputs. It would be misleading to pad or pack unrelated states merely to pretend the same context construction was used.

- **Primary `raw128`:** every domain uses exactly the last128 raw-content tokens. Reasoning inputs include original question/answer text, FineWeb uses document content, and Formal Proof uses raw state + newline + reference tactic. No domain-specific chat instructions are included. Formal pairs require at least128 total tokens and a2..48-token tactic, ensuring the tail contains state content and the full tactic. This induces a longer-state selection bias. The last128 tokens of long reasoning conversations are usually response content, not the original question.
- **`native_context` control:** retain at most the last2048 tokens of native-format input. For Formal Proof, preserve full provided proof/retrieval context and append its reference tactic. Activation positions are restricted to the actual current state and reference tactic, excluding instruction, in-context example and retrieval positions. Other reasoning domains use native chat formatting and eligible response positions; FineWeb uses raw document context. Select the same four relative positions from the last128 eligible content tokens. This control intentionally has varying context lengths and is not a reproduction of the paper's original calibration prefix truncation.

Each view has128 groups/domain and4 sampled token positions/group:2,560 points. The raw primary sampled Formal Proof points contain437 state,72 tactic and3 boundary positions. Token positions are causally contextualized and correlated within each source; they are not independent proof examples.

Model: unmodified BF16 REAL-Prover-v1; one-based Transformer block14 output, before final model normalization. A full28-block forward validates the hook against hidden_states[14], then execution can stop at block14 after an exact prefix check. No training or quantizer changes.

All five domains jointly fit a single t-SNE **without PCA or feature normalization**, random initialization, LR200, perplexity30,1000iterations, fixed seeds42/2/3. Primary seed42 is fixed before the forward. Separate views/seeds have unrelated t-SNE coordinate systems.

## Interpretation

This can test whether Formal Proof inputs occupy a visibly different representation distribution and whether that pattern persists without templates. It cannot prove the model knows how to prove a theorem, or establish a separate knowledge module. The original REAL-Prover training corpus already contains state–tactic pairs; adding them to QAT calibration concerns preserving existing proof ability.

To test calibration necessity, the stronger follow-up is paired BF16/quantized activation or predictive-distribution error on Formal Proof, then independent Lean theorem evaluation. Those training/accuracy experiments are not replaced by this image.

## Display, isolation and evidence

Colors are categorical plotting IDs: Code0,Math1,Science2,Formal Proof3,Pre-training5. The0..5 Viridis bar has no entropy or confidence meaning. General label/font layout is adapted to fit five domains. Display uses the same disclosed global rotation/reflection and common scaling as v3; Proof is not moved independently. Native and display coordinates are saved together.

GPU3 only, batch1,32GiB allocator cap, approximately50% duty cycle. The existing supervisor checks retrieval health and real query latency, memory reserves and the active production fingerprint; only this observation child may be stopped. No production source, environments, weights or evaluation results are changed. Only existing packages are used.

Local `inputs.jsonl` and high-dimensional `*_features.npz` are not intended for Git. Publish small metadata, coordinates, PNG/PDF/SVG, methods and source code. Scripts assume this folder sits under the existing Reasoning workspace and uses the existing isolated observation environments.

Sources: [official ReasoningQAT data processing](https://github.com/yasu0001/ReasoningQAT/blob/main/datautils_block.py), [ReasoningQAT Fig.2(b)](https://proceedings.iclr.cc/paper_files/paper/2026/file/209423f076b6479ab3a4f45886e30306-Paper-Conference.pdf), [official state–tactic dataset](https://huggingface.co/datasets/FrenzyMath/state_tactic_pairs).
