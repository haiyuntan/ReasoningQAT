# Expanded representation observation v4

Isolated from formal miniF2F evaluation. No model training, checkpoint changes, production environment changes, or production process control.

## Frozen comparison design

- Qwen3-1.7B BF16: four domains, 512 source groups/domain; original 128/domain retained.
- REAL-Prover-v1 BF16: five domains, same expanded four-domain source groups retokenized, plus 512 unique raw Lean states paired with reference tactics from official training data. Original 128 proof states retained by deterministic nested sampling.
- Primary enlargement: 512 groups/domain × 4 tokens/group: 8,192 Qwen points; 10,240 REAL-Prover points.
- Matched pilot: original 128 groups/domain × 4 tokens/group.
- Density control: original 128 groups/domain × 16 tokens/group. Same total point counts as enlargement, different source diversity.
- Extract 16 positions once; first four positions reproduce pilot positions. No comparison of different token subsets disguised as a sample-size-only change.
- Qwen view: native calibration chat, first 2,048 tokens, uniformly sampled positions after first 32 tokens.
- REAL primary: last 128 raw-content tokens, uniformly sampled positions16..127. Native-context control: last up to2,048 tokens, positions confined to actual content/current state/reference tactic.
- Hook: decoder block14 output before final model norm, checked against hidden_states[14]; executing first14 blocks must match full28 exactly.
- Each dot is one token hidden vector, Qwen2048 dimensions / REAL3584 dimensions. No mean pooling in these plots.
- Joint t-SNE across domains, no PCA or feature normalization; random initialization, LR200, perplexity30, 1,000 iterations; seeds42,2,3. Domain labels do not enter t-SNE fitting.
- All scenario plots use identical canvas, marker size36, alpha0.2, axes and colorbar. Presentation applies label-informed global rotation/reflection, common translation and one uniform scale. No separate movement of domains. Native coordinates retained.
- Colorbar is categorical domain-color encoding, not entropy or activation magnitude. t-SNE axes have no physical units.

## Biases and limits

The paper does not publish exact Fig2b sampling/plotting parameters. This is a reconstruction and sample-size sensitivity study, not an exact author-protocol replication. FineWeb sequential source-order bias and long-response reasoning eligibility remain. Proof requires128 content tokens and tactic2..48 tokens; this biases toward longer states. Different states can belong to the same theorem; source-theorem independence and semantic leakage are unknown. Original exact-match miniF2F screen is reused; no test outcome enters selection. Tokens from one group are correlated. A separated cluster does not establish theorem-solving ability or a knowledge module.

## Resource controls and execution

`pipeline.py` runs under the existing read-only forward environment. Qwen and REAL GPU stages run sequentially on GPU3, with batch1, 50% duty, 12GiB/32GiB allocator caps respectively, reserve16GiB free GPU and64GiB available host memory. Health/probe/fingerprint/new-error checks reuse the existing guard. Only the observation child process group may be stopped. No automatic retry after guard failure. CPU t-SNE begins after GPU extraction children exit and uses2 threads. All old outputs remain intact.

Status: `status.json`; controller log: `nohup.log`; GPU logs: `{qwen,real}/logs/`; figures: `{qwen,real}/plots/`. Completed marker is written only after all27 panels finish. Existing per-panel completion files allow CPU plotting resume; finished feature caches avoid repeated extraction.
