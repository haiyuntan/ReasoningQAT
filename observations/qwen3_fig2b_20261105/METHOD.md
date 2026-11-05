# Qwen3-1.7B Figure 2(b) reconstruction

Primary reference: [ReasoningQAT, ICLR 2026, Figure 2(b)](https://proceedings.iclr.cc/paper_files/paper/2026/file/209423f076b6479ab3a4f45886e30306-Paper-Conference.pdf).

The paper uses a full-precision Qwen3-1.7B and the 14th Transformer block, with Pre-training, Code, Math and Science domains. No quantization training is needed for this panel. The paper and released repository do not specify a complete plotting protocol (sample identities/count, pooling, token selection, preprocessing and t-SNE hyperparameters); this is a disclosed reconstruction, not an exact author-script reproduction.

Earlier local REAL-Prover plots used a different model, Lean in place of Science, a shared chat wrapper, 128-token windows, mean pooling and L2-normalization before PCA. They are a transfer observation and cannot be compared as an identical replication of Fig. 2(b).

This run uses local unquantized Qwen3-1.7B, FineWeb-Edu and OpenThoughts3. It takes 256 unique eligible human prompts per reasoning domain, and 256 pre-training documents. Deterministic first-eligible source order is a limitation. It compares raw unwrapped inputs, 128-token and up-to-512-token contexts, block-output mean/last/eight evenly spaced token representations, with/without L2 normalization. Sequences shorter than 128 tokens are excluded. Shorter-than-512 sequences are retained without padding; length is a potential confound. Prompt-only input is an explicit assumption; the original paper does not state whether responses were included. No theorem evaluation samples or success labels are used.

The declared main plot is `mean_512_none_seed2.png`. Seeds 3/42 are controls, never selected to imitate the paper. Token plots use correlated tokens and must not be interpreted as independent samples. `statistics.json` reports sequence-level high-dimensional silhouette and within-domain dispersion; a visual match alone does not validate the implementation or establish independent knowledge modules. All methods share a joint PCA/t-SNE fit within each panel. Different panels have unrelated axes.

`launch.py` reuses the existing resource guard without modifying its source or environment, points it to the active v15 campaign, and writes only this directory. GPU3 only, batch 1, allocator cap 12 GiB, approximately 50% forward duty cycle. Retrieval health/real query latency, free GPU memory, host memory and new production retrieval failures are monitored; only the observation child can be terminated. Shared GPU scheduling does not guarantee strict hardware QoS.

Files: `supervisor.log`, `logs/qwen3_fig2b.log`, `logs/guard-qwen3_fig2b.jsonl`, `inputs.jsonl`, `protocol.json`, `model_identity.json`, `features.npz`, coordinates/plots, `statistics.json`, `completed.json`. A successful launch is not completion; check `manifests/qwen3_fig2b.json` and `completed.json`.

No changes to production Git repositories, model weights, prior observation results or formal evaluation configuration.
