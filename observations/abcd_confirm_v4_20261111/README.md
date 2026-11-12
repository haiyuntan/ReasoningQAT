# REAL-Prover ABCD independent confirmation v4

This is a new, isolated run. It does not overwrite `observation_calibration_v3_20261111/` or formal FATE-M results. The previously published pilot branch remains unchanged. Only GPU3 UUID is visible to observation workers; existing retrieval and GPU0–2 evaluation are never restarted or reconfigured. Models are reused read-only.

## Preregistered scope

- 128 miniF2F-valid theorems, statically hash-selected from the212 IDs remaining after excluding all32 pilot valid theorems; no test/FATE-M outcomes used.
- A fits rank4 teacher reference-tactic Jacobian subspaces on128 new exact-state-disjoint S-OBS-DISC groups. Layer28 is primary because it was suggested by the pilot; layers7/14/21 are secondary. Total NMSE/PCA/random controls share the same output, state and aggregation. Test paired correlation differences, not separate CI significance.
- B has64 distinct typed base-rule instances, eight structural families, four types and two used constants3/7, with fresh bijective alpha-renaming. Each pair is kernel-compiled; history theorem name is identical in each pair. Primary analysis matches input/action token lengths and bootstraps rule families, not token/tactic counts.
- C freezes actual top10 retrieval records and the six actually rendered records. Intervene on the first three. Removing one never refills from record7. Compare removal with exact-token-length lexical distractor controls, actual-record replacement, order reversal and an empty-list control. Frozen `apply premise` candidate transitions label demonstrated applicability, not semantic necessity; unvalidated is not called irrelevant.
- D covers32 predetermined confirmation roots and all28 decoder blocks. Store29 shared hidden boundaries per case rather than duplicating input/output tensors. Strict same-input teacher replay (max abs<=1e-5) precedes the algebraic decomposition. Primary geometry is prompt-prefix/last-prompt-token only; zero-norm cosines and cancellation are null. Functional projections exist only where A bases were fit.
- Primary models P3/P4/P6; P5 is a preregistered secondary external baseline. P0 is the common reference. Nine primary A/B/C×model comparisons use Bonferroni-adjusted paired/cluster bootstrap intervals;10,000 replicates. All-layer D curves and other A layers remain exploratory.

## Numeric and resource repairs

Each model first measures identical-input repeats, masked-padding perturbations and same-length duplicate-batch perturbations on four OLD pilot development cases. Only identical-input repeats gate the primary path (tolerance1e-5). Shape perturbations are diagnostics, not silently treated as zero or pooled into confirmation data. Primary scoring stays batch1, unpadded, with deterministic algorithms/explicit cuBLAS workspace. Gradients are for observations only: no optimizer or parameter saves.

The new environment is a complete independent filesystem copy of the already isolated pilot conda environment, validated at its own prefix. All writable caches, temporary Lean files, logs and results remain here. Existing Lean dependency packages and binary are shared read-only. The only external use of production retrieval is sparse snapshot queries (<=0.2/s), health checks and sparse probes; model scoring never calls retrieval.

GPU3 allocation cap56GiB, duty0.90, free reserve8GiB, available host reserve64GiB. The guard checks service identity, health, real-query latency and new P3/P4/P5 retrieval errors. It may kill only the observation child group. A method failure is recorded and other methods may proceed; retrieval-priority failure stops the observation campaign. No hardware QoS guarantee.

## Run/status

`nohup <base-python> -u src/pipeline.py` in this workspace. The controller waits for environment copy, prepares/fixes inputs, collects independent Lean roots/candidates, runs data-contract tests, scores P0/P3/P4/P6/P5 serially, replays after each student, then analyzes/plots. It skips only verified completed stages. Per-case score/replay receipts allow continuation without losing completed cases. Missing replay caches fail explicitly. Student activation caches are reclaimed after durable replay receipts; teacher caches only after every replay completes. No old assets are deleted.

Read `manifests/controller.json`, stage manifests and `logs/pipeline.nohup.log`. `metrics/confirmation_summary.json` and `plots/ABCD_confirmation.*` are produced only after analysis. M0/M10/M20 training and theorem-level proof-search evaluation are outside this run. Unknown source-theorem or semantic overlap of upstream SFT data remains UNKNOWN.

An auxiliary `src/prefit.py` can use GPU idle time while the main controller collects Lean roots on CPU. It waits for the static prepare gate, then runs only numeric development checks and the128-group discovery fit. It never scores confirmation states. All GPU children take the same lock, so only one observation model resides on GPU3. The main P0 scorer reuses the input-hash/protocol/basis-hash-verified fit. Auxiliary status is `manifests/prefit_controller.json`.
