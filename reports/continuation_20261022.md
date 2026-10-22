# Independent four-GPU continuation after v8 failure

The user authorizes repair and formal continuation without another full smoke campaign, prioritizes isolation of individual failures, and explicitly disables P5 per-block checkpoint storage. No P7 is invented without a defined treatment.

## What failed and what can be reused

v8 stopped because OpenThoughts global-shuffle sample 138 has a 4479-token prompt. Right truncation at 2048 retained no answer tokens. P4 and P6 rejected it before optimizer updates. The old controller then terminated independent P5 while its final block was still running.

P1, P2 and P3 have complete preparation manifests. Reuse verifies every original evidence hash, compares the source commits against current quantization/config files, and checks AST equality for P3's calibration, reconstruction and entrypoint branch. Checkpoint binaries are immutable hardlinks, JSON files are private copies, and a new preparation manifest records source fingerprints and review evidence. No old evaluation receipt is promoted to a current PASS.

P5 has no saved model/optimizer/checkpoint: its old directory contains only identity, invocation and lifecycle records. Activation caches cannot reconstruct trained weights. It must restart independently from the same BF16 REAL-Prover base; P3 is not substituted for the independent EfficientQAT baseline.

## Stage2 data repair

All P4/P5/P6 use the same 32768 globally shuffled samples (seed 2), the same maximum length 2048, final-assistant-only mask and original training hyperparameters. Normal samples retain their original right-truncated tokens exactly. If the prompt would leave fewer than two token positions, preserve its tail including the assistant prefix and reserve up to half of the context for the answer head. No examples are silently filtered or replaced. Empty answers and invalid template prefixes remain errors. This is an explicit change to the migration's truncation engineering policy, not a claim of exact published preprocessing parity.

A CPU preflight audits all examples and publishes a token/mask cache atomically only after completion. The cache records input/config/tokenizer/policy identities, exact token/mask fingerprints, supervised token totals and long-prompt counts. All three methods use that same hash-verified cache, avoiding repeated selection and tokenization on GPU workers.

## Failure isolation and scheduling

`scripts/resume_four_gpu.py` replaces v8's global phase barrier. Each method has independent preparation/evaluation states. A failed preparation blocks only that method's evaluation; a failed evaluation does not invalidate its checkpoint or stop another method. P4/P6 depend on successful P3 preparation, not successful P3 evaluation. A failed slot can immediately accept another eligible job. Method exceptions are caught in polling and never enter controller-wide cancellation. Only an explicit controller signal or a global code/config integrity failure cancels all controller-owned children.

With the validated data cache ready, initial assignment is GPU 0: P1 miniF2F evaluation; GPU 1: independent P5 preparation; GPU 2: P4 preparation; GPU 3: shared retrieval. P2/P3 evaluations and P6 preparation are queued. All slots are reassigned on completion/failure. Shared retrieval is stopped and its GPU becomes available for training when no evaluation is active or ready; it starts again only when both a retrieval slot and an evaluator slot are available. It never shares a GPU with a training worker. vLLM utilization remains 0.85 and all miniF2F/search limits remain unchanged.

If the full CPU data audit is still running when the controller starts, all three already-prepared P1/P2/P3 evaluations may start immediately on GPUs 0/1/2, sharing retrieval on GPU 3. Stage2 training remains blocked until atomic publication of the audited cache. An existing CPU preflight can be observed by a small monitor process so selection/tokenization are not repeated. The actual per-method assignments and data preflight state are written to control/status.json and may differ from the ready-cache example above.

P5 saves no per-block checkpoints, as requested. Stage2 keeps only one latest recovery file per method containing trainable scales, optimizer/scheduler state, random state, history and next sample; it atomically replaces the previous file and deletes it after the final export/reload/parity checks pass. The scale checkpoint does not contain a second full base model. Recovery reconstructs the method's original initialization and resumes updates without replaying completed optimizer steps. For P5, absence of a completed Block-AP checkpoint still means an interrupted block phase must restart.

## Verification and cleanup

Real child-process tests ensure a failed P4 process leaves a second GPU task alive and able to finish. Scheduling tests check exclusive GPU ownership, mixed evaluation/training, shared retrieval reservation and reuse of a failed slot. A GPU optimizer test interrupts after an actual saved update and verifies resumed scales are bitwise equal to uninterrupted training, with no repeated updates. Real sample 138 now retains 1024 supervised answer tokens at length 2048. Cache round-trip and corruption rejection are tested.

Cleanup inventory/open-file checks are saved in `runs/resume_repair_20261022`. Four orphaned activation directories totaling 393.846 GiB were removed after confirming no process held files inside them. Original data, base weights, completed P1/P2/P3 artifacts and logs are preserved. The complete old v8 failure evidence remains available.
