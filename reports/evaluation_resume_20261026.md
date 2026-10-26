# Resume formal evaluation after the host-memory fix

Create a new v10 campaign from the stopped v9 campaign. Do not modify v9 models,
attempts, telemetry or historical fingerprints. Prepared P1-P6 models are inherited
only after old artifact hashes and unchanged training/quantization code are checked.
Binary weights use immutable hardlinks to avoid duplicate storage. JSON metadata
and result attempts use independent copies, with explicit source hashes/lineage.

The new attempt reader validates filenames, slot range, theorem identity, result
schema and full Lean verification for any claimed success. A theorem is finished
only after a verified success or all 64 slots exist. Partial theorems skip occupied
slots and evaluate only missing slots. Previously recorded timeouts remain failed
attempts and still consume their slots. Atomic, exclusive publication cannot
overwrite an existing slot, including after an interrupted write. Corrupt or
unverified inputs fail closed instead of silently increasing the budget.

The aggregate completion criterion now also requires success or exhausted retry
budget; one failed attempt no longer means a completed theorem. Smoke aggregation
uses the actual smoke retry limit rather than assuming 64. Final receipts bind the
evaluation resume manifest and all inherited/new raw results.

GPU 3 serves shared retrieval. GPUs 0/1/2 resume P1/P2/P3, then take P4/P5/P6 when
free. The existing controller isolates individual task failures. miniF2F's 244 test
problems, pass@64, 64 candidates per step, all search/sampling limits and vLLM 0.85
are unchanged. There is no retraining. The vLLM finished-request ownership repair
is active. A resumed process starts with seed 2; the previous in-memory RNG state
was not checkpointed, so bitwise equivalence to an uninterrupted run is not claimed.

The initializer records original progress under control/previous_progress.json,
then nohups the existing controller. A fresh fingerprint and batch-bound authorized
waiver reference targeted resume tests and the passing real-GPU memory regression.
No new full smoke PASS is claimed. Old resource-induced timeout failures remain
visible, so the combined result should be described as a continuation including
historical infrastructure failures, not a clean rerun under the repaired engine.

Entry point: scripts/launch_evaluation_resume.py --source runs/realprover_formal_four_gpu_v9
--results runs/realprover_formal_four_gpu_v10 (paths relative to project root).
