# Qwen3-1.7B W2g128 reproduction

Source branch: `reproduce/qwen3-1p7b-w2g128`. `origin` is the user's fork; `upstream` is the official repository. Changes and reviewed compact results are pushed to this branch. Main is unchanged until the reproduction is reviewed.

## Preparation

Assets are stored outside this Git repository, in its parent workspace. Models, datasets, caches, tensor checkpoints, raw generations, full logs and credentials are never committed. Frozen revisions and file SHA256 values are available in `repro_results/sources.json` once collected.

Preparation uses CPU only. The original GPU workload is never stopped automatically. `check_preparation.py` reports missing assets and environments; it cannot declare an incomplete download ready.

## Review commands without starting GPU work

```bash
bash repro_scripts/run_gated.sh --mode smoke --dry-run
bash repro_scripts/run_gated.sh --mode formal --dry-run
```

## After explicit user release of GPU

```bash
RQ_GPU_RELEASED=1 bash repro_scripts/run_gated.sh --mode smoke
RQ_GPU_RELEASED=1 bash repro_scripts/run_gated.sh --mode formal
```

The runner still checks for active GPU processes and exits if any exist. It acquires a workspace lock, uses separate output/log roots for smoke/formal, executes R0 → R1 → R2 → R3 → R4 → R4a → R5 → R6_rr → R6_kl, and stops on the first failure. It never stops an existing GPU process. R6 can be omitted with `--runs R0 R1 R2 R3 R4 R4a R5`.

Every Stage 2 run loads the same R3 artifact in its session. Smoke retains short real conversation spans and uses two optimizer steps. Its metrics validate execution only. Formal uses the full frozen calibration/training sizes and epochs; unresolved author settings in `evaluation.json` cause NOT_COMPARABLE before any formal GPU work.

To resume completed stages, pass the previous `--session <id>`. Completed summaries must have the same source/configuration fingerprint. A failed/stale session requires a fresh session or explicit investigation; it is never silently accepted.

## Evidence for web analysis

`repro_results/` stores configuration, revisions/hashes, actual CPU checks and one compact JSON per experiment. All five benchmark scores for each seed, mean, fixed paper references, gate status, code fingerprint, wall time and generation peak memory are preserved. Full predictions stay local.

```bash
../env/bootstrap/bin/python repro_scripts/publish_results.py --commit --push
```

Git authentication is supplied externally; no credential is embedded in a script or remote. The publishing command rejects binary/large evidence, credential-like strings and unrelated staged changes. It does not fabricate missing measurements or pass status.
