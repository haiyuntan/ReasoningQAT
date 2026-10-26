# Host memory growth in the four-GPU evaluation

The user requested stopping all project processes and repairing increasing host
memory on 2026-10-26. The v9 controller, three evaluator trees and shared retrieval
were stopped. Existing models, prepared evidence, generated attempts and logs are
preserved. The campaign status is STOPPED_BY_USER; this repair does not restart it.

## Evidence

At 01:18 UTC, the cgroup used 515,395,076,096 bytes against a
515,396,075,520-byte limit. Anonymous memory was 490,746,273,792 bytes.
The three evaluator worker RSS readings immediately before stopping were:

| PID | Method | RSS KiB |
| --- | --- | ---: |
| 28782 | P2 | 222917372 |
| 28781 | P1 | 160986408 |
| 138090 | P3 | 110016960 |

GPU utilization was zero on all four cards. Lean children waited in
`wait_on_page_bit_common` with repeated major page faults. All evaluators repeatedly
consumed the existing 1200-second Lean response timeout. This is consistent with
host-memory exhaustion evicting Lean's mapped file pages, rather than a shortage of
GPU KV cache. After process exit, available host memory recovered to roughly
479 GiB. No shared system cache was manually dropped.

## Root cause and repair

The installed vLLM 0.6.6 `ParallelSampleSequenceGroup.add_request` inserts all n
child request IDs into `LLMEngine.seq_id_to_seq_group`. `RequestOutput.from_seq_group`
finishes and assembles their results but never removes these entries. Completed
groups retain their sequences, output token IDs and per-token logprobs. With n=64,
thousands of generate calls retain hundreds of thousands of completed sequences.
Resetting the project's generator.calls does not release the engine's references.
Installed source hashes and pre-change Git heads are in
`runs/memory_repair_20261026/diagnosis.json`.

`manager/thirdparty/generation_memory.py` now removes only references to fully
finished, already-consumed parallel groups after synchronous `LLM.generate`
returns. It preserves unfinished requests and unconsumed groups and does nothing
on engine versions without this internal map. Both single-state and batched tactic
generation use the wrapper. The installed environment is not modified.

The repair does not change sampling parameters, generation order, RNG state,
candidate filtering, scoring, search limits or timeout policy. Returned outputs
continue owning their values after redundant engine references are removed.
Historical attempts remain associated with the old experiment fingerprint;
resource-induced timeout failures must be disclosed when interpreting them. A
future continuation needs a new audited fingerprint and must preserve old evidence.

## Validation

Targeted tests cover 64 child references per group, mixed finished/live groups,
engine-busy protection, other-version compatibility, exact returned object and
parameter preservation, and 2000 repeated request lifecycles. Existing generator
filtering, timeout handling, Interactive transport and aggregation tests also run.
Results: `runs/memory_repair_20261026/unit_tests.json`.

The GPU regression script reproduces the installed leak without cleanup, then
checks repeated n=64 generation with cleanup, including 256-token requests.
It records RSS, retained engine keys, and exact hashes of text, token IDs and both
cumulative/per-token logprobs before/after cleanup. Its final status and measured
growth are saved in `runs/memory_repair_20261026/live_gpu.json`.

Validation completed successfully: 22 targeted unit tests passed. The real A800
GPU test reproduced 64, 128, ..., 512 retained child keys over eight unpatched
requests. All 24 repaired 64-token requests and both 256-token requests ended with
zero retained keys and unchanged output hashes. After five warmup requests, RSS
growth over the remaining short requests was 0.00390625 MiB (4 KiB). This is a
bounded regression test, not a multi-day soak test. Both full-length checks also
passed. The diagnostic process exited with code 0; no formal experiment restarted.
