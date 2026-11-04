# Shared retrieval acceleration and continuation

The original service held one lock across GPU encoding and a single-query CPU
FAISS flat search. The index search averaged 1.25 seconds on this host; simply
raising OpenMP threads from 1 to 16 did not materially improve it.

The replacement keeps the original immutable FP32 CPU index, partitions its
vectors into eight exact flat indexes, searches them concurrently and merges
their neighbors. It requests k+1 results and falls back to the original CPU
routine whenever selected scores tie, including the cutoff. Batched input also
uses the original routine. A bounded LRU stores only query keys and neighbor IDs
(2048 entries, 16 MiB accounted bytes); no model tensors are retained.

## Validation on this machine

128 real Lean states, drawn across P3/P4/P5, passed bitwise equality of neighbor
IDs, order and FP32 scores. Full HTTP responses also matched all 128 original
reference responses. A separate 12-query serial/concurrent comparison matched
the previous service's response bodies in every case.

| Measurement | Original | Updated |
|---|---:|---:|
| CPU index search mean | 1.251 s | 0.171 s |
| Cold HTTP, one client | 1.36–1.40 s | 0.225 s |
| Cold HTTP, three continuous clients | 3.662 s | 0.695 s |
| Warm cache, three clients | not cached | 0.0029 s |

Repeated cache requests left service RSS unchanged at 10042004 KiB. This is a
short regression check, not proof of unlimited-duration memory stability.
30 focused tests cover exact search, tied scores, bounded caching, timing,
fingerprint recovery, process cleanup, failure isolation and result inheritance.
These service speedups are not an end-to-end theorem evaluation speedup claim.

GPU FAISS initialization did not finish during the diagnostic window. That
backend remains disabled; no GPU nearest-neighbor results enter formal runs.

## Safeguards and profiling

`manifests/retrieval_acceleration.json` pins the validated implementation and
runtime config hashes. Service startup rejects changed or unvalidated sources;
health responses bind the runtime config hash. Original retrieval asset hashes,
model/tokenizer, miniF2F, pass@64 and vLLM 0.85 are preserved.

The controller saves per-component fingerprint hashes. On mismatch it freezes
only its own evaluator trees and records the changed components; two complete
reads must match the original fingerprint before resuming. A persistent change
still aborts. The previous transient incident's root cause is not established.

Formal workers now accumulate retrieval, generation, Lean tactic/state and final
verifier time. They publish periodic `eval/performance.jsonl` snapshots and add
`pipeline_timing` to new attempt JSON. Optional timing-log I/O failures do not
change search results. Retrieval timing is in
`runs/realprover_retrieval/requests.jsonl`. No seven-day stage deadline is added.

## Repeating the saved-vector parity check

Use the configured PROVER interpreter with FAISS and NumPy:

```bash
python research/scripts/validate_retrieval_search.py \
  --vectors runs/retrieval_repair_20261104/query_vectors.npy \
  --reference runs/retrieval_repair_20261104/cpu_reference.npz \
  --output runs/retrieval_repair_20261104/recheck.json --shards 8
```

Local evidence, original queries and exact diagnostic launch sources are under
`runs/retrieval_repair_20261104`. New formal results inherit valid attempts into
a new fingerprint; old result files and completed P1/P2 receipts are preserved.
