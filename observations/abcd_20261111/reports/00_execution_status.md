# Execution status — 2026-11-11

P6 FATE-M was stopped by user request; GPU3 production retrieval PID 3222326 remains running. P3/P4/P5 on GPU0/1/2 remain running.

The independent observation environment passed imports at its own prefix: Python3.11, torch2.2.2+cu121, transformers4.52.4, NumPy1.26.4, local Parquet/statistics dependencies. Offline conda clone could not find local archives; an independent filesystem copy plus matching local CPU packages was used instead. No production package prefix was modified.

The resource guard fault-injection test passed (only the owned observation child reaped). An actual GPU3 P0 forward/gradient/teacher-block replay engineering check passed, with layer7/14/21/28 replay max-abs errors {'7': 0.0, '14': 0.0, '21': 0.0, '28': 0.0}. Peak PyTorch allocated memory: 14.303 GiB. No optimizer or parameter save was used. This is not ABCD external evidence.

Independent Lean sanity passed: exact h.symm valid transition and rfl rejected for a=b ⊢ b=a. A real retrieval schema query returned 10 premise records compatible with the frozen official prompt builder.

The nohup controller PID 3094470 is executing the CPU audit. P0/P3/P4/P6 checkpoint hashes have been recorded; full state–tactic length/template/group audit is ongoing. Subsequent stages are automatic only if their gates pass. Full A/B/C/D pilot and G5 confirmation are not yet completed; M0/M10/M20 are not started. Referenced standalone ABCD SOP was not found locally; the v3 revision and configs/protocol.json define this pilot.

Logs: logs/pipeline.nohup.log, logs/audit.log, logs/gpu_sanity.log, logs/guard.jsonl. Active stage receipts: manifests/*.json. Large/generated outputs remain local and gitignored.

CPU audit completed: 56,407 rows; 55,585 unique screened states; S-CAL 44,653 and S-OBS 10,932 exact-state groups. Median input length 914 tokens, only187 raw inputs reach2048 tokens. This blocks claims of ready-to-train M calibration; it does not block fixed-prefix observation. The dataset revision is91bb339e61ff045d3ebf97cd94d1df1d755dd745, checked against the original acquisition receipt and matching Parquet SHA. The controller has advanced to independent valid-root/Lean candidate collection; at least the first selected valid theorem was frozen successfully. Full ABCD results remain pending.

Final pilot status: PILOT_DONE. P0/P3/P4/P6 each scored all48 cases (32 valid roots +16 transformed-rule instances), no skipped cases. P3/P4/P6 each completed8-theorem ×4-layer teacher replay with zero max-abs same-input error. Analysis completed. G5 confirmation and M training remain unexecuted. Earlier paragraphs record startup history, not current pending work.
