# Corrected paired entropy diagnostics

Primary: Low minus High mean (log_tail_Q-log_tail_T), no epsilon, equal-weight state groups with both bins; 2000 cluster bootstrap draws, seed2, 95% CI. Thresholds recomputed using stable FP64 entropy from separate BF16-only calibration state groups. Full-vocabulary statistics, BF16 model, gold-prefix teacher forcing.

method,state_pairs,low_minus_high_log_rho,ci_low,ci_high,population,estimand
P3,274,3.4819113400570267,3.0731674737334473,3.9066593214892382,in_training_distribution,equal-weight states having both Low and High eligible positions
P4,274,3.7604366411618857,3.397368796128041,4.155056881992575,in_training_distribution,equal-weight states having both Low and High eligible positions
P5,274,3.650669265403056,3.307062783780696,3.9797014170513783,in_training_distribution,equal-weight states having both Low and High eligible positions
P6,274,4.020295443883553,3.590548062749984,4.444296837292948,in_training_distribution,equal-weight states having both Low and High eligible positions


Legacy float32 cancellation is reconstructed on the SAME new logits in tail_numeric_audit.csv and epsilon_sensitivity.csv; legacy results are superseded for tail conclusions. Report absolute probability changes and actual T1.5/top_p.9 sampler tails alongside relative ratios. A relative change does not establish sampling or verifier harm. Token categories remain lexical heuristics. Independent dev verification, frequency controls and native-scale gradients remain incomplete; no theorem ranking inferred.
