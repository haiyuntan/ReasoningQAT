# DTR publication figures

Run `bash src/figures/run_all_figures.sh` from the isolated DTR workspace. Canonical scripts live outside plots_final so the whole output directory can be regenerated after deletion. Generated plots_final/scripts wrappers also support `python build_fig1.py --input metrics/per_tactic_metrics.csv --outdir plots_final/main`. CPU only; no new model forward, Lean evaluation or experiment change.

Main figures use all160 confirmation P4 tactics across20 roots, with aligned P0 same-token scores. P0 is NOT an independently generated pool: its length and Lean labels are identical. Figures1/2 are descriptive adaptations of the paper's Figure1/2, not a replication on long CoT.

Fig1: tie-preserving quintile boundaries, micro tactic validity, theorem-cluster95% CI,10000 resamples seed2; Spearman is descriptive. Equal values never split by labels. Fig2: seeded uniform choice among confirmation valid/invalid cases; shared[0,ln2] JSD nats scale, normalized lens, g=.5/rho=.85, cutoff24; at most32 tokens in main, full text in A2. V1/V2 discordant natural-generation classes are NOT_APPLICABLE, because only P4 was generated; never fabricate different Lean verdicts for replayed identical strings.

Fig3: same-state concordance averaged equally by theorem. P4 validity point estimates and intervals reused exactly from frozen primary analysis; the primary effective count is8, below the prespecified20 threshold, so INCONCLUSIVE_LOW_POWER even if the displayed interval is negative. Goals-closed is secondary. Fig4: identical-token shift grouped into mutually exclusive valid-goals-remain/goals-closed/invalid; violin shows tactic distribution, black points theorem means. Higher DTR is not automatically better reasoning.

A1: outcome distributions; A2: full tactic cases; A3: separate confirmation and development panels, only complete192-candidate P3/P6 DEV pools if available, P5 NOT_RUN; A4: heuristic BPE-level categories, not verified semantic labels.

Inputs: metrics/per_tactic/*.json, metrics/per_token/*.json, metrics/paired_replay/*.json, optional complete secondary_replay_P3/P6, frozen case/split manifests and primary summary. Derived metrics/per_tactic_metrics.csv, per_token_metrics.csv, per_layer_jsd.jsonl are exported and validated without re-tokenization. Every displayed value and data/script SHA is in tables/. No unknown verdict is counted as invalid. These are tactic-level measurements, not theorem pass@64 or evidence that SEM training works.
