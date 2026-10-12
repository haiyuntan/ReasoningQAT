# Compact experiment records for analysis

This directory contains reviewed JSON/CSV/Markdown only. It must not contain weights, datasets, checkpoint tensors, full predictions, credentials or large logs.

Each result must identify run/mode, code commit, upstream revisions, model/data hashes, actual configuration, loss/trainable checks, five task scores for every seed, mean, paper reference, gate decision, wall time and peak GPU memory. Missing measurements remain null; smoke metrics are never paper reproduction metrics.

Recommended files:

- `preparation_status.json`: completed and outstanding preparation checks.
- `cpu_checks.json`: actual CPU test results.
- `environment.json` and small pip lock files: environment provenance.
- `sources.json`: model/data revisions and hashes, without data contents.
- `table2.csv`, `table4.csv`, `table6.csv`: experiment comparisons.
- `<mode>/<run>/summary.json`: structured run results and gate decisions.
- `reproduction_report.md`: conclusions, limitations and troubleshooting.

Raw predictions and full logs stay in the parent workspace's outputs/logs directories. Never mark an unexecuted run PASS.
