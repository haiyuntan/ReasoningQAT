# Verification and reproduction

Run `python verify_results.py` inside this directory to validate artifact hashes and completion coverage without a GPU.

To redraw, install numpy and `configs/requirements-plot.txt` in an isolated Python environment, then run `python src/plots.py`. This reads the published summary and writes PNG/PDF.

Full model recomputation additionally requires the read-only P0/P3/P4/P5/P6 checkpoints described in manifests/models.json, the original upstream calibration dataset, an isolated Torch/Transformers environment (reports/environment_after_plot_repair.txt), Lean 4.16 and interactive binary/dependency tree, and the pinned retrieval assets/service. Machine paths and service/GPU settings in src/common.py, configs/protocol.json and the stage manifests document the original machine; adapt them in a new workspace and create a new protocol identity rather than reuse old completion receipts. The copied environment, model weights, Lean binaries and activation caches are intentionally excluded. This publication is an audit/result package, not a self-contained runnable model distribution. See README.md and reports/PREREGISTRATION.md for data selection, baselines and limitations.
