# Storage and retention

The pipeline records each run's file sizes before and after cleanup in its compact result JSON. Large artifacts stay outside Git. The shared filesystem's free space is not a user quota guarantee.

| Run | Local artifacts | Retention after successful evaluation |
| --- | --- | --- |
| R0 | Evaluation predictions, reviews, reports, configuration, generation telemetry | Keep evidence; reuse the downloaded BF16 model |
| R1 | Packed GPTQ W2 checkpoint, tokenizer, quantization metadata, evaluation artifacts | Keep formal checkpoint; remove passed smoke checkpoint |
| R2 | AWQ calibration scales and materialized W2 checkpoint, evaluation artifacts | Keep formal checkpoint; remove passed smoke checkpoint |
| R3 | Native block-QAT checkpoint, calibration cache, temporary HF export, evaluation artifacts | Keep native shared starting point; remove verified duplicate HF export |
| R4/R4a/R5/R6_rr/R6_kl | Native final checkpoint, at most one intermediate trainer snapshot, temporary HF export, trainer state, evaluation artifacts | Keep formal native final checkpoint; remove intermediate snapshot and verified duplicate export; remove passed smoke final checkpoint |

BF16 base weights occupy 4.06 GB (decimal). Each materialized or native checkpoint is approximately another 4 GB, with exact bytes measured at runtime. Packed W2 GPTQ is smaller but includes unquantized layers and metadata. Complete downloaded training inputs require approximately 30.6 GB; Arrow caches can duplicate dataset storage. Formal predictions and reviews can occupy several additional GB because generation permits 32,768 tokens. Reserve at least 150 GB for this project initially and revise the budget from measured caches, checkpoints and evaluation files. Block-QAT activation buffers also require substantial host RAM; disk capacity does not guarantee sufficient RAM.

Cleanup runs only after the run passes export reload verification, evaluation and its gates. Before removing an HF duplicate, the pipeline saves the native checkpoint SHA256 manifest and export verification metadata under `repro_results`. Rebuild an export with `repro_scripts/export_qat.py --input <native-directory> --output <export-directory>`. Failed runs retain diagnostic artifacts. Downloads, `.partial` files, original datasets, the R3 shared starting point and unrelated experiments are never removed by this cleanup.

Git receives compact scores, protocol and configuration, timing, memory measurements, checkpoint hashes and storage/cleanup evidence. Models, raw datasets, raw predictions/reviews and full logs remain local.
