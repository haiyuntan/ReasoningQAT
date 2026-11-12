# Paper alignment

Checked primary sources https://arxiv.org/pdf/2602.13517 v2 2026-07-06 and PMLR chen26di. Eq1–6: own-final full-vocab JSD, cummin/first crossing; natural-log main g=.5, rho=.85, layer24 cutoff. Architecture-aware lens applies checkpoint final RMSNorm at every block then same head. Raw sensitivity applies no norm at all28 blocks with its own raw-final reference. No early raw/late normalized mixing. Self-Certainty=KL(uniform||p). Original paper long-CoT, Think@n48/eta50%/prefix50 differs from8 short Lean tactics. No exact Think@n claim or cost-saving claim.

Original GPU3-only and prohibition on stopping production evaluations are superseded by explicit user instruction on2026-11-12 to pause evaluations and allocate resources for one hour. DTR retains a single GPU1 model process; P4 then P0 same-token replay. No training.
