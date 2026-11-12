# DTR transfer, one-hour observation

User explicitly paused formal GPU0–2 evaluations on 2026-11-12; overrides SOP prohibition on stopping those evaluations and original GPU3-only scheduling. Shared retrieval remains on GPU3. DTR uses GPU1, one model process with P4 generation followed by P0 same-token replay, separate environment/workdir. ReQAT occupies GPU0/GPU2 for primary confirmation when gated.

P4-generated frozen8-candidate pools on24 independent DEV roots (old ABCD data), then20 independently reserved confirmation roots, disjoint from ReQAT64. Primary normalized all-layer full-vocabulary JSD first crossing at g=.5 nats, rho=.85, cutoff24. Raw/nats/bits/persistent sensitivity separate; SC uses KL(uniform||p). No training. Tactic-level only, not Think@n reproduction or pass@64. Full-prefix canonical math prevents prior BF16 shape artifacts. Production max256 vs observation cap48 is a deviation. No P0 natural pools or intermediate rollouts within deadline; explicitly NOT_RUN.
