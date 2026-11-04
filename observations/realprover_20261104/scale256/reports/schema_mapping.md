# Physical input mapping

`input` includes full proof/state/retrieval content, with two observed template families. Current STATE is extracted only for exact-state grouping and raw-domain control. Entropy preserves full content, unwraps a legacy deepseek User/Assistant wrapper when present, and applies the pinned formal runtime qwen wrapper exactly once. The gold tactic is appended once and only its token positions are eligible. No EOS; overlength pairs rejected, not spliced.

{
  "selected": 507,
  "variants": {
    "native_incontext": 408,
    "deepseek_wrapped": 99
  },
  "rejected": 5
}
