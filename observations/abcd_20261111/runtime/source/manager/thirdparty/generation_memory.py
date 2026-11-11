"""Release completed vLLM 0.6 parallel-sampling ownership after synchronous output."""


def release_completed_parallel_groups(llm):
    """Drop only consumed, fully finished groups; preserve any live engine state.

    vLLM 0.6.6 inserts each n>1 child into seq_id_to_seq_group but never removes
    the entries. The groups retain all token/logprob data across LLM.generate
    calls. RequestOutput already owns the returned values, so removing these
    redundant engine references after synchronous generation changes no output.
    Other vLLM versions without this map need no compatibility cleanup.
    """
    engine = getattr(llm, 'llm_engine', None)
    groups = getattr(engine, 'seq_id_to_seq_group', None)
    if not isinstance(groups, dict) or not groups:
        return 0
    # Never clean between decoding steps or while async processing still owns
    # requests. This helper is called only after the public generate returns.
    if engine.has_unfinished_requests():
        return 0
    finished = set()
    for group in groups.values():
        if id(group) in finished:
            continue
        assembled = getattr(group, 'assembled_seq_group', None)
        if (getattr(group, 'output_produced', False)
                and not getattr(group, 'to_be_finished', {None: None})
                and assembled is not None and assembled.is_finished()):
            finished.add(id(group))
    keys = [key for key, group in groups.items() if id(group) in finished]
    for key in keys:
        del groups[key]
    return len(keys)


def generate_with_cleanup(llm, *args, **kwargs):
    outputs = llm.generate(*args, **kwargs)
    release_completed_parallel_groups(llm)
    return outputs
