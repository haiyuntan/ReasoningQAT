"""Preserve the selected sample and a causal completion at fixed sequence length."""


def completion_window(full, prefix, maximum):
    if maximum < 2:
        raise ValueError('Need a context token and a supervised token')
    if full[:len(prefix)] != prefix:
        raise ValueError('Chat template prompt is not a token prefix')
    answer = full[len(prefix):]
    if not prefix or not answer:
        raise ValueError('Empty prompt or completion in source sample')
    if len(prefix) < maximum - 1:
        ids = full[:maximum]
        labels = [-100] * len(prefix) + ids[len(prefix):]
        adjusted = False
    else:
        # Long prompts previously consumed every token. Retain their tail
        # (including the assistant header), reserving up to half for the answer.
        reserve = min(len(answer), maximum // 2)
        context = prefix[-(maximum - reserve):]
        completion = answer[:maximum - len(context)]
        ids = context + completion
        labels = [-100] * len(context) + completion
        adjusted = True
    if not any(label != -100 for label in labels[1:]):
        raise ValueError('No causal completion tokens')
    return ids, labels, adjusted
