"""Shifted/masked RR and truncated KL matching pinned ReasoningQAT semantics.

RR = p_teacher(reference)*CE; alpha multiplies the RR loss, not the
probability interpolation. KL uses full-vocabulary normalization then teacher
top-k selection, without renormalizing the selected mass (upstream beta=1).
"""
import torch
import torch.nn.functional as F


def objective(student, labels, teacher=None, alpha=0.2, beta=1.0, top_k=20):
    s = student[:, :-1]
    y = labels[:, 1:]
    mask = y != -100
    if not mask.any():
        raise ValueError('No assistant targets survive truncation')
    # Select supervised positions before vocabulary-sized float32 allocations.
    s, y = s[mask].float(), y[mask]
    ce = F.cross_entropy(s, y, reduction='none')
    if teacher is None:
        return ce.mean(), {'ce': float(ce.detach().mean()), 'rr': None, 'kl': 0., 'tokens': int(mask.sum())}
    t = teacher[:, :-1][mask].float().detach()
    log_s = s.log_softmax(-1)
    log_t = t.log_softmax(-1)
    probability = log_t.gather(-1, y[:, None]).squeeze(-1).exp()
    rr = (ce * probability).mean()
    selected, indices = log_t.topk(min(top_k, t.shape[-1]), dim=-1)
    truncated_kl = (selected.exp() * (selected - log_s.gather(-1, indices))).sum(-1).mean()
    loss = alpha * rr + beta * truncated_kl
    return loss, {'ce': float(ce.detach().mean()), 'rr': float(rr.detach()),
                  'kl': float(truncated_kl.detach()), 'tokens': int(mask.sum())}
