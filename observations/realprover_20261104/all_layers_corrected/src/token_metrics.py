"""Stable full-vocabulary FP64 statistics from BF16 logits, without 1-p cancellation."""
import torch

def stable_log_probs(logits):
    z=logits.double()
    maximum,top=z.max(-1,keepdim=True)
    shifted=z-maximum
    other=shifted.scatter(-1,top,float('-inf'))
    log_other=torch.logsumexp(other,-1)
    normalizer=torch.log1p(log_other.exp())
    return shifted-normalizer[:,None],top.squeeze(-1),log_other-normalizer

def excluded_log_mass(log_probs,indices):
    return torch.logsumexp(log_probs.scatter(-1,indices[:,None],float('-inf')),-1)

def distribution(logits, gold, teacher=None, temperature=1.5, top_p=.9):
    z=logits.double()
    lp,idx,log_tail=stable_log_probs(z);p=lp.exp()
    gold_lp=lp.gather(-1,gold[:,None]).squeeze(-1)
    top_lp,top_idx=lp.topk(min(20,z.shape[-1]),dim=-1)
    legacy=logits.float().log_softmax(-1)
    z32=logits.float();shift32=z32-z32.max(-1,keepdim=True).values
    rest32=torch.logsumexp(shift32.scatter(-1,idx[:,None],float('-inf')),-1)
    stable_tail32=rest32-torch.log1p(rest32.exp())
    result={'entropy':-(p*lp).sum(-1),'gold_logp':gold_lp,'top1':idx,
            'own_log_tail':log_tail,'own_tail':log_tail.exp(),
            'legacy_own_tail':-torch.expm1(legacy.max(-1).values).double(),
            'stable_fp32_log_tail':stable_tail32.double(),
            'top20_logp':top_lp,'top20_ids':top_idx}
    # Same nucleus filter as vLLM: remove items with cumulative lower-prob mass <= 1-p.
    ordered,order=(z/temperature).sort(dim=-1,descending=False)
    ordered_lp,_,_=stable_log_probs(ordered)
    prob=ordered_lp.exp()
    removed=prob.cumsum(-1)<=1-top_p
    removed[:,-1]=False
    ordered=ordered.masked_fill(removed,float('-inf'))
    filtered_lp,_,_=stable_log_probs(ordered)
    sample_lp=torch.empty_like(filtered_lp).scatter(-1,order,filtered_lp)
    result['sample_gold']=sample_lp.gather(-1,gold[:,None]).squeeze(-1).exp()
    if teacher is not None:
        tid=teacher['top1'].to(z.device).long()
        t_gold=teacher['gold_logp'].to(z.device).double()
        t_top=teacher['top20_logp'].to(z.device).double()
        t_ids=teacher['top20_ids'].to(z.device).long()
        fixed=excluded_log_mass(lp,tid)
        result.update(fixed_log_tail=fixed,fixed_tail=fixed.exp(),sample_fixed_tail=excluded_log_mass(sample_lp,tid).exp(),delta_nll=t_gold-gold_lp,rr=-(t_gold.exp()*gold_lp),kl_top20=(t_top.exp()*(t_top-lp.gather(-1,t_ids))).sum(-1))
    else:
        result['sample_fixed_tail']=excluded_log_mass(sample_lp,idx).exp()
    return result
