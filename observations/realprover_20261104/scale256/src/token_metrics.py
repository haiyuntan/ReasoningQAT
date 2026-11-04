"""Exact full-vocabulary metrics. Float32 within bounded token chunks."""
import torch

def distribution(logits, gold, teacher=None, temperature=1.5, top_p=.9):
    z=logits.float()
    lp=z.log_softmax(-1);p=lp.exp()
    best,idx=lp.max(-1)
    gold_lp=lp.gather(-1,gold[:,None]).squeeze(-1)
    top_lp,top_idx=lp.topk(min(20,z.shape[-1]),dim=-1)
    result={'entropy':-(p*lp).sum(-1),'gold_logp':gold_lp,'top1':idx,'own_tail':-torch.expm1(best),'top20_logp':top_lp,'top20_ids':top_idx}
    # Same nucleus filter as vLLM: remove items with cumulative lower-prob mass <= 1-p.
    ordered,order=(z/temperature).sort(dim=-1,descending=False)
    prob=ordered.softmax(-1)
    removed=prob.cumsum(-1)<=1-top_p
    removed[:,-1]=False
    ordered=ordered.masked_fill(removed,float('-inf'))
    sampled=ordered.softmax(-1).scatter(-1,order,ordered.softmax(-1))
    result['sample_gold']=sampled.gather(-1,gold[:,None]).squeeze(-1)
    if teacher is not None:
        tid=teacher['top1'].to(z.device).long()
        t_gold=teacher['gold_logp'].to(z.device)
        t_top=teacher['top20_logp'].to(z.device)
        t_ids=teacher['top20_ids'].to(z.device).long()
        result.update(fixed_tail=-torch.expm1(lp.gather(-1,tid[:,None]).squeeze(-1)),sample_fixed_tail=1-sampled.gather(-1,tid[:,None]).squeeze(-1),delta_nll=t_gold-gold_lp,rr=-(t_gold.exp()*gold_lp),kl_top20=(t_top.exp()*(t_top-lp.gather(-1,t_ids))).sum(-1))
    else:
        result['sample_fixed_tail']=1-sampled.gather(-1,idx[:,None]).squeeze(-1)
    return result
