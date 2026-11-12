"""Full vocabulary pre-sampler entropy, stable tails, reproducible CDF sampling."""
import torch
def entropy(z,dtype=torch.float32):
    lp=z.to(dtype).log_softmax(-1); h=-(lp.exp()*lp).sum(-1)
    assert torch.isfinite(h).all() and (h>=-1e-7).all()
    return h.clamp_min(0)
def logtail(z,excluded=None):
    z=z.double();i=int(z.argmax()) if excluded is None else int(excluded)
    other=z.clone();other[i]=-torch.inf
    return torch.logsumexp(other,-1)-torch.logsumexp(z,-1)
def distribution(z,temperature,top_p):
    # HF/vLLM ascending top-p convention. CPU sorting/cumsum avoids Torch2.2
    # CUDA cumsum's nondeterministic-kernel rejection. Keep the pinned CPU sort
    # tie order and sample in token-ID CDF order; do not disable determinism.
    logits=z.detach().float().cpu()/temperature
    vals,idx=torch.sort(logits,descending=False)
    remove=vals.softmax(-1).cumsum(-1)<=(1-top_p);remove[-1]=False
    mask=torch.zeros_like(remove);mask.scatter_(0,idx,remove)
    return logits.masked_fill(mask,-torch.inf).softmax(-1)
def sample(p,u):
    assert 0<=u<1
    cdf=p.double().cumsum(-1)
    return min(int(torch.searchsorted(cdf,cdf[-1]*u)),p.numel()-1)
def noise(z,sigma,eta):return z+sigma*z*eta
def classify(h,threshold):
    # Explicit disjoint tie handling. Equal q25/q75 is invalid calibration.
    assert threshold['q25']<threshold['q75']
    return 'LOW' if h<=threshold['q25'] else ('HIGH' if h>=threshold['q75'] else 'MID')
def token_category(text):
    import re
    t=text.strip()
    if t in {'exact','apply','rw','simp','simpa','intro','rintro','have','refine','constructor','linarith','nlinarith','ring','norm_num','aesop','rfl'}:return 'tactic_keyword'
    if not t:return 'whitespace'
    if re.fullmatch(r'\d+(\.\d+)?',t):return 'numeric'
    if re.fullmatch(r'[\[\](){}⟨⟩,:;]+',t):return 'syntax'
    if re.fullmatch(r'[+*/^=<>≤≥≠↔→∧∨¬-]+',t):return 'operator'
    if re.fullmatch(r'[\w]+(?:\.[\w]+)+',t):return 'theorem_reference'
    if re.fullmatch(r'[\w]+',t):return 'identifier'
    return 'UNKNOWN'
def paired_metrics(t,q,target=None):
    ti=int(t.argmax());qi=int(q.argmax());lt=logtail(t);lq=logtail(q);la=logtail(q,ti)
    out=dict(H_T=float(entropy(t)),H_Q=float(entropy(q)),top1_T=ti,top1_Q=qi,top1_flip=ti!=qi,
             log_tail_T=float(lt),log_tail_Q=float(lq),log_rho=float(lq-lt),delta_tail=float(lq.exp()-lt.exp()),delta_tail_teacher_anchor=float(la.exp()-lt.exp()),
             epsilon_log_ratios={str(e):float(torch.logaddexp(lq,torch.tensor(e,dtype=torch.float64,device=q.device).log())-torch.logaddexp(lt,torch.tensor(e,dtype=torch.float64,device=q.device).log())) for e in [0,1e-12,1e-10,1e-8]})
    if target is not None:out['delta_nll']=float(t.float().log_softmax(-1)[target]-q.float().log_softmax(-1)[target])
    for name,T,p in [('task',1.5,.9),('paper',.6,.95)]:
        pt=distribution(t,T,p);pq=distribution(q,T,p)
        out[name+'_delta_sampling_tail']=float(pt[ti]-pq[ti]);out[name+'_support_T']=int((pt>0).sum());out[name+'_support_Q']=int((pq>0).sum())
    return out
