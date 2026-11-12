"""Paper first crossing, full-vocabulary JSD and direction-correct confidence."""
import math,torch
def jsd(lp,lq):
    mix=torch.logaddexp(lp,lq)-math.log(2)
    d=.5*((lp.exp()*(lp-mix)).sum(-1)+(lq.exp()*(lq-mix)).sum(-1))
    assert torch.isfinite(d).all() and (d>=-1e-5).all() and (d<=math.log(2)+1e-5).all()
    return d.clamp_min(0)
def depths(values,g=.5,rho=.85):
    if len(values)==0:raise ValueError('INVALID_EMPTY_SEQUENCE')
    L=len(values[0]);cutoff=math.ceil(rho*L);first=[];persistent=[]
    for row in values:
        assert abs(row[-1])<1e-5
        first.append(next(i+1 for i,v in enumerate(row) if v<=g))
        persistent.append(next(i+1 for i in range(L) if max(row[i:])<=g))
    return dict(dtr=sum(x>=cutoff for x in first)/len(first),first_crossing=first,persistent=persistent,mean_settling_depth=sum(first)/len(first),cutoff=cutoff)
def certainty(lp):return -math.log(lp.shape[-1])-lp.mean(-1)
