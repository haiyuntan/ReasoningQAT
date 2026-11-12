"""Paired/clustered bootstrap utilities; no model-dependent selection."""
import numpy as np
from scipy.stats import rankdata

def rho(x,y):
 x=np.asarray(x);y=np.asarray(y)
 if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0:return None
 a=rankdata(x);b=rankdata(y);return float(np.corrcoef(a,b)[0,1])
def paired_correlation(xf,xn,y,replicates,alpha=.05):
 n=len(y);rf=rho(xf,y);rn=rho(xn,y)
 if rf is None or rn is None:return {'n':n,'status':'INSUFFICIENT_VARIATION'}
 rng=np.random.default_rng(2);deltas=[]
 for _ in range(replicates):
  idx=rng.integers(0,n,n);f=rho(np.asarray(xf)[idx],np.asarray(y)[idx]);g=rho(np.asarray(xn)[idx],np.asarray(y)[idx])
  if f is not None and g is not None:deltas.append(f-g)
 return {'n':n,'rho_functional':rf,'rho_nmse':rn,'paired_difference':rf-rn,
  'ci95':np.quantile(deltas,[.025,.975]).tolist(),'simultaneous_ci':np.quantile(deltas,[alpha/2,1-alpha/2]).tolist(),'simultaneous_alpha':alpha,'valid_bootstrap':len(deltas)}
def cluster_mean(records,key,cluster,replicates,alpha=.05):
 groups={}
 for row in records:
  if row.get(key) is not None:groups.setdefault(row[cluster],[]).append(float(row[key]))
 values=np.array([np.mean(v) for _,v in sorted(groups.items())])
 if len(values)<3:return {'clusters':len(values),'rows':sum(map(len,groups.values())),'status':'INSUFFICIENT_CLUSTERS'}
 rng=np.random.default_rng(2);boot=np.array([rng.choice(values,len(values),replace=True).mean() for _ in range(replicates)])
 return {'clusters':len(values),'rows':sum(map(len,groups.values())),'mean':float(values.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),'cluster_unit':cluster,'simultaneous_ci':np.quantile(boot,[alpha/2,1-alpha/2]).tolist(),'simultaneous_alpha':alpha}
