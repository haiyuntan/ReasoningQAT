"""Compact pilot statistics; theorem-cluster bootstrap, no layer selection."""
from common import *
def main():
 import numpy as np
 import torch
 from scipy.stats import spearmanr
 basis=torch.load(W/'caches/A_basis.pt',weights_only=True)
 teacher={r['theorem_id']:r for r in read_rows(W/'metrics/P0_scores.jsonl') if r['status']=='PILOT'}
 tv=torch.load(W/'caches/P0_vectors.pt',weights_only=True)
 results={};flat=[]
 def bootstrap(values):
  values=np.asarray(values,dtype=float)
  if len(values)<3:return {'n':len(values),'status':'INSUFFICIENT_EVIDENCE'}
  rng=np.random.default_rng(2);means=[float(rng.choice(values,len(values),replace=True).mean()) for _ in range(2000)]
  return {'n':len(values),'mean':float(values.mean()),'ci95':np.quantile(means,[.025,.975]).tolist(),'status':'PILOT_NOT_CONFIRMATORY'}
 for method in P['models'][1:]:
  path=W/'metrics'/f'{method}_scores.jsonl'
  if not path.exists():continue
  student={r['theorem_id']:r for r in read_rows(path) if r['status']=='PILOT'}
  qv=torch.load(W/'caches'/f'{method}_vectors.pt',weights_only=True)
  for name,q in student.items():
   if name in teacher:
    assert q['input_ids_sha']==teacher[name]['input_ids_sha'], 'CANDIDATE_INPUT_PARITY_BLOCKED'
    assert q['candidates']==teacher[name]['candidates'], 'CANDIDATE_IDENTITY_BLOCKED'
  stats={'A':{},'B':{},'C':{},'scope':'engineering pilot; no claim of validated mechanism'}
  for li in basis:
   rows=[]
   for name,q in student.items():
    if q['split']!='V-REAL' or name not in teacher or name not in qv or name not in tv:continue
    t=teacher[name];delta=qv[name]['vectors'][li]-tv[name]['vectors'][li];norm=float(tv[name]['vectors'][li].square().sum())+1e-20
    row={'theorem_id':name,'model':method,'layer':li+1,'action_drift':float(np.mean(np.abs(np.array(q['scores']['full'])-np.array(t['scores']['full'])))),
      'total_nmse':float(delta.square().sum())/norm}
    for key,U in basis[li].items():row[key+'_normalized_error']=float((delta@U).square().sum())/norm
    rows.append(row);flat.append(row)
   columns={}
   for key in ['total_nmse','functional_normalized_error','pca_normalized_error','random_normalized_error']:
    x=[r[key] for r in rows];y=[r['action_drift'] for r in rows]
    rho=float(spearmanr(x,y).statistic) if len(rows)>=3 and len(set(x))>1 and len(set(y))>1 else None
    rng=np.random.default_rng(2);boot=[]
    for _ in range(2000):
     idx=rng.integers(0,len(rows),len(rows)) if rows else []
     bx=[x[i] for i in idx];by=[y[i] for i in idx]
     if len(set(bx))>1 and len(set(by))>1:boot.append(float(spearmanr(bx,by).statistic))
    columns[key]={'spearman':rho,'theorems':len(rows),'ci95':np.quantile(boot,[.025,.975]).tolist() if boot else None}
   stats['A'][str(li+1)]=columns
  b=[]
  for name,q in student.items():
   if q.get('variant')!='original':continue
   other=name.replace('_original','_renamed')
   if other not in student or name not in teacher or other not in teacher:continue
   qs=np.array(q['scores']['full']);qr=np.array(student[other]['scores']['full'])
   ts=np.array(teacher[name]['scores']['full']);tr=np.array(teacher[other]['scores']['full'])
   b.append(float(np.mean(np.abs(qs-qr))-np.mean(np.abs(ts-tr))))
  stats['B']['alpha_excess_sensitivity']=bootstrap(b)
  for control in ['remove_first','same_length_distractor','order_reverse']:
   values=[]
   for name,q in student.items():
    if q['split']!='V-REAL' or name not in teacher:continue
    t=teacher[name]
    if control not in q['scores'] or control not in t['scores']:continue
    iq=np.array(q['scores']['full'])-np.array(q['scores'][control]);it=np.array(t['scores']['full'])-np.array(t['scores'][control])
    values.append(float(np.mean(np.abs(iq-it))))
   stats['C'][control]=bootstrap(values)
  results[method]=stats
 save(W/'metrics/ABCD_summary.json',results);write_rows(W/'metrics/A_per_theorem.jsonl',flat)
 (W/'reports/06_ABCD_external_validation.md').write_text('# ABCD pilot\nSee metrics/ABCD_summary.json, A_per_theorem.jsonl and D_*.jsonl. A bases were fit only on S-OBS-DISC; V-REAL roots are held out. B is a small typed alpha-renaming family, not broad symmetry evidence. C uses frozen true retrieval plus remove/order/token-length controls. D requires numeric replay parity before interpretation. All results are exploratory engineering pilot; G5 external confirmation NOT COMPLETED.\n')
 (W/'reports/07_decision_memo.md').write_text('# Decision memo\nM0/M10/M20 NOT_STARTED. G2 complete span-aware calibration parity remains blocked, and exact-state separation does not certify source-theorem or semantic disjointness. Valid has frozen root states and static candidates, not gold actions or proof-search metrics. A fit uses upstream BF16 training-distribution data and valid only evaluates; B uses a small independent rule family; C cannot alone prove semantic relevance; D reports exact algebra and replay tolerances. Do not infer novelty or final theorem success from this pilot. Referenced full ABCD SOP missing; no replacement for future independent held-out confirmation.\n')
 record('analysis','DONE',methods=list(results))
if __name__=='__main__':main()
