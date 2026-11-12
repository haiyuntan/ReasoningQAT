"""Confirmation statistics, all cases accounted for; explicit controls and multiplicity."""
from common import *

def main():
 import numpy as np
 import torch
 from statistics_abcd import paired_correlation,cluster_mean,rho
 cases={r['theorem_id']:r for r in rows(W/'manifests/frozen_cases.jsonl')}
 target=[r for r in cases.values() if r['split']=='V-REAL-CONFIRM'];assert len(target)==128
 teacher={k:read(W/'metrics/scores/P0'/(k+'.json')) for k in cases};basis=torch.load(W/'caches/A_basis.pt',weights_only=True)
 summary={};Arows=[];Brows=[];Crows=[];Drows=[]
 for method in P['models'][1:]:
  paths=[W/'metrics/scores'/method/(k+'.json') for k in cases]
  if not all(p.exists() for p in paths):summary[method]={'status':'INCOMPLETE','missing_scores':sum(not p.exists() for p in paths)};continue
  student={k:read(p) for k,p in zip(cases,paths)}
  for k,q in student.items():
   assert q['input_ids_sha']==teacher[k]['input_ids_sha'] and q['candidate_actions']==teacher[k]['candidate_actions']
  per_layer={l:[] for l in P['A_layers']}
  for case in target:
   k=case['theorem_id'];t=teacher[k];q=student[k]
   tv=torch.load(W/'caches/vectors/P0'/(k+'.pt'),weights_only=True)['vectors'];qv=torch.load(W/'caches/vectors'/method/(k+'.pt'),weights_only=True)['vectors']
   drift=float(np.mean(np.abs(np.array(q['scores']['full'][:8])-np.array(t['scores']['full'][:8]))))
   for l in P['A_layers']:
    delta=qv[l]-tv[l];normal=float(tv[l].square().sum())+1e-30
    row={'theorem_id':k,'method':method,'layer':l,'action_drift_static8':drift,'total_nmse':float(delta.square().sum())/normal}
    for key,U in basis[l].items():row[key]=float((delta@U).square().sum())/normal
    per_layer[l].append(row);Arows.append(row)
  astats={}
  for l,records in per_layer.items():
   result=paired_correlation([r['functional'] for r in records],[r['total_nmse'] for r in records],[r['action_drift_static8'] for r in records],P['bootstrap_replicates'],alpha=P['primary_familywise_alpha']/P['primary_comparisons'])
   result.update(primary=l==P['primary_layer'] and method in P['primary_models'],random_rho=rho([r['random'] for r in records],[r['action_drift_static8'] for r in records]),pca_rho=rho([r['pca'] for r in records],[r['action_drift_static8'] for r in records]))
   astats[str(l)]=result
  brows=[]
  for case in cases.values():
   if case.get('variant')!='original':continue
   k=case['theorem_id'];other=k.replace('_original','_renamed')
   qs=np.array(student[k]['scores']['full']);qr=np.array(student[other]['scores']['full']);ts=np.array(teacher[k]['scores']['full']);tr=np.array(teacher[other]['scores']['full'])
   record={'base_group':case['base_group'],'family_id':case['family_id'],'method':method,'length_matched':case['B_length_matched'],'excess_sensitivity':float(np.mean(np.abs(qs-qr))-np.mean(np.abs(ts-tr)))}
   brows.append(record);Brows.append(record)
  bstats={'primary_length_matched_family_clustered':cluster_mean([r for r in brows if r['length_matched']],'excess_sensitivity','family_id',P['bootstrap_replicates'],alpha=P['primary_familywise_alpha']/P['primary_comparisons']),
   'all_instances_family_clustered':cluster_mean(brows,'excess_sensitivity','family_id',P['bootstrap_replicates']),'pairs':len(brows),'matched_pairs':sum(r['length_matched'] for r in brows)}
  crecords=[]
  for case in target:
   k=case['theorem_id'];q=student[k]['scores'];t=teacher[k]['scores'];removed=[];sham=[];valid=[];unvalidated=[]
   for control in case['C_controls']:
    j=control['premise_index']
    diffs={}
    for variant in [f'remove_{j}',f'same_length_{j}',f'record_replace_{j}']:
     iq=np.array(q['full'])-np.array(q[variant]);it=np.array(t['full'])-np.array(t[variant]);diffs[variant]=float(np.mean(np.abs(iq-it)))
    r=diffs[f'remove_{j}'];s=diffs[f'same_length_{j}'];removed.append(r);sham.append(s)
    (valid if control['valid_application'] else unvalidated).append(r)
    Crows.append({'theorem_id':k,'method':method,'premise_index':j,'valid_application':control['valid_application'],**diffs})
   rec={'theorem_id':k,'method':method,'remove_drift':float(np.mean(removed)),'same_length_drift':float(np.mean(sham)),
    'remove_minus_same_length':float(np.mean(removed)-np.mean(sham)),'valid_minus_unvalidated':float(np.mean(valid)-np.mean(unvalidated)) if valid and unvalidated else None,
    'order_drift':float(np.mean(np.abs((np.array(q['full'])-np.array(q['order_reverse']))-(np.array(t['full'])-np.array(t['order_reverse'])))))}
   crecords.append(rec)
  cstats={name:cluster_mean(crecords,name,'theorem_id',P['bootstrap_replicates'],alpha=P['primary_familywise_alpha']/P['primary_comparisons'] if name=='remove_minus_same_length' else .05) for name in ['remove_drift','same_length_drift','remove_minus_same_length','valid_minus_unvalidated','order_drift']}
  replay_files=list((W/'metrics/D'/method).glob('*.json'));dstats={'status':'INCOMPLETE','theorems':len(replay_files)}
  if len(replay_files)==P['D_theorems']:
   data=[]
   for path in replay_files:
    r=read(path);assert len(r['layers'])==28
    for row in r['layers']:
     assert row['replay_max_abs']<=1e-5 and row['algebra_max_abs']<1e-10
     record={'theorem_id':r['theorem_id'],'method':method,**row};data.append(record);Drows.append(record)
   dstats={'status':'EXACT_REPLAY_PASS','theorems':32,'layers':28,'records':len(data),
    'per_layer':{str(l):{key:cluster_mean([{'theorem_id':r['theorem_id'],key:r['last_prompt'][key]} for r in data if r['layer']==l],key,'theorem_id',P['bootstrap_replicates']) for key in ['cosine','cancellation','excess_over_orthogonal','energy_relative_to_orthogonal']} for l in range(1,29)}}
  summary[method]={'status':'CONFIRMATION_SCORES_DONE','model_role':'PRIMARY_CONFIRMATION' if method in P['primary_models'] else 'SECONDARY_BASELINE','A':astats,'B':bstats,'C':cstats,'D':dstats,
   'numeric_controls':read(W/'metrics'/f'{method}_numeric.json'),'limits':'candidate scores, not theorem proof success; semantic relevance/overlap unknown where not verified'}
 jsonl(W/'metrics/A_per_theorem.jsonl',Arows);jsonl(W/'metrics/B_per_pair.jsonl',Brows);jsonl(W/'metrics/C_per_premise.jsonl',Crows);jsonl(W/'metrics/D_all_layers.jsonl',Drows)
 save(W/'metrics/confirmation_summary.json',summary)
 (W/'reports/results.md').write_text('# ABCD confirmation\nActual estimates and coverage are in metrics/confirmation_summary.json. A primary=layer28 paired rho difference, Bonferroni CIs across9 A/B/C ×P3/P4/P6 primary comparisons; other layers/P5 secondary. B family-clustered, length-matched primary. C deletion minus matched-length control; lexical distractor relevance is not certified. D strict replay before interpreting prefix/last-token geometry; undefined cosines/cancellation stay null. No proof-search accuracy or M-training claim.\n')
 status('analysis','DONE',models=list(summary))
if __name__=='__main__':main()
