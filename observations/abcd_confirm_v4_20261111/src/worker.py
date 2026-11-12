"""Resume-safe per-case scoring and separate discovery fit; no optimizer exists."""
import argparse,fcntl
from common import *
from engine import Engine

def fit(engine):
 import torch
 path=W/'caches/A_basis.pt';receipt=W/'manifests/A_fit.json'
 if receipt.exists() and path.exists():
  r=read(receipt);assert r['input_sha']==sha(W/'manifests/A_fit_inputs.jsonl') and r['basis_sha']==sha(path) and r['groups']==128 and r['protocol_sha']==protocol_sha();return
 gradients={l:[] for l in P['A_layers']};vectors={l:[] for l in P['A_layers']};accepted=[]
 for row in rows(W/'manifests/A_fit_inputs.jsonl'):
  r=engine.score(row['prompt'],row['reference_or_candidate_action'],grad=True,capture=True)
  for l in P['A_layers']:gradients[l].append(r['gradients'][l]);vectors[l].append(r['vectors'][l])
  accepted.append(row['state_id']);status('A_fit_progress','RUNNING',groups=len(accepted),target=128)
 assert len(accepted)==128
 basis={}
 for l in P['A_layers']:
  G=torch.stack(gradients[l]);X=torch.stack(vectors[l]);rank=P['A_rank']
  _,_,V=torch.linalg.svd(G,full_matrices=False);_,_,PC=torch.linalg.svd(X-X.mean(0),full_matrices=False)
  R=torch.linalg.qr(torch.randn(X.shape[1],rank,generator=torch.Generator().manual_seed(2+l))).Q
  basis[l]={'functional':V[:rank].T.contiguous(),'pca':PC[:rank].T.contiguous(),'random':R}
 atomic_torch(path,basis);save(receipt,{'groups':128,'protocol_sha':protocol_sha(),'state_ids':accepted,'input_sha':sha(W/'manifests/A_fit_inputs.jsonl'),'basis_sha':sha(path),'rank':4,'layers':P['A_layers'],'split':'S-OBS-DISC BF16_TRAIN_DISTRIBUTION','frozen_before_confirm_scores':True})

def main():
 import torch
 ap=argparse.ArgumentParser();ap.add_argument('--method',required=True,choices=P['models']);ap.add_argument('--prepare-only',action='store_true');args=ap.parse_args();method=args.method
 lock=(W/'tmp/gpu3.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
 models=read(W/'manifests/models.json');engine=Engine(models[method]);engine.numeric_controls(method);engine.native_parity(method)
 if method=='P0':fit(engine)
 if args.prepare_only:
  assert method=='P0';status('prefit_P0','DONE',groups=128);return
 cases=rows(W/'manifests/frozen_cases.jsonl');total=0
 for case in cases:
  identifier=case['theorem_id'];output=W/'metrics/scores'/method/(identifier+'.json');vector_path=W/'caches/vectors'/method/(identifier+'.pt');block_path=W/'caches/D'/method/(identifier+'.pt')
  if output.exists():
   previous=read(output);assert previous['protocol_sha']==protocol_sha() and previous['checkpoint_sha']==models[method]['checkpoint_sha']
   assert previous['case_sha']==digest(json.dumps(case,sort_keys=True))
   assert vector_path.exists()
   if case['D_selected'] and not block_path.exists() and not (W/'metrics/D'/method/(identifier+'.json')).exists():
    r=engine.score(case['input_ids']['full'],case['candidates'][0],capture=True,blocks=True);atomic_torch(block_path,r['blocks'])
   total+=1;status(method+'_progress','RUNNING',completed=total,total=len(cases),resumed=True);continue
  item={'theorem_id':identifier,'base_group':case['base_group'],'family_id':case.get('family_id'),'variant':case.get('variant'),'B_length_matched':case.get('B_length_matched'),
   'split':case['split'],'model_id':method,'protocol_sha':protocol_sha(),'checkpoint_sha':models[method]['checkpoint_sha'],'case_sha':digest(json.dumps(case,sort_keys=True)),
   'candidate_actions':case['candidates'],'candidate_origin':case['candidate_origin'],'scores':{},'input_ids_sha':{},'status':'DONE','numeric_path':'single_unpadded_BF16','retrieval_sha':case['retrieval_sha']}
  vectors=None;d_cache=None
  for variant,ids in case['input_ids'].items():
   item['scores'][variant]=[];item['input_ids_sha'][variant]=[]
   for j,action in enumerate(case['candidates']):
    capture=variant=='full' and j==0
    r=engine.score(ids,action,capture=capture,blocks=capture and case['D_selected'])
    item['scores'][variant].append(r['score']);item['input_ids_sha'][variant].append(r['input_ids_sha'])
    if capture:
     vectors={'vectors':r['vectors'],'position':r['prompt_tokens']-1,'input_ids_sha':r['input_ids_sha']};d_cache=r['blocks']
  assert vectors is not None
  atomic_torch(vector_path,vectors)
  if d_cache is not None:
   atomic_torch(block_path,d_cache)
   save(W/'manifests/D_cache'/method/(identifier+'.json'),{'sha256':sha(block_path),'case_sha':item['case_sha'],'path':str(block_path),'temporary_owned_cache':True})
  save(output,item);total+=1;status(method+'_progress','RUNNING',completed=total,total=len(cases))
 save(W/'metrics'/f'{method}_runtime.json',{'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'GPU':P['GPU'],'batch':1,'training':False})
 status('score_'+method,'DONE',cases=total)
if __name__=='__main__':main()
