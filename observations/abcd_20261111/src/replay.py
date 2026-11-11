"""Exact BF16 teacher block replay of cached student inputs, one model resident."""
import argparse,time
from common import *
def main():
 import fcntl
 gpu_lock=(W/'tmp/gpu3.lock').open('a');fcntl.flock(gpu_lock,fcntl.LOCK_EX)
 import torch
 from transformers import AutoModelForCausalLM
 ap=argparse.ArgumentParser();ap.add_argument('--student',required=True);a=ap.parse_args()
 torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(resources()['allocator_cap_gib']*1024**3/torch.cuda.get_device_properties(0).total_memory,0)
 ids=json.loads((W/'manifests/models.json').read_text())
 teacher=AutoModelForCausalLM.from_pretrained(ids['P0']['path'],torch_dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True).to('cuda').eval()
 teacher.requires_grad_(False);rows=[]
 def cuda(v):
  if torch.is_tensor(v):return v.to('cuda')
  if isinstance(v,tuple):return tuple(cuda(x) for x in v)
  return v
 cases=read_rows(W/'manifests/V_real_states.jsonl')
 for case in [r for r in cases if r['split']=='V-REAL'][:8]:
  name=case['theorem_id'];tpath=W/'caches'/f'P0_{name}_blocks.pt';qpath=W/'caches'/f'{a.student}_{name}_blocks.pt'
  if not tpath.exists() or not qpath.exists():continue
  T=torch.load(tpath,weights_only=True);Q=torch.load(qpath,weights_only=True)
  for li in T:
   start=time.monotonic()
   with torch.inference_mode():
    layer=teacher.model.layers[li];kw={k:cuda(v) for k,v in T[li]['kwargs'].items()}
    tf=layer(T[li]['input'].to('cuda'),**kw);tf=tf[0] if isinstance(tf,tuple) else tf
    tq=layer(Q[li]['input'].to('cuda'),**kw);tq=tq[0] if isinstance(tq,tuple) else tq
    parity=float((tf.cpu().float()-T[li]['output'].float()).abs().max())
    if parity>.125:raise RuntimeError('REPLAY_PARITY_BLOCKED')
    t=T[li]['output'].float();q=Q[li]['output'].float();mid=tq.cpu().float()
    prop=mid-t;inj=q-mid;actual=q-t;res=actual-prop-inj
    denom=float(actual.square().sum())+1e-20
    cosine=float((prop*inj).sum()/(prop.norm()*inj.norm()+1e-20))
    cancellation=1-float(actual.norm()/(prop.norm()+inj.norm()+1e-20))
    basis=torch.load(W/'caches/A_basis.pt',weights_only=True)[li]['functional']
    vecs=torch.load(W/'caches/P0_vectors.pt',weights_only=True);pos=vecs[name]['prefix_tokens']-1
    projected={k:float((v[0,pos]@basis).square().sum()) for k,v in [('propagation',prop),('injection',inj),('actual',actual)]}
    row={'theorem_id':name,'split':'V-REAL','model_id':a.student,'layer':li+1,'replay_max_abs':parity,
      'algebra_residual_max_abs':float(res.abs().max()),'algebra_relative_squared_residual':float(res.square().sum())/denom,
      'propagation_squared':float(prop.square().sum()),'injection_squared':float(inj.square().sum()),
      'actual_squared':float(actual.square().sum()),'cosine':cosine,'cancellation_coefficient':cancellation,
      'functional_projection_last_prompt':projected,'status':'PILOT_EXACT_REPLAY_PASS'}
    assert row['algebra_residual_max_abs']<1e-5
    rows.append(row)
   torch.cuda.synchronize();elapsed=time.monotonic()-start;time.sleep(max(.05,elapsed*(1/resources()['duty_cycle']-1)))
  write_rows(W/'metrics'/f'D_{a.student}.jsonl',rows)
 record('D_'+a.student,'DONE',records=len(rows),independent_theorems=len(set(r['theorem_id'] for r in rows)))
if __name__=='__main__':main()
