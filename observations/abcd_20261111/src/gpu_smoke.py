"""Small actual GPU forward/gradient/replay sanity, no training or weight writes."""
import fcntl,time
from common import *
def main():
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 lock=(W/'tmp/gpu3.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
 assert torch.cuda.device_count()==1;torch.set_num_threads(2)
 torch.cuda.set_per_process_memory_fraction(P['allocator_cap_gib']*1024**3/torch.cuda.get_device_properties(0).total_memory,0)
 modelpath=json.loads((W/'manifests/models.json').read_text())['P0']['path']
 tok=AutoTokenizer.from_pretrained(modelpath,local_files_only=True)
 model=AutoModelForCausalLM.from_pretrained(modelpath,torch_dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True).to('cuda').eval()
 model.requires_grad_(False);model.config.use_cache=False
 ids=tok('Prove that n = n in Lean.\nTACTIC:\n',return_tensors='pt').to('cuda')
 captures={};hooks=[]
 for li in [n-1 for n in P['layers_one_based']]:
  def pre(m,args,kw,li=li):captures.setdefault(li,{}).update(input=args[0].detach().clone(),kwargs=kw)
  def post(m,args,out,li=li):captures[li]['output']=out[0] if isinstance(out,tuple) else out
  hooks.extend([model.model.layers[li].register_forward_pre_hook(pre,with_kwargs=True),model.model.layers[li].register_forward_hook(post)])
 hooks.append(model.model.embed_tokens.register_forward_hook(lambda m,i,o:o.detach().requires_grad_(True)))
 start=time.monotonic();hidden=model.model(**ids,use_cache=False).last_hidden_state
 score=model.lm_head(hidden[:,-1:]).float().log_softmax(-1)[0,0,0]
 grads=torch.autograd.grad(score,[captures[li]['output'] for li in sorted(captures)])
 assert all(torch.isfinite(g).all() for g in grads)
 for hook in hooks:hook.remove()
 residual={}
 with torch.inference_mode():
  for li,cap in captures.items():
   replay=model.model.layers[li](cap['input'],**cap['kwargs']);replay=replay[0] if isinstance(replay,tuple) else replay
   delta=float((replay-cap['output']).abs().max());assert delta<=.125
   residual[str(li+1)]=delta
 torch.cuda.synchronize();time.sleep(max(.05,time.monotonic()-start))
 save(W/'metrics/gpu_smoke.json',{'status':'PASS','single_visible_gpu':True,'gradient_layers':P['layers_one_based'],
  'same_input_replay_max_abs':residual,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
  'training':False,'note':'small engineering check only; not external ABCD evidence'})
 print('GPU_FORWARD_GRADIENT_REPLAY_PASS',residual,flush=True)
if __name__=='__main__':main()
