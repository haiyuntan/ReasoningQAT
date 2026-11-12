"""One resident model; fixed batch1 scores; explicit numerical and exact replay paths."""
import time
from common import *

class Engine:
 def __init__(self,identity,dtype_name="bf16"):
  import torch
  from transformers import AutoModelForCausalLM,AutoTokenizer
  self.torch=torch;assert torch.cuda.device_count()==1
  torch.set_num_threads(2);torch.manual_seed(P['seed']);torch.backends.cuda.matmul.allow_tf32=False
  torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
  torch.cuda.set_per_process_memory_fraction(P['GPU']['allocator_cap_gib']*1024**3/torch.cuda.get_device_properties(0).total_memory,0)
  self.tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/REAL-Prover-v1-bf16',local_files_only=True)
  self.model=AutoModelForCausalLM.from_pretrained(identity['path'],torch_dtype=torch.float32 if dtype_name=='fp32' else torch.bfloat16,attn_implementation='eager',local_files_only=True).to('cuda').eval()
  self.model.requires_grad_(False);self.model.config.use_cache=False
  assert len(self.model.model.layers)==28
 def pace(self,start):
  self.torch.cuda.synchronize();elapsed=time.monotonic()-start
  time.sleep(max(.002,elapsed*(1/P['GPU']['duty']-1)))
 def tokens(self,prompt):return prompt if isinstance(prompt,list) else self.tokenizer.encode(prompt,add_special_tokens=False)
 def action(self,action):return self.tokenizer.encode(action,add_special_tokens=False)
 def score(self,prompt,action,grad=False,capture=False,blocks=False):
  torch=self.torch;prefix=self.tokens(prompt);target=self.action(action)
  if not prefix or not target or len(prefix)+len(target)>P['max_sequence_length']:raise ValueError('Invalid frozen sequence length')
  ids=torch.tensor([prefix+target],device='cuda');mask=torch.ones_like(ids);position=len(prefix)-1
  hooks=[];outputs={};vectors={};boundaries={};kwargs={};memo={}
  def cpu(v):
   if torch.is_tensor(v):
    key=id(v)
    if key not in memo:memo[key]=v.detach().cpu()
    return memo[key]
   if isinstance(v,dict):return {k:cpu(x) for k,x in v.items()}
   if isinstance(v,tuple):return tuple(cpu(x) for x in v)
   if isinstance(v,list):return [cpu(x) for x in v]
   if v is None or isinstance(v,(str,int,float,bool)):return v
   raise TypeError('Unsupported replay object '+str(type(v)))
  active_layers=range(28) if blocks else [l-1 for l in P['A_layers']]
  if grad or capture or blocks:
   for li in active_layers:
    if blocks:
     def pre(layer,args,kw,li=li):
      if li==0:boundaries[0]=args[0].detach().cpu()
      kwargs[li]=cpu(kw)
     hooks.append(self.model.model.layers[li].register_forward_pre_hook(pre,with_kwargs=True))
    def post(layer,args,out,li=li):
     hidden=out[0] if isinstance(out,tuple) else out
     if li+1 in P['A_layers']:
      vectors[li+1]=hidden[0,position].detach().float().cpu()
      if grad:outputs[li+1]=hidden
     if blocks:boundaries[li+1]=hidden.detach().cpu()
    hooks.append(self.model.model.layers[li].register_forward_hook(post))
  if grad:hooks.append(self.model.model.embed_tokens.register_forward_hook(lambda m,i,o:o.detach().requires_grad_(True)))
  start=time.monotonic()
  try:
   with torch.set_grad_enabled(grad):
    hidden=self.model.model(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state
    logits=self.model.lm_head(hidden[:,position:position+len(target)]).float()
    score=logits.log_softmax(-1)[0,torch.arange(len(target),device='cuda'),torch.tensor(target,device='cuda')].mean()
    gradients={}
    if grad:
     values=torch.autograd.grad(score,[outputs[l] for l in P['A_layers']])
     gradients={l:g[0,position].detach().float().cpu() for l,g in zip(P['A_layers'],values)}
    cache={'boundaries':boundaries,'kwargs':kwargs,'position':position,'input_ids_sha':digest(ids.cpu().numpy().tobytes())} if blocks else None
    return {'score':float(score.detach()),'vectors':vectors,'gradients':gradients,'blocks':cache,'input_ids_sha':digest(ids.cpu().numpy().tobytes()),'prompt_tokens':len(prefix),'target_tokens':len(target)}
  finally:
   for hook in hooks:hook.remove()
   self.pace(start)
 def shape_score(self,prompt,action,pad_extra=0,batch=1):
  torch=self.torch;prefix=self.tokens(prompt);target=self.action(action);original=prefix+target
  pad=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
  ids=torch.tensor([original+[pad]*pad_extra]*batch,device='cuda');mask=torch.tensor([[1]*len(original)+[0]*pad_extra]*batch,device='cuda')
  start=time.monotonic()
  try:
   with torch.inference_mode():
    hidden=self.model.model(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state
    logits=self.model.lm_head(hidden[:,len(prefix)-1:len(prefix)-1+len(target)]).float()
    logp=logits.log_softmax(-1);target_ids=torch.tensor(target,device='cuda')
    return [float(logp[i,torch.arange(len(target),device='cuda'),target_ids].mean()) for i in range(batch)]
  finally:self.pace(start)
 def numeric_controls(self,method):
  records=[]
  for case in rows(W/'manifests/numeric_development.jsonl'):
   for action in P['static_actions'][:2]:
    prompt=case['prompts']['full'];values=[self.score(prompt,action)['score'] for _ in range(P['numeric_repeats'])]
    padded={str(k):self.shape_score(prompt,action,pad_extra=k)[0] for k in [8,32]}
    batch=self.shape_score(prompt,action,batch=2)
    records.append({'development_case':case['theorem_id'],'action':action,'identical_repeats':values,
     'repeat_max_abs':max(abs(v-values[0]) for v in values),'padding_score_delta':{k:v-values[0] for k,v in padded.items()},
     'same_length_batch2_delta':[v-values[0] for v in batch]})
  maximum=max(r['repeat_max_abs'] for r in records)
  save(W/'metrics'/f'{method}_numeric.json',{'status':'PASS' if maximum<=P['same_input_noise_tolerance'] else 'BLOCKED','records':records,'same_input_max_abs':maximum,'primary_batch':1,'primary_padding':False,'shape_changes':'diagnostic only, never pooled into confirm scores'})
  assert maximum<=P['same_input_noise_tolerance'],'NUMERIC_REPEAT_BLOCKED'
 def native_parity(self,method):
  if method=='P0':return
  import ast
  torch=self.torch
  source=ROOT/'research/adapters/quantization.py';tree=ast.parse(source.read_text())
  prompts=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PROMPTS' for t in n.targets))
  path=ROOT/'runs/realprover_formal_four_gpu_v16/formal'/method/'train/native_probe.pt';ref=torch.load(path,map_location='cpu',weights_only=True)
  maxima=[];means=[];correct=[]
  for prompt,expected in zip(prompts,ref['logits']):
   start=time.monotonic()
   with torch.inference_mode():
    inp=self.tokenizer(prompt,return_tensors='pt').to('cuda');hidden=self.model.model(**inp,use_cache=False).last_hidden_state[:,-1:]
    actual=self.model.lm_head(hidden)[:,0].float().cpu();delta=(actual-expected.float()).abs()
    maxima.append(float(delta.max()));means.append(float(delta.mean()));correct.append(bool(torch.equal(actual.argmax(-1),expected.argmax(-1))))
   self.pace(start)
  receipt={'native_probe_sha':sha(path),'max_abs':max(maxima),'mean_abs':sum(means)/len(means),'top1':sum(correct)/len(correct)}
  save(W/'metrics'/f'{method}_native_parity.json',receipt)
  assert receipt['max_abs']<=.125 and receipt['mean_abs']<=.01 and receipt['top1']==1.,'NATIVE_PARITY_BLOCKED'
