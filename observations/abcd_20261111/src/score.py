"""Read-only paired ABCD pilot: no optimizer, no generation/search or weight writes."""
import argparse,gc,importlib.util,time,math
from common import *

def main():
 import fcntl
 gpu_lock=(W/'tmp/gpu3.lock').open('a');fcntl.flock(gpu_lock,fcntl.LOCK_EX)
 import numpy as np
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 ap=argparse.ArgumentParser();ap.add_argument('--method',required=True);a=ap.parse_args()
 method=a.method;identities=json.loads((W/'manifests/models.json').read_text());identity=identities[method]
 assert torch.cuda.device_count()==1
 torch.set_num_threads(2);torch.manual_seed(P['seed'])
 torch.cuda.set_per_process_memory_fraction(resources()['allocator_cap_gib']*1024**3/torch.cuda.get_device_properties(0).total_memory,0)
 tokenizer=AutoTokenizer.from_pretrained(identity['path'],local_files_only=True)
 model=AutoModelForCausalLM.from_pretrained(identity['path'],torch_dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True).to('cuda').eval()
 model.requires_grad_(False);model.config.use_cache=False
 layers=[n-1 for n in P['layers_one_based']]
 def pace(start):
  torch.cuda.synchronize();elapsed=time.monotonic()-start;time.sleep(max(.05,elapsed*(1/resources()['duty_cycle']-1)))
 def forward(prompt,action,grad=False,capture=False):
  prefix=prompt if isinstance(prompt,list) else tokenizer.encode(prompt,add_special_tokens=False);target=tokenizer.encode(action,add_special_tokens=False)
  if not prefix or not target or len(prefix)+len(target)>P['max_sequence_length']:raise ValueError('OVERLENGTH_OR_EMPTY_PILOT')
  ids=torch.tensor([prefix+target],device='cuda');mask=torch.ones_like(ids);pos=len(prefix)-1
  outputs={};inputs={};kwargs={};hooks=[]
  if capture:
   for li in layers:
    def pre(mod,args,kw,li=li):
     inputs[li]=args[0].detach().cpu()
     kwargs[li]={k:cpu(v) for k,v in kw.items()}
    def post(mod,args,out,li=li):outputs[li]=out[0] if isinstance(out,tuple) else out
    hooks.extend([model.model.layers[li].register_forward_pre_hook(pre,with_kwargs=True),model.model.layers[li].register_forward_hook(post)])
  if grad:hooks.append(model.model.embed_tokens.register_forward_hook(lambda m,i,o:o.detach().requires_grad_(True)))
  start=time.monotonic()
  try:
   with torch.set_grad_enabled(grad):
    hidden=model.model(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state
    selected=hidden[:,pos:pos+len(target)]
    logits=model.lm_head(selected).float()
    targets=torch.tensor(target,device='cuda')
    score=torch.log_softmax(logits,dim=-1)[0,torch.arange(len(target),device='cuda'),targets].mean()
    vectors={li:outputs[li][0,pos].detach().float().cpu() for li in outputs}
    gradients={}
    if grad:
     values=torch.autograd.grad(score,[outputs[li] for li in layers])
     gradients={li:value[0,pos].detach().float().cpu() for li,value in zip(layers,values)}
    full={li:{'input':inputs[li],'output':outputs[li].detach().cpu(),'kwargs':kwargs[li]} for li in outputs} if capture else {}
    return float(score.detach()),vectors,gradients,full,digest(ids.cpu().numpy().tobytes()),len(prefix),len(target)
  finally:
   for h in hooks:h.remove()
   pace(start)
 def cpu(v):
  if torch.is_tensor(v):return v.detach().cpu()
  if isinstance(v,tuple):return tuple(cpu(x) for x in v)
  return v
 def cuda(v):
  if torch.is_tensor(v):return v.to('cuda')
  if isinstance(v,tuple):return tuple(cuda(x) for x in v)
  return v
 def batch_scores(prompt,actions):
  prefix=prompt if isinstance(prompt,list) else tokenizer.encode(prompt,add_special_tokens=False)
  targets=[tokenizer.encode(action,add_special_tokens=False) for action in actions]
  if any(not t or len(prefix)+len(t)>P['max_sequence_length'] for t in targets):raise ValueError('OVERLENGTH_OR_EMPTY_PILOT')
  width=max(len(prefix)+len(t) for t in targets);pad=tokenizer.pad_token_id
  if pad is None:pad=tokenizer.eos_token_id
  ids=torch.full((len(actions),width),pad,dtype=torch.long,device='cuda');mask=torch.zeros_like(ids)
  for i,target in enumerate(targets):
   seq=prefix+target;ids[i,:len(seq)]=torch.tensor(seq,device='cuda');mask[i,:len(seq)]=1
  start=time.monotonic()
  try:
   with torch.inference_mode():
    hidden=model.model(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state
    pos=len(prefix)-1;scores=[];hashes=[]
    for i,target in enumerate(targets):
     logits=model.lm_head(hidden[i:i+1,pos:pos+len(target)]).float()
     target_ids=torch.tensor(target,device='cuda')
     scores.append(float(logits.log_softmax(-1)[0,torch.arange(len(target),device='cuda'),target_ids].mean()))
     hashes.append(digest(ids[i:i+1,:len(prefix)+len(target)].cpu().numpy().tobytes()))
    return scores,hashes
  finally:pace(start)
 batch_limit=resources()['candidate_batch_size']
 # Validate on the longest frozen prompt and all static actions before enabling batching.
 cases=read_rows(W/'manifests/V_real_states.jsonl')
 eligible=[r for r in cases if len(tokenizer.encode(r['prompts']['full'],add_special_tokens=False))+max(len(tokenizer.encode(t,add_special_tokens=False)) for t in r['candidates'])<=P['max_sequence_length']]
 assert eligible,'NO_ELIGIBLE_STRESS_CASE'
 test=max(eligible,key=lambda r:len(tokenizer.encode(r['prompts']['full'],add_special_tokens=False)))
 test_prompt=test['prompts']['full'];test_actions=test['candidates']
 if len(tokenizer.encode(test_prompt,add_special_tokens=False))+max(len(tokenizer.encode(t,add_special_tokens=False)) for t in test_actions)<=P['max_sequence_length']:
  singles=[];start=time.monotonic()
  with torch.inference_mode():
   for action in test_actions:singles.append(forward(test_prompt,action)[0])
  single_seconds=time.monotonic()-start;trials=[]
  for size in [n for n in (2,4,8) if n<=batch_limit]:
   try:
    start=time.monotonic();values=[]
    for begin in range(0,len(test_actions),size):values.extend(batch_scores(test_prompt,test_actions[begin:begin+size])[0])
    elapsed=time.monotonic()-start;error=max(abs(x-y) for x,y in zip(singles,values))
    trials.append({'batch':size,'seconds':elapsed,'max_score_abs':error,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'valid':error<=resources()['batch_score_max_abs_tolerance']})
   except torch.cuda.OutOfMemoryError:
    torch.cuda.empty_cache();trials.append({'batch':size,'valid':False,'reason':'OOM'});break
  valid=[t for t in trials if t['valid']]
  winner=min(valid,key=lambda t:t['seconds']) if valid else None
  batch_limit=winner['batch'] if winner and winner['seconds']<single_seconds else 1
  save(W/'metrics'/f'{method}_batch_stress.json',{'single_seconds':single_seconds,'trials':trials,'selected_batch':batch_limit,'runtime_resources':resources(),'parity_case':test['theorem_id']})
 else:batch_limit=1
 # Dense export is compared to independent native probes, not to itself.
 if method!='P0':
  import ast
  path=ROOT/'runs/realprover_formal_four_gpu_v16/formal'/method/'train/native_probe.pt'
  reference=torch.load(path,map_location='cpu',weights_only=True)
  tree=ast.parse((ROOT/'research/adapters/quantization.py').read_text())
  prompts=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PROMPTS' for t in n.targets))
  maxima=[];means=[];agree=[]
  with torch.inference_mode():
   for prompt,expected in zip(prompts,reference['logits']):
    start=time.monotonic();inp=tokenizer(prompt,return_tensors='pt').to('cuda')
    hidden=model.model(**inp,use_cache=False).last_hidden_state[:,-1:]
    actual=model.lm_head(hidden)[:,0].float().cpu();delta=(actual-expected.float()).abs()
    maxima.append(float(delta.max()));means.append(float(delta.mean()));agree.append(bool(torch.equal(actual.argmax(-1),expected.argmax(-1))));pace(start)
  parity={'max_abs':max(maxima),'mean_abs':sum(means)/len(means),'top1':sum(agree)/len(agree),'native_probe_sha':sha(path)}
  save(W/'metrics'/f'{method}_parity.json',parity)
  assert parity['max_abs']<=.125 and parity['mean_abs']<=.01 and parity['top1']==1.,'NATIVE_PARITY_BLOCKED'
 if method=='P0':
  gradients={li:[] for li in layers};fit_vectors={li:[] for li in layers};accepted=[]
  spec=importlib.util.spec_from_file_location('frozen_prompt',W/'runtime/source/manager/manage/prompt_manage.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
  for row in read_rows(W/'manifests/discovery_pilot.jsonl'):
   prompt=row['input'] if '<|im_start|>' in row['input'] else mod.PromptManage.chat_template_to_prompt([{'role':'user','content':row['input']}],'qwen')
   try:s,v,g,_,ih,_,_=forward(prompt,row['reference_or_candidate_action'],grad=True,capture=True)
   except ValueError:continue
   for li in layers:gradients[li].append(g[li]);fit_vectors[li].append(v[li])
   accepted.append({'state_id':row['state_id'],'input_ids_sha':ih})
   record('fit_A','RUNNING',groups=len(accepted))
  assert len(accepted)>=8,'INSUFFICIENT_DISCOVERY_GROUPS'
  basis={}
  for li in layers:
   G=torch.stack(gradients[li]);X=torch.stack(fit_vectors[li]);rank=min(P['functional_rank'],len(accepted))
   _,_,vh=torch.linalg.svd(G,full_matrices=False);U=vh[:rank].T.contiguous()
   _,_,pv=torch.linalg.svd(X-X.mean(0),full_matrices=False)
   gen=torch.Generator().manual_seed(2+li);R=torch.linalg.qr(torch.randn(X.shape[1],rank,generator=gen)).Q
   basis[li]={'functional':U,'pca':pv[:rank].T.contiguous(),'random':R}
  torch.save(basis,W/'caches/A_basis.pt');save(W/'manifests/A_fit.json',{'groups':accepted,'split':'S-OBS-DISC','rank':P['functional_rank'],'layers':P['layers_one_based'],'frozen_before_valid_forward':True,'target':'mean reference-tactic log probability gradient at last prompt token'})
 # All methods use exactly the same frozen cases, prompts and candidate strings.
 rows=read_rows(W/'manifests/V_real_states.jsonl');summaries=read_rows(W/'metrics'/f'{method}_scores.jsonl') if (W/'metrics'/f'{method}_scores.jsonl').exists() else [];vectors=torch.load(W/'caches'/f'{method}_vectors.pt',weights_only=True) if (W/'caches'/f'{method}_vectors.pt').exists() else {};d_ids=[r['theorem_id'] for r in rows if r['split']=='V-REAL'][:8]
 for row in rows:
  if any(previous['theorem_id']==row['theorem_id'] for previous in summaries):continue
  item={'theorem_id':row['theorem_id'],'base_group':row['base_group'],'split':row['split'],'rule_id':row['rule_id'],
   'variant':row.get('variant'),'model_id':method,'checkpoint_sha':identity['checkpoint_sha'],
   'protocol_sha':sha(W/'configs/protocol.json'),'state_id':row['state_id'],'retrieval_sha':row['retrieval_sha'],
   'runtime_resources':resources(),'candidate_batch_limit':batch_limit,'candidate_origin':row['candidate_origin'],'candidates':row['candidates'],'scores':{},'input_ids_sha':{},'status':'PILOT'}
  prompts=dict(row['prompts'])
  # Same encoded prompt length, preserve suffix (state/history/action marker).
  # This is an explicit token-level negative control, not a valid retrieved theorem.
  if row['premises']:
   fullids=tokenizer.encode(prompts['full'],add_special_tokens=False);minusids=tokenizer.encode(prompts['remove_first'],add_special_tokens=False)
   common=0
   while common<min(len(fullids),len(minusids)) and fullids[common]==minusids[common]:common+=1
   suffix=0
   while suffix<min(len(fullids),len(minusids))-common and fullids[-suffix-1]==minusids[-suffix-1]:suffix+=1
   end=len(fullids)-suffix if suffix else len(fullids)
   token=tokenizer.encode(' irrelevant',add_special_tokens=False)[0]
   prompts['same_length_distractor']=fullids[:common]+[token]*(end-common)+fullids[end:]
   assert len(prompts['same_length_distractor'])==len(fullids)
   item['distractor_control']='TOKEN_LEVEL_SAME_LENGTH_REPLACEMENT; synthetic irrelevant content, not a retrieved theorem'
   item['distractor_span']=[common,end]
   item['full_prompt_tokens']=len(fullids);item['removed_prompt_tokens']=len(minusids)
  try:
   for variant,prompt in prompts.items():
    item['scores'][variant]=[];item['input_ids_sha'][variant]=[]
    begin=0
    if variant=='full':
     action=row['candidates'][0]
     with torch.inference_mode():s,v,_,full,ih,pl,tl=forward(prompt,action,capture=True)
     item['scores'][variant].append(s);item['input_ids_sha'][variant].append(ih)
     vectors[row['theorem_id']]={'vectors':v,'prefix_tokens':pl,'target_tokens':tl}
     if row['theorem_id'] in d_ids:torch.save(full,W/'caches'/f'{method}_{row["theorem_id"]}_blocks.pt')
     begin=1
    for ci in range(begin,len(row['candidates']),batch_limit):
     actions=row['candidates'][ci:ci+batch_limit]
     values,hashes=batch_scores(prompt,actions)
     item['scores'][variant].extend(values);item['input_ids_sha'][variant].extend(hashes)
   item['best_full']=int(np.argmax(item['scores']['full']))
  except ValueError as e:item.update(status='SKIPPED_OVERLENGTH',reason=str(e))
  summaries.append(item);write_rows(W/'metrics'/f'{method}_scores.jsonl',summaries)
  torch.save(vectors,W/'caches'/f'{method}_vectors.pt')
  record(method+'_scores','RUNNING',cases=len(summaries))
 save(W/'metrics'/f'{method}_runtime.json',{'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'gpu_uuid':P['gpu_uuid'],'training':False,'method':method,'runtime_resources':resources(),'selected_batch':batch_limit})
 record(method,'FORWARD_DONE',cases=len(summaries))
 del model;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
