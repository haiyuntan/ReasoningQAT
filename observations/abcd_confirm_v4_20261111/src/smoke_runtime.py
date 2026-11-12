"""Engineering check for all28 serialized block boundaries and exact replay."""
import fcntl
from common import *
from engine import Engine

def main():
 import torch
 lock=(W/'tmp/gpu3.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
 engine=Engine(read(W/'manifests/models.json')['P0'])
 case=rows(W/'manifests/numeric_development.jsonl')[0]
 value=engine.score(case['prompts']['full'],'rfl',capture=True,blocks=True)
 cache=W/'caches/engineering_blocks.pt';atomic_torch(cache,value['blocks']);data=torch.load(cache,weights_only=True)
 assert set(data['boundaries'])==set(range(29)) and set(data['kwargs'])==set(range(28))
 assert set(value['vectors'])==set(P['A_layers'])
 def cuda(v):
  if torch.is_tensor(v):return v.to('cuda')
  if isinstance(v,tuple):return tuple(cuda(x) for x in v)
  if isinstance(v,list):return [cuda(x) for x in v]
  if isinstance(v,dict):return {k:cuda(x) for k,x in v.items()}
  return v
 maxima=[]
 for li in range(28):
  start=time.monotonic()
  with torch.inference_mode():
   out=engine.model.model.layers[li](data['boundaries'][li].to('cuda'),**cuda(data['kwargs'][li]));out=out[0] if isinstance(out,tuple) else out
   maximum=float((out.cpu().float()-data['boundaries'][li+1].float()).abs().max());maxima.append(maximum)
   assert maximum<=1e-5
  engine.pace(start)
 save(W/'metrics/engineering_runtime.json',{'status':'PASS','all28_replay_max_abs':maxima,'input_ids_sha':value['input_ids_sha'],'cache_serialization':'weights_only PASS','cache_bytes':cache.stat().st_size,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
 cache.unlink();status('engineering_runtime','DONE',layers=28,max_abs=max(maxima))
if __name__=='__main__':main()
