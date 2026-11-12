"""FP32-only development probe for shape sensitivity; never alters a BF16 engine."""
import fcntl
from common import *
from engine import Engine

def main():
 import torch
 lock=(W/'tmp/gpu3.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
 engine=Engine(read(W/'manifests/models.json')['P0'],dtype_name='fp32')
 baseline=read(W/'metrics/P0_numeric.json');results=[]
 for case in rows(W/'manifests/numeric_development.jsonl'):
  for action in P['static_actions'][:2]:
   prompt=case['prompts']['full'];score=engine.score(prompt,action)['score']
   padded={str(k):engine.shape_score(prompt,action,pad_extra=k)[0]-score for k in [8,32]}
   batched=[v-score for v in engine.shape_score(prompt,action,batch=2)]
   results.append({'development_case':case['theorem_id'],'action':action,'fp32_base':score,'fp32_padding_delta':padded,'fp32_batch2_delta':batched})
 bf_pad=max(abs(v) for r in baseline['records'] for v in r['padding_score_delta'].values());bf_batch=max(abs(v) for r in baseline['records'] for v in r['same_length_batch2_delta'])
 fp_pad=max(abs(v) for r in results for v in r['fp32_padding_delta'].values());fp_batch=max(abs(v) for r in results for v in r['fp32_batch2_delta'])
 save(W/'metrics/P0_precision_shape_probe.json',{'status':'DIAGNOSTIC_COMPLETE','BF16_padding_max_abs':bf_pad,'BF16_batch2_max_abs':bf_batch,'FP32_padding_max_abs':fp_pad,'FP32_batch2_max_abs':fp_batch,
  'precision_dependence_supported_on_development':fp_pad<1e-4 and fp_batch<1e-4,'records':results,'primary_precision_unchanged':'BF16 batch1 unpadded','peak_allocated_bytes':torch.cuda.max_memory_allocated(),'source_checkpoint_unchanged':True})
 status('precision_probe','DONE',BF16_pad=bf_pad,FP32_pad=fp_pad,BF16_batch=bf_batch,FP32_batch=fp_batch)
if __name__=='__main__':main()
