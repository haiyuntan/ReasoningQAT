import argparse
import hashlib
import json
import os
from pathlib import Path
import torch
from runtime_compat import ensure_dtensor_alias
ensure_dtensor_alias()
from transformers import AutoModelForCausalLM, AutoTokenizer
from awq.quantize.pre_quant import run_awq
from awq.quantize.quantizer import pseudo_quantize_model_weight

p=argparse.ArgumentParser()
p.add_argument('--model',required=True)
p.add_argument('--output',required=True)
p.add_argument('--samples',type=int,default=128)
p.add_argument('--seqlen',type=int,default=512)
a=p.parse_args()
os.environ.setdefault('RQ_DATA',str(Path(__file__).resolve().parents[2]/'datasets'))
out=Path(a.output)
out.mkdir(parents=True,exist_ok=True)
tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True)
cfg={'zero_point':True,'q_group_size':128}
original_use_cache=model.config.use_cache
model.config.use_cache=False
scales=run_awq(model,tokenizer,w_bit=2,q_config=cfg,n_samples=a.samples,seqlen=a.seqlen,calib_data='numinamath_local')
model.config.use_cache=original_use_cache
torch.save(scales,out/'awq_scales.pt')
pseudo_quantize_model_weight(model,w_bit=2,q_config=cfg)
model=model.cuda().eval()
inputs=tokenizer('What is 2 + 2?',return_tensors='pt').to('cuda')
with torch.no_grad():
    before=model(**inputs).logits.float().cpu()
model.save_pretrained(out,safe_serialization=True)
tokenizer.save_pretrained(out)
del model
torch.cuda.empty_cache()
reloaded=AutoModelForCausalLM.from_pretrained(out,torch_dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True).cuda().eval()
with torch.no_grad():
    after=reloaded(**inputs).logits.float().cpu()
assert torch.allclose(before,after,atol=1e-4,rtol=1e-4),'AWQ fake export parity failed'
source=Path(__file__).resolve().parents[2]/'datasets/numinamath_128/calib.jsonl'
metadata={'method':'AWQ','bits':2,'group_size':128,'storage':'BF16 W2 pseudo-quantized weights',
          'samples':a.samples,'seqlen':a.seqlen,'calibration_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
          'reload_logits_max_diff':(before-after).abs().max().item()}
(out/'repro_quantization.json').write_text(json.dumps(metadata,indent=2))
print(json.dumps(metadata,indent=2))
