import argparse
import hashlib
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from gptqmodel import GPTQModel, QuantizeConfig

p=argparse.ArgumentParser()
p.add_argument('--model',required=True)
p.add_argument('--output',required=True)
p.add_argument('--samples',type=int,default=128)
p.add_argument('--seqlen',type=int,default=512)
a=p.parse_args()
root=Path(__file__).resolve().parents[2]
source=root/'datasets/numinamath_128/calib.jsonl'
rows=[json.loads(line) for line in source.read_text().splitlines()]
assert len(rows)==128 and a.samples in (2,128)
tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
calib=[]
for row in rows[:a.samples]:
    tokens=tokenizer.encode(row['text'],add_special_tokens=True)[:a.seqlen]
    tokens += [tokenizer.eos_token_id]*(a.seqlen-len(tokens))
    # Exactly the same padded token IDs as AWQ's local loader.
    calib.append({'input_ids':torch.tensor([tokens]), 'attention_mask':torch.ones(1,len(tokens),dtype=torch.long)})
cfg=QuantizeConfig(bits=2,group_size=128,desc_act=False,sym=True)
model=GPTQModel.load(a.model,cfg)
model.quantize(calib,batch_size=1)
model.save(a.output)
metadata={'method':'GPTQ','bits':2,'group_size':128,'samples':a.samples,'seqlen':a.seqlen,
          'desc_act':False,'sym':True,'calibration_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
          'comparability':'Author exact GPTQ settings/NuminaMath selection unavailable'}
Path(a.output,'repro_quantization.json').write_text(json.dumps(metadata,indent=2))
print(json.dumps(metadata,indent=2))
