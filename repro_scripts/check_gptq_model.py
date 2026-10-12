"""Reload the two-sample GPTQ component checkpoint and verify finite GPU inference."""
import json
from pathlib import Path
import time
import torch
from runtime_compat import ensure_dtensor_alias
ensure_dtensor_alias()
from gptqmodel import GPTQModel
from gptqmodel.utils.backend import BACKEND
from transformers import AutoTokenizer
from storage import prune

root=Path(__file__).resolve().parents[2]
path=root/'outputs/component_checks/R1'
config=json.loads((path/'quantize_config.json').read_text())
assert config['bits']==2 and config['group_size']==128
start=time.monotonic()
model=GPTQModel.load(str(path),backend=BACKEND.TORCH,device='cuda:0')
tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
inputs=tokenizer('What is 2 + 2?',return_tensors='pt').to('cuda')
with torch.no_grad():
    logits=model(**inputs).logits
    assert torch.isfinite(logits).all()
    generated=model.generate(**inputs,max_new_tokens=2,do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
assert generated.shape[1]>inputs['input_ids'].shape[1]
result={'status':'PASS','meaning':'Two-sample W2 GPTQ quantize/save/reload/inference component check; not R1 evaluation',
        'bits':2,'group_size':128,'backend':'TORCH','samples':2,'sequence_length':128,
        'checkpoint_bytes':sum(p.stat().st_size for p in path.rglob('*') if p.is_file()),
        'peak_memory_bytes':torch.cuda.max_memory_allocated(),'reload_inference_seconds':time.monotonic()-start}
result['quantization_metadata']=json.loads((path/'repro_quantization.json').read_text())
result['quantize_config']=config
result['cleanup']=prune(path,'Passed reproducible two-sample GPTQ component check')
(root/'results/gptq-component-check.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
