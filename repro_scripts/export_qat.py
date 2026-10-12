"""Restore original QAT modules, materialize W2 weights and check reload parity."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from quantize.int_linear_fake import load_quantized_model,QuantLinear
from quantize.utils import set_quant_state,set_op_by_name

p=argparse.ArgumentParser()
p.add_argument('--input',required=True)
p.add_argument('--output',required=True)
a=p.parse_args()
model,tokenizer=load_quantized_model(a.input,2,128,replace=True,strict=True,use_flash_attn=False,device='cpu')
model=model.cuda().eval()
set_quant_state(model,weight_quant=True)
inputs=tokenizer('What is 2 + 2?',return_tensors='pt').to('cuda')
with torch.no_grad():
    before=model(**inputs).logits.float().cpu()
    count=0
    for name,module in list(model.named_modules()):
        if isinstance(module,QuantLinear):
            linear=torch.nn.Linear(module.in_features,module.out_features,bias=module.bias is not None,
                                   device=module.weight.device,dtype=module.weight.dtype)
            linear.weight.copy_(module.weight_quantizer(module.weight))
            if module.bias is not None:
                linear.bias.copy_(module.bias)
            set_op_by_name(model,name,linear)
            count+=1
    assert count>0
    materialized=model(**inputs).logits.float().cpu()
    assert torch.allclose(before,materialized,atol=1e-4,rtol=1e-4)
model.save_pretrained(a.output,safe_serialization=True)
tokenizer.save_pretrained(a.output)
dtype=next(model.parameters()).dtype
del model
torch.cuda.empty_cache()
reloaded=AutoModelForCausalLM.from_pretrained(a.output,torch_dtype=dtype,attn_implementation='eager',local_files_only=True).cuda().eval()
with torch.no_grad():
    after=reloaded(**inputs).logits.float().cpu()
assert torch.allclose(before,after,atol=1e-4,rtol=1e-4),'QAT export parity failed'
meta={'method':'ReasoningQAT','bits':2,'group_size':128,'storage':str(dtype)+' materialized W2 weights',
      'replaced_linears':count,'reload_logits_max_diff':(before-after).abs().max().item()}
Path(a.output,'repro_quantization.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta,indent=2))
