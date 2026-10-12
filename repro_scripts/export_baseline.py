"""Strict native baseline reload, materialization, and HF reload parity check."""
import argparse
import json
from pathlib import Path
import torch
from torch import nn
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer
from runtime_compat import ensure_dtensor_alias
from baseline_ops import portable_packed_class,packed_weight,bitdistiller_ops,set_module
ensure_dtensor_alias()

def restore_native(path,device='cpu'):
    path=Path(path)
    metadata=json.loads((path/'baseline_adapter.json').read_text())
    config=AutoConfig.from_pretrained(path,local_files_only=True)
    model=AutoModelForCausalLM.from_config(config,torch_dtype=torch.bfloat16,attn_implementation='eager')
    if metadata['run']=='R8':
        Packed=portable_packed_class()
        for index,layer in enumerate(model.model.layers):
            for name,module in list(layer.named_modules()):
                if isinstance(module,nn.Linear):
                    packed=Packed(2,128,module.in_features,module.out_features,module.bias is not None)
                    # E2E-QP trained scales in FP32; preserve rather than cast.
                    packed.scales.data=packed.scales.data.float()
                    set_module(layer,name,packed)
    else:
        linear,_,_,_=bitdistiller_ops()
        model,_=linear.convertModelToQuant(model,quant_type='int2-asym',q_group_size=128)
    state=torch.load(path/'baseline_state.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True)
    del state
    return model.to(device).eval(),metadata

def materialize(model,run):
    count=0
    for name,module in list(model.named_modules()):
        if run=='R8' and hasattr(module,'qweight') and hasattr(module,'scales'):
            weight=packed_weight(module).t().contiguous()
            linear=nn.Linear(module.infeatures,module.outfeatures,bias=module.bias is not None,
                             device=weight.device,dtype=next(model.parameters()).dtype)
        elif run=='R9' and hasattr(module,'weight_quantizer'):
            weight=module.weight_quantizer(module.weight.to(module.compute_dtype))
            linear=nn.Linear(module.in_features,module.out_features,bias=module.bias is not None,
                             device=weight.device,dtype=weight.dtype)
        else: continue
        with torch.no_grad():
            linear.weight.copy_(weight)
            if module.bias is not None: linear.bias.copy_(module.bias)
        set_module(model,name,linear);count+=1
    assert count>0
    return count

def export(path,output,device='cuda'):
    model,metadata=restore_native(path,device)
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    tokens=tokenizer('What is 2 + 2?',return_tensors='pt').to(device)
    with torch.no_grad():
        before=model(**tokens).logits.float().cpu()
        count=materialize(model,metadata['run'])
        materialized=model(**tokens).logits.float().cpu()
    assert torch.isfinite(before).all()
    assert torch.equal(before,materialized),f'Native materialization changed logits: {(before-materialized).abs().max().item()}'
    model.save_pretrained(output,safe_serialization=True)
    tokenizer.save_pretrained(output)
    del model
    if device=='cuda': torch.cuda.empty_cache()
    reloaded=AutoModelForCausalLM.from_pretrained(output,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='eager').to(device).eval()
    with torch.no_grad(): after=reloaded(**tokens).logits.float().cpu()
    assert torch.equal(before,after),f'Baseline HF reload changed logits: {(before-after).abs().max().item()}'
    evidence={'method':metadata['run'],'bits':2,'group_size':128,'replaced_linears':count,
              'reload_logits_max_diff':float((before-after).abs().max()),
              'native_materialization_max_diff':float((before-materialized).abs().max()),
              'strict_native_reload':True,'storage':'BF16 materialized W2 weights; native checkpoint authoritative'}
    Path(output,'repro_quantization.json').write_text(json.dumps(evidence,indent=2))
    print(json.dumps(evidence,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    export(args.input,args.output)
