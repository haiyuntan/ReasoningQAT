"""Independent Qwen3 smoke adapters for pinned EfficientQAT and BitDistiller."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl.trainer.utils import DataCollatorForChatML
from baseline_ops import efficient_ops, portable_packed_class, bitdistiller_ops, set_module

ROOT=Path(__file__).resolve().parents[2]
REPO=ROOT/'ReasoningQAT'
sys.path.insert(0,str(REPO))
from repro_scripts.local_datasets import load_local_or_hf_openthoughts
from runtime_compat import ensure_dtensor_alias
ensure_dtensor_alias()

def finite_step(loss,optimizer,parameters,clip_norm=1.0):
    parameters=list(parameters)
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite baseline loss')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grads=[p.grad for p in parameters if p.grad is not None]
    if not grads or not all(torch.isfinite(grad).all() for grad in grads):
        raise FloatingPointError('Missing or nonfinite baseline gradients')
    norm=(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.detach().float()) for g in grads]))
          if clip_norm is None else torch.nn.utils.clip_grad_norm_(parameters,clip_norm))
    optimizer.step()
    return float(norm)

def autocast(device):
    return torch.autocast(device.type,dtype=torch.bfloat16)

def layer_output(layer,hidden,kwargs):
    result=layer(hidden,**kwargs)
    return result[0] if isinstance(result,tuple) else result

class CaptureDone(Exception):
    pass

def capture_inputs(model,sequences,device):
    """Capture each sample's Qwen3 rotary embeddings and mask explicitly."""
    original=model.model.layers[0]
    captured=[]
    def hook(module,args,kwargs):
        def cpu(value):
            if isinstance(value,torch.Tensor): return value.detach().cpu()
            if isinstance(value,tuple): return tuple(cpu(v) for v in value)
            return value
        captured.append((args[0].detach().cpu(),{k:cpu(v) for k,v in kwargs.items()}))
        raise CaptureDone()
    handle=original.register_forward_pre_hook(hook,with_kwargs=True)
    try:
        with torch.no_grad():
            for tokens in sequences:
                try: model(input_ids=tokens.to(device),use_cache=False)
                except CaptureDone: pass
    finally: handle.remove()
    assert len(captured)==len(sequences)
    return captured

def to_device_kwargs(kwargs,device):
    def move(value):
        if isinstance(value,torch.Tensor): return value.to(device)
        if isinstance(value,tuple): return tuple(move(v) for v in value)
        return value
    return {k:move(v) for k,v in kwargs.items()}

def block_cache(mode):
    matrix=json.loads((REPO/'repro_configs/matrix.json').read_text())
    cfg={**matrix['formal'],**(matrix['smoke'] if mode=='smoke' else {})}
    cache=ROOT/'hf_cache/block_qat'/mode
    train,val=[torch.load(cache/f'dataloader_Qwen3-1.7B_sweep_0.8_{cfg["block_train_size"]}_{cfg["block_val_size"]}_{cfg["block_sequence_length"]}_{name}.cache',map_location='cpu',weights_only=False)
               for name in ('train','val')]
    return cfg,train,val

def stage2_batches(tokenizer,mode):
    matrix=json.loads((REPO/'repro_configs/matrix.json').read_text())
    cfg={**matrix['formal'],**(matrix['smoke'] if mode=='smoke' else {})}
    ds=load_local_or_hf_openthoughts().shuffle(seed=2).select(range(cfg['e2e_dataset_size']))
    examples=[]
    for sample in ds:
        messages=[]
        for entry in sample['conversations']:
            role='user' if entry['from']=='human' else 'assistant'
            text=entry['value']
            if mode=='smoke':
                limit=32 if role=='user' else 96
                text=tokenizer.decode(tokenizer.encode(text,add_special_tokens=False)[:limit],skip_special_tokens=True)
            messages.append({'role':role,'content':text})
        examples.append({'messages':messages})
    collator=DataCollatorForChatML(tokenizer=tokenizer,max_length=cfg['e2e_max_length'])
    batches=[collator([example]) for example in examples]
    # ChatML collator includes prompt tensors for teacher RR. These baselines
    # use the fixed original trajectory; only standard causal-LM fields apply.
    batches=[{k:v for k,v in batch.items() if k in ('input_ids','attention_mask','labels')} for batch in batches]
    digest=hashlib.sha256()
    for batch in batches:
        for key in ('input_ids','attention_mask','labels'):
            digest.update(batch[key].numpy().tobytes())
    evidence={'samples':len(batches),'tokens_after_mask':sum(int((b['labels']!=-100).sum()) for b in batches),
              'causal_target_tokens':sum(int((b['labels'][:,1:]!=-100).sum()) for b in batches),
              'collator':'TRL DataCollatorForChatML, same shortening and seed as R4',
              'tensor_sha256':digest.hexdigest(),'trajectory':'Fixed dataset assistant trajectory; no regeneration'}
    assert evidence['tokens_after_mask']>0
    return batches,evidence

def efficient_block_ap(model,sequences,val_sequences,device,cfg):
    """Port original one-block MSE reconstruction with independent initialization."""
    Fake,_,_=efficient_ops()
    Packed=portable_packed_class()
    fp=capture_inputs(model,sequences+val_sequences,device)
    quant=[(hidden.clone(),kwargs) for hidden,kwargs in fp]
    history=[]
    model.requires_grad_(False)
    for index,original in enumerate(model.model.layers):
        with torch.no_grad(),autocast(device):
            targets=[layer_output(original,hidden.to(device),to_device_kwargs(kwargs,device)).float().cpu()
                     for hidden,kwargs in fp]
        layer=copy.deepcopy(original).float()
        for name,module in list(layer.named_modules()):
            if isinstance(module,nn.Linear):
                set_module(layer,name,Fake(module,2,128))
        weight_params=[];quant_params=[]
        for module in layer.modules():
            if isinstance(module,Fake):
                module.set_quant_state(True)
        # Match upstream weight_parameters/set_weight_parameters exactly:
        # includes RMSNorm/q_norm/k_norm weights, not only projection weights.
        weight_names=[];quant_names=[]
        for name,param in layer.named_parameters():
            if 'scale' in name or 'zero_point' in name:
                param.requires_grad_(True);quant_params.append(param);quant_names.append(name)
            elif 'weight' in name:
                param.requires_grad_(True);weight_params.append(param);weight_names.append(name)
        optimizer=torch.optim.AdamW([{'params':weight_params,'lr':cfg['block_weight_lr']},
                                     {'params':quant_params,'lr':cfg['block_quant_lr']}],weight_decay=0.)
        total=cfg['block_epochs']*len(sequences)
        schedulers=[torch.optim.lr_scheduler.CosineAnnealingLR(
            torch.optim.AdamW([torch.tensor(0.)],lr=lr),T_max=total,eta_min=lr/20)
            for lr in (cfg['block_weight_lr'],cfg['block_quant_lr'])]
        for epoch in range(cfg['block_epochs']):
            losses=[]
            for sample,(hidden,kwargs) in enumerate(quant[:len(sequences)]):
                with autocast(device):
                    output=layer_output(layer,hidden.to(device),to_device_kwargs(kwargs,device))
                    loss=torch.nn.functional.mse_loss(output.float(),targets[sample].to(device))
                norm=finite_step(loss,optimizer,weight_params+quant_params,clip_norm=None)
                losses.append(float(loss.detach()))
                for group,scheduler in zip(optimizer.param_groups,schedulers):
                    scheduler.step();group['lr']=scheduler.get_last_lr()[0]
            with torch.no_grad(),autocast(device):
                val_loss=sum(float(torch.nn.functional.mse_loss(
                    layer_output(layer,hidden.to(device),to_device_kwargs(kwargs,device)).float(),targets[i].to(device)))
                    for i,(hidden,kwargs) in enumerate(quant) if i>=len(sequences))/len(val_sequences)
            if not math.isfinite(val_loss): raise FloatingPointError('Block validation is nonfinite')
            history.append({'block':index,'epoch':epoch,'mse':sum(losses)/len(losses),'val_mse':val_loss,'grad_norm':norm,
                            'weight_parameter_names':weight_names,'quant_parameter_names':quant_names})
        del optimizer
        # Use upstream native pack exactly. Quantize in FP16 as its Block-AP
        # does; packed scales are FP16. PyTorch unpack is only a kernel port.
        layer.half()
        with torch.no_grad():
            for name,module in list(layer.named_modules()):
                if not isinstance(module,Fake): continue
                module.weight.copy_(module.weight_quantizer(module.weight))
                scales=module.weight_quantizer.scale.clamp(1e-4,1e4).detach().reshape(module.out_features,-1).t().contiguous().cpu()
                zeros=module.weight_quantizer.zero_point.round().clamp(0,3).detach().reshape(module.out_features,-1).t().contiguous().cpu()
                packed=Packed(2,128,module.in_features,module.out_features,module.bias is not None)
                packed.pack(module.cpu(),scales.float(),zeros.float())
                packed.to(device)
                set_module(layer,name,packed)
            layer.to(dtype=next(model.parameters()).dtype,device=device)
            with autocast(device):
                quant=[(layer_output(layer,hidden.to(device),to_device_kwargs(kwargs,device)).detach().cpu(),kwargs)
                       for hidden,kwargs in quant]
        fp=[(targets[i].to(quant[i][0].dtype),kwargs) for i,(_,kwargs) in enumerate(fp)]
        model.model.layers[index]=layer
        print(json.dumps(history[-1]),flush=True)
        del original
    return history

def bitdistiller_clip(model,sequences,device,range_output=None):
    _,_,_,clip=bitdistiller_ops()
    captured=capture_inputs(model,sequences,device)
    history=[];ranges=[]
    for index,layer in enumerate(model.model.layers):
        features={};handles=[]
        for name,module in layer.named_modules():
            if isinstance(module,nn.Linear) and not any(key in name for key in ('q_','k_','query','key','Wqkv')):
                features[name]=[]
                def hook(module,inputs,output,name=name): features[name].append(inputs[0].detach().cpu())
                handles.append(module.register_forward_hook(hook))
        try:
            with torch.no_grad():
                next_inputs=[(layer_output(layer,hidden.to(device),to_device_kwargs(kwargs,device)).detach().cpu(),kwargs)
                             for hidden,kwargs in captured]
        finally:
            for handle in handles: handle.remove()
        # Upstream propagates the unmodified block output before applying clips.
        for name,values in features.items():
            module=layer.get_submodule(name)
            upper,lower=clip(module.weight,torch.cat(values,dim=0).to(device),2,
                             {'q_group_size':128,'quant_type':'int'},
                             n_sample_token=8 if os.environ.get('RQ_RUN_MODE')=='smoke' else 512)
            with torch.no_grad():
                shape=module.weight.shape
                module.weight.copy_(module.weight.reshape(shape[0],-1,128).clamp(lower,upper).reshape(shape))
            history.append({'block':index,'linear':name,'asymmetric':True,'grid':20,'max_shrink':.5,
                            'requested_sample_tokens':8 if os.environ.get('RQ_RUN_MODE')=='smoke' else 512})
            ranges.append({'module':f'model.layers.{index}.{name}','upper':upper.detach().cpu(),'lower':lower.detach().cpu()})
        captured=next_inputs
        print('CLIPPED_BLOCK',index,flush=True)
    if range_output is not None:
        range_output=Path(range_output)
        range_output.parent.mkdir(parents=True,exist_ok=True)
        torch.save(ranges,range_output)
    return history

def stage2_train(model,teacher,batches,run,device,cfg):
    losses=[]
    if run=='R8':
        model.requires_grad_(False)
        for name,param in model.named_parameters():
            if name.endswith('.scales'):
                param.data=param.data.float();param.requires_grad_(True)
        names=[name for name,p in model.named_parameters() if p.requires_grad]
        assert names and all(name.endswith('.scales') for name in names)
        beta=None
    else:
        # Original BitDistiller performs full weight QAT, not scale-only QAT.
        model.requires_grad_(True)
        names=[name for name,p in model.named_parameters() if p.requires_grad]
        _,_,cakld,_=bitdistiller_ops()
        probabilities=[]
        with torch.no_grad():
            for batch in batches:
                inputs={k:v.to(device) for k,v in batch.items() if k!='labels'}
                logits=teacher(**inputs).logits.float()
                # Upstream coefficient uses all positions, including prompts.
                probabilities.append(float(logits.softmax(-1).amax(-1).mean()))
        beta=sum(probabilities)/len(probabilities)
        assert 0<=beta<=1
    parameters=[p for p in model.parameters() if p.requires_grad]
    probes={name:p.detach().reshape(-1)[:64].float().cpu().clone()
            for name,p in model.named_parameters() if p.requires_grad}
    optimizer=torch.optim.AdamW(parameters,lr=cfg['e2e_learning_rate'],weight_decay=0.)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=2)
    model.train()
    for step in range(2):
        batch={k:v.to(device) for k,v in batches[step%len(batches)].items()}
        with autocast(device):
            if run=='R8': loss=model(**batch).loss.float()
            else:
                with torch.no_grad(): teacher_logits=teacher(input_ids=batch['input_ids'],attention_mask=batch['attention_mask']).logits.float()
                student_logits=model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask']).logits.float()
                loss=cakld(None,batch['labels'],student_logits,teacher_logits,beta)
        norm=finite_step(loss,optimizer,parameters)
        scheduler.step()
        losses.append({'step':step,'loss':float(loss.detach()),'grad_norm':norm})
        print(json.dumps(losses[-1]),flush=True)
    changed=sum(int((p.detach().reshape(-1)[:64].float().cpu()!=probes[name]).sum())
                for name,p in model.named_parameters() if p.requires_grad)
    if changed==0: raise RuntimeError('Baseline optimizer made no observed parameter updates')
    return {'objective':'CE' if run=='R8' else 'Original CAKLD, unshifted label mask, sequence sum/batch mean',
            'trainable_names':names,'steps':losses,'confidence_beta':beta,
            'confidence_estimator':'Mean of actual batches; upstream off-by-one/fixed-denominator/distributed-only bugs corrected',
            'optimizer':'AdamW','learning_rate':cfg['e2e_learning_rate'],'scheduler':'cosine',
            'teacher_used':run=='R9','updated_parameter_probe_values':changed}

def save_native(model,tokenizer,out,run,evidence):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    model.config.save_pretrained(out)
    tokenizer.save_pretrained(out)
    # Native state dictionaries retain packed R8 tensors or R9 latent weights.
    torch.save(model.cpu().state_dict(),out/'baseline_state.pt')
    evidence.update({'run':run,'bits':2,'group_size':128,'format':'upstream_packed' if run=='R8' else 'upstream_QLinear_latent',
                     'storage':'Native state dictionary; no duplicate model snapshot',
                     'formal_comparable':False})
    (out/'baseline_adapter.json').write_text(json.dumps(evidence,indent=2))

def train_baseline(run,mode,out):
    if mode!='smoke':
        raise RuntimeError('NOT_COMPARABLE: author baseline hyperparameters and token budget remain unresolved')
    if os.environ.get('RQ_RUN_MODE')!='smoke': raise RuntimeError('Explicit smoke environment required')
    set_seed(2)
    device=torch.device('cuda')
    model_path=ROOT/'models/Qwen3-1.7B'
    model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='eager').to(device)
    model.config.use_cache=False
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    cfg,train,val=block_cache(mode)
    batches,tokens=stage2_batches(tokenizer,mode)
    if run=='R8':
        initialization=efficient_block_ap(model,[p[0] for p in train],[p[0] for p in val],device,cfg)
    else:
        initialization=bitdistiller_clip(model,[p[0] for p in train],device,Path(out)/'clip_ranges.pt')
        linear,_,_,_=bitdistiller_ops()
        model,replaced=linear.convertModelToQuant(model,quant_type='int2-asym',q_group_size=128)
        assert replaced
    # Exercise the Stage 1 -> Stage 2 checkpoint boundary, not just RAM transfer.
    stage1=Path(out)/'stage1'
    save_native(model,tokenizer,stage1,run,{'mode':mode,'initialization':initialization})
    del model
    torch.cuda.empty_cache()
    from export_baseline import restore_native
    model,_=restore_native(stage1,device)
    model.config.use_cache=False
    if run=='R9':
        teacher=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='eager').to(device).eval()
        teacher.requires_grad_(False);teacher.config.use_cache=False
    else: teacher=None
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    stage2=stage2_train(model,teacher,batches,run,device,cfg)
    if teacher is not None: del teacher
    model.eval();model.config.use_cache=True
    save_native(model,tokenizer,out,run,{'initialization':initialization,'stage2':stage2,'token_budget':tokens,
                                      'mode':mode,'seed':2,'independent_initialization':True,
                                      'stage1_strict_reload':True,
                                      'peak_training_memory_bytes':torch.cuda.max_memory_allocated()})
