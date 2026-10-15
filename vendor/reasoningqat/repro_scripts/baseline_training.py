"""Independent Qwen3 training adapters for pinned EfficientQAT and BitDistiller."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
# Stage2 uses the migration collator; no TRL import required.
from baseline_ops import efficient_ops, portable_packed_class, bitdistiller_ops, set_module

ROOT=Path(__file__).resolve().parents[4]
REPO=Path(__file__).resolve().parents[1]
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

class DiskSamples:
    """Bound activation memory independently of the formal calibration count."""
    def __init__(self,values=()):
        cache=ROOT/'hf_cache/baseline_activations'
        cache.mkdir(parents=True,exist_ok=True)
        self.directory=tempfile.TemporaryDirectory(prefix='block-',dir=cache)
        self.count=0
        for value in values: self.append(value)
    def append(self,value):
        torch.save(value,Path(self.directory.name)/f'{self.count}.pt');self.count+=1
    def __len__(self): return self.count
    def __getitem__(self,index):
        if isinstance(index,slice): return (self[i] for i in range(*index.indices(self.count)))
        if index<0 or index>=self.count: raise IndexError(index)
        return torch.load(Path(self.directory.name)/f'{index}.pt',map_location='cpu',weights_only=False)

class CaptureDone(Exception):
    pass

def capture_inputs(model,sequences,device):
    """Capture each sample's Qwen3 rotary embeddings and mask explicitly."""
    original=model.model.layers[0]
    captured=DiskSamples()
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
    collator=DataCollatorForChatML(tokenizer=tokenizer,max_length=cfg['e2e_max_length'])
    class Batches:
        def __len__(self): return math.ceil(len(ds)/cfg['e2e_micro_batch'])
        def __getitem__(self,index):
            if index<0 or index>=len(self): raise IndexError(index)
            examples=[]
            start=index*cfg['e2e_micro_batch']
            for sample in ds.select(range(start,min(start+cfg['e2e_micro_batch'],len(ds)))):
                messages=[]
                for entry in sample['conversations']:
                    role={'human':'user','gpt':'assistant'}[entry['from']]
                    text=entry['value']
                    if mode=='smoke':
                        limit=32 if role=='user' else 96
                        text=tokenizer.decode(tokenizer.encode(text,add_special_tokens=False)[:limit],skip_special_tokens=True)
                    messages.append({'role':role,'content':text})
                examples.append({'messages':messages})
            return {k:v for k,v in collator(examples).items() if k in ('input_ids','attention_mask','labels')}
    batches=Batches()
    evidence={'samples':len(ds),'micro_batches':len(batches),'max_length':cfg['e2e_max_length'],
              'lazy_tokenization':True,'dataset_fingerprint':ds._fingerprint,
              'trajectory':'Fixed dataset assistant trajectory; no regeneration'}
    if mode=='smoke':
        digest=hashlib.sha256();masked=causal=0
        for batch in batches:
            masked+=int((batch['labels']!=-100).sum());causal+=int((batch['labels'][:,1:]!=-100).sum())
            for key in ('input_ids','attention_mask','labels'): digest.update(batch[key].numpy().tobytes())
        evidence.update(tokens_after_mask=masked,causal_target_tokens=causal,tensor_sha256=digest.hexdigest())
        assert masked>0
    return batches,evidence

def efficient_block_ap(model,sequences,val_sequences,device,cfg):
    """Port original one-block MSE reconstruction with independent initialization."""
    Fake,_,_=efficient_ops()
    Packed=portable_packed_class()
    fp=capture_inputs(model,sequences+val_sequences,device)
    quant=DiskSamples((hidden.clone(),kwargs) for hidden,kwargs in fp)
    history=[]
    model.requires_grad_(False)
    for index,original in enumerate(model.model.layers):
        with torch.no_grad(),autocast(device):
            targets=DiskSamples(layer_output(original,hidden.to(device),to_device_kwargs(kwargs,device)).float().cpu()
                     for hidden,kwargs in fp)
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
        batch_size=cfg.get('block_batch_size',1)
        total=cfg['block_epochs']*math.ceil(len(sequences)/batch_size)
        schedulers=[torch.optim.lr_scheduler.CosineAnnealingLR(
            torch.optim.AdamW([torch.tensor(0.)],lr=lr),T_max=total,eta_min=lr/20)
            for lr in (cfg['block_weight_lr'],cfg['block_quant_lr'])]
        for epoch in range(cfg['block_epochs']):
            losses=[]
            for start in range(0,len(sequences),batch_size):
                end=min(start+batch_size,len(sequences))
                optimizer.zero_grad(set_to_none=True)
                for sample in range(start,end):
                    hidden,kwargs=quant[sample]
                    with autocast(device):
                        output=layer_output(layer,hidden.to(device),to_device_kwargs(kwargs,device))
                        loss=torch.nn.functional.mse_loss(output.float(),targets[sample].to(device))
                    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite Block-AP loss')
                    (loss/(end-start)).backward();losses.append(float(loss.detach()))
                grads=[p.grad for p in weight_params+quant_params if p.grad is not None]
                if not grads or not all(torch.isfinite(g).all() for g in grads): raise FloatingPointError('Invalid Block-AP gradients')
                norm=float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.float()) for g in grads])))
                optimizer.step()
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
                quant=DiskSamples((layer_output(layer,hidden.to(device),to_device_kwargs(kwargs,device)).detach().cpu(),kwargs)
                       for hidden,kwargs in quant)
        fp=DiskSamples((targets[i].to(quant[i][0].dtype),kwargs) for i,(_,kwargs) in enumerate(fp))
        model.model.layers[index]=layer
        print(json.dumps(history[-1]),flush=True)
        del original
    return history

def bitdistiller_clip(model,sequences,device,range_output=None):
    _,_,_,clip=bitdistiller_ops()
    sample_tokens=8 if os.environ.get('RQ_RUN_MODE')=='smoke' else 512
    stride=max(1,sum(tokens.numel() for tokens in sequences)//sample_tokens)
    captured=capture_inputs(model,sequences,device)
    history=[];ranges=[]
    for index,layer in enumerate(model.model.layers):
        features={};handles=[];offsets={}
        for name,module in layer.named_modules():
            if isinstance(module,nn.Linear) and not any(key in name for key in ('q_','k_','query','key','Wqkv')):
                features[name]=[];offsets[name]=0
                def hook(module,inputs,output,name=name):
                    rows=inputs[0].detach().reshape(-1,inputs[0].shape[-1])
                    first=(-offsets[name])%stride
                    features[name].append(rows[first::stride].cpu())
                    offsets[name]+=len(rows)
                handles.append(module.register_forward_hook(hook))
        try:
            with torch.no_grad():
                next_inputs=DiskSamples((layer_output(layer,hidden.to(device),to_device_kwargs(kwargs,device)).detach().cpu(),kwargs)
                             for hidden,kwargs in captured)
        finally:
            for handle in handles: handle.remove()
        # Upstream propagates the unmodified block output before applying clips.
        for name,values in features.items():
            module=layer.get_submodule(name)
            sampled=torch.cat(values,dim=0)
            # Already selected exactly the upstream global stride positions.
            upper,lower=clip(module.weight,sampled.to(device),2,
                             {'q_group_size':128,'quant_type':'int'},
                             n_sample_token=len(sampled))
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

def stage2_train(model,teacher,batches,run,device,cfg,checkpoint=None):
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
        confidence_batches=cfg.get('confidence_batches',10)
        if confidence_batches<1: raise ValueError('Confidence budget must be positive')
        generator=torch.Generator().manual_seed(cfg.get('seed',2))
        selected=torch.randperm(len(batches),generator=generator).tolist()[:confidence_batches]
        batch_confidences=[]
        with torch.no_grad():
            for index in selected:
                batch=batches[index]
                inputs={k:v.to(device) for k,v in batch.items() if k!='labels'}
                logits=teacher(**inputs).logits.float()
                # Upstream averages maximum probabilities over prompt and padding.
                confidence=logits.softmax(-1).amax(-1)
                batch_confidences.append(float(confidence.mean()))
                del logits,confidence,inputs
        if not batch_confidences: raise ValueError('Empty confidence sample')
        # Divide by the actual count, avoiding the upstream 11/10 loop defect.
        beta=sum(batch_confidences)/len(batch_confidences)
        assert 0<=beta<=1
    parameters=[p for p in model.parameters() if p.requires_grad]
    probes={name:p.detach().reshape(-1)[:64].float().cpu().clone()
            for name,p in model.named_parameters() if p.requires_grad}
    settings=cfg.get('stage2_optimizer',{})
    optimizer=torch.optim.AdamW(parameters,lr=cfg['e2e_learning_rate'],
        betas=tuple(settings.get('betas',[0.9,0.999])),eps=settings.get('eps',1e-8),
        weight_decay=settings.get('weight_decay',0.))
    accumulation=cfg.get('e2e_accumulation',1)
    epochs=cfg.get('e2e_epochs',2)
    if accumulation<1 or epochs<1 or not len(batches): raise ValueError('Invalid training budget')
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs*math.ceil(len(batches)/accumulation))
    model.train()
    step=0;token_count=0;resume_epoch=0;resume_start=0
    signature={'run':run,'config':cfg,'batches':len(batches),
               'trainer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if checkpoint is not None and Path(checkpoint).exists():
        state=torch.load(checkpoint,map_location=device,weights_only=False)
        if state['signature']!=signature: raise ValueError('Resume configuration changed')
        model.load_state_dict(state['model'],strict=True)
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        step=state['step'];token_count=state['token_count'];losses=state['losses']
        resume_epoch=state['epoch'];resume_start=state['next_batch']
        torch.set_rng_state(state['rng'].cpu())
        if device.type=='cuda': torch.cuda.set_rng_state(state['cuda_rng'].cpu(),device)
        del state
    for epoch in range(resume_epoch,epochs):
        for start in range(resume_start if epoch==resume_epoch else 0,len(batches),accumulation):
            end=min(start+accumulation,len(batches))
            optimizer.zero_grad(set_to_none=True)
            window_loss=0.;window_tokens=0
            for index in range(start,end):
                batch={k:v.to(device) for k,v in batches[index].items()}
                count=int((batch['labels'][:,1:]!=-100).sum())
                if count==0: raise ValueError('Batch has no causal targets after truncation')
                window_tokens+=count
                with autocast(device):
                    if run=='R8': loss=model(**batch).loss.float()
                    else:
                        with torch.no_grad(): teacher_logits=teacher(input_ids=batch['input_ids'],attention_mask=batch['attention_mask']).logits.float()
                        student_logits=model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask']).logits.float()
                        loss=cakld(None,batch['labels'],student_logits,teacher_logits,beta)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite baseline loss')
                (loss/(end-start)).backward()
                window_loss+=float(loss.detach())/(end-start)
                del loss,batch
                if run=='R9': del teacher_logits,student_logits
            grads=[p.grad for p in parameters if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads): raise FloatingPointError('Invalid accumulated gradients')
            norm=torch.nn.utils.clip_grad_norm_(parameters,settings.get('max_grad_norm',1.))
            if not torch.isfinite(norm): raise FloatingPointError('Nonfinite gradient norm')
            optimizer.step();scheduler.step();token_count+=window_tokens
            losses.append({'step':step,'epoch':epoch,'micro_batches':end-start,'loss':window_loss,
                           'grad_norm':float(norm),'causal_target_tokens':window_tokens})
            print(json.dumps(losses[-1]),flush=True);step+=1
            if checkpoint is not None and (step%64==0 or end==len(batches)):
                checkpoint=Path(checkpoint);checkpoint.parent.mkdir(parents=True,exist_ok=True)
                temporary=checkpoint.with_suffix('.tmp')
                torch.save({'signature':signature,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                            'scheduler':scheduler.state_dict(),'epoch':epoch if end<len(batches) else epoch+1,
                            'next_batch':end if end<len(batches) else 0,'step':step,'token_count':token_count,
                            'losses':losses,'rng':torch.get_rng_state(),
                            'cuda_rng':torch.cuda.get_rng_state(device) if device.type=='cuda' else None},temporary)
                temporary.replace(checkpoint)
    changed=sum(int((p.detach().reshape(-1)[:64].float().cpu()!=probes[name]).sum())
                for name,p in model.named_parameters() if p.requires_grad)
    if changed==0: raise RuntimeError('Baseline optimizer made no observed parameter updates')
    return {'objective':'CE' if run=='R8' else 'Original CAKLD, unshifted label mask, sequence sum/batch mean',
            'trainable_names':names,'steps':losses,'confidence_beta':beta,
            'epochs':epochs,'gradient_accumulation_steps':accumulation,'optimizer_steps':step,
            'causal_target_tokens_processed':token_count,
            'confidence_estimator':'Mean of batch maximum teacher probabilities; prompts/padding included; actual-count normalization',
            'confidence_selected_batches':selected if run=='R9' else None,
            'confidence_batch_means':batch_confidences if run=='R9' else None,
            'optimizer':'AdamW','learning_rate':cfg['e2e_learning_rate'],'scheduler':'cosine',
            'optimizer_settings':{'betas':list(optimizer.defaults['betas']),'eps':optimizer.defaults['eps'],'weight_decay':optimizer.defaults['weight_decay'],'warmup_steps':0},
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
    if os.environ.get('RQ_RUN_MODE')!=mode: raise RuntimeError('Explicit matching environment required')
    if mode=='formal':
        from pipeline import check_protocol
        check_protocol(mode)
        baseline=json.loads((REPO/'repro_configs/qat_baselines.json').read_text())[run]
        if baseline['blockers']: raise RuntimeError('Unresolved baseline settings: '+ '; '.join(baseline['blockers']))
    set_seed(2)
    device=torch.device('cuda')
    model_path=ROOT/'models/Qwen3-1.7B'
    model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='eager').to(device)
    model.config.use_cache=False
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    cfg,train,val=block_cache(mode)
    contract=json.loads((REPO/'repro_configs/qat_baselines.json').read_text())
    cfg['e2e_epochs']=1 if mode=='smoke' else contract['shared_contract']['stage2_epochs']
    cfg['stage2_optimizer']=contract[run]['stage2_optimizer']
    cfg['confidence_batches']=contract[run].get('confidence_batches',10)
    cfg['seed']=2
    if mode=='formal' and (contract[run]['stage1_samples']!=len(train) or contract[run]['stage1_sequence_length']!=cfg['block_sequence_length']):
        raise ValueError('Baseline stage 1 calibration budget differs from frozen configuration')
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
    checkpoint=Path(out)/'training_resume.pt' if mode=='formal' else None
    stage2=stage2_train(model,teacher,batches,run,device,cfg,checkpoint)
    if teacher is not None: del teacher
    model.eval();model.config.use_cache=True
    save_native(model,tokenizer,out,run,{'initialization':initialization,'stage2':stage2,'token_budget':tokens,
                                      'mode':mode,'seed':2,'independent_initialization':True,
                                      'stage1_strict_reload':True,
                                      'peak_training_memory_bytes':torch.cuda.max_memory_allocated()})
    if checkpoint is not None: checkpoint.unlink(missing_ok=True)
