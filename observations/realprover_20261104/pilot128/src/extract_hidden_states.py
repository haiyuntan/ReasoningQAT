"""Bounded observation forward, single model, no gradients or parameter writes.

Called only by resource_guard. Checkpoint/data identities are prerequisites.
"""
import argparse
import ast
import hashlib
import json
import os
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import OBS, ROOT, read, atomic, sha
from token_metrics import distribution

def paced(start):
    torch.cuda.synchronize()
    elapsed=time.monotonic()-start
    duty=read(OBS/'configs/observation_protocol.json')['resource_policy']['duty_cycle']
    time.sleep(max(.05,elapsed*(1/duty-1))) # software pacing, not CUDA hard QoS

def load_model(path):
    assert torch.cuda.device_count()==1
    torch.set_num_threads(2)
    total=torch.cuda.get_device_properties(0).total_memory
    cap=read(OBS/'configs/observation_protocol.json')['resource_policy']['process_allocator_limit_gib']
    torch.cuda.set_per_process_memory_fraction(cap*1024**3/total,0)
    model=AutoModelForCausalLM.from_pretrained(str(path),torch_dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True,device_map={'':'cuda:0'})
    assert model.config.model_type=='qwen2' and len(model.model.layers)==28
    model.eval();model.requires_grad_(False)
    model.config.use_cache=False
    return model

def parity(model, tokenizer, method):
    """Compare with SHA-verified native probe recorded before original export.

    No native package/environment mutation. Reuses frozen native logits from the
    exact 16 preregistered prefixes, not a dense-export self comparison.
    """
    if method=='P0':return {'status':'BF16_REFERENCE'}
    native=ROOT/'runs/realprover_formal_four_gpu_v12/formal'/method/'train/native_probe.pt'
    if not native.exists():return {'status':'BLOCKED','reason':'No independent native logits for this method'}
    identities=read(OBS/'manifests/checkpoint_identities.json')[method]
    prepared=read(ROOT/'runs/realprover_formal_four_gpu_v12/formal'/method/'prepared.json')
    expected=next(e['sha256'] for e in prepared['evidence'] if e['path']==str(native))
    assert sha(native)==expected
    reference=torch.load(native,map_location='cpu',weights_only=True)
    source=OBS/'upstream/User-ReasoningQAT/adapters/quantization.py'
    tree=ast.parse(source.read_text())
    prompts=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PROMPTS' for t in n.targets))
    assert len(prompts)==len(reference['logits'])==16
    maximum=0.;absolute=0.;count=0;top1=0
    with torch.inference_mode():
        for prompt,expected_logits in zip(prompts,reference['logits']):
            start=time.monotonic()
            inputs=tokenizer(prompt,return_tensors='pt').to('cuda')
            actual=model(**inputs).logits[:,-1].float().cpu()
            delta=(actual-expected_logits.float()).abs()
            maximum=max(maximum,float(delta.max()));absolute+=float(delta.sum());count+=delta.numel()
            top1+=int(torch.equal(actual.argmax(-1),expected_logits.argmax(-1)))
            paced(start)
    config=read(OBS/'configs/observation_protocol.json')['parity']
    result={'status':'PASS' if maximum<=config['max_abs_tolerance'] and absolute/count<=config['mean_abs_tolerance'] and top1/16>=config['top1_agreement_min'] else 'BLOCKED','prefixes':16,'native_probe_sha256':expected,'max_abs':maximum,'mean_abs':absolute/count,'top1_agreement':top1/16,'limitation':'Frozen native reference from original export; independent runtime dense validation, not a new native forward in this environment.'}
    atomic(OBS/'metrics'/(method+'_parity.json'),result)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',required=True,choices=['P0','P1','P2','P3','P4','P5','P6'])
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--kind',choices=['domains','entropy'],default='domains')
    args=parser.parse_args()
    identity=read(OBS/'manifests/checkpoint_identities.json')[args.method]['inventory']
    assert identity['ready']
    started=time.monotonic()
    tokenizer=AutoTokenizer.from_pretrained(identity['source_path'],local_files_only=True)
    model=load_model(identity['source_path'])
    if args.smoke:
        # 16-token first smoke; no long sequence or large batch trial on retrieval GPU.
        for length in [16,64,256,512,1024,2048]:
            start=time.monotonic()
            ids=tokenizer('n : ℕ\n⊢ n = n\n',return_tensors='pt')['input_ids'].repeat(1,300)[:,:length].cuda()
            with torch.inference_mode():out=model.model(input_ids=ids,use_cache=False)
            assert torch.isfinite(out.last_hidden_state).all()
            paced(start)
        atomic(OBS/'metrics/gpu_smoke.json',{'status':'PASS','lengths':[16,64,256,512,1024,2048],'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'elapsed_s':time.monotonic()-started,'model':identity,'cuda':torch.version.cuda,'torch':torch.__version__})
        return
    result=parity(model,tokenizer,args.method)
    if result['status'] not in ['PASS','BF16_REFERENCE']:
        raise RuntimeError(f'Native parity not passed: {result}')
    input_path=OBS/'manifests'/('domain_inputs.jsonl' if args.kind=='domains' else 'entropy_inputs.jsonl')
    samples=[json.loads(line) for line in input_path.read_text().splitlines()]
    assert all(0<len(s['input_ids'])<=(256 if args.kind=='domains' else 2048) for s in samples)
    output=OBS/'metrics/raw_summaries'/args.kind/args.method
    output.mkdir(parents=True,exist_ok=True)
    fingerprints={'protocol_sha256':sha(OBS/'configs/observation_protocol.json'),'inputs_sha256':sha(input_path),'model_evidence_sha256':identity['evidence_sha256']}
    hook_values={}
    def capture(layer):
        def hook(module,inputs,outputs):hook_values[layer]=outputs[0].detach()
        return hook
    handles=[model.model.layers[k-1].register_forward_hook(capture(k)) for k in [1,7,14,21,28]]
    teacher_dir=OBS/'metrics/raw_summaries/entropy/P0'
    try:
        for index,sample in enumerate(samples):
            target=output/(sample['sample_id']+'.npz')
            meta=target.with_suffix('.json')
            if target.exists() and meta.exists() and read(meta).get('fingerprints')==fingerprints:continue
            start=time.monotonic()
            ids=torch.tensor([sample['input_ids']],device='cuda')
            arrays={}
            with torch.inference_mode():
                hidden=model.model(input_ids=ids,use_cache=False).last_hidden_state
                if args.kind=='domains':
                    for layer,value in hook_values.items():
                        arrays['mean_'+str(layer)]=value[0].float().mean(0).cpu().numpy()
                        arrays['last_'+str(layer)]=value[0,-1].float().cpu().numpy()
                    # Same raw token window, no chat wrapper: template control.
                    raw=torch.tensor([sample['raw_input_ids']],device='cuda')
                    raw_hidden=model.model(input_ids=raw,use_cache=False).last_hidden_state
                    for layer,value in hook_values.items():
                        arrays['raw_mean_'+str(layer)]=value[0].float().mean(0).cpu().numpy()
                    del raw_hidden
                else:
                    positions=sample['prediction_positions']
                    gold=torch.tensor(sample['gold_ids'],device='cuda')
                    teacher=None
                    if args.method!='P0':
                        teacher=dict(np.load(teacher_dir/(sample['sample_id']+'.npz')))
                    collected={}
                    for offset in range(0,len(positions),16):
                        end=min(offset+16,len(positions))
                        logits=model.lm_head(hidden[0,positions[offset:end]])
                        reference=None if teacher is None else {k:torch.from_numpy(v[offset:end]) for k,v in teacher.items() if k in ['top1','gold_logp','top20_logp','top20_ids']}
                        stats=distribution(logits,gold[offset:end],reference)
                        for key,value in stats.items():collected.setdefault(key,[]).append(value.cpu().numpy())
                    arrays={key:np.concatenate(values) for key,values in collected.items()}
            assert all(np.isfinite(a).all() for a in arrays.values())
            tmp=target.with_suffix('.partial')
            with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
            tmp.replace(target)
            atomic(meta,{'sample_id':sample['sample_id'],'fingerprints':fingerprints,'method':args.method,'peak_allocated':torch.cuda.max_memory_allocated(),'format':identity['checkpoint_type']})
            hook_values.clear();del hidden
            paced(start)
            if index%16==0:print(args.method,args.kind,index+1,'/',len(samples),flush=True)
    finally:
        for handle in handles:handle.remove()
    atomic(output/'completed.json',{'status':'DONE','samples':len(samples),'fingerprints':fingerprints,'peak_allocated':torch.cuda.max_memory_allocated(),'elapsed_s':time.monotonic()-started})

if __name__=='__main__':main()
