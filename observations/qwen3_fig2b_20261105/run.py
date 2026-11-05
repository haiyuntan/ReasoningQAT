"""Fig.2b reconstruction, not author's unreleased exact plotting protocol."""
import os
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
import json, hashlib, time, sys
from pathlib import Path
# Read-only reuse of the existing observation data packages (same NumPy 1.26.4).
# Append, so the forward environment's Torch/Transformers retain precedence.
sys.path.append(str(Path(__file__).resolve().parent.parent/'observation_20261104_all_layers/envs/data/lib/python3.11/site-packages'))
import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
DOMAINS=['Pre-training','Code','Math','Science']
def save(name, value):
    (OUT/name).write_text(json.dumps(value,indent=2))
def main():
    torch.set_num_threads(2)
    tok=AutoTokenizer.from_pretrained(ROOT/'models/Qwen3-1.7B',local_files_only=True)
    rows=[]; seen=set(); counts=dict.fromkeys(DOMAINS,0)
    def add(domain,text,origin):
        if counts[domain]>=256:return
        digest=hashlib.sha256(text.encode()).hexdigest()
        if digest in seen:return
        ids=tok.encode(text,add_special_tokens=False,truncation=True,max_length=512)
        if len(ids)<128:return
        seen.add(digest);counts[domain]+=1
        rows.append(dict(domain=domain,sha256=digest,origin=origin,ids=ids))
    for p in sorted((ROOT/'datasets/fineweb-edu').rglob('*.parquet')):
        for b in pq.ParquetFile(p).iter_batches(batch_size=128,columns=['text']):
            for r in b.to_pylist():add('Pre-training',r['text'],str(p))
            if counts['Pre-training']==256:break
        if counts['Pre-training']==256:break
    for p in sorted((ROOT/'datasets/openthoughts3').rglob('*.parquet')):
        pf=pq.ParquetFile(p)
        # Domain column is small; skip shards that cannot fill missing domains.
        available={str(x).lower() for x in pf.read(columns=['domain']).column('domain').to_pylist()}
        if not any(d.lower() in available and counts[d]<256 for d in DOMAINS[1:]):continue
        for b in pf.iter_batches(batch_size=128,columns=['domain','conversations']):
            for r in b.to_pylist():
                d=str(r['domain']).title()
                if d in counts:
                    text=next((c['value'] for c in r['conversations'] if c['from']=='human'),'')
                    add(d,text,str(p))
            if all(counts[d]==256 for d in DOMAINS[1:]):break
        print('selection',p.name,counts,flush=True)
        if all(counts[d]==256 for d in DOMAINS[1:]):break
    assert all(n==256 for n in counts.values()), counts
    manifest='\n'.join(json.dumps(r) for r in rows)+'\n'
    (OUT/'inputs.jsonl').write_text(manifest)
    save('protocol.json',dict(model='Qwen3-1.7B BF16',block_one_based=14,
        hook='decoder block output before final model norm',counts=counts,
        input_sha256=hashlib.sha256(manifest.encode()).hexdigest(),
        selection='first eligible unique human prompt, deterministic shard order; not random corpus sample',
        contexts=[128,512],pooling=['mean','last','token'],normalizations=['none','l2'],
        seeds=[2,3,42],perplexity=30,pca=50,iterations=1500,
        caveat='Author pooling, sampling and t-SNE settings unavailable; sensitivity study, not exact numerical reproduction. Prompt-only assumption; no chat wrapper or generated answers.'))
    model_path=ROOT/'models/Qwen3-1.7B'
    save('model_identity.json',{'config':json.loads((model_path/'config.json').read_text()),
        'files':[dict(name=p.name,bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in sorted(model_path.iterdir()) if p.is_file()]})
    torch.cuda.set_per_process_memory_fraction(12*1024**3/torch.cuda.get_device_properties(0).total_memory)
    model=AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,
        local_files_only=True,attn_implementation='sdpa',device_map={'':'cuda:0'})
    model.eval().requires_grad_(False)
    assert model.config.model_type=='qwen3' and len(model.model.layers)==28
    captured={}
    def hook(m,i,o):captured['h']=(o[0] if isinstance(o,tuple) else o).detach()
    handle=model.model.layers[13].register_forward_hook(hook)
    features={}
    try:
        for length in [128,512]:
            means=[];last=[];tokens=[]
            for i,r in enumerate(rows):
                started=time.monotonic()
                ids=torch.tensor([r['ids'][:length]],device='cuda')
                with torch.inference_mode():model.model(input_ids=ids,use_cache=False)
                h=captured.pop('h')[0].float()
                means.append(h.mean(0).cpu().numpy());last.append(h[-1].cpu().numpy())
                # Eight evenly spaced tokens per input; sensitivity, not independent samples.
                positions=torch.linspace(0,len(h)-1,8,device='cuda').long()
                tokens.append(h[positions].cpu().numpy())
                del h,ids
                torch.cuda.synchronize()
                time.sleep(max(.02,time.monotonic()-started))  # 50% duty cycle
                if (i+1)%64==0:print('forward',length,i+1,len(rows),flush=True)
            for key,value in [('mean',means),('last',last),('token',tokens)]:
                features[f'{key}_{length}']=np.stack(value)
    finally:handle.remove()
    save('forward.json',{'status':'DONE','samples':len(rows),'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
    np.savez_compressed(OUT/'features.npz',**features)
    del model
    torch.cuda.empty_cache()
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE,trustworthiness
    from sklearn.metrics import silhouette_score
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=np.array([r['domain'] for r in rows]); stats=[]
    # Main: sequence means, 512-token maximum, no normalization, fixed layer14/seed2.
    for key,x in features.items():
        token=x.ndim==3
        x=x.reshape(-1,x.shape[-1]); y=np.repeat(labels,8) if token else labels
        for norm in ['none','l2']:
            z=x if norm=='none' else x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
            # Sequence-level high-dimensional metrics; tokens are correlated within source.
            if not token:
                stats.append(dict(feature=key,norm=norm,silhouette=float(silhouette_score(z,y)),
                    dispersion={d:float(np.mean(np.sum((z[y==d]-z[y==d].mean(0))**2,axis=1))) for d in DOMAINS}))
            pca=PCA(n_components=50,random_state=2).fit_transform(z)
            for seed in ([2] if token else [2,3,42]):
                xy=TSNE(perplexity=30,random_state=seed,init='pca',learning_rate='auto',max_iter=1500).fit_transform(pca)
                tag=f'{key}_{norm}_seed{seed}'
                np.savez_compressed(OUT/(tag+'.npz'),xy=xy,labels=y)
                fig,ax=plt.subplots(figsize=(7,5))
                for d,c in zip(DOMAINS,['#4c78a8','#f58518','#54a24b','#e45756']):
                    mask=y==d;ax.scatter(xy[mask,0],xy[mask,1],s=5 if token else 10,alpha=.55,label=d,c=c)
                ax.set(xticks=[],yticks=[],title='Qwen3-1.7B BF16 | Block 14')
                ax.legend(frameon=False)
                fig.text(.5,.01,f'Reconstruction | {key} | {norm} | seed {seed} | author plotting settings unavailable',ha='center',fontsize=7)
                fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(OUT/(tag+'.png'),dpi=200);plt.close(fig)
                print('plot',tag,flush=True)
            save('statistics.json',stats)
    save('completed.json',{'status':'DONE','main_figure':'mean_512_none_seed2.png','warning':'Do not choose seeds to match paper; compare high-dimensional statistics and all controls.'})
if __name__=='__main__':main()
