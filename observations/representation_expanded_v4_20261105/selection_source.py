"""Calibration-aligned Fig2b reconstruction, all analysis choices declared upfront."""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
import sys,json,time,hashlib,inspect
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
sys.path.append(str(ROOT/'observation_20261104_all_layers/envs/data/lib/python3.11/site-packages'))
import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM

DOMAINS=['Pre-training','Code','Math','Science']
N=128
LENGTH=2048
def save(name,value):
    path=OUT/name;tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def select(tok):
    rng=np.random.default_rng(2);rows=[];seen=set();counts=dict.fromkeys(DOMAINS,0)
    # Four random shards per reasoning domain, maximum 32 samples from each.
    paths=sorted((ROOT/'datasets/openthoughts3').rglob('*.parquet'));rng.shuffle(paths)
    for p in paths:
        if all(counts[d]==N for d in DOMAINS[1:]):break
        pf=pq.ParquetFile(p)
        labels=np.array(pf.read(columns=['domain']).column('domain').to_pylist())
        wanted=[]
        for domain in DOMAINS[1:]:
            if counts[domain]>=N:continue
            candidates=np.flatnonzero(labels==domain.lower());rng.shuffle(candidates)
            wanted.extend(candidates[:256].tolist())
        if not wanted:continue
        # One shard at a time, rather than materializing the full corpus.
        table=pf.read(columns=['domain','source','conversations'])
        selected=table.take(wanted).to_pylist();rng.shuffle(selected);del table
        local=dict.fromkeys(DOMAINS,0)
        for r in selected:
            d=r['domain'].title()
            if counts[d]>=N or local[d]>=(64 if d=='Science' else 32):continue
            conversations=r['conversations']
            messages=[dict(role={'human':'user','gpt':'assistant'}[c['from']],content=c['value']) for c in conversations if c['from'] in ['human','gpt']]
            if not any(m['role']=='assistant' for m in messages):continue
            text=tok.apply_chat_template(messages,tokenize=False)
            raw='\n\n'.join(m['content'] for m in messages)
            response='\n\n'.join(m['content'] for m in messages if m['role']=='assistant')
            digest=hashlib.sha256(raw.encode()).hexdigest()
            if digest in seen:continue
            views={}
            for name,content in [('calibration_chat',text),('raw_conversation',raw),('response_only',response)]:
                views[name]=tok(content,add_special_tokens=False,truncation=True,max_length=LENGTH)['input_ids']
            if any(len(v)!=LENGTH for v in views.values()):continue
            seen.add(digest);counts[d]+=1;local[d]+=1
            rows.append(dict(domain=d,source=r['source'],path=str(p),sha256=digest,views=views))
        print('selected',p.name,counts,flush=True)
    assert all(counts[d]==N for d in DOMAINS[1:]),counts
    # Official FineWeb buffering: BOS + document + EOS, discard remainder after cut.
    bos=tok.bos_token or tok.additional_special_tokens[0];eos=tok.eos_token
    buffer='';origins=[]
    for p in sorted((ROOT/'datasets/fineweb-edu').rglob('*.parquet')):
        pf=pq.ParquetFile(p)
        for batch in pf.iter_batches(batch_size=128,columns=['text']):
            for r in batch.to_pylist():
                text=r['text'];buffer+=bos+text+eos;origins.append(hashlib.sha256(text.encode()).hexdigest())
                ids=tok(buffer,add_special_tokens=False,truncation=True,max_length=LENGTH)['input_ids']
                if len(ids)!=LENGTH:continue
                digest=hashlib.sha256(buffer.encode()).hexdigest()
                if digest not in seen:
                    seen.add(digest);counts['Pre-training']+=1
                    rows.append(dict(domain='Pre-training',source='FineWeb-Edu sample10BT',path=str(p),sha256=digest,documents=origins,views={name:ids for name in ['calibration_chat','raw_conversation','response_only']}))
                buffer='';origins=[]
                if counts['Pre-training']==N:break
            if counts['Pre-training']==N:break
        if counts['Pre-training']==N:break
    assert all(v==N for v in counts.values()),counts
    # Domain-balanced sample IDs, no raw text persisted.
    rows.sort(key=lambda r:(DOMAINS.index(r['domain']),r['sha256']))
    manifest='\n'.join(json.dumps(r) for r in rows)+'\n';(OUT/'inputs.jsonl').write_text(manifest)
    save('selection.json',{'counts':counts,'input_sha256':hashlib.sha256(manifest.encode()).hexdigest(),
        'reasoning':'seed2 random shard order and random candidate rows, cap32 per shard/domain, identical eligible groups across views',
        'pretraining':'official sequential buffer/BOS/EOS; not random corpus sample',
        'sampling_limitation':'Not exact HF whole-corpus shuffle; longer-than2048 response eligibility bias; FineWeb source-order bias'})
    return rows

def forward(rows):
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(12*1024**3/torch.cuda.get_device_properties(0).total_memory)
    model=AutoModelForCausalLM.from_pretrained(ROOT/'models/Qwen3-1.7B',torch_dtype=torch.bfloat16,local_files_only=True,attn_implementation='sdpa',device_map={'':'cuda:0'})
    model.eval().requires_grad_(False)
    assert model.config.model_type=='qwen3' and len(model.model.layers)==28
    captured={}
    layer=model.model.layers[13]
    def pre(m,inputs,kwargs):captured['block_in']=(inputs[0] if inputs else kwargs['hidden_states']).detach()
    def post(key):
        def hook(m,i,o):captured[key]=(o[0] if isinstance(o,tuple) else o).detach()
        return hook
    handles=[layer.register_forward_pre_hook(pre,with_kwargs=True),layer.register_forward_hook(post('block_out')),layer.post_attention_layernorm.register_forward_hook(post('mlp_in'))]
    with torch.inference_mode():
        ids=torch.tensor([rows[0]['views']['calibration_chat'][:128]],device='cuda')
        full=model.model(input_ids=ids,use_cache=False,output_hidden_states=True)
        assert torch.equal(captured['block_in'],full.hidden_states[13])
        assert torch.equal(captured['block_out'],full.hidden_states[14])
        reference={k:v.clone() for k,v in captured.items()}
        # Execute only through block14; causal prefix blocks are unchanged.
        model.model.layers=torch.nn.ModuleList(list(model.model.layers[:14]))
        model.model(input_ids=ids,use_cache=False)
        errors={k:float((v-captured[k]).abs().max()) for k,v in reference.items()}
        assert max(errors.values())==0,errors
        save('hook_validation.json',{'status':'PASS','block_one_based':14,'input_matches_hidden_states_index':13,'output_matches_hidden_states_index':14,'full28_vs_prefix14_max_abs':errors,'probe_tokens':128})
        del full,reference,ids
    stats={'peak_allocated_bytes':0};metadata=[]
    for view in ['calibration_chat','raw_conversation','response_only']:
        cache=OUT/(view+'_features.npz')
        if cache.exists():continue
        arrays={f'{h}_{pool}':[] for h in ['block_in','block_out','mlp_in'] for pool in ['mean','token','last']}
        started_view=time.monotonic()
        for i,row in enumerate(rows):
            start=time.monotonic();ids=torch.tensor([row['views'][view]],device='cuda')
            with torch.inference_mode():model.model(input_ids=ids,use_cache=False)
            # Identical seeded positions per source/view; exclude first32 boundary tokens.
            positions=np.random.default_rng(int(row['sha256'][:16],16)).choice(np.arange(32,LENGTH),size=16,replace=False)
            for key,h in captured.items():
                value=h[0].float()
                arrays[key+'_mean'].append(value.mean(0).cpu().numpy())
                arrays[key+'_last'].append(value[-1].cpu().numpy())
                arrays[key+'_token'].append(value[torch.as_tensor(positions,device='cuda')].cpu().numpy())
            captured.clear();del ids,value,h
            torch.cuda.synchronize();time.sleep(max(.02,time.monotonic()-start))
            if (i+1)%32==0:print('forward',view,i+1,len(rows),flush=True)
        np.savez_compressed(cache,**{k:np.stack(v) for k,v in arrays.items()})
        stats[view]={'samples':len(rows),'elapsed_s':time.monotonic()-started_view}
        stats['peak_allocated_bytes']=torch.cuda.max_memory_allocated();save('forward.json',stats)
    for h in handles:h.remove()
    del model;torch.cuda.empty_cache()

def analyze(rows):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE,trustworthiness
    from sklearn.metrics import silhouette_score
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=np.array([r['domain'] for r in rows]);colors={'Pre-training':'#e2cf18','Code':'#6a3495','Math':'#425fa3','Science':'#2495a7'}
    plots=OUT/'plots';plots.mkdir(exist_ok=True);stats=[]
    for view in ['calibration_chat','raw_conversation','response_only']:
        with np.load(OUT/(view+'_features.npz')) as saved:
            for key in saved.files:
                x=saved[key];token=x.ndim==3
                x=x.reshape(-1,x.shape[-1]);y=np.repeat(labels,16) if token else labels
                for norm in ['none','l2']:
                    z=x if norm=='none' else x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
                    dispersion={d:float(np.mean(np.sum((z[y==d]-z[y==d].mean(0))**2,axis=1))) for d in DOMAINS}
                    stats.append(dict(view=view,feature=key,norm=norm,points=len(z),dispersion=dispersion,
                        reasoning_to_pretraining_dispersion={d:dispersion[d]/dispersion['Pre-training'] for d in DOMAINS[1:]},
                        silhouette=float(silhouette_score(z,y,sample_size=min(len(z),2048),random_state=2))))
                    save('statistics.json',stats)
                    # Every location/pooling has a panel in the fixed overview, no posthoc winner.
                    # Main token outputs get three seeds; other controls a fixed seed2.
                    seeds=[2,3,42] if view=='calibration_chat' and key=='block_out_token' and norm=='none' else [2]
                    pca=PCA(n_components=50,random_state=2).fit_transform(z)
                    for seed in seeds:
                        tag=f'{view}_{key}_{norm}_seed{seed}'
                        if (plots/(tag+'.png')).exists():continue
                        xy=TSNE(perplexity=30,init='pca',learning_rate='auto',random_state=seed,max_iter=1500,n_jobs=2).fit_transform(pca)
                        np.savez_compressed(plots/(tag+'.npz'),xy=xy,labels=y)
                        fig,ax=plt.subplots(figsize=(5,5))
                        for d in DOMAINS:
                            mask=y==d;ax.scatter(xy[mask,0],xy[mask,1],s=5 if token else 10,alpha=.45,c=colors[d],label=d,linewidths=0)
                        ax.set(xticks=[],yticks=[],title='Qwen3-1.7B BF16 | Block 14');ax.legend(loc='upper center',bbox_to_anchor=(.5,1.02),ncol=2,frameon=False,fontsize=8)
                        fig.text(.5,.01,f'{view} | {key} | {norm} | seed{seed}\n2048 tokens/input; reconstruction, not author script',ha='center',fontsize=7)
                        fig.tight_layout(rect=[0,.06,1,.96]);fig.savefig(plots/(tag+'.png'),dpi=180);plt.close(fig)
                        print('plot',tag,flush=True)
    save('completed.json',{'status':'DONE','main':'plots/calibration_chat_block_out_token_none_seed2.png','panels':len(list(plots.glob('*.png'))),'caution':'Calibration aligned, author exact plotting protocol unknown. No similarity-based panel selection.'})

def main():
    protocol={'model':'Qwen3-1.7B BF16','block':14,'samples_per_domain':N,'tokens_per_input':LENGTH,
        'views':['calibration_chat','raw_conversation','response_only'],'hooks':['block_in','block_out','mlp_in'],
        'pooling':['token16_after_first32','mean','last'],'normalization':['none','l2'],
        'pca':50,'perplexity':30,'iterations':1500,'main':'calibration_chat_block_out_token_none_seed2',
        'main_seeds':[2,3,42],'controls_seed':2,
        'official_source':'https://raw.githubusercontent.com/yasu0001/ReasoningQAT/main/datautils_block.py',
        'script_sha256':sha(Path(__file__)),'limitation':'Fig2b exact inputs and hook/pooling not released; calibration match is a hypothesis, not proof of original figure protocol.'}
    if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())==protocol
    else:save('protocol.json',protocol)
    tok=AutoTokenizer.from_pretrained(ROOT/'models/Qwen3-1.7B',local_files_only=True)
    if (OUT/'inputs.jsonl').exists():rows=[json.loads(s) for s in (OUT/'inputs.jsonl').read_text().splitlines()]
    else:rows=select(tok)
    forward(rows);analyze(rows)
if __name__=='__main__':main()
