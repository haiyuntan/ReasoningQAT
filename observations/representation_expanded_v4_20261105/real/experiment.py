"""REAL-Prover BF16 five-domain, v3 no-PCA representation observation.

Read-only production inputs; all outputs remain in this directory.
"""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
import sys,json,hashlib,time,argparse,gc
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.append(str(ROOT/'observation_20261104_all_layers/envs/data/lib/python3.11/site-packages'))
import numpy as np

DOMAINS=['Pre-training','Code','Math','Science','Formal Proof']
IDS={'Pre-training':5,'Code':0,'Math':1,'Science':2,'Formal Proof':3}
def save(name,value):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.partial');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8*1024**2),b''):h.update(part)
    return h.hexdigest()
def digest(text):return hashlib.sha256(text.encode()).hexdigest()
def raw_state(text):
    marker='Here is the current STATE:\n';ending='\n\n**Please generate a TACTIC in lean4 to solve the state.**'
    if marker in text:
        assert ending in text
        return text.rsplit(marker,1)[1].split(ending,1)[0]
    if '\nSTATE:\n' in text and '\nTACTIC:' in text:return text.rsplit('\nSTATE:\n',1)[1].split('\nTACTIC:',1)[0].rstrip()
    if '⊢' in text and 'FORMAL PROOF' not in text:return text
    raise ValueError('Unknown formal input schema')
def unwrap(text):
    if text.startswith('User: ') and text.rstrip().endswith('Assistant:'):
        return text[len('User: '):].rstrip()[:-len('Assistant:')].rstrip()
    return text
def proof_prompt(text):
    return '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'+unwrap(text)+'<|im_end|>\n<|im_start|>assistant\n'

def prepare():
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/REAL-Prover-v1-bf16',local_files_only=True)
    tokenizer.truncation_side='left'
    frozen=[json.loads(s) for s in (OUT.parent/'qwen/inputs.jsonl').read_text().splitlines()]
    originals={r['sha256']:r for r in frozen if r['domain']!='Pre-training'}
    recovered={}
    for name in sorted({Path(r['path']).name for r in originals.values()}):
        p=ROOT/'datasets/openthoughts3/data'/name
        wanted={k for k,r in originals.items() if Path(r['path']).name==name}
        for batch in pq.ParquetFile(p).iter_batches(batch_size=64,columns=['domain','source','conversations']):
            for r in batch.to_pylist():
                messages=[{'role':{'human':'user','gpt':'assistant'}[c['from']],'content':c['value']} for c in r['conversations'] if c['from'] in ['human','gpt']]
                raw='\n\n'.join(m['content'] for m in messages);key=digest(raw)
                if key in wanted:
                    native=tokenizer.apply_chat_template(messages,tokenize=False)
                    response=next(m['content'] for m in reversed(messages) if m['role']=='assistant')
                    start=native.rfind(response);assert start>=0
                    recovered[key]={'domain':r['domain'].title(),'group':key,'raw':raw,'native':native,'native_spans':[(start,start+len(response))],
                        'source':r['source'],'shard':name}
            if wanted.issubset(recovered):break
        assert wanted.issubset(recovered),name
        print('recovered reasoning',name,len(recovered),flush=True)
    general=[r for r in frozen if r['domain']=='Pre-training']
    wanted_docs={h for r in general for h in r['documents']};documents={}
    for p in sorted((ROOT/'datasets/fineweb-edu').rglob('*.parquet')):
        for batch in pq.ParquetFile(p).iter_batches(batch_size=128,columns=['text']):
            for r in batch.to_pylist():
                key=digest(r['text'])
                if key in wanted_docs:documents[key]=r['text']
            if wanted_docs.issubset(documents):break
        if wanted_docs.issubset(documents):break
    assert wanted_docs.issubset(documents)
    for r in general:
        text='\n\n'.join(documents[key] for key in r['documents'])
        recovered[r['sha256']]={'domain':'Pre-training','group':r['sha256'],'raw':text,'native':text,'native_spans':[(0,len(text))],
            'source':'FineWeb-Edu sample10BT','shard':Path(r['path']).name}
    assert len(recovered)==2048
    # Use the already audited official training corpus, not miniF2F evaluation text.
    acquisition=json.loads((ROOT/'observation_20261102/manifests/acquisition.json').read_text())['dataset']['result']
    parquet=ROOT/'observation_20261104_all_layers/assets/state_tactic_pairs/data/train-00000-of-00001.parquet'
    expected=next(r['sha256'] for r in acquisition['files'] if Path(r['path']).name==parquet.name)
    assert sha(parquet)==expected
    leakage_path=ROOT/'observation_20261104_all_layers/manifests/leakage_report.json'
    leakage=json.loads(leakage_path.read_text());assert not leakage['matched_rows']
    table=pq.read_table(parquet,columns=['input','tactic']);rng=np.random.default_rng(2)
    order=rng.permutation(table.num_rows);pairs=table.to_pylist();proof=[];seen=set()
    for i in order:
        r=pairs[int(i)];state=raw_state(r['input']);key=digest(state)
        if key in seen:continue
        tactic=r['tactic'];tactic_tokens=tokenizer.encode(tactic,add_special_tokens=False)
        if not 2<=len(tactic_tokens)<=48:continue
        raw=state+'\n'+tactic
        if len(tokenizer.encode(raw,add_special_tokens=False))<128:continue
        prefix=proof_prompt(r['input']);native=prefix+tactic
        pos=native.rfind(state,0,len(prefix));assert pos>=0
        spans=[(pos,pos+len(state)),(len(prefix),len(native))]
        encoded=tokenizer(native,add_special_tokens=False,truncation=True,max_length=2048,return_offsets_mapping=True)
        eligible=[j for j,(a,b) in enumerate(encoded['offset_mapping']) if b>a and any(a>=lo and b<=hi for lo,hi in spans)]
        if len(eligible)<128:continue
        seen.add(key);proof.append({'domain':'Formal Proof','group':key,'raw':raw,'native':native,'native_spans':spans,'source':'FrenzyMath/state_tactic_pairs',
            'shard':parquet.name,'row':int(i),'state_chars':len(state),'tactic_chars':len(tactic),'tactic_tokens':len(tactic_tokens)})
        if len(proof)==512:break
    assert len(proof)==512
    rows=sorted(list(recovered.values())+proof,key=lambda r:(DOMAINS.index(r['domain']),r['group']))
    inputs=[];sampled_proof_roles={'state':0,'tactic':0,'boundary':0};lengths={d:[] for d in DOMAINS}
    for r in rows:
        raw_encoded=tokenizer(r['raw'],add_special_tokens=False,truncation=True,max_length=128,return_offsets_mapping=True)
        assert len(raw_encoded['input_ids'])==128
        native_encoded=tokenizer(r['native'],add_special_tokens=False,truncation=True,max_length=2048,return_offsets_mapping=True)
        eligible=[i for i,(a,b) in enumerate(native_encoded['offset_mapping']) if b>a and any(a>=lo and b<=hi for lo,hi in r['native_spans'])]
        assert len(eligible)>=128,(r['domain'],r['group'])
        rng_positions=np.random.default_rng(int(r['group'][:16],16))
        first=rng_positions.choice(np.arange(16,128),size=4,replace=False)
        relative=np.concatenate([first,rng_positions.choice(np.setdiff1d(np.arange(16,128),first),size=12,replace=False)])
        positions=[eligible[-128:][i] for i in relative]
        roles=[]
        for i in relative:
            a,b=raw_encoded['offset_mapping'][i]
            role='content'
            if r['domain']=='Formal Proof':
                role='state' if b<=r['state_chars'] else 'tactic' if a>=r['state_chars']+1 else 'boundary'
                sampled_proof_roles[role]+=1
            roles.append(role)
        item={'domain':r['domain'],'group':r['group'],'source':r['source'],'shard':r['shard'],'views':{
            'raw128':{'ids':raw_encoded['input_ids'],'positions':relative.tolist()},
            'native_context':{'ids':native_encoded['input_ids'],'positions':positions}},'raw_point_roles':roles}
        if r['domain']=='Formal Proof':item.update(row=r['row'],tactic_tokens=r['tactic_tokens'])
        inputs.append(item);lengths[r['domain']].append(len(native_encoded['input_ids']))
    path=OUT/'inputs.jsonl';path.write_text('\n'.join(json.dumps(r) for r in inputs)+'\n')
    save('selection.json',{'status':'PASS','counts':{d:sum(r['domain']==d for r in inputs) for d in DOMAINS},'input_sha256':sha(path),
        'nonproof_groups':'Same2048 source groups as expanded Qwen; original512 groups retained; retokenized using REAL-Prover tokenizer, no reuse of Qwen3 token IDs',
        'proof_source_revision':acquisition['revision'],'proof_parquet_sha256':expected,'leakage_screen_sha256':sha(leakage_path),
        'leakage_status':'Reused exact-match screen; source-theorem identity/semantic overlap unknown. Training-distribution observation.',
        'proof_selection':'Seed2 shuffled rows, unique raw-state hashes, raw state+newline+tactic >=128 tokens, tactic2..48tokens; longer-state selection bias',
        'raw128':'All domains last128 raw content tokens; no domain-specific instructions or chat wrapper; sample16 uniform positions16..127',
        'native_context':'Native-format context, keep last2048 tokens; sample corresponding16 relative positions from last128 eligible content tokens. Proof restricts to actual current state and gold tactic, excluding instructions/examples/retrieval token positions.',
        'native_context_length':{d:{'min':min(v),'median':float(np.median(v)),'max':max(v)} for d,v in lengths.items()},
        'raw_sampled_proof_token_roles':sampled_proof_roles,'no_test_outcome_or_model_success_used':True})
    save('protocol.json',{'model':'REAL-Prover-v1 BF16 (Qwen2ForCausalLM)','block_one_based':14,'views':['raw128','native_context'],
        'primary':'raw128, joint five-domain no-PCA t-SNE, random init, LR200, perplexity30,1000iterations,seed42',
        'seeds':[42,2,3],'groups_per_domain':512,'tokens_per_group':16,'fit_points':40960,
        'comparison_to_qwen_v3':'Same no-PCA t-SNE/hook method; primary context deliberately128 rather than2048 because raw state/tactic pairs are short. Not an exact Fig2b experiment.',
        'knowledge_limit':'Distributional observation, not proof of possession of a knowledge module or theorem-solving ability.',
        'tokenizer_sha256':sha(ROOT/'models/REAL-Prover-v1-bf16/tokenizer.json'),'input_sha256':sha(path),'script_sha256':sha(Path(__file__))})
    print('PREPARED',len(inputs),'groups',sampled_proof_roles,flush=True)

def forward():
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(32*1024**3/torch.cuda.get_device_properties(0).total_memory)
    model=AutoModelForCausalLM.from_pretrained(ROOT/'models/REAL-Prover-v1-bf16',torch_dtype=torch.bfloat16,local_files_only=True,
        attn_implementation='sdpa',device_map={'':'cuda:0'})
    model.eval().requires_grad_(False);assert model.config.model_type=='qwen2' and len(model.model.layers)==28
    captured={}
    def hook(m,i,o):captured['h']=(o[0] if isinstance(o,tuple) else o).detach()
    handle=model.model.layers[13].register_forward_hook(hook)
    rows=[json.loads(s) for s in (OUT/'inputs.jsonl').read_text().splitlines()]
    with torch.inference_mode():
        ids=torch.tensor([rows[0]['views']['raw128']['ids']],device='cuda')
        full=model.model(input_ids=ids,use_cache=False,output_hidden_states=True)
        assert torch.equal(captured['h'],full.hidden_states[14])
        reference=captured['h'].clone();del full
        model.model.layers=torch.nn.ModuleList(list(model.model.layers[:14]))
        model.model(input_ids=ids,use_cache=False);error=float((reference-captured['h']).abs().max());assert error==0
        save('hook_validation.json',{'status':'PASS','block':14,'hidden_states_index':14,'full28_vs_prefix14_max_abs':error})
        del reference,ids
    statistics={}
    try:
        for view in ['raw128','native_context']:
            arrays=[];means=[];start_view=time.monotonic()
            for i,r in enumerate(rows):
                start=time.monotonic();spec=r['views'][view]
                ids=torch.tensor([spec['ids']],device='cuda')
                with torch.inference_mode():model.model(input_ids=ids,use_cache=False)
                h=captured.pop('h')[0].float();assert torch.isfinite(h).all()
                arrays.append(h[torch.tensor(spec['positions'],device='cuda')].cpu().numpy());means.append(h.mean(0).cpu().numpy())
                del h,ids
                torch.cuda.synchronize();time.sleep(max(.02,time.monotonic()-start))
                if (i+1)%32==0:print('FORWARD',view,i+1,len(rows),flush=True)
            target=OUT/(view+'_features.npz');temporary=target.with_suffix('.partial')
            with temporary.open('wb') as f:np.savez_compressed(f,tokens=np.stack(arrays),means=np.stack(means))
            temporary.replace(target)
            statistics[view]={'groups':len(rows),'elapsed_s':time.monotonic()-start_view,'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
            save('forward.json',statistics)
    finally:handle.remove()
    del model;captured.clear();gc.collect();torch.cuda.empty_cache()

def analyze():
    from sklearn.manifold import TSNE
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    sys.path.insert(0,str(ROOT/'fig2b_qwen3_v3_20261105'))
    from explore import display_transform
    rows=[json.loads(s) for s in (OUT/'inputs.jsonl').read_text().splitlines()]
    labels=np.repeat(np.array([r['domain'] for r in rows]),4);groups=np.repeat(np.array([r['group'] for r in rows]),4)
    out=OUT/'plots';out.mkdir(exist_ok=True);stats=[];transforms={}
    for view in ['raw128','native_context']:
        with np.load(OUT/(view+'_features.npz')) as data:x=data['tokens'].reshape(-1,3584)
        assert x.shape==(2560,3584) and np.isfinite(x).all()
        var=lambda a:float(np.square(a-a.mean(0)).sum(1).mean())
        high_dim={'within_domain_dispersion':{d:var(x[labels==d]) for d in DOMAINS},
            'centroid_distance_to_formal_proof':{d:float(np.linalg.norm(x[labels==d].mean(0)-x[labels=='Formal Proof'].mean(0))) for d in DOMAINS}}
        for seed in [42,2,3]:
            fit=TSNE(n_components=2,init='random',learning_rate=200,perplexity=30,random_state=seed,max_iter=1000,n_jobs=2)
            xy=fit.fit_transform(x);shown,transform=display_transform(xy,labels,orient=True,scale=True)
            name=f'realprover_five_domain_{view}_seed{seed}'
            np.savez_compressed(out/(name+'.npz'),native_xy=xy,display_xy=shown,labels=labels,groups=groups)
            transforms[name]=transform
            fig=plt.figure(figsize=(7.2,6),dpi=100);ax=fig.add_axes([.085,.10,.775,.76]);cax=fig.add_axes([.90,.10,.025,.76])
            order=np.concatenate([np.flatnonzero(labels==d) for d in ['Code','Math','Science','Formal Proof','Pre-training']])
            artist=ax.scatter(shown[order,0],shown[order,1],c=np.array([IDS[d] for d in labels[order]]),cmap='viridis',norm=Normalize(0,5),s=36,alpha=.2,linewidths=.5)
            ax.set_xlim(-40,40);ax.set_ylim(-40,40);ax.set_xticks(np.arange(-40,41,20));ax.set_yticks(np.arange(-40,41,10))
            fig.colorbar(artist,cax=cax,ticks=range(6))
            fig.text(.085,.965,'Dataset Domain',fontsize=14,fontweight='bold')
            for d,left in zip(DOMAINS,[.085,.29,.41,.53,.69]):
                fig.text(left,.91,d,fontsize=11,fontweight='bold',color=plt.colormaps['viridis'](IDS[d]/5))
            for ext in ['png','pdf','svg']:
                target=out/(name+'.'+ext);fig.savefig(target,dpi=200)
                if ext=='svg':
                    before=target.read_text();after='\n'.join(line.rstrip() for line in before.splitlines())+'\n';assert before.split()==after.split();target.write_text(after)
            plt.close(fig)
            stats.append({'view':view,'seed':seed,'points':len(x),'kl_divergence':float(fit.kl_divergence_),'high_dimensional':high_dim,
                'two_dimensional_within_domain_dispersion':{d:var(xy[labels==d]) for d in DOMAINS}})
            save('statistics.json',stats);save('display_transforms.json',transforms);print('PLOT',name,flush=True)
    save('completed.json',{'status':'DONE','plots':6,'primary':'plots/realprover_five_domain_raw128_seed42.png','domain_name':'Formal Proof (Lean State–Tactic)',
        'not_evidence_of_theorem_solving_ability':True,'display':'One global orientation/reflection and uniform scale; native coordinates retained'})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if args.prepare:prepare()
    else:forward()
