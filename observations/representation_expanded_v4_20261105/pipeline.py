"""Isolated, restartable expansion. GPU stages guarded; t-SNE CPU only."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
import sys,json,time,hashlib,importlib.util,subprocess,gc
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
OLD=ROOT/'observation_20261104_all_layers'
sys.path.append(str(OLD/'envs/data/lib/python3.11/site-packages'))
import numpy as np

def write(p,v):
    p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);tmp=p.with_suffix(p.suffix+'.partial');tmp.write_text(json.dumps(v,indent=2));tmp.replace(p)
def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def prepare_qwen():
    dest=OUT/'qwen'
    if (dest/'prepared.json').exists():return
    from transformers import AutoTokenizer
    source=load_module(OUT/'selection_source.py','qwen_original')
    source.OUT=dest;source.N=512
    tok=AutoTokenizer.from_pretrained(ROOT/'models/Qwen3-1.7B',local_files_only=True)
    candidates=source.select(tok)
    old=[json.loads(s) for s in (ROOT/'fig2b_qwen3_v2_20261105/inputs.jsonl').read_text().splitlines()]
    selected=[]
    for d in source.DOMAINS:
        base=[r for r in old if r['domain']==d];seen={r['sha256'] for r in base}
        extra=[r for r in candidates if r['domain']==d and r['sha256'] not in seen][:384]
        assert len(base)==128 and len(extra)==384
        selected.extend(base+extra)
    selected.sort(key=lambda r:(source.DOMAINS.index(r['domain']),r['sha256']))
    (dest/'inputs.jsonl').write_text('\n'.join(json.dumps(r) for r in selected)+'\n')
    write(dest/'prepared.json',{'groups_per_domain':512,'original_groups_retained':512,'total_groups':2048,'selection':'Original128/domain retained; add384/domain from seed2 sampler, cap32/shard for Code/Math, cap64/shard for Science (only10 source shards). FineWeb source-order buffering bias retained.'})

def qwen_forward():
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(12*1024**3/torch.cuda.get_device_properties(0).total_memory)
    rows=[json.loads(s) for s in (OUT/'qwen/inputs.jsonl').read_text().splitlines()]
    model=AutoModelForCausalLM.from_pretrained(ROOT/'models/Qwen3-1.7B',local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda:0'})
    model.eval().requires_grad_(False);captured={}
    def hook(m,i,o):captured['h']=(o[0] if isinstance(o,tuple) else o).detach()
    handle=model.model.layers[13].register_forward_hook(hook)
    with torch.inference_mode():
        ids=torch.tensor([rows[0]['views']['calibration_chat'][:128]],device='cuda');full=model.model(input_ids=ids,use_cache=False,output_hidden_states=True)
        assert torch.equal(captured['h'],full.hidden_states[14]);reference=captured['h'].clone();del full
        model.model.layers=torch.nn.ModuleList(list(model.model.layers[:14]));model.model(input_ids=ids,use_cache=False)
        assert torch.equal(reference,captured['h']);del reference,ids
    write(OUT/'qwen/hook_validation.json',{'status':'PASS','block':14,'full_vs_prefix_exact':True})
    arrays=[];started=time.time()
    for i,r in enumerate(rows):
        begin=time.monotonic();positions=np.random.default_rng(int(r['sha256'][:16],16)).choice(np.arange(32,2048),16,replace=False)
        ids=torch.tensor([r['views']['calibration_chat']],device='cuda')
        with torch.inference_mode():model.model(input_ids=ids,use_cache=False)
        h=captured.pop('h')[0];value=h[torch.as_tensor(positions,device='cuda')].float().cpu().numpy();assert np.isfinite(value).all();arrays.append(value)
        del h,ids,value;torch.cuda.synchronize();time.sleep(max(.02,time.monotonic()-begin))
        if (i+1)%32==0:print('QWEN FORWARD',i+1,len(rows),flush=True)
    target=OUT/'qwen/features.npz';tmp=target.with_suffix('.partial')
    with tmp.open('wb') as f:np.savez_compressed(f,tokens=np.stack(arrays))
    tmp.replace(target);write(OUT/'qwen/forward.json',{'status':'DONE','groups':len(rows),'elapsed_s':time.time()-started,'peak_bytes':torch.cuda.max_memory_allocated()})
    handle.remove();del model;gc.collect();torch.cuda.empty_cache()

def analyze(kind):
    from sklearn.manifold import TSNE
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    sys.path.insert(0,str(ROOT/'fig2b_qwen3_v3_20261105'));from explore import display_transform
    folder=OUT/kind;plot=folder/'plots';plot.mkdir(exist_ok=True)
    if kind=='qwen':
        rows=[json.loads(s) for s in (folder/'inputs.jsonl').read_text().splitlines()];domains=['Pre-training','Code','Math','Science'];views=['calibration_chat'];dim=2048
        old=[json.loads(s) for s in (ROOT/'fig2b_qwen3_v2_20261105/inputs.jsonl').read_text().splitlines()];oldids={r['sha256'] for r in old};key='sha256'
    else:
        rows=[json.loads(s) for s in (folder/'inputs.jsonl').read_text().splitlines()];domains=['Pre-training','Code','Math','Science','Formal Proof'];views=['raw128','native_context'];dim=3584
        old=[json.loads(s) for s in (ROOT/'realprover_five_domain_v3_20261105/inputs.jsonl').read_text().splitlines()];oldids={r['group'] for r in old};key='group'
    labels=np.array([r['domain'] for r in rows]);groups=np.array([r[key] for r in rows]);base=np.array([g in oldids for g in groups]);assert base.sum()==128*len(domains)
    ids={'Pre-training':5,'Code':0,'Math':1,'Science':2,'Formal Proof':3};results=[]
    for view in views:
        feature=folder/('features.npz' if kind=='qwen' else view+'_features.npz')
        with np.load(feature) as f:features=f['tokens']
        assert features.shape==(512*len(domains),16,dim) and np.isfinite(features).all()
        for scenario,mask,tokens in [('baseline128x4',base,4),('density128x16',base,16),('expanded512x4',np.ones(len(rows),dtype=bool),4)]:
            x=features[mask,:tokens].reshape(-1,dim);y=np.repeat(labels[mask],tokens);g=np.repeat(groups[mask],tokens)
            high={d:float(np.square(x[y==d]-x[y==d].mean(0)).sum(1).mean()) for d in domains}
            for seed in [42,2,3]:
                name=f'{kind}_{view}_{scenario}_seed{seed}';done=plot/(name+'.json')
                if done.exists():results.append(json.loads(done.read_text()));continue
                begin=time.time();fit=TSNE(n_components=2,init='random',learning_rate=200,perplexity=30,random_state=seed,max_iter=1000,n_jobs=2)
                xy=fit.fit_transform(x);shown,transform=display_transform(xy,y,orient=True,scale=True)
                np.savez_compressed(plot/(name+'.npz'),native_xy=xy,display_xy=shown,labels=y,groups=g)
                fig=plt.figure(figsize=(7.2,6),dpi=100);ax=fig.add_axes([.085,.10,.775,.76]);cax=fig.add_axes([.90,.10,.025,.76])
                order=np.concatenate([np.flatnonzero(y==d) for d in ['Code','Math','Science','Formal Proof','Pre-training'] if d in domains])
                artist=ax.scatter(shown[order,0],shown[order,1],c=[ids[d] for d in y[order]],cmap='viridis',norm=Normalize(0,5),s=36,alpha=.2,linewidths=.5)
                ax.set(xlim=(-40,40),ylim=(-40,40),xticks=np.arange(-40,41,20),yticks=np.arange(-40,41,10));fig.colorbar(artist,cax=cax,ticks=range(6));fig.text(.085,.965,'Dataset Domain',fontsize=14,fontweight='bold')
                for d,left in zip(domains,[.085,.29,.41,.53,.69]):fig.text(left,.91,d,fontsize=11,fontweight='bold',color=plt.colormaps['viridis'](ids[d]/5))
                for ext in ['png','pdf']:fig.savefig(plot/(name+'.'+ext),dpi=200)
                plt.close(fig)
                result={'name':name,'points':len(x),'groups_per_domain':int(mask.sum()/len(domains)),'tokens_per_group':tokens,'seed':seed,'high_dimensional_within_domain_dispersion':high,'kl':float(fit.kl_divergence_),'elapsed_s':time.time()-begin,'display_transform':transform}
                write(done,result);results.append(result);write(folder/'statistics.json',results);print('PLOT',name,flush=True)
    write(folder/'completed.json',{'status':'DONE','plots':len(results)})

def guarded(kind,command):
    sys.path.insert(0,str(OLD/'src'));import common
    common.OBS=OUT/kind
    import resource_guard as guard
    guard.OBS=OUT/kind;guard.CAMPAIGN=ROOT/'runs/realprover_formal_four_gpu_v15'
    for name in ['configs','manifests','logs','tmp','cache']:(guard.OBS/name).mkdir(exist_ok=True)
    (guard.OBS/'configs/observation_protocol.json').write_text((OLD/'configs/observation_protocol.json').read_text())
    result=guard.supervise(command,guard.baseline(),kind+'_expanded_v4');assert result['status']=='DONE',result

def main():
    started=time.time();write(OUT/'status.json',{'status':'RUNNING','stage':'prepare_qwen','pid':os.getpid()})
    prepare_qwen()
    write(OUT/'status.json',{'status':'RUNNING','stage':'qwen_forward','pid':os.getpid()})
    if not (OUT/'qwen/features.npz').exists():guarded('qwen',[sys.executable,str(Path(__file__)),'qwen_forward'])
    write(OUT/'status.json',{'status':'RUNNING','stage':'prepare_real','pid':os.getpid()})
    if not (OUT/'real/inputs.jsonl').exists():subprocess.run([sys.executable,str(OUT/'real/experiment.py'),'--prepare'],check=True)
    write(OUT/'status.json',{'status':'RUNNING','stage':'real_forward','pid':os.getpid()})
    if not (OUT/'real/native_context_features.npz').exists():guarded('real',[sys.executable,str(OUT/'real/experiment.py')])
    # GPU children have exited before CPU t-SNE starts.
    for kind in ['qwen','real']:
        write(OUT/'status.json',{'status':'RUNNING','stage':'analyze_'+kind,'pid':os.getpid()});analyze(kind)
    write(OUT/'status.json',{'status':'DONE','elapsed_s':time.time()-started})
if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='qwen_forward':qwen_forward()
    else:
        try:main()
        except BaseException as e:
            write(OUT/'status.json',{'status':'FAILED','error':str(e),'time':time.time()});raise
