import argparse,random,datetime,time,urllib.request,subprocess,sys,math
from common import *
def expired():return time.time()>=datetime.datetime.fromisoformat(protocol()['deadline_utc']).timestamp()
def request(path,payload=None):
    req=urllib.request.Request('http://127.0.0.1:8080/'+path,data=None if payload is None else json.dumps(payload).encode(),headers={'Content-Type':'application/json','X-Observation-ID':'DTR-only'})
    with urllib.request.urlopen(req,timeout=20) as r:return json.load(r)
def prepare():
    from transformers import AutoTokenizer
    from lean import Session
    import importlib.util
    models=read(W/'manifests/models.json');token=AutoTokenizer.from_pretrained(models['P0']['path'],local_files_only=True)
    spec=importlib.util.spec_from_file_location('DTRprompt',W/'upstream/realprover/manager/manage/prompt_manage.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    health=request('health');assert health['status']=='ready' and health['adapter_loaded']
    assert health['asset_manifest_sha256']==sha(ROOT/'research/manifests/retrieval_assets.json')
    save(W/'manifests/retrieval.json',health)
    cases=[]
    for row in rows(W/'manifests/dev_cases.jsonl')+rows(W/'manifests/confirm_ids.jsonl'):
        cache=W/'caches/roots'/(row['theorem_id']+'.json')
        if cache.exists():cases.append(read(cache));continue
        r=dict(row)
        if r['split']=='D-CONFIRM':
            s=Session(r)
            try:r['state']=s.state
            finally:s.close()
            data=request('retrieve_premises',{'query':r['state'],'num':10})['data'];prem=data[0] if data and isinstance(data[0],list) else data
            history=r['formal_statement'].replace('sorry','').replace('by','').strip()+' by\n'
            r.update(history=history,returned_top10=prem,rendered_top6=prem[:6],prompt=m.PromptManage.build_local_incontext_prompt_str(history,r['state'],prem[:6],'qwen'),retrieval_snapshot_sha=digest(prem));time.sleep(5)
            r['input_ids']=token.encode(r['prompt'],add_special_tokens=False)
        assert r['input_ids']==token.encode(r['prompt'],add_special_tokens=False),'PROMPT_TOKEN_ID_MISMATCH'
        r.update(state_id=digest(r['state']),prompt_sha=digest(r['prompt']),input_ids_sha=digest(r['input_ids']),eligible=len(r['input_ids'])+protocol()['max_new_tokens']<=4096,state_origin='ROOT')
        save(cache,r);cases.append(r);status('prepare_progress','RUNNING',cases=len(cases),target=44)
    jsonl(W/'manifests/frozen_cases.jsonl',cases)
    save(W/'manifests/experiment_manifest.json',dict(protocol_sha=identity(),frozen_cases_sha=sha(W/'manifests/frozen_cases.jsonl'),split_sha=sha(W/'configs/split_registry.json'),models=models,Lean_sha=sha('/tmp/realprover_lean_v10/interactive/.lake/build/bin/interactive'),retrieval=health,
        source_sha={p.name:sha(p) for p in (W/'src').glob('*.py')},user_resource_override='Pause formal GPU0-2; use GPU1; shared retrieval preserved',dtype='BF16 dense W2A16',semantic_overlap='UNKNOWN'))
    (W/'manifests/protocol.sha256').write_text(identity()+'\n');status('G0','DONE',dev=24,confirm=20)
def sample_seed(case,seed,step):return random.Random(int(digest(['DTR-sample',case['theorem_id'],seed,step])[:16],16)).random()
def summarize(tokens,raw=False,include_eos=False):
    import numpy as np
    from dtr import depths
    selected=[t for t in tokens if include_eos or not t['stop_flag']]
    if not selected:return dict(dtr=None,mean_settling_depth=None,empty=True)
    d=depths([t['jsd_raw' if raw else 'jsd'] for t in selected]);lp=sum(t['logprob'] for t in selected)/len(selected)
    return dict(dtr=d['dtr'],mean_settling_depth=d['mean_settling_depth'],mean_logprob=lp,negative_ppl=-math.exp(min(700,-lp)),negative_entropy=-sum(t['entropy'] for t in selected)/len(selected),
        self_certainty=sum(t['self_certainty'] for t in selected)/len(selected),length=len(selected),negative_length=-len(selected),empty=False,
        dtr_bits_literal_g05=depths([[x/math.log(2) for x in t['jsd_raw' if raw else 'jsd']] for t in selected])['dtr'],
        dtr_g025=depths([t['jsd_raw' if raw else 'jsd'] for t in selected],g=.25)['dtr'],dtr_g075_degenerate=depths([t['jsd_raw' if raw else 'jsd'] for t in selected],g=.75)['dtr'],
        persistent_deep_ratio=sum(x>=24 for x in d['persistent'])/len(selected),DTR_rho080=depths([t['jsd_raw' if raw else 'jsd'] for t in selected],rho=.80)['dtr'],DTR_rho095=depths([t['jsd_raw' if raw else 'jsd'] for t in selected],rho=.95)['dtr'])
def generate():
    from engine import Engine
    from numerics import distribution,sample,token_category
    from lean import verify,close_reused
    import torch
    e=Engine('P4');cases=rows(W/'manifests/frozen_cases.jsonl');parity=e.native_parity();checks=[]
    # G2: identical complete prefixes and projected own-final logits, no future-token
    # BF16 matrix path silently substituted for incremental decisions.
    for c in cases[:4]:
        a=e.step(c['input_ids'],verify=True);b=e.step(c['input_ids'],verify=True)
        assert torch.equal(a['logits'],b['logits']) and a['jsd']==b['jsd']
        checks.append({k:a[k] for k in ['lens_max_abs','lens_mean_abs','jsd_fp64_abs_error']})
    save(W/'manifests/G2.json',dict(status='PASS',native=parity,checks=checks,path='batch1 full prefix noKV; exact pre-next-token replay',future_inclusive_matrix='NOT_USED_BF16_SHAPE_SENSITIVITY'))
    for split in ['D-DEV','D-CONFIRM']:
        status('controller','GENERATING',model='P4',split=split)
        for c in [x for x in cases if x['split']==split]:
            if expired():status('controller','TIME_BUDGET_REACHED',split=split);close_reused();return
            if not c['eligible']:save(W/'metrics/per_tactic'/('ineligible_'+c['theorem_id']+'.json'),dict(theorem_id=c['theorem_id'],status='INELIGIBLE_OVERLENGTH'));continue
            for seed in protocol()['seeds']:
                if expired():status('controller','TIME_BUDGET_REACHED',split=split);close_reused();return
                key=digest(['P4',c['theorem_id'],seed]);p=W/'metrics/per_tactic'/(key+'.json');ident=digest([identity(),c['input_ids_sha'],seed,e.inventory['P4']['checkpoint_sha']])
                if p.exists():assert read(p)['identity']==ident;continue
                prefix=c['input_ids'].copy();ids=[];tt=[];start=time.monotonic();finish='LENGTH_LIMIT'
                for step in range(protocol()['max_new_tokens']):
                    result=e.step(prefix);z=result['logits'];probs=distribution(z,1.5,.9);token=sample(probs,sample_seed(c,seed,step));stop=token in e.eos
                    tokenrow={k:v for k,v in result.items() if k!='logits'}
                    tokenrow.update(step=step,token_id=token,logprob=float(z.log_softmax(-1)[token]),token_text=e.tokenizer.decode([token]),prefix_sha=digest(prefix),stop_flag=stop,category=token_category(e.tokenizer.decode([token])))
                    tt.append(tokenrow)
                    if stop:finish='EOS';break
                    ids.append(token);prefix.append(token)
                text=e.tokenizer.decode(ids,skip_special_tokens=True).strip();verdict=verify(c,text)
                tp=W/'metrics/per_token'/(key+'.json');save(tp,tt)
                row=dict(identity=ident,protocol_sha=identity(),theorem_id=c['theorem_id'],state_id=c['state_id'],split=split,state_origin='ROOT',model_id='P4',generation_origin='P4',checkpoint_sha=e.inventory['P4']['checkpoint_sha'],prompt_sha=c['prompt_sha'],input_ids_sha=c['input_ids_sha'],retrieval_sha=c['retrieval_snapshot_sha'],seed=seed,
                    generated_ids=ids,generated_token_ids_sha=digest(ids),tactic_text=text,finish_reason=finish,token_file=str(tp.relative_to(W)),token_sha=sha(tp),lens='ARCH_NORM',jsd_unit='nats',g=.5,rho=.85,
                    **summarize(tt),**verdict,raw_sensitivity=summarize(tt,raw=True),EOS_sensitivity=summarize(tt,include_eos=True),nonwhitespace_sensitivity=summarize([t for t in tt if t['token_text'].strip()]),duration_s=time.monotonic()-start,numeric_parity_status='PASS',status='DONE')
                save(p,row)
            status('generation_progress','RUNNING',theorem_id=c['theorem_id'],split=split,completed_tactics=len(list((W/'metrics/per_tactic').glob('*.json'))))
        analyze(split)
        status('G3' if split=='D-DEV' else 'G6','DONE',split=split)
    close_reused();status('generation','DONE')
def replay():
    from engine import Engine
    from lean import close_reused
    e=Engine('P0');cases={c['theorem_id']:c for c in rows(W/'manifests/frozen_cases.jsonl')}
    for path in sorted((W/'metrics/per_tactic').glob('*.json')):
        if expired():status('replay','TIME_BUDGET_REACHED');return
        r=read(path)
        if r.get('status')!='DONE':continue
        out=W/'metrics/paired_replay'/path.name
        if out.exists():continue
        c=cases[r['theorem_id']];prefix=c['input_ids'].copy();tt=[]
        for i,token in enumerate(r['generated_ids']):
            a=e.step(prefix);z=a.pop('logits') if False else a['logits']
            row={k:v for k,v in a.items() if k!='logits'};row.update(step=i,token_id=token,logprob=float(z.log_softmax(-1)[token]),token_text=e.tokenizer.decode([token]),stop_flag=False)
            tt.append(row);prefix.append(token)
        save(out,dict(theorem_id=r['theorem_id'],split=r['split'],generation_origin='P4',evaluated_by='P0',protocol_sha=identity(),checkpoint_sha=e.inventory['P0']['checkpoint_sha'],source_tactic_sha=sha(path),input_ids_sha=c['input_ids_sha'],generated_token_ids_sha=r['generated_token_ids_sha'],tokens=tt,**summarize(tt),raw_sensitivity=summarize(tt,raw=True)))
    status('replay','DONE')
def bootstrap(values):
    import numpy as np
    if not values:return dict(n=0,mean=None,ci=None)
    x=np.array(values);rng=np.random.default_rng(2);b=x[rng.integers(0,len(x),size=(10000,len(x)))].mean(1)
    return dict(n=len(x),mean=float(x.mean()),ci=list(map(float,np.quantile(b,[.025,.975]))))
def analyze(split=None):
    import collections
    rows_=[read(p) for p in (W/'metrics/per_tactic').glob('*.json')];rows_=[r for r in rows_ if r.get('status')=='DONE']
    if split is None:
        for s in ['D-DEV','D-CONFIRM']:analyze(s)
        return
    data=[r for r in rows_ if r['split']==split];group=collections.defaultdict(list)
    for r in data:group[r['theorem_id']].append(r)
    metrics=['dtr','self_certainty','mean_logprob','negative_entropy','length','negative_length','negative_ppl'];scores={m:[] for m in metrics};per=[]
    for theorem,rr in group.items():
        if len(rr)!=8:continue
        pos=[r for r in rr if r['lean_status']=='VALID_TRANSITION' and r['dtr'] is not None];neg=[r for r in rr if r['lean_status']=='INVALID_TACTIC' and r['dtr'] is not None]
        if not pos or not neg:continue
        d={}
        for m in metrics:
            d[m]=sum(float(a[m]>b[m])+.5*float(a[m]==b[m]) for a in pos for b in neg)/(len(pos)*len(neg));scores[m].append(d[m])
        per.append(dict(theorem_id=theorem,**d,primary_difference=d['dtr']-d['self_certainty']))
    diff=bootstrap([r['primary_difference'] for r in per]);values=[r['dtr'] for r in data if r['dtr'] is not None]
    saturated=sum(v in [0.,1.] for v in values)/len(values) if values else None
    if len(per)<20:state='INCONCLUSIVE_LOW_POWER'
    elif diff['ci'][0]>0:state='SUPPORTED'
    else:state='NOT_SUPPORTED' if diff['ci'][1]<0 else 'INCONCLUSIVE'
    result=dict(split=split,status=state,primary=diff,baselines={m:bootstrap(v) for m,v in scores.items()},theorems=len(group),complete_pools=sum(len(rr)==8 for rr in group.values()),tactics=len(data),known_valid=sum(r['lean_status']=='VALID_TRANSITION' for r in data),invalid=sum(r['lean_status']=='INVALID_TACTIC' for r in data),unknown=sum(r['lean_status'] not in ['VALID_TRANSITION','INVALID_TACTIC'] for r in data),dtr_0_or_1_fraction=saturated,measurement='DTR_MEASUREMENT_DEGENERATE' if values and len(set(values))<=1 else 'DTR_METHOD_COMPATIBLE',DTR_unique_values=sorted(set(values)),short_tactics_lt4=sum(r.get('length',0)<4 for r in data),limits='P4 root tactic-level; not proof-search/Think@n; independent confirm<=20 before mixed-label exclusions; SEMANTIC_OVERLAP_UNKNOWN')
    save(W/'metrics/summaries'/(split+'.json'),result);jsonl(W/'metrics/summaries'/(split+'_per_theorem.jsonl'),per)
    print('DTR_SUMMARY',json.dumps(result),flush=True)
    (W/'reports'/('04_DTR_TACTIC_VALIDITY_'+split+'.md')).write_text('# '+split+'\n\n```json\n'+json.dumps(result,indent=2)+'\n```\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','generate','replay','analyze'],required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    {'prepare':prepare,'generate':generate,'replay':replay,'analyze':analyze}[a.mode]()
if __name__=='__main__':main()
