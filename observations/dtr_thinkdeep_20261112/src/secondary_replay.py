"""Post-primary DEV-only P3/P6 same-token diagnostics, not new confirmation."""
import time,datetime,gc,ast,subprocess
from common import *
def main():
    import torch
    from engine import Engine
    from run import summarize,bootstrap
    cutoff=datetime.datetime(2026,11,12,7,8,tzinfo=datetime.timezone.utc).timestamp()
    source={p.name:read(p) for p in (W/'metrics/per_tactic').glob('*.json') if read(p).get('split')=='D-DEV' and read(p).get('status')=='DONE'}
    cases={c['theorem_id']:c for c in rows(W/'manifests/frozen_cases.jsonl')};summary={}
    for model in ['P3','P6']:
        if time.time()>=cutoff:break
        e=Engine(model)
        tree=ast.parse((ROOT/'research/adapters/quantization.py').read_text())
        prompts=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PROMPTS' for t in n.targets))
        probe=ROOT/'runs/realprover_formal_four_gpu_v16/formal'/model/'train/native_probe.pt';expected=torch.load(probe,map_location='cpu',weights_only=True);checks=[]
        for prompt,ref in zip(prompts,expected['logits']):
            z=e.step(e.tokenizer.encode(prompt))['logits'];d=(z-ref.float().flatten()).abs();checks.append(dict(max_abs=float(d.max()),mean_abs=float(d.mean()),top1=int(z.argmax())==int(ref.argmax())))
        assert all(r['max_abs']<=.125 and r['mean_abs']<=.01 and r['top1'] for r in checks)
        save(W/'manifests'/('secondary_native_'+model+'.json'),dict(status='PASS',checks=checks,probe_sha=sha(probe)))
        done=0
        for name,r in sorted(source.items()):
            if time.time()>=cutoff:break
            out=W/'metrics'/('secondary_replay_'+model)/name
            if out.exists():done+=1;continue
            prefix=cases[r['theorem_id']]['input_ids'].copy();tokens=[]
            for t in r['generated_ids']:
                z=e.step(prefix);lp=z['logits'].log_softmax(-1);row={k:v for k,v in z.items() if k!='logits'};row.update(token_id=t,logprob=float(lp[t]),stop_flag=False,token_text=e.tokenizer.decode([t]));tokens.append(row);prefix.append(t)
            result=dict(model=model,theorem_id=r['theorem_id'],split='D-DEV',generation_origin='P4',generated_token_ids_sha=r['generated_token_ids_sha'],source_sha=sha(W/'metrics/per_tactic'/name),checkpoint_sha=e.inventory[model]['checkpoint_sha'],tokens=tokens,**summarize(tokens))
            save(out,result);done+=1
            if done%16==0:
                import urllib.request
                with urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=5) as response:health=json.load(response)
                assert health['status']=='ready'
                status('secondary_progress','RUNNING',model=model,records=done,target=192)
        diffs={}
        for name,r in source.items():
            p=W/'metrics'/('secondary_replay_'+model)/name
            if p.exists() and r['dtr'] is not None and read(p)['dtr'] is not None:diffs.setdefault(r['theorem_id'],[]).append(read(p)['dtr']-r['dtr'])
        summary[model]=dict(records=done,target=192,status='DONE' if done==192 else 'TIME_BUDGET_PARTIAL',delta_DTR_vs_P4=bootstrap([sum(v)/len(v) for v in diffs.values()]),scope='post-primary DEV only, same P4-generated tokens; exploratory, not generation accuracy')
        for hook in e.hooks:hook.remove()
        del e;gc.collect();torch.cuda.empty_cache()
    save(W/'metrics/summaries/secondary_P3_P6_DEV.json',summary)
    (W/'reports/10_SECONDARY_P3_P6_DEV.md').write_text('# Exploratory same-token DEV diagnostics\n\nNot independent confirmation. P4-generated tokens are identical across evaluations; no natural-generation accuracy claim.\n\n```json\n'+json.dumps(summary,indent=2)+'\n```\n')
    status('secondary','DONE',coverage=summary)
if __name__=='__main__':main()
