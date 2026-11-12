"""Exploratory P1 same-token replay of the already selected illustration only."""
from common import *
import copy


def main():
    import torch
    from engine import Engine
    selection=read(W/'plots_paper_style/source_and_validation.json')
    cid=selection['candidate_id']
    source=W/'metrics/per_tactic'/f'{cid}.json';r=read(source)
    case=next(c for c in rows(W/'manifests/frozen_cases.jsonl') if c['theorem_id']==r['theorem_id'])
    assert digest(case['input_ids'])==r['input_ids_sha']
    assert r['generated_ids']==selection['token_ids']
    inventory=read(W/'manifests/models.json')
    path=ROOT/'runs/realprover_formal_four_gpu_v16/formal/P1/eval_model'
    export=read(path/'export.json')
    assert export['method']=='P1' and export['bits']==2 and export['group_size']==128
    assert export['materialized_logits_max_error']==0 and export['reload_logits_max_error']==0
    files=[]
    for p in sorted(path.glob('*.safetensors')):
        files.append(dict(path=str(p),size=p.stat().st_size,sha256=sha(p)))
    assert len(files)==4
    assert sha(path/'tokenizer.json')==sha(Path(inventory['P0']['path'])/'tokenizer.json')
    inventory['P1']=dict(path=str(path),files=files,checkpoint_sha=digest(files),export_sha=sha(path/'export.json'))
    save(W/'manifests/models_P1_figure.json',inventory['P1'])
    e=Engine('P1',inventory=inventory)
    prefix=case['input_ids'].copy();tokens=[]
    for i,t in enumerate(r['generated_ids']):
        z=e.step(prefix,verify=(i==0));lp=z['logits'].log_softmax(-1)
        row={k:v for k,v in z.items() if k!='logits'}
        row.update(token_id=t,token_text=e.tokenizer.decode([t]),logprob=float(lp[t]))
        tokens.append(row);prefix.append(t)
        print('P1 token',i+1,'/',len(r['generated_ids']),flush=True)
    from dtr import depths
    result=dict(model='P1',status='DONE',candidate_id=cid,theorem_id=r['theorem_id'],
        input_ids_sha=r['input_ids_sha'],generated_token_ids_sha=r['generated_token_ids_sha'],
        source_sha=sha(source),checkpoint_sha=inventory['P1']['checkpoint_sha'],
        generation_origin='P4',measurement='same-token replay; single illustrative case, no new confirmation',
        token_ids=r['generated_ids'],token_strings=[t['token_text'] for t in tokens],
        layer_ids=list(range(1,29)),tokens=tokens,
        jsd_matrix=[list(x) for x in zip(*[t['jsd'] for t in tokens])],
        tactic_text=r['tactic_text'],dtr=depths([t['jsd'] for t in tokens])['dtr'])
    save(W/'metrics/P1_figure_replay.json',result)
    print('P1_DONE',result['dtr'],flush=True)


if __name__=='__main__':main()
