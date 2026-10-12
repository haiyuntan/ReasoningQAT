import argparse
import json
import math
from pathlib import Path
from evalscope import TaskConfig,run_task

p=argparse.ArgumentParser()
p.add_argument('--mode',choices=['smoke','formal'],required=True)
p.add_argument('--output',required=True)
p.add_argument('--port',type=int,default=18080)
a=p.parse_args()
repo=Path(__file__).resolve().parents[1]
protocol=json.loads((repo/'repro_configs/evaluation.json').read_text())
if a.mode=='formal' and (not protocol['confirmed_paper_protocol'] or protocol['unresolved']):
    raise SystemExit('NOT_COMPARABLE: paper evaluation protocol has unresolved fields')
out=Path(a.output)
out.mkdir(parents=True,exist_ok=True)
seeds=protocol['seeds'] if a.mode=='formal' else protocol['seeds'][:1]
records=[]
for seed in seeds:
    dataset_args={name:{'few_shot_num':0} for name in protocol['datasets']}
    dataset_args['live_code_bench']['subset_list']=['release_v1']
    # Formal must supply a reviewed, paper-comparable slice and all overrides.
    for key,value in protocol.get('dataset_args',{}).items():
        dataset_args[key].update(value)
    config=TaskConfig(model='rq-model',eval_type='openai_api',api_url=f'http://127.0.0.1:{a.port}/v1',api_key='EMPTY',
                      datasets=protocol['datasets'],dataset_args=dataset_args,eval_batch_size=1,
                      generation_config={'temperature':0.6,'top_p':1.0,'max_tokens':32768 if a.mode=='formal' else 64,
                                         'seed':seed,'stream':False,'timeout':7200,
                                         'extra_body':{'top_k':20,'chat_template_kwargs':{'enable_thinking':True}}},
                      limit=None if a.mode=='formal' else 1,seed=seed,work_dir=str(out/f'seed_{seed}'),no_timestamp=True)
    run_task(config)
    scores={}
    report_details={}
    for path in (out/f'seed_{seed}'/'reports').rglob('*.json'):
        payload=json.loads(path.read_text())
        if not isinstance(payload,dict) or payload.get('dataset_name') not in protocol['datasets']:
            continue
        identity=payload.get('primary_metric_identity')
        metrics=[m for m in payload.get('metrics',[]) if m.get('identity')==identity]
        if len(metrics)!=1 or metrics[0]['num']<=0:
            raise ValueError(f'Missing or empty primary metric: {path}')
        completion=payload.get('execution_summary') or {}
        if completion.get('incomplete') or completion.get('errored',0):
            raise ValueError(f'Incomplete evaluation: {path}')
        dataset=payload['dataset_name']
        if dataset in scores:
            raise ValueError(f'Duplicate reports: {dataset}')
        score=float(metrics[0]['score'])*100
        if not math.isfinite(score):
            raise ValueError('Non-finite score')
        scores[dataset]=score
        report_details[dataset]={'metric_identity':identity,'num':metrics[0]['num'],'execution_summary':completion}
    if set(scores)!=set(protocol['datasets']):
        raise ValueError(f'Missing benchmarks: {set(protocol["datasets"])-set(scores)}')
    records.append({'seed':seed,'scores_percent':scores,'report_details':report_details})
means={task:sum(r['scores_percent'][task] for r in records)/len(records) for task in protocol['datasets']}
result={'mode':a.mode,'status':'PASS','seeds':records,'means_percent':means,
        'average_percent':sum(means.values())/5,'protocol':protocol}
(out/'metrics.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
