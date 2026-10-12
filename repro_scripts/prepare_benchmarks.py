"""Prefetch all five EvalScope benchmark inputs with CPU only, no model calls."""
import json
from pathlib import Path
from evalscope import TaskConfig
from evalscope.api.registry import get_benchmark

repo=Path(__file__).resolve().parents[1]
protocol=json.loads((repo/'repro_configs/evaluation.json').read_text())
dataset_args={name:{'few_shot_num':0} for name in protocol['datasets']}
dataset_args['live_code_bench']['subset_list']=['release_v1']
cfg=TaskConfig(model='unused',eval_type='mock_llm',datasets=protocol['datasets'],dataset_args=dataset_args,limit=None)
records={}
for name in protocol['datasets']:
    adapter=get_benchmark(name,cfg)
    data=adapter.load_dataset()
    records[name]={'source':adapter.benchmark_meta.dataset_id,
                   'evaluation_version':adapter.benchmark_meta.evaluation_version,
                   'subset_counts':{k:len(v) for k,v in data.items()},
                   'comparability':'Author dataset revision not yet confirmed'}
    assert sum(records[name]['subset_counts'].values())>0
    print(name,json.dumps(records[name]),flush=True)
    (repo.parent/'datasets/manifests/benchmarks.json').write_text(json.dumps(records,indent=2))
