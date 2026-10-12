"""CPU contract tests: dependency paths, smoke limits and hard failure gates."""
import copy
import json
import os
from pathlib import Path
from unittest.mock import patch
from repro_scripts import pipeline

def rejects(function,fragment):
    try:
        function()
    except (ValueError,RuntimeError) as exc:
        assert fragment in str(exc),str(exc)
    else:
        raise AssertionError(f'Expected failure containing {fragment}')

with patch.dict(os.environ,{'RQ_GPU_RELEASED':'0'}):
    rejects(pipeline.check_gpu_release,'GPU_NOT_RELEASED')
with patch.dict(os.environ,{'RQ_GPU_RELEASED':'1'}),patch('subprocess.check_output',return_value='45274\n'):
    rejects(pipeline.check_gpu_release,'GPU_BUSY')
rejects(lambda:pipeline.check_protocol('formal'),'NOT_COMPARABLE')
for run in ['R4','R4a','R5','R6_rr','R6_kl']:
    out=pipeline.ROOT/'outputs/formal/cpu_test'/run
    formal=pipeline.train_command(run,'formal',out,'cpu_test')
    smoke=pipeline.train_command(run,'smoke',out,'cpu_test')
    assert formal[formal.index('--model')+1]==str(out.parent/'R3')
    assert formal[formal.index('--dataset_size')+1]=='32768'
    assert '--max_steps' not in formal and '--smoke_short_text' not in formal
    assert smoke[smoke.index('--max_steps')+1]=='2'
    assert smoke[smoke.index('--gradient_accumulation_steps')+1]=='1'
    assert '--smoke_short_text' in smoke
    if run=='R5':
        assert '--use_teacher_weight' not in formal and formal[formal.index('--kl_weight')+1]=='0.0'
protocol=json.loads((pipeline.REPO/'repro_configs/evaluation.json').read_text())
metrics={'status':'PASS','seeds':[{}, {}, {}],'average_percent':65.2,'protocol':protocol,
         'means_percent':dict(zip(protocol['datasets'],[89.,53.6,74.7,38.4,70.4]))}
assert pipeline.gate('R0','formal',metrics)['status']=='PASS'
bad=copy.deepcopy(metrics)
bad['average_percent']=70
rejects(lambda:pipeline.gate('R0','formal',bad),'Avg outside')
bad=copy.deepcopy(metrics)
bad['means_percent']['math_500']=0.
rejects(lambda:pipeline.gate('R0','formal',bad),'math_500')
bad=copy.deepcopy(metrics)
bad['seeds']=[{}]
rejects(lambda:pipeline.gate('R0','formal',bad),'three seeds')
assert pipeline.gate('R0','smoke',bad)['status']=='PASS'
result={'status':'PASS','device':'cpu','checks':['Explicit user GPU release required',
        'Active GPU process blocks execution without stopping it','Unresolved protocol blocks formal runs',
        'All Stage 2 runs use the same R3; smoke has two steps; formal retains 32768 samples',
        'R5 excludes Teacher-RR/KL','Average and individual score failures stop formal gate',
        'Formal requires three seeds; smoke remains execution-only']}
dest=pipeline.ROOT/'results/cpu-pipeline-check.json'
dest.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
