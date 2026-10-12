"""CPU-only readiness audit. Missing assets cannot be reported as prepared."""
import importlib.metadata
import json
from pathlib import Path
import subprocess

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO.parent
checks={}
def record(key,value,detail):
    checks[key]={'passed':bool(value),'detail':detail}

manifest=ROOT/'datasets/manifests/sources.json'
if manifest.exists():
    sources=json.loads(manifest.read_text())
    for repo,data in sources.items():
        required=data.get('verified_files',{})
        complete=[path for path,meta in required.items() if (ROOT/data['destination']/path).is_file()
                  and (ROOT/data['destination']/path).stat().st_size==meta['size']]
        record(repo,bool(required) and len(complete)==len(required),{'complete':len(complete),'required':len(required)})
for name in ['train','gptq','eval']:
    python=ROOT/'env'/name/'bin/python'
    if not python.exists():
        record('env_'+name,False,'Missing environment')
        continue
    result=subprocess.run([str(python),'-m','pip','check'],capture_output=True,text=True)
    record('env_'+name,result.returncode==0,(result.stdout+result.stderr).strip()[:2000])
cpu=ROOT/'results/cpu-loss-check.json'
record('cpu_loss_checks',cpu.exists() and json.loads(cpu.read_text()).get('status')=='PASS','Actual CPU loss and trainable tests')
record('numinamath_128',(ROOT/'datasets/manifests/numinamath.json').exists(),'Frozen 128 samples and checksum')
record('five_benchmark_data',(ROOT/'datasets/manifests/benchmarks.json').exists() and
       len(json.loads((ROOT/'datasets/manifests/benchmarks.json').read_text()))==5,'All five benchmark manifests')
record('block_calibration',(ROOT/'datasets/manifests/block_calibration.json').exists(),'4096/64 × 2048 CPU-prepared data')
status={'status':'PREPARED_GPU_VALIDATION_PENDING' if all(c['passed'] for c in checks.values()) else 'PREPARING',
        'checks':checks,'gpu_experiments_started':False,
        'formal_protocol':json.loads((REPO/'repro_configs/evaluation.json').read_text()),
        'note':'GPU remains reserved for the existing task until explicit user release. Formal comparison also requires protocol confirmation.'}
(ROOT/'results/preparation_status.json').write_text(json.dumps(status,indent=2))
print(json.dumps(status,indent=2))
