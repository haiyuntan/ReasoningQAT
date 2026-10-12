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
baseline_manifest=ROOT/'results/qat-baselines-preparation.json'
baseline_assets=json.loads(baseline_manifest.read_text()) if baseline_manifest.exists() else {}
record('qat_baseline_assets',all(run in baseline_assets and baseline_assets[run]['dependency_check_passed']
       and (ROOT/baseline_assets[run]['paper']['path']).is_file() for run in ('R8','R9')),
       'Official source revisions, papers and Qwen3 runtime environments')
baseline_cfg=json.loads((REPO/'repro_configs/qat_baselines.json').read_text())
record('qat_baseline_adapters',all(baseline_cfg[run]['adapter_verified'] for run in ('R8','R9')),
       'Independent R8/R9 adapters must pass smoke before reporting all preparations complete')
gpu_check=ROOT/'results/gpu-model-check.json'
gpu_pass=gpu_check.exists() and json.loads(gpu_check.read_text()).get('status')=='PASS'
status={'status':('PREPARED' if gpu_pass else 'PREPARED_GPU_VALIDATION_PENDING')
        if all(c['passed'] for c in checks.values()) else 'PREPARING',
        'checks':checks,'gpu_validation_passed':gpu_pass,
        'gpu_released_by_user':(ROOT/'results/gpu_handoff.json').exists(),
        'formal_protocol':json.loads((REPO/'repro_configs/evaluation.json').read_text()),
        'note':'GPU use is authorized. A passed model component check is not completed R0–R6 evaluation. Formal comparison also requires protocol confirmation.'}
(ROOT/'results/preparation_status.json').write_text(json.dumps(status,indent=2))
print(json.dumps(status,indent=2))
