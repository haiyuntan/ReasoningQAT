"""Continue authorized smoke pipeline when complete datasets and CPU caches are ready."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
REPO=ROOT/'ReasoningQAT'
env=os.environ.copy()
env.update({'PYTHONPATH':str(REPO),'HF_HOME':str(ROOT/'hf_cache'),
            'MODELSCOPE_CACHE':str(ROOT/'hf_cache/modelscope'),'RQ_GPU_RELEASED':'1'})
state=ROOT/'results/queued-smoke.json'
while True:
    assets=(ROOT/'results/download-complete.json').exists()
    benchmarks=ROOT/'datasets/manifests/benchmarks.json'
    ready=benchmarks.exists() and len(json.loads(benchmarks.read_text()))==5
    status={'status':'WAITING_FOR_INPUTS','all_downloads_complete':assets,'five_benchmarks_ready':ready,
            'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    state.write_text(json.dumps(status,indent=2))
    if assets and ready:
        break
    time.sleep(50)
try:
    env['CUDA_VISIBLE_DEVICES']=''
    status['status']='PREPARING_CPU_CALIBRATION'
    state.write_text(json.dumps(status,indent=2))
    subprocess.run([str(ROOT/'env/train/bin/python'),str(REPO/'repro_scripts/prepare_block_calibration.py')],
                   cwd=REPO,env=env,check=True)
    # Do not compete with an already running GPU job or change someone else's processes.
    while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
        time.sleep(50)
    status['status']='RUNNING_SMOKE'
    state.write_text(json.dumps(status,indent=2))
    subprocess.run(['/bin/bash',str(REPO/'repro_scripts/run_gated.sh'),'--mode','smoke'],cwd=REPO,env=env,check=True)
    status['status']='SMOKE_COMPLETE'
except Exception as exc:
    status['status']='FAIL'
    status['error']=f'{type(exc).__name__}: {exc}'
    raise
finally:
    state.write_text(json.dumps(status,indent=2))
