"""Detached serial smoke -> complete data -> gated formal campaign."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from audit_formal import audit

ROOT=Path(__file__).resolve().parents[2]
REPO=ROOT/'ReasoningQAT'
STATE=ROOT/'results/background-campaign.json'
PYTHON=str(ROOT/'env/bootstrap/bin/python')

def write_state(status,**details):
    data={'status':status,'pid':os.getpid(),'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**details}
    temporary=STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2));temporary.replace(STATE)
    print(json.dumps(data),flush=True)

def run(command,env,log):
    with log.open('a') as stream:
        subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)

def publish(env):
    try:
        subprocess.run([PYTHON,str(REPO/'repro_scripts/publish_results.py'),'--commit','--push'],
                       cwd=ROOT,env=env,check=True,timeout=180)
    except Exception as exc:
        print('REPORT_PUSH_FAILED',repr(exc),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--session',required=True)
    args=parser.parse_args()
    (ROOT/'logs').mkdir(exist_ok=True)
    lock=(ROOT/'outputs/background_campaign.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=os.environ.copy()
    env.update({'RQ_GPU_RELEASED':'1','RQ_DOWNLOAD_WORKERS':'1','OMP_NUM_THREADS':'4',
                'TOKENIZERS_PARALLELISM':'false','GIT_TERMINAL_PROMPT':'0',
                'GIT_ASKPASS':'/tmp/rq-git-askpass.py','PYTHONPATH':str(REPO),
                'HF_HOME':str(ROOT/'hf_cache'),'MODELSCOPE_CACHE':str(ROOT/'hf_cache/modelscope')})
    runs=['R3','R4','R4a','R5','R6_rr','R6_kl','R9','R8','R0','R1','R2']
    smoke=args.session+'_smoke';formal=args.session+'_formal'
    context={'session':args.session,'smoke_session':smoke,'formal_session':formal,'runs':runs,
             'download_policy':'One stream with HTTP Range resume; retain all completed/partial files'}
    try:
        initial=audit()
        write_state('RUNNING_SMOKE',formal_audit=initial['status'],formal_blockers=initial['blockers'],**context)
        run(['/bin/bash',str(REPO/'repro_scripts/run_gated.sh'),'--mode','smoke','--runs',*runs,'--session',smoke],
            env,ROOT/'logs'/f'{args.session}.smoke.log')
        summaries=REPO/'repro_results/smoke'/smoke
        assert all(json.loads((summaries/f'{run}.json').read_text())['status']=='PASS' for run in runs)
        while True:
            run([PYTHON,str(REPO/'repro_scripts/download_status.py')],env,ROOT/'logs'/f'{args.session}.data.log')
            downloaded=json.loads((ROOT/'results/download-status.json').read_text())
            ready=all(item['complete_files']==item['required_files'] for item in downloaded['repositories'].values())
            if ready: break
            write_state('SMOKE_PASS_WAITING_FOR_FULL_DATA',download_progress=downloaded,**context)
            time.sleep(50)
        previous_blockers=None
        while True:
            # Fresh process reads any later source/configuration corrections.
            run([PYTHON,str(REPO/'repro_scripts/audit_formal.py')],env,ROOT/'logs'/f'{args.session}.formal-audit.log')
            final=json.loads((ROOT/'results/formal-code-audit.json').read_text())
            if final['status']=='READY': break
            write_state('BLOCKED_FORMAL',blockers=final['blockers'],smoke_status='PASS',**context)
            if final['blockers']!=previous_blockers:
                publish(env);previous_blockers=final['blockers']
            time.sleep(50)
        write_state('PREPARING_FORMAL_CALIBRATION',**context)
        run([str(ROOT/'env/train/bin/python'),str(REPO/'repro_scripts/prepare_block_calibration.py'),'--mode','formal'],
            {**env,'CUDA_VISIBLE_DEVICES':''},ROOT/'logs'/f'{args.session}.formal-data.log')
        while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
            write_state('WAITING_FOR_IDLE_GPU',**context);time.sleep(50)
        write_state('RUNNING_FORMAL',**context)
        run(['/bin/bash',str(REPO/'repro_scripts/run_gated.sh'),'--mode','formal','--runs',*runs,'--session',formal],
            env,ROOT/'logs'/f'{args.session}.formal.log')
        summaries=REPO/'repro_results/formal'/formal
        assert all(json.loads((summaries/f'{run}.json').read_text())['status']=='PASS' for run in runs)
        write_state('FORMAL_COMPLETE',**context)
    except Exception as exc:
        write_state('FAIL',error=f'{type(exc).__name__}: {exc}',**context)
        publish(env)
        raise
    publish(env)

if __name__=='__main__': main()
