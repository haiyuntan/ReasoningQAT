import subprocess,time,signal,os,sys
from common import *
active=None
def health():
    import urllib.request
    start=time.monotonic()
    with urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=5) as r:h=json.load(r)
    assert h['status']=='ready' and h['adapter_loaded'] and h['asset_manifest_sha256']==sha(ROOT/'research/manifests/retrieval_assets.json')
    free=int(subprocess.check_output(['nvidia-smi','--id='+protocol()['GPU_uuid'],'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    assert free>=8192,'GPU_RESERVE_BREACHED'
    with (W/'logs/resource_guard.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),health_s=time.monotonic()-start,free_gpu_mib=free))+'\n')
def owned(pid):
    result={pid};parents={}
    for p in Path('/proc').glob('[0-9]*/stat'):
        try:parents[int(p.parent.name)]=int(p.read_text().split(') ',1)[1].split()[1])
        except (OSError,ValueError,IndexError):pass
    while True:
        more={x for x,y in parents.items() if y in result}
        if more<=result:return result
        result|=more
def stop():
    if active is None:return
    ids=owned(active.pid)
    for sig in [signal.SIGTERM,signal.SIGKILL]:
        for pid in ids:
            try:os.kill(pid,sig)
            except ProcessLookupError:pass
        try:active.wait(timeout=3)
        except subprocess.TimeoutExpired:pass
def stage(name,command,gpu=False):
    global active
    receipt=W/'manifests'/('stage_'+name+'.json')
    if receipt.exists() and read(receipt)['status']=='DONE':return
    env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=protocol()['GPU_uuid'] if gpu else '',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME=str(W/'caches/hf'),XDG_CACHE_HOME=str(W/'caches/xdg'),MPLCONFIGDIR=str(W/'caches/mpl'),TORCH_HOME=str(W/'caches/torch'),TMPDIR=str(W/'tmp'),
        CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',LEAN_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',WANDB_DISABLED='true')
    status('pipeline','RUNNING',current_stage=name);health()
    with (W/'logs'/(name+'.log')).open('a') as f:
        active=subprocess.Popen([str(W/'envs/dtr_forward/bin/python'),'-B','-u',*command],cwd=W,env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        save(receipt,dict(status='RUNNING',pid=active.pid,command=command,protocol_sha=identity()))
        try:
            last=time.monotonic()
            while active.poll() is None:
                time.sleep(2)
                if time.monotonic()-last>30:health();last=time.monotonic()
            code=active.returncode
            save(receipt,dict(status='DONE' if code==0 else 'FAILED',exit_code=code,command=command,protocol_sha=identity()))
            assert code==0,'STAGE_FAILED '+name
        finally:stop();active=None
def main():
    def term(sig,frame):raise KeyboardInterrupt(str(sig))
    signal.signal(signal.SIGTERM,term);signal.signal(signal.SIGINT,term)
    try:
        assert read(W/'manifests/environment_copy_done.json')['status']=='DONE'
        for n in ['hf','xdg','mpl','torch']:(W/'caches'/n).mkdir(exist_ok=True)
        stage('math_tests',[str(W/'tests/test_math.py')])
        stage('prepare',[str(W/'src/run.py'),'--mode','prepare'])
        stage('generate',[str(W/'src/run.py'),'--mode','generate'],True)
        stage('replay',[str(W/'src/run.py'),'--mode','replay'],True)
        stage('analyze',[str(W/'src/run.py'),'--mode','analyze'])
        stage('report',[str(W/'src/report.py')])
        status('pipeline','DONE',proof_search='NOT_RUN',training='NOT_RUN')
    except BaseException as e:stop();status('pipeline','BLOCKED' if not isinstance(e,KeyboardInterrupt) else 'STOPPED',error=repr(e));raise
if __name__=='__main__':main()
