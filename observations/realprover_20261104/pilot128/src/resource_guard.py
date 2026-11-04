"""Fail closed on retrieval regressions. Can terminate only its own child process.

Shared CUDA does NOT provide strict QoS. No claims of hardware isolation.
The health endpoint alone is insufficient: measure a sparse real query too.
"""
import json
import os
import signal
import subprocess
import time
import urllib.request
from common import OBS, ROOT, atomic, read

CAMPAIGN = ROOT/'runs/realprover_formal_four_gpu_v12'
DEFAULT_POLICY = {'latency_floor_s':10.,'latency_baseline_multiplier':3.,
                  'latency_soft_cap_s':15.,'latency_hard_limit_s':20.,
                  'latency_consecutive_limit':3,'health_limit_s':2.,
                  'health_consecutive_limit':3,'min_free_gpu_mib':16384,
                  'min_available_host_gib':64}

class LatencyWindow:
    def __init__(self, baseline_seconds, policy=None):
        self.policy={**DEFAULT_POLICY,**(policy or {})}
        self.soft=min(self.policy['latency_soft_cap_s'],max(self.policy['latency_floor_s'],baseline_seconds*self.policy['latency_baseline_multiplier']))
        self.consecutive=0

    def observe(self, seconds):
        self.consecutive=self.consecutive+1 if seconds>self.soft else 0
        if seconds>=self.policy['latency_hard_limit_s']:
            raise RuntimeError(f'Retrieval hard latency limit: {seconds:.3f}s')
        if self.consecutive>=self.policy['latency_consecutive_limit']:
            raise RuntimeError(f'Retrieval latency above {self.soft:.3f}s for {self.consecutive} consecutive probes')

def host_available_gib():
    from pathlib import Path
    lines=Path('/proc/meminfo').read_text().splitlines()
    return int(next(line.split()[1] for line in lines if line.startswith('MemAvailable:')))/1024**2

def request(path='health', payload=None, timeout=2):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request('http://127.0.0.1:8080/'+path,data=data,headers={'Content-Type':'application/json'})
    start = time.monotonic()
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=timeout) as response:
        value = json.load(response)
    return value, time.monotonic()-start

def probe(timeout=8):
    value, elapsed = request('retrieve_premises',{'query':['n : ℕ\n⊢ n = n'],'num':1},timeout)
    if not value.get('data'):
        raise RuntimeError('Retrieval query returned empty data')
    return elapsed

def gpu():
    text = subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=uuid,memory.used,memory.free','--format=csv,noheader,nounits'],text=True)
    uuid,used,free = [x.strip() for x in text.split(',')]
    return {'uuid':uuid,'used_mib':int(used),'free_mib':int(free)}

def tail_events():
    latest={}
    for method in ['P3','P4','P5','P6']:
        p=CAMPAIGN/'formal'/method/'eval/retrieval.jsonl'
        if not p.exists():continue
        with p.open('rb') as f:
            f.seek(max(0,p.stat().st_size-32768))
            lines=f.read().splitlines()
        for line in lines:
            try:e=json.loads(line)
            except json.JSONDecodeError:continue
            if e.get('success') is False:latest[method]=max(latest.get(method,0),e['time'])
    return latest

def supervise(command, baseline, run_id):
    """Baseline must be measured immediately before this child, on production GPU3."""
    lock = open(OBS/'tmp/gpu3.lock','w')
    __import__('fcntl').flock(lock,__import__('fcntl').LOCK_EX|__import__('fcntl').LOCK_NB)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=baseline['gpu']['uuid'],PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HOME=str(OBS/'cache/hf'),XDG_CACHE_HOME=str(OBS/'cache/xdg'),TMPDIR=str(OBS/'tmp'))
    log_path=OBS/'logs'/(run_id+'.log')
    started=time.time(); old_errors=tail_events(); last_probe=time.monotonic(); reason=None
    window=LatencyWindow(baseline['probe_p95'],baseline.get('policy'))
    policy=window.policy;health_bad=0
    with log_path.open('w') as log:
        child=subprocess.Popen(command,env=env,cwd=OBS,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        os.setpriority(os.PRIO_PROCESS,child.pid,15)
        atomic(OBS/'manifests/active_gpu_child.json',{'pid':child.pid,'command':command,'baseline':baseline,'started':started})
        try:
            while child.poll() is None:
                time.sleep(2)
                state=read(CAMPAIGN/'control/status.json')
                if state['status']!='RUNNING' or state['fingerprint']!=baseline['fingerprint']:
                    raise RuntimeError('Production state changed')
                health,latency=request(timeout=5)
                if health.get('status')!='ready':raise RuntimeError('Retrieval service not ready')
                health_bad=health_bad+1 if latency>policy['health_limit_s'] else 0
                if health_bad>=policy['health_consecutive_limit']:raise RuntimeError('Retrieval health persistently slow')
                memory=gpu()
                if memory['uuid']!=baseline['gpu']['uuid'] or memory['free_mib']<policy['min_free_gpu_mib']:
                    raise RuntimeError('GPU3 free-memory reserve breached')
                available=host_available_gib()
                if available<policy['min_available_host_gib']:raise RuntimeError('Host available-memory reserve breached')
                new_errors=tail_events()
                if any(v>old_errors.get(k,0) for k,v in new_errors.items()):raise RuntimeError('New production retrieval failure')
                elapsed=None
                if time.monotonic()-last_probe>=30:
                    elapsed=probe(timeout=policy['latency_hard_limit_s'])
                    window.observe(elapsed)
                    last_probe=time.monotonic()
                with (OBS/'logs'/('guard-'+run_id+'.jsonl')).open('a') as f:f.write(json.dumps({'time':time.time(),'gpu':memory,'host_available_gib':available,'health_s':latency,'probe_s':elapsed,'probe_soft_limit_s':window.soft,'consecutive_slow_probes':window.consecutive})+'\n')
        except BaseException as error:
            reason=str(error)
            if child.poll() is None:
                # This process group was created here; never signal production.
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait()
        result={'run_id':run_id,'status':'STOPPED_RETRIEVAL_PRIORITY' if reason else ('DONE' if child.returncode==0 else 'FAILED'),'reason':reason,'returncode':child.returncode,'elapsed_s':time.time()-started,'log':str(log_path)}
    atomic(OBS/'manifests'/(run_id+'.json'),result)
    return result

def baseline():
    times=[probe() for _ in range(5)]
    state=read(CAMPAIGN/'control/status.json')
    configured=read(OBS/'configs/observation_protocol.json')['resource_policy']
    result={'time':time.time(),'probe_seconds':times,'probe_p95':max(times),'gpu':gpu(),'fingerprint':state['fingerprint'],'policy':{key:configured.get(key,value) for key,value in DEFAULT_POLICY.items()}}
    if result['gpu']['free_mib']<48000:raise RuntimeError('Insufficient GPU3 headroom')
    atomic(OBS/'manifests/retrieval_baseline.json',result)
    return result
