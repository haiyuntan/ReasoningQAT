"""nohup sequential ABCD pilot, GPU3 only; retrieval has priority."""
import fcntl,json,os,signal,subprocess,sys,time,urllib.request,shutil
from common import *
active=None

def request(path,payload=None,timeout=5):
 req=urllib.request.Request('http://127.0.0.1:8080/'+path,data=None if payload is None else json.dumps(payload).encode(),headers={'Content-Type':'application/json','X-Observation-ID':'ABCD-v3-guard'})
 start=time.monotonic()
 with urllib.request.urlopen(req,timeout=timeout) as response:value=json.load(response)
 return value,time.monotonic()-start

def memory():
 raw=subprocess.check_output(['nvidia-smi','--id='+P['gpu_uuid'],'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
 host=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))/1024**2
 return int(raw.strip()),host

def stop_child():
 global active
 if active is not None:
  try:os.killpg(active.pid,signal.SIGTERM)
  except ProcessLookupError:pass
  try:active.wait(timeout=5)
  except subprocess.TimeoutExpired:
   os.killpg(active.pid,signal.SIGKILL);active.wait()
  active=None

def run(name,script,*args,gpu=False):
 global active
 receipt=W/'manifests'/f'{name}.json'
 if receipt.exists() and json.loads(receipt.read_text()).get('status')=='DONE':
  print('REUSE_COMPLETED',name,flush=True);return
 log=W/'logs'/f'{name}.log'
 if log.exists():
  log.rename(W/'logs'/f'{name}.before_throughput.{int(time.time())}.log')
 env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=P['gpu_uuid'] if gpu else '',
  HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',HF_HOME=str(W/'caches/hf'),
  XDG_CACHE_HOME=str(W/'caches/xdg'),TORCH_HOME=str(W/'caches/torch'),MPLCONFIGDIR=str(W/'caches/matplotlib'),
  TMPDIR=str(W/'tmp'),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 command=[str(W/'envs/forward/bin/python'),'-u',str(W/'src'/script),*args]
 if (W/'manifests/controller.json').exists() and json.loads((W/'manifests/controller.json').read_text()).get('pid')==os.getpid():
  save(W/'manifests/controller.json',{'pid':os.getpid(),'status':'RUNNING','stage':name,'protocol_sha':sha(W/'configs/protocol.json')})
 record(name,'RUNNING',command=command,gpu=gpu)
 with (W/'logs'/f'{name}.log').open('x') as stream:
  active=subprocess.Popen(command,cwd=W,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
  os.setpriority(os.PRIO_PROCESS,active.pid,10)
  slow=0;healthbad=0;lastprobe=time.monotonic()
  try:
   while active.poll() is None:
    time.sleep(2)
    if not gpu:continue
    health,elapsed=request('health')
    if health.get('status')!='ready' or health.get('adapter_loaded') is not True:raise RuntimeError('RETRIEVAL_NOT_READY')
    healthbad=healthbad+1 if elapsed>2 else 0
    if healthbad>=3:raise RuntimeError('RETRIEVAL_HEALTH_SLOW')
    free,host=memory()
    if free<resources()['gpu_free_reserve_mib'] or host<P['host_available_reserve_gib']:raise RuntimeError('RETRIEVAL_RESOURCE_RESERVE')
    probe=None
    if time.monotonic()-lastprobe>P['retrieval_probe_interval_s']:
     value,probe=request('retrieve_premises',{'query':'n : ℕ\n⊢ n = n','num':1},P['retrieval_latency_hard_s'])
     if not value.get('data'):raise RuntimeError('RETRIEVAL_EMPTY')
     slow=slow+1 if probe>P['retrieval_latency_soft_s'] else 0
     if probe>=P['retrieval_latency_hard_s'] or slow>=P['retrieval_consecutive_limit']:raise RuntimeError('RETRIEVAL_LATENCY_REGRESSION')
     lastprobe=time.monotonic()
    with (W/'logs/guard.jsonl').open('a') as f:f.write(json.dumps({'stage':name,'time':time.time(),'health_seconds':elapsed,'free_gpu_mib':free,'available_host_gib':host,'query_seconds':probe})+'\n')
   code=active.returncode
   if code:raise RuntimeError(f'{name} exited {code}; see its isolated log')
   record(name,'DONE',exit_code=code)
  finally:stop_child()

def main():
 lock=(W/'tmp/pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 def interrupt(sig,frame):raise KeyboardInterrupt(f'signal {sig}')
 signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
 save(W/'manifests/controller.json',{'pid':os.getpid(),'status':'STARTING','protocol_sha':sha(W/'configs/protocol.json')})
 try:
  # Environment copy is asynchronous; never use a partial copy for experiments.
  prep=json.loads((W/'manifests/environment_copy.json').read_text())
  while Path('/proc/'+str(prep['pid'])+'/stat').exists():
   stat=Path('/proc/'+str(prep['pid'])+'/stat').read_text().split(') ',1)[1]
   if stat.startswith('Z'):break
   time.sleep(5)
  if (W/'logs/env_copy.log').stat().st_size:raise RuntimeError('Environment copy emitted errors; inspect env_copy.log')
  # Reuse CPU-only audit packages from a matching Python3.11/NumPy1.26 observation prefix.
  if not (W/'manifests/environment.json').exists():
   source=ROOT/'observation_20261104_all_layers/envs/data/lib/python3.11/site-packages'
   destination=W/'envs/forward/lib/python3.11/site-packages'
   packages=('pyarrow','pandas','pandas.libs','scipy','scipy.libs','dateutil','pytz','tzdata','six.py')
   for name in packages:
    src=source/name;dst=destination/name
    if src.is_dir():shutil.copytree(src,dst,dirs_exist_ok=True)
    elif src.is_file():shutil.copy2(src,dst)
   env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES='')
   probe=subprocess.check_output([str(W/'envs/forward/bin/python'),'-c','import sys,torch,transformers,pyarrow,pandas,scipy,numpy; print(sys.prefix); print(torch.__version__); print(transformers.__version__); print(numpy.__version__)'],env=env,text=True)
   assert str(W/'envs/forward') in probe
   save(W/'manifests/environment.json',{'mode':'independent filesystem copy of prior observation conda prefix; production untouched','probe':probe,'source':str(ROOT/'observation_20261104_all_layers/envs/forward'),'audit_packages_copied_from_matching_observation_data_prefix':list(packages),'no_network_install':True})
  run('audit','audit.py')
  run('collect','collect.py')
  run('contract_tests','../tests/test_contract.py')
  frozen={str(p.relative_to(W)):sha(p) for p in [W/'configs/protocol.json',W/'manifests/V_real_states.jsonl',W/'manifests/discovery_pilot.jsonl',W/'manifests/models.json']}
  save(W/'manifests/frozen_inputs.json',frozen)
  (W/'manifests/frozen_protocol.sha256').write_text(digest(json.dumps(frozen,sort_keys=True))+'\n')
  for method in P['models']:
   run('forward_'+method,'score.py','--method',method,gpu=True)
   if method!='P0':run('replay_'+method,'replay.py','--student',method,gpu=True)
  run('analysis','analyze.py')
  record('campaign','PILOT_DONE',G5='NOT_COMPLETED',M='NOT_STARTED')
  save(W/'manifests/controller.json',{'pid':os.getpid(),'status':'PILOT_DONE','protocol_sha':sha(W/'configs/protocol.json')})
 except BaseException as error:
  stop_child();record('campaign','STOPPED' if isinstance(error,KeyboardInterrupt) else 'BLOCKED',error=repr(error))
  save(W/'manifests/controller.json',{'pid':os.getpid(),'status':'BLOCKED','error':repr(error)})
  raise
if __name__=='__main__':main()
