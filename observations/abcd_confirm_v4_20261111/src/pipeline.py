"""Protected resumable confirmation pipeline; never signals production processes."""
import fcntl,os,signal,subprocess,sys,urllib.request,shutil
from common import *
active=None
class RetrievalPriorityStop(RuntimeError):pass

def request(path,payload=None,timeout=5):
 req=urllib.request.Request('http://127.0.0.1:8080/'+path,data=None if payload is None else json.dumps(payload).encode(),headers={'Content-Type':'application/json','X-Observation-ID':'ABCD-confirm-guard'})
 start=time.monotonic()
 with urllib.request.urlopen(req,timeout=timeout) as r:value=json.load(r)
 return value,time.monotonic()-start

def resources():
 text=subprocess.check_output(['nvidia-smi','--id='+P['GPU']['uuid'],'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
 host=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))/1024**2
 return int(text.strip()),host

def production_errors():
 result={}
 for m in ['P3','P4','P5']:
  path=ROOT/'runs/realprover_fate_m_20261109'/m/'eval/retrieval.jsonl'
  if not path.exists():continue
  with path.open('rb') as f:f.seek(max(0,path.stat().st_size-16384));data=f.read().splitlines()
  for line in data:
   try:r=json.loads(line)
   except ValueError:continue
   if r.get('success') is False:result[m]=max(result.get(m,0),r['time'])
 return result

def owned_processes(pid):
 table={}
 for path in Path('/proc').glob('[0-9]*/stat'):
  try:
   fields=path.read_text().rsplit(') ',1)[1].split();table[int(path.parent.name)]=(int(fields[1]),int(fields[2]))
  except (OSError,ValueError,IndexError):continue
 owned={pid}|{p for p,(_,group) in table.items() if group==pid}
 while True:
  expanded=owned|{p for p,(parent,_) in table.items() if parent in owned}
  if expanded==owned:return owned
  owned=expanded

def stop():
 global active
 if active:
  owned=owned_processes(active.pid)
  # Lean launches independent sessions. Reap our descendants, never production.
  for sig in (signal.SIGTERM,signal.SIGKILL):
   for pid in sorted(owned,reverse=True):
    try:os.kill(pid,sig)
    except ProcessLookupError:pass
   if sig==signal.SIGTERM:
    try:active.wait(timeout=2)
    except subprocess.TimeoutExpired:pass
  active.wait();active=None

def write_controller(state):
 state=dict(state,updated_at=time.time())
 try:save(W/'manifests'/os.environ.get('ABCD_STATUS_FILE','controller.json'),state)
 except OSError as error:print('WARNING shared controller status unavailable; worker preserved:',error,flush=True)
 mirror=Path('/tmp')/('abcd-confirm-'+digest(str(W)+os.environ.get('ABCD_STATUS_FILE','controller.json'))[:16]+'.json')
 try:mirror.write_text(json.dumps(state))
 except OSError as error:print('WARNING local status unavailable',error,flush=True)

def check_frozen():
 if not (W/'manifests/frozen_identity.json').exists():return
 for name,expected in read(W/'manifests/frozen_identity.json').items():
  actual=protocol_sha() if name=='protocol' else sha(W/'manifests'/name)
  assert actual==expected,'Frozen input changed: '+name

def run(name,script,*args,gpu=False,required=False):
 global active
 receipt=W/'manifests'/f'{name}.json'
 if receipt.exists() and read(receipt)['status']=='DONE':
  check_frozen();print('RESUME_SKIP',name,flush=True);return True
 check_frozen();path=W/'logs'/f'{name}.log'
 if path.exists():path.rename(W/'logs'/f'{name}.prior.{int(time.time())}.log')
 env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=P['GPU']['uuid'] if gpu else '',
  HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME=str(W/'caches/hf'),XDG_CACHE_HOME=str(W/'caches/xdg'),
  TORCH_HOME=str(W/'caches/torch'),MPLCONFIGDIR=str(W/'caches/matplotlib'),TMPDIR=str(W/'tmp'),TOKENIZERS_PARALLELISM='false',
  CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',LEAN_NUM_THREADS='2')
 command=[str(W/'envs/forward/bin/python'),'-u',str(W/'src'/script),*args]
 baseline_errors=production_errors();slow=0;bad=0;lastprobe=time.monotonic();reason=None
 status(name,'RUNNING',command=command,gpu=gpu,protocol_sha=protocol_sha());write_controller({'pid':os.getpid(),'status':'RUNNING','stage':name})
 with path.open('x') as log:
  active=subprocess.Popen(command,cwd=W,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  os.setpriority(os.PRIO_PROCESS,active.pid,10)
  try:
   while active.poll() is None:
    time.sleep(2)
    if not gpu:continue
    h,elapsed=request('health');assert h['status']=='ready' and h['adapter_loaded']
    if h['asset_manifest_sha256']!=sha(ROOT/'research/manifests/retrieval_assets.json'):raise RetrievalPriorityStop('Retrieval identity changed')
    bad=bad+1 if elapsed>2 else 0
    if bad>=3:raise RetrievalPriorityStop('Health persistently slow')
    free,host=resources()
    if free<P['GPU']['free_reserve_mib'] or host<P['GPU']['host_reserve_gib']:raise RetrievalPriorityStop('Resource reserve breached')
    probe=None
    if time.monotonic()-lastprobe>30:
     try:result,probe=request('retrieve_premises',{'query':'n : ℕ\n⊢ n = n','num':1},timeout=P['GPU']['latency_hard_s'])
     except Exception as e:raise RetrievalPriorityStop('Retrieval probe failed: '+repr(e)) from e
     if not result.get('data'):raise RetrievalPriorityStop('Empty retrieval response')
     slow=slow+1 if probe>P['GPU']['latency_soft_s'] else 0
     if probe>=P['GPU']['latency_hard_s'] or slow>=P['GPU']['slow_limit']:raise RetrievalPriorityStop('Retrieval latency regression')
     if any(t>baseline_errors.get(k,0) for k,t in production_errors().items()):raise RetrievalPriorityStop('New production retrieval failure')
     lastprobe=time.monotonic()
    with (W/'logs/guard.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),'stage':name,'health_s':elapsed,'probe_s':probe,'free_gpu_mib':free,'host_available_gib':host})+'\n')
   code=active.returncode
   if code:
    status(name,'FAILED',exit_code=code,log=str(path))
    if required:raise RuntimeError('Required stage failed: '+name)
    return False
   status(name,'DONE',exit_code=0);return True
  finally:stop()

def environment():
 meta=read(W/'manifests/environment_copy.json')
 while Path('/proc/'+str(meta['pid'])+'/stat').exists():
  if Path('/proc/'+str(meta['pid'])+'/stat').read_text().split(') ',1)[1].startswith('Z'):break
  time.sleep(5)
 assert not (W/'logs/environment_copy.log').read_text(),'Environment copy errors'
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES='')
 output=subprocess.check_output([str(W/'envs/forward/bin/python'),'-c','import sys,torch,transformers,pyarrow,scipy,pandas,matplotlib; print(sys.prefix); print(torch.__version__); print(transformers.__version__); print("matplotlib="+matplotlib.__version__)'],env=env,text=True)
 assert str(W/'envs/forward') in output
 save(W/'manifests/environment.json',{'source':meta['source'],'mode':'independent file copy with isolated plotting dependency install','probe':output})

def main():
 lock=(W/'tmp/pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 def interrupt(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
 signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
 write_controller({'pid':os.getpid(),'status':'PREPARING_ENVIRONMENT'})
 try:
  environment()
  run('prepare','prepare.py',required=True)
  run('collect','collect.py',required=True)
  run('contracts','../tests/test_contract.py',required=True)
  run('score_P0','worker.py','--method','P0',gpu=True,required=True)
  failed=[]
  for method in P['models'][1:]:
   if not run('score_'+method,'worker.py','--method',method,gpu=True):failed.append(method);continue
   if not run('replay_'+method,'replay.py','--method',method,gpu=True):failed.append(method+'_replay')
  run('analysis','analyze.py',required=True)
  run('plots','plots.py',required=True)
  # Teacher caches can be reclaimed only when every model replay succeeded.
  reclaimed=0
  if not failed:
   for p in (W/'caches/D/P0').glob('*.pt'):reclaimed+=p.stat().st_size;p.unlink()
  status('campaign','CONFIRMATION_DONE' if not failed else 'DONE_WITH_FAILURES',failed=failed,teacher_cache_reclaimed_bytes=reclaimed,M='NOT_STARTED')
  write_controller({'pid':os.getpid(),'status':'CONFIRMATION_DONE' if not failed else 'DONE_WITH_FAILURES','failed':failed})
 except BaseException as error:
  stop();state='STOPPED_RETRIEVAL_PRIORITY' if isinstance(error,RetrievalPriorityStop) else ('STOPPED' if isinstance(error,KeyboardInterrupt) else 'BLOCKED')
  status('campaign',state,error=repr(error));write_controller({'pid':os.getpid(),'status':state,'error':repr(error)});raise
if __name__=='__main__':main()
