"""All-layer representations and corrected tails, sharing GPU3 with retrieval."""
import datetime
import fcntl
import json
import os
import subprocess
from common import OBS,ROOT,atomic,read,sha
from resource_guard import baseline,supervise,request

def main():
    lock=open(OBS/'tmp/queue.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status={'status':'RUNNING','pid':os.getpid(),'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'tasks':[],
            'git_commit':subprocess.check_output(['git','-C',str(OBS),'rev-parse','HEAD'],text=True).strip(),
            'protocol_sha256':sha(OBS/'configs/observation_protocol.json'),
            'domain_inputs_sha256':sha(OBS/'manifests/domain_inputs.jsonl'),
            'entropy_inputs_sha256':sha(OBS/'manifests/entropy_inputs.jsonl')}
    path=OBS/'manifests/queue_status.json'
    def publish():
        status['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();atomic(path,status)
    def gpu(method,kind):
        run_id=method+'_'+kind;print('START',run_id,flush=True)
        result=supervise([str(OBS/'envs/forward/bin/python'),str(OBS/'src/extract_hidden_states.py'),'--method',method,'--kind',kind],baseline(),run_id)
        status['tasks'].append(result);publish();print(result,flush=True)
        if result['status']=='STOPPED_RETRIEVAL_PRIORITY':raise RuntimeError('Retrieval guard stopped new observation queue')
        return result['status']=='DONE'
    def cpu(script,*args):
        status['cpu_phase']=script;publish();print('CPU',script,flush=True)
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',XDG_CACHE_HOME=str(OBS/'cache/xdg'))
        subprocess.run([str(OBS/'envs/data/bin/python'),str(OBS/'src'/script),*args],env=env,check=True)
        status.pop('cpu_phase',None);publish()
    publish()
    try:
        assert read(OBS/'manifests/preflight.json')['status']=='PASS'
        if not gpu('P0','domains'):raise RuntimeError('BF16 representation reference failed')
        cpu('analyze_all_layers.py')
        if not gpu('P0','entropy'):raise RuntimeError('Corrected BF16 entropy reference failed')
        cpu('analyze_entropy_failures.py','--freeze-only')
        for method in ['P3','P4','P6','P5']:
            gpu(method,'domains')
            gpu(method,'entropy')
        cpu('analyze_all_layers.py','--quantization')
        cpu('analyze_entropy_failures.py')
        status['status']='COMPLETED' if all(t['status']=='DONE' for t in status['tasks']) else 'COMPLETED_WITH_LIMITATIONS'
    except BaseException as error:
        import traceback
        status.update(status='BLOCKED',reason=str(error),traceback=traceback.format_exc());print(status['traceback'],flush=True)
    finally:
        try:status['retrieval_health']=request()[0]
        except Exception as error:status['retrieval_health']={'error':str(error)}
        publish();print('QUEUE',status['status'],flush=True)

if __name__=='__main__':main()
