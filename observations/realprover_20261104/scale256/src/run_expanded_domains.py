"""Domain-only 256/group expansion; immutable pilot and formal evaluation inputs."""
import datetime
import fcntl
import json
import os
import subprocess
from common import OBS, ROOT, read, atomic, sha
from resource_guard import baseline, supervise, request

def main():
    lock=open(OBS/'tmp/queue.lock','w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status={'status':'RUNNING','pid':os.getpid(),'started':datetime.datetime.now(datetime.timezone.utc).isoformat(),'tasks':[],
            'git_commit':subprocess.check_output(['git','-C',str(OBS),'rev-parse','HEAD'],text=True).strip(),
            'protocol_sha256':sha(OBS/'configs/observation_protocol.json'),
            'inputs_sha256':sha(OBS/'manifests/domain_inputs.jsonl')}
    target=OBS/'manifests/queue_status.json'
    atomic(target,status)
    try:
        audit=read(OBS/'manifests/expansion_validation.json')
        assert audit['status']=='PASS' and audit['samples']==1024
        for method in ['P0','P3','P4','P6','P5']:
            print('START',method,'domains 1024',flush=True)
            result=supervise([str(OBS/'envs/forward/bin/python'),str(OBS/'src/extract_hidden_states.py'),'--method',method,'--kind','domains'],baseline(),method+'_domains')
            status['tasks'].append(result);atomic(target,status)
            print(result,flush=True)
            if result['status']=='STOPPED_RETRIEVAL_PRIORITY':raise RuntimeError('Retrieval priority guard stopped expansion')
            if method=='P0' and result['status']!='DONE':raise RuntimeError('BF16 reference failed')
            if result['status']=='DONE':
                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',XDG_CACHE_HOME=str(OBS/'cache/xdg'))
                subprocess.run([str(OBS/'envs/data/bin/python'),str(OBS/'src/analyze_domains.py')],env=env,check=True)
        status['status']='COMPLETED' if all(t['status']=='DONE' for t in status['tasks']) else 'COMPLETED_WITH_LIMITATIONS'
    except BaseException as error:
        import traceback
        status.update(status='BLOCKED',reason=str(error),traceback=traceback.format_exc())
        print(status['traceback'],flush=True)
    finally:
        status['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        try:status['retrieval_health']=request()[0]
        except Exception as error:status['retrieval_health']={'error':str(error)}
        atomic(target,status)
        print('QUEUE',status['status'],flush=True)

if __name__=='__main__':main()
