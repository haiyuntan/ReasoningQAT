"""Isolated serial queue. Retrieval regression halts observation only, never production."""
import datetime
import json
import os
import subprocess
import sys
import traceback
from common import OBS, ROOT, read, atomic, sha
from resource_guard import baseline,supervise,request

def cpu(script,*args):
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',RAYON_NUM_THREADS='2',XDG_CACHE_HOME=str(OBS/'cache/xdg'),HF_HOME=str(OBS/'cache/hf'),PYTHONNOUSERSITE='1',TMPDIR=str(OBS/'tmp'))
    subprocess.run([str(OBS/'envs/data/bin/python'),str(OBS/'src'/script),*args],env=env,check=True,cwd=OBS)

def main():
    import fcntl
    lock=open(OBS/'tmp/queue.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status={'status':'PREFLIGHT','pid':os.getpid(),'started':datetime.datetime.now(datetime.timezone.utc).isoformat(),'tasks':[]}
    target=OBS/'manifests/queue_status.json'
    def publish():atomic(target,status)
    publish()
    try:
        acquisition=read(OBS/'manifests/acquisition.json')
        assert len(acquisition)==11 and all(x['status']=='DONE' for x in acquisition.values())
        assert read(OBS/'manifests/selection_gate.json')['status']=='PASS'
        assert all(x['inventory']['ready'] for x in read(OBS/'manifests/checkpoint_identities.json').values())
        for name in ['data','forward']:
            assert (OBS/'configs'/('observation_env_'+name+'.yml')).exists()
        commit=subprocess.check_output(['git','-C',str(OBS),'rev-parse','HEAD'],text=True).strip()
        status.update(status='RUNNING',git_commit=commit,protocol_sha256=sha(OBS/'configs/observation_protocol.json'))
        publish()
        def run(method,kind='domains',smoke=False):
            run_id=method+'_'+('smoke' if smoke else kind)
            print('START',run_id,flush=True)
            b=baseline()
            cmd=[str(OBS/'envs/forward/bin/python'),str(OBS/'src/extract_hidden_states.py'),'--method',method,'--kind',kind]
            if smoke:cmd.append('--smoke')
            result=supervise(cmd,b,run_id)
            status['tasks'].append(result);publish();print(result,flush=True)
            if result['status']=='STOPPED_RETRIEVAL_PRIORITY':raise RuntimeError('Retrieval-priority circuit breaker: observation queue stopped, no automatic retry')
            return result['status']=='DONE'
        if not run('P0',smoke=True):raise RuntimeError('GPU smoke failed; no observation run authorized by gates')
        if not run('P0'):raise RuntimeError('P0 domains failed')
        cpu('analyze_domains.py')
        if not run('P0','entropy'):raise RuntimeError('P0 entropy failed')
        cpu('analyze_entropy_failures.py','--freeze-only')
        for method in ['P3','P4','P6','P5','P1','P2']:
            if run(method):run(method,'entropy')
            cpu('analyze_domains.py')
            cpu('analyze_entropy_failures.py')
        status['status']='COMPLETED_WITH_LIMITATIONS' if any(t['status']!='DONE' for t in status['tasks']) else 'PILOT_COMPLETED'
    except BaseException as error:
        status.update(status='BLOCKED',reason=str(error),traceback=traceback.format_exc())
        print(status['traceback'],flush=True)
    finally:
        try:
            health,seconds=request();status['final_retrieval_health']=health;status['health_latency_s']=seconds
        except Exception as error:status['final_retrieval_health']={'error':str(error)}
        status['production_status']=read(ROOT/'runs/realprover_formal_four_gpu_v12/control/status.json')['status']
        publish()
        (OBS/'reports/06_idea_decision_memo.md').write_text('# Idea decision memo\n\nRecommendation: CONDITIONAL.\n\n'+json.dumps({'observation_status':status['status'],'reason':status.get('reason'),'formal_evaluation':'WAITING_FINAL_EVAL','limitations':['Training-distribution observations only; source-theorem provenance incomplete','Figure pilot 128/domain; no main256 claim','Native-scale gradient diagnosis not executed on shared retrieval GPU','No new quantization, training, or test-set tuning','New W2 runtime compared with original SHA-verified frozen native reference; PTQ without reference remains blocked']},indent=2)+'\n\nRead actual effect sizes/CI in metrics/entropy_bootstrap.csv if available. No claim of winning method or new loss effectiveness.\n')
        print('QUEUE',status['status'],flush=True)

if __name__=='__main__':main()
