"""Wait for independent tests and I/O-bound preflight, then launch once.

This process never allocates CUDA. It does not retry a retrieval-protection stop.
"""
import datetime
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

OBS=Path(__file__).resolve().parents[1]

def main():
    import fcntl
    lock=open(OBS/'tmp/validation-launch.lock','w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_path=OBS/'manifests/validation_launcher.json'
    def state(value):
        value.update(pid=os.getpid(),updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        tmp=state_path.with_suffix('.partial');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(state_path)
    while True:
        missing=[]
        for name,required in [('data',6),('forward',10)]:
            path=OBS/'logs'/('tests-'+name+'.log')
            text=path.read_text() if path.exists() else ''
            if re.search(r'\d+ failed|ERROR collecting',text):
                state({'status':'BLOCKED_TEST_FAILURE','environment':name,'log':str(path)});return
            match=re.search(r'(\d+) passed',text)
            if not match or int(match.group(1))<required:missing.append(name+' tests')
        try:
            if json.loads((OBS/'manifests/preflight.json').read_text())['status']!='PASS':missing.append('preflight')
        except (OSError,ValueError,KeyError):missing.append('preflight')
        if not missing:break
        state({'status':'WAITING_VALIDATION','waiting_for':missing,'gpu_started':False})
        print('WAITING',', '.join(missing),flush=True);time.sleep(20)
    for key,value in [('user.name','haiyuntan'),('user.email','haiyuntan@users.noreply.github.com')]:
        subprocess.run(['git','-C',str(OBS),'config',key,value],check=True)
    subprocess.run(['git','-C',str(OBS),'add','src','tests','configs','manifests','reports','README.md','.gitignore'],check=True)
    subprocess.run(['git','-C',str(OBS),'commit','-m','fix: stable tail statistics and exploratory all-layer representation analysis'],check=True)
    env=os.environ.copy();env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPYCACHEPREFIX=str(OBS/'cache/pycache'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',RAYON_NUM_THREADS='2',HF_HOME=str(OBS/'cache/hf'),XDG_CACHE_HOME=str(OBS/'cache/xdg'),TMPDIR=str(OBS/'tmp'))
    command=[str(OBS/'envs/data/bin/python'),'-u',str(OBS/'src/run_corrected_observations.py')]
    with (OBS/'logs/observations.nohup.log').open('ab') as log:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,cwd=OBS,env=env,start_new_session=True,preexec_fn=lambda:signal.signal(signal.SIGHUP,signal.SIG_IGN))
    state({'status':'LAUNCHED','controller_pid':child.pid,'command':command})
    print('LAUNCHED',child.pid,flush=True)

if __name__=='__main__':main()
