"""CPU simulation of campaign ordering and formal failure/blocked gates."""
import json
from pathlib import Path
import sys
import tempfile
import background_campaign as campaign

ROOT=Path(__file__).resolve().parents[2]
checks=[]
class EndSimulation(BaseException): pass

def scenario(name,smoke_fails=False,blocked=False,data_waits=False):
    with tempfile.TemporaryDirectory(prefix='rq-campaign-',dir='/tmp') as folder:
        root=Path(folder);repo=root/'ReasoningQAT'
        for path in (root/'results',root/'logs',root/'outputs',repo): path.mkdir(parents=True,exist_ok=True)
        campaign.ROOT=root;campaign.REPO=repo;campaign.STATE=root/'results/state.json'
        commands=[];counters={'data':0,'sleep':0}
        runs=['R9','R8','R0','R1','R2','R3','R4','R4a','R5','R6_rr','R6_kl']
        def fake_audit(): return {'status':'BLOCKED' if blocked else 'READY','blockers':['test-only author config'] if blocked else []}
        def fake_run(command,env,log):
            commands.append(command)
            if '--mode' in command and '--session' in command:
                mode=command[command.index('--mode')+1]
                if mode=='smoke' and smoke_fails: raise RuntimeError('Simulated smoke failure')
                session=command[command.index('--session')+1]
                out=repo/'repro_results'/mode/session;out.mkdir(parents=True,exist_ok=True)
                for run in runs: (out/f'{run}.json').write_text(json.dumps({'status':'PASS'}))
            elif any(str(arg).endswith('download_status.py') for arg in command):
                counters['data']+=1
                complete=0 if data_waits and counters['data']==1 else 1
                (root/'results/download-status.json').write_text(json.dumps({'repositories':{'fixture':{'complete_files':complete,'required_files':1}}}))
            elif any(str(arg).endswith('audit_formal.py') for arg in command):
                (root/'results/formal-code-audit.json').write_text(json.dumps(fake_audit()))
        def fake_sleep(seconds):
            counters['sleep']+=1
            if blocked: raise EndSimulation()
        campaign.audit=fake_audit;campaign.run=fake_run;campaign.publish=lambda env:None
        campaign.time.sleep=fake_sleep;campaign.subprocess.check_output=lambda *args,**kwargs:''
        sys.argv=['campaign','--session','fixture']
        try: campaign.main()
        except (RuntimeError,EndSimulation): pass
        modes=[cmd[cmd.index('--mode')+1] for cmd in commands if '--mode' in cmd and '--session' in cmd]
        if smoke_fails:
            assert modes==['smoke']
            assert json.loads(campaign.STATE.read_text())['status']=='FAIL'
        elif blocked:
            assert modes==['smoke'] and counters['sleep']==1
            assert json.loads(campaign.STATE.read_text())['status']=='BLOCKED_FORMAL'
        else:
            assert modes==['smoke','formal']
            assert json.loads(campaign.STATE.read_text())['status']=='FORMAL_COMPLETE'
            if data_waits: assert counters['data']==2 and counters['sleep']==1
        checks.append({'case':name,'status':'PASS','simulated_modes':modes})

scenario('Smoke failure prevents formal',smoke_fails=True)
scenario('Unknown author config prevents formal',blocked=True)
scenario('Ready flow starts formal after smoke')
scenario('Incomplete data waits before formal',data_waits=True)
result={'status':'PASS','device':'CPU','meaning':'Controller simulations only; no real experiment results', 'checks':checks}
(ROOT/'results/background-campaign-check.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
