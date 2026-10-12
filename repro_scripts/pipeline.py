"""Explicit R0–R6 pipeline; fails on missing evidence and requires GPU release."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
try:
    from .storage import inventory, cleanup_verified_run
except ImportError:
    from storage import inventory, cleanup_verified_run

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO.parent
MATRIX=json.loads((REPO/'repro_configs/matrix.json').read_text())
ORDER=['R0','R1','R2','R3','R4','R4a','R5','R6_rr','R6_kl']
TABLE2={'R0':65.2,'R1':4.2,'R2':9.5,'R4':28.4}
TABLE2_TASKS={'R0':[89.,53.6,74.7,38.4,70.4], 'R1':[2.1,0.,5.8,4.7,8.5],
              'R2':[0.,0.,27.3,8.1,12.3], 'R4':[48.6,6.5,40.1,14.5,32.2]}

def code_fingerprint():
    digest=hashlib.sha256()
    files=[REPO/'main_block_qat.py',REPO/'main_e2e_distill.py',REPO/'datautils_block.py',
           *sorted((REPO/'repro_scripts').glob('*.py')),*sorted((REPO/'repro_configs').glob('*.json')),
           *sorted((REPO/'quantize').rglob('*.py'))]
    for path in files:
        digest.update(str(path.relative_to(REPO)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()

def check_gpu_release():
    if os.environ.get('RQ_GPU_RELEASED')!='1':
        raise RuntimeError('GPU_NOT_RELEASED: set RQ_GPU_RELEASED=1 only after user authorizes GPU use')
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    if active:
        raise RuntimeError(f'GPU_BUSY: active PIDs {active}; no process will be stopped')

def check_protocol(mode):
    if mode=='formal':
        protocol=json.loads((REPO/'repro_configs/evaluation.json').read_text())
        if not protocol['confirmed_paper_protocol'] or protocol['unresolved']:
            raise RuntimeError('NOT_COMPARABLE: unresolved author settings: '+ '; '.join(protocol['unresolved']))

def train_command(run,mode,out,session):
    cfg={**MATRIX['formal'],**(MATRIX['smoke'] if mode=='smoke' else {})}
    python=str(ROOT/'env/train/bin/python')
    model=str(ROOT/'models/Qwen3-1.7B')
    logs=str(ROOT/'logs'/mode/session/run)
    if run in ('R1','R2'):
        python=str(ROOT/'env/gptq/bin/python') if run=='R1' else python
        script='ptq_gptq_w2.py' if run=='R1' else 'ptq_awq_w2.py'
        return [python,str(REPO/'repro_scripts'/script),'--model',model,'--output',str(out),
                '--samples',str(cfg['calibration_samples']),'--seqlen',str(cfg['ptq_sequence_length'])]
    if run=='R3':
        return [python,'main_block_qat.py','--model',model,'--net','Qwen3-1.7B','--wbits','2','--group_size','128',
                '--calib_dataset','sweep_0.8','--seed','2','--train_size',str(cfg['block_train_size']),
                '--val_size',str(cfg['block_val_size']),'--training_seqlen',str(cfg['block_sequence_length']),
                '--epochs',str(cfg['block_epochs']),'--batch_size',str(cfg['block_batch_size']),
                '--weight_lr',str(cfg['block_weight_lr']),'--quant_lr',str(cfg['block_quant_lr']),
                '--nblock','1','--max_memory','70GiB','--cache_dir',str(ROOT/'hf_cache/block_qat'/mode),
                '--save_quant_dir',str(out),'--output_dir',logs]
    if run in MATRIX['formal']['runs']:
        rcfg=MATRIX['formal']['runs'][run]
        args=[python,'main_e2e_distill.py','--model',str(out.parent/'R3'),'--teacher_model',model,
              '--wbits','2','--group_size','128','--dataset_type','openthoughts','--seed','2',
              '--dataset_size',str(cfg['e2e_dataset_size']),'--epochs',str(1 if mode=='smoke' else rcfg['epochs']),
              '--learning_rate',str(cfg['e2e_learning_rate']),'--optim','adamw_torch',
              '--lr_scheduler_type','cosine','--per_device_train_batch_size',str(cfg['e2e_micro_batch']),
              '--gradient_accumulation_steps',str(cfg['e2e_accumulation']),'--max_length',str(cfg['e2e_max_length']),
              '--cross_entropy_weight',str(rcfg['alpha']),'--kl_weight',str(rcfg['kl']),
              '--kd_loss_type','jsd','--top_k','20','--enable_efficient_qat',
              '--save_quant_dir',str(out),'--output_dir',logs]
        if rcfg['teacher_rr']:
            args+=['--use_teacher_weight']
        if mode=='smoke':
            args+=['--max_steps','2','--smoke_short_text']
        return args
    return None

def execute(command,log,environment):
    log.parent.mkdir(parents=True,exist_ok=True)
    print('EXECUTE',json.dumps(command),flush=True)
    with log.open('w') as stream:
        subprocess.run(command,cwd=REPO,env=environment,stdout=stream,stderr=subprocess.STDOUT,check=True,
                       timeout=int(os.environ.get('RQ_STAGE_TIMEOUT','604800')))

def gate(run,mode,metrics):
    if metrics['status']!='PASS':
        raise ValueError('Evaluation was not completed')
    if mode=='smoke':
        return {'status':'PASS','meaning':'Pipeline execution only; not paper reproduction'}
    if len(metrics['seeds'])!=3:
        raise ValueError('Formal evaluation requires three seeds')
    tolerance=2.
    if run in TABLE2:
        if abs(metrics['average_percent']-TABLE2[run])>tolerance:
            raise ValueError(f'Table 2 Avg outside fixed ±{tolerance}pp tolerance')
        refs=TABLE2_TASKS[run]
        for task,ref in zip(metrics['protocol']['datasets'],refs):
            if abs(metrics['means_percent'][task]-ref)>tolerance:
                raise ValueError(f'Table 2 task {task} outside fixed ±{tolerance}pp tolerance')
    if run in ('R4a','R5'):
        refs=[38.13,5.75] if run=='R4a' else [22.70,0.]
        for task,ref in zip(['math_500','live_code_bench'],refs):
            if abs(metrics['means_percent'][task]-ref)>tolerance:
                raise ValueError(f'Table 4 task {task} outside fixed tolerance')
    if run in ('R6_rr','R6_kl'):
        refs=[22.8,1.5,32.7,15.7,25.0] if run=='R6_rr' else [2.2,0.0,6.6,2.0,8.0]
        for task,ref in zip(metrics['protocol']['datasets'],refs):
            if abs(metrics['means_percent'][task]-ref)>tolerance:
                raise ValueError(f'Table 6 task {task} outside fixed tolerance')
    return {'status':'PASS','meaning':'Paper protocol and fixed score gates passed'}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['smoke','formal'],required=True)
    parser.add_argument('--runs',nargs='+',choices=ORDER,default=ORDER)
    parser.add_argument('--session',default=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    parser.add_argument('--dry-run',action='store_true')
    a=parser.parse_args()
    session_root=ROOT/'outputs'/a.mode/a.session
    compact=REPO/'repro_results'/a.mode/a.session
    for run in a.runs:
        command=train_command(run,a.mode,session_root/run,a.session)
        print(json.dumps({'run':run,'command':command or 'Five benchmark evaluation','output':str(session_root/run)},indent=2))
    if a.dry_run:
        return
    check_protocol(a.mode)
    check_gpu_release()
    compact.mkdir(parents=True,exist_ok=True)
    session_root.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'outputs/pipeline.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    fingerprint=code_fingerprint()
    env=os.environ.copy()
    env.update({'CUDA_VISIBLE_DEVICES':'0','HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1',
                'PYTHONPATH':str(REPO)+os.pathsep+str(ROOT/'ptq/llm-awq'),
                'RQ_ROOT':str(ROOT),'RQ_DATA':str(ROOT/'datasets'),
                'RQ_OPENTHOUGHTS_PARQUET_GLOB':str(ROOT/'datasets/openthoughts3/data/**/*.parquet'),
                'RQ_FINEWEB_PARQUET':str(ROOT/'datasets/fineweb-edu/sample/10BT/000_00000.parquet')})
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    for run in a.runs:
        summary_path=compact/f'{run}.json'
        if summary_path.exists():
            saved=json.loads(summary_path.read_text())
            if saved.get('status')=='PASS' and saved.get('code_fingerprint')==fingerprint:
                print('RESUME_SKIP',run)
                continue
            raise RuntimeError(f'Existing failed/stale run; choose a new session: {run}')
        dependency='R3' if run in MATRIX['formal']['runs'] else None
        if dependency:
            dep=json.loads((compact/f'{dependency}.json').read_text())
            if dep['status']!='PASS' or dep['code_fingerprint']!=fingerprint:
                raise RuntimeError('R3 dependency did not pass with the same code/configuration')
        check_gpu_release()
        out=session_root/run
        logdir=ROOT/'logs'/a.mode/a.session/run
        logdir.mkdir(parents=True,exist_ok=True)
        summary={'run':run,'mode':a.mode,'session':a.session,'git_commit':commit,'code_fingerprint':fingerprint,
                 'status':'RUNNING','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 'command':train_command(run,a.mode,out,a.session),'configuration':MATRIX[a.mode]}
        start=time.monotonic()
        server=None
        try:
            if summary['command']:
                execute(summary['command'],logdir/'train.console.log',env)
                assert (out/'config.json').exists(), 'Checkpoint not saved'
            if run in ('R0','R1','R2'):
                eval_model=ROOT/'models/Qwen3-1.7B' if run=='R0' else out
            else:
                eval_model=session_root/(run+'_hf')
                execute([str(ROOT/'env/train/bin/python'),str(REPO/'repro_scripts/export_qat.py'),
                         '--input',str(out),'--output',str(eval_model)],logdir/'export.console.log',env)
            port=18080
            server_python=ROOT/('env/gptq/bin/python' if run=='R1' else 'env/train/bin/python')
            server_command=[str(server_python),str(REPO/'repro_scripts/serve_model.py'),'--model',str(eval_model),
                            '--backend','gptq' if run=='R1' else 'hf','--port',str(port),
                            '--telemetry',str(logdir/'generation_telemetry.jsonl')]
            server_log=(logdir/'server.console.log').open('w')
            server=subprocess.Popen(server_command,cwd=REPO,env=env,stdout=server_log,stderr=subprocess.STDOUT)
            for _ in range(300):
                if server.poll() is not None:
                    raise RuntimeError('Generation server exited; inspect its log')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models',timeout=2) as response:
                        assert json.load(response)['data'][0]['id']=='rq-model'
                    break
                except (OSError,AssertionError):
                    time.sleep(2)
            else:
                raise TimeoutError('Generation server startup timeout')
            evaldir=session_root/(run+'_eval')
            cpu_env={**env,'CUDA_VISIBLE_DEVICES':''}
            execute([str(ROOT/'env/eval/bin/python'),str(REPO/'repro_scripts/eval_paper_5tasks.py'),
                     '--mode',a.mode,'--output',str(evaldir),'--port',str(port)],logdir/'eval.console.log',cpu_env)
            metrics=json.loads((evaldir/'metrics.json').read_text())
            summary['metrics']=metrics
            summary['gate']=gate(run,a.mode,metrics)
            summary['status']='PASS'
            telemetry=[json.loads(line) for line in (logdir/'generation_telemetry.jsonl').read_text().splitlines()]
            summary['peak_generation_memory_bytes']=max(row['peak_memory_bytes'] for row in telemetry)
        except Exception as exc:
            summary['status']='FAIL'
            summary['error']=f'{type(exc).__name__}: {exc}'
            raise
        finally:
            if server is not None:
                server.terminate()
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
                server_log.close()
            summary['wall_seconds']=time.monotonic()-start
            summary['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
            paths=[out,session_root/(run+'_hf'),session_root/(run+'_eval'),logdir]
            summary['storage_before_cleanup']=inventory(paths)
            if summary['status']=='PASS':
                try:
                    summary['cleanup']=cleanup_verified_run(run,a.mode,out,eval_model,logdir,compact)
                except Exception as exc:
                    summary['cleanup_error']=f'{type(exc).__name__}: {exc}'
            summary['storage_after_cleanup']=inventory(paths)
            summary_path.write_text(json.dumps(summary,indent=2))
            if os.environ.get('RQ_AUTO_PUBLISH','1')=='1':
                publish_env=os.environ.copy()
                publish_env['GIT_TERMINAL_PROMPT']='0'
                if Path('/tmp/rq-git-askpass.py').exists():
                    publish_env.setdefault('GIT_ASKPASS','/tmp/rq-git-askpass.py')
                try:
                    subprocess.run([str(ROOT/'env/bootstrap/bin/python'),
                                    str(REPO/'repro_scripts/publish_results.py'),'--commit','--push'],
                                   cwd=REPO,env=publish_env,check=True,timeout=180)
                except (subprocess.SubprocessError,OSError) as exc:
                    summary['publish_error']=str(exc)
                    summary_path.write_text(json.dumps(summary,indent=2))
                    print('RESULT_PUBLICATION_FAILED',run,str(exc),flush=True)
    print('PIPELINE_COMPLETE',a.mode,a.session)

if __name__=='__main__':
    main()
