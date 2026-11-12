"""Resume the paused FATE-M campaign with shared retrieval and isolated workers."""
import datetime
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'adapters'))
from common import ROOT, load, save, sha, terminate_job
import retrieval_service


def main():
    campaign = ROOT / 'runs/realprover_fate_m_20261109'
    lock = (campaign / 'campaign_resume.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    original = load(campaign / 'status.json')
    old = Path(f"/proc/{original['pid']}/stat")
    if old.exists() and old.read_text().split(') ', 1)[1].split()[0] != 'T':
        raise RuntimeError('Original controller must remain stopped')
    for proc in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            command = proc.read_bytes()
        except OSError:
            continue
        if (b'experiment.run\0' in command or b'/adapters/evaluation.py\0' in command) and str(campaign).encode() in command:
            raise RuntimeError(f'Campaign evaluator already alive: {proc}')
    os.environ['REALPROVER_SHARED_RETRIEVAL'] = '1'
    retrieval_service.ready(timeout=30)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    old_env = load(Path(original['models_from']) / 'control/status.json')['lean_environment']
    state = {'status': 'RUNNING', 'pid': os.getpid(), 'started_at': stamp,
             'original_controller_paused': original['pid'], 'methods': {},
             'retrieval': 'EXISTING_SHARED_SERVICE_UNCHANGED'}
    active = {}

    def record():
        state['updated_at'] = time.time()
        save(campaign / 'campaign_resume_status.json', state)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for method, spec in original['methods'].items():
            directory = campaign / method
            archive = directory / 'history' / stamp
            archive.mkdir(parents=True)
            evidence = [{'path': str(p), 'sha256': sha(p)} for p in sorted(
                (directory / 'eval/native/generated').glob('*/*.json'))]
            save(archive / 'inherited_attempts.json', evidence)
            for name in ('retrieval.jsonl', 'evaluation.json'):
                path = directory / 'eval' / name
                if path.exists():
                    path.rename(archive / name)
            profile = load(directory / 'vllm_runtime.json')
            assert profile['gpu_memory_utilization'] == (.70 if spec['gpu'] == 3 else .85)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(spec['gpu']),
                REALPROVER_SHARED_RETRIEVAL='1', REALPROVER_OUTPUT=str(directory),
                REALPROVER_VLLM_RUNTIME=str(directory / 'vllm_runtime.json'),
                OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', USE_TF='0', USE_FLAX='0')
            env.update({k: v for k, v in old_env.items() if v})
            stream = (directory / f'evaluate.resume.{stamp}.log').open('x')
            process = subprocess.Popen(spec['command'], cwd=ROOT, env=env,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True)
            active[method] = process, stream, evidence
            state['methods'][method] = dict(status='RUNNING', worker_pid=process.pid,
                gpu=spec['gpu'], log=stream.name, inherited_attempts=len(evidence),
                archive=str(archive), engine_options=profile)
            record()
        while active:
            for method, (process, stream, evidence) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                terminate_job(process.pid)
                stream.close()
                unchanged = all(sha(item['path']) == item['sha256'] for item in evidence)
                state['methods'][method].update(status='DONE' if code == 0 and unchanged else 'FAILED',
                    exit_code=code, inherited_attempts_unchanged=unchanged)
                del active[method]
            record()
            if active:
                time.sleep(10)
        state['status'] = 'DONE' if all(s['status'] == 'DONE' for s in state['methods'].values()) else 'DONE_WITH_FAILURES'
    finally:
        for method, (process, stream, _) in active.items():
            terminate_job(process.pid)
            process.wait()
            stream.close()
            state['methods'][method]['status'] = 'STOPPED'
        if active:
            state['status'] = 'STOPPED'
        record()
        # Never signal the shared retrieval process, including on worker failures.


if __name__ == '__main__':
    main()
