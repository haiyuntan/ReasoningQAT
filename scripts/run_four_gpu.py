"""P1-P6: four independent preparation GPUs, then three evaluators + shared retrieval."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from experiment_pipeline import ROOT, RESEARCH, fingerprint, formal_smoke_gate, validate_receipt
sys.path.insert(0, str(RESEARCH / 'adapters'))
from common import load, save

RUNS = ['P1', 'P2', 'P3', 'P5', 'P4', 'P6']


def eligible(run, completed):
    return run not in ('P4', 'P6') or 'P3' in completed


def terminate_tree(pid):
    """Include Lean's independent process groups when cancelling our own children."""
    children = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().split(') ', 1)[1].split()
            children[int(path.parent.name)] = int(fields[1])
        except (OSError, ValueError, IndexError):
            pass
    owned = {pid}
    while True:
        expanded = owned | {p for p, parent in children.items() if parent in owned}
        if expanded == owned:
            break
        owned = expanded
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for child in sorted(owned, reverse=True):
            try:
                os.kill(child, sig)
            except ProcessLookupError:
                pass
        if sig == signal.SIGTERM:
            time.sleep(2)


class Campaign:
    def __init__(self, results, digest):
        self.results, self.digest = results, digest
        self.active = {}
        self.state = {'status': 'STARTING', 'pid': os.getpid(), 'fingerprint': digest,
                      'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      'P0': 'EXCLUDED_BY_USER', 'methods': {}, 'retrieval_gpu': 3,
                      'commits': {name: subprocess.check_output(
                          ['git', '-C', str(ROOT / name), 'rev-parse', 'HEAD'], text=True).strip()
                          for name in ('research', 'REALPROVER')},
                      'lean_environment': {key: os.environ.get(key) for key in
                          ('REALPROVER_LEAN_WORKSPACE', 'REALPROVER_INTERACTIVE_ROOT',
                           'REALPROVER_JIXIA_BIN', 'REALPROVER_LEAN_BIN')},
                      'engine_options': load(ROOT / 'REALPROVER/configs/vllm_runtime.json')}

    def record(self):
        self.state['updated_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save(self.results / 'control/status.json', self.state)

    def phase(self, phase, gpus):
        pending, completed = list(RUNS), set()
        self.state['status'] = phase.upper() + '_RUNNING'
        while pending or self.active:
            if fingerprint() != self.digest:
                raise RuntimeError('Code/config changed during campaign')
            for gpu, (run, process, stream) in list(self.active.items()):
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                del self.active[gpu]
                if code:
                    raise RuntimeError(f'{run} {phase} exited {code}; inspect {phase}.log')
                out = self.results / 'formal' / run
                if phase == 'evaluate':
                    validate_receipt(load(out / 'receipt.json'), run, 'formal', self.digest)
                else:
                    prepared = load(out / 'prepared.json')
                    if (any(prepared.get(k) != v for k, v in
                            {'run': run, 'mode': 'formal', 'fingerprint': self.digest}.items())
                            or not prepared.get('evidence')):
                        raise RuntimeError(f'{run}: preparation failed identity check')
                completed.add(run)
                self.state['methods'][run].update(phase=phase + '_DONE', exit_code=0)
                print(f'DONE {run} {phase} GPU {gpu}', flush=True)
            for gpu in gpus:
                if gpu in self.active:
                    continue
                ready = [r for r in pending if phase == 'evaluate' or eligible(r, completed)]
                if not ready:
                    continue
                run = ready[0]
                pending.remove(run)
                out = self.results / 'formal' / run
                out.mkdir(parents=True, exist_ok=True)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), REALPROVER_RUN=run,
                           REALPROVER_MODE='formal', REALPROVER_OUTPUT=str(out),
                           REALPROVER_FINGERPRINT=self.digest, REALPROVER_SHARED_RETRIEVAL='1',
                           OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                           USE_TF='0', USE_FLAX='0', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
                command = [sys.executable, '-u', str(RESEARCH / 'adapters/run_experiment.py'),
                           '--run', run, '--mode', 'formal', '--phase', phase]
                stream = (out / f'{phase}.log').open('x')
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                           stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                self.active[gpu] = (run, process, stream)
                save(out / f'{phase}_invocation.json', {'command': command, 'gpu': gpu,
                     'pid': process.pid, 'fingerprint': self.digest})
                self.state['methods'][run] = {'phase': phase, 'gpu': gpu, 'pid': process.pid}
                print(f'START {run} {phase} GPU {gpu} PID {process.pid}', flush=True)
            self.record()
            if pending or self.active:
                time.sleep(10)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', type=Path, required=True)
    ap.add_argument('--smoke-waiver', type=Path, required=True)
    args = ap.parse_args()
    results = args.results.resolve()
    digest = fingerprint()
    formal_smoke_gate(results, digest, args.smoke_waiver)
    if (results / 'formal').exists():
        raise RuntimeError('Refusing existing formal outputs; use a new batch')
    locks = []
    for name in ('realprover_autorun.lock', 'realprover_pipeline.lock'):
        lock = (ROOT / 'runs' / name).open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
    campaign = Campaign(results, digest)
    # Signals use the same cleanup path as an adapter failure.
    def interrupted(signum, frame):
        raise RuntimeError(f'Campaign interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    retrieval_started = False
    try:
        import retrieval_service
        retrieval_service.stop()
        campaign.phase('prepare', [0, 1, 2, 3])
        os.environ['CUDA_VISIBLE_DEVICES'] = '3'
        # Only this controller owns shared service lifecycle; clients just check health.
        retrieval_started = True
        retrieval_service.ready()
        campaign.phase('evaluate', [0, 1, 2])
        campaign.state['status'] = 'PASS'
    except BaseException as exc:
        campaign.state.update(status='FAILED', error=str(exc))
        raise
    finally:
        for run, process, stream in campaign.active.values():
            terminate_tree(process.pid)
            process.wait()
            stream.close()
        if retrieval_started:
            retrieval_service.stop()
        campaign.record()


if __name__ == '__main__':
    main()
