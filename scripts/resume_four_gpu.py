"""Independent failure domains and adaptive GPU scheduling for audited P1-P6 restarts."""
import argparse
import datetime
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from run_four_gpu import Campaign, terminate_tree
from experiment_pipeline import ROOT, RESEARCH, fingerprint, formal_smoke_gate, validate_receipt
from common import load, save, sha, data_environment, python_for
import retrieval_service

RUNS = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']


def prepared(out, run, digest):
    record = load(out / 'prepared.json')
    if any(record.get(k) != v for k, v in {'run': run, 'mode': 'formal', 'fingerprint': digest}.items()):
        raise RuntimeError(f'{run}: preparation identity mismatch')
    if not record.get('evidence'):
        raise RuntimeError(f'{run}: missing artifact hashes')
    for item in record['evidence']:
        if sha(Path(item['path'])) != item['sha256']:
            raise RuntimeError(f'{run}: prepared artifact changed: {item["path"]}')


class Adaptive(Campaign):
    def __init__(self, results, digest, training_first=True):
        super().__init__(results, digest)
        self.training_first = training_first
        self.retrieval_gpu = None
        self.cpu = None
        self.jobs = {run: {'prepare': 'PENDING', 'evaluate': 'PENDING'} for run in RUNS}
        self.state.update(status='RUNNING', policy='independent failures; training first' if training_first else 'independent failures; mixed training/evaluation',
                          methods=self.jobs, retrieval_gpu=None, data_preflight='PENDING')

    def fail(self, run, phase, error):
        self.jobs[run][phase] = 'FAILED'
        self.jobs[run].setdefault('errors', []).append({'phase': phase, 'error': str(error)})
        if phase == 'prepare':
            self.jobs[run]['evaluate'] = 'BLOCKED'
        print(f'FAILED {run} {phase}: {error}; independent jobs continue', flush=True)

    def poll_jobs(self):
        for gpu, job in list(self.active.items()):
            run, phase, process, stream = job
            code = process.poll()
            if code is None:
                continue
            stream.close()
            del self.active[gpu]
            try:
                if code:
                    raise RuntimeError(f'exit {code}; see {self.jobs[run]["log"]}')
                out = self.results / 'formal' / run
                if phase == 'prepare':
                    prepared(out, run, self.digest)
                else:
                    validate_receipt(load(out / 'receipt.json'), run, 'formal', self.digest)
                self.jobs[run][phase] = 'DONE'
                print(f'DONE {run} {phase} GPU {gpu}', flush=True)
            except Exception as error:
                self.fail(run, phase, error)

    def start_job(self, run, phase, gpu):
        out = self.results / 'formal' / run
        out.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        log = out / f'{phase}.{stamp}.log'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), REALPROVER_RUN=run,
                   REALPROVER_MODE='formal', REALPROVER_OUTPUT=str(out),
                   REALPROVER_FINGERPRINT=self.digest, REALPROVER_SHARED_RETRIEVAL='1',
                   REALPROVER_STAGE2_CACHE=str(self.results / 'stage2_cache'),
                   OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                   USE_TF='0', USE_FLAX='0', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
        command = [sys.executable, '-u', str(RESEARCH / 'adapters/run_experiment.py'),
                   '--run', run, '--mode', 'formal', '--phase', phase]
        stream = log.open('x')
        try:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        except Exception:
            stream.close()
            raise
        self.active[gpu] = (run, phase, process, stream)
        self.jobs[run].update({phase: 'RUNNING', 'gpu': gpu, 'pid': process.pid, 'log': str(log)})
        save(out / f'{phase}.{stamp}.invocation.json', {'command': command, 'gpu': gpu,
             'pid': process.pid, 'fingerprint': self.digest})
        print(f'START {run} {phase} GPU {gpu} PID {process.pid}', flush=True)

    def start_data(self):
        path = self.results / 'stage2_cache'
        if (path / 'manifest.json').exists():
            self.state['data_preflight'] = 'DONE'
            return
        log = self.results / 'control/stage2_data.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        stream = log.open('a')
        env = data_environment('formal')
        env['CUDA_VISIBLE_DEVICES'] = ''
        env.pop('REALPROVER_STAGE2_CACHE', None)
        external = os.environ.get('REALPROVER_EXISTING_STAGE2_PREFLIGHT_PID')
        command = ([sys.executable, '-u', str(RESEARCH / 'scripts/wait_stage2_cache.py'),
                    '--pid', external, '--cache', str(path)] if external else
                   [python_for('train'), '-u', str(RESEARCH / 'scripts/prepare_stage2_cache.py'), '--output', str(path)])
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        self.cpu = process, stream
        self.state.update(data_preflight='RUNNING', data_preflight_pid=process.pid)

    def poll_data(self):
        if self.cpu is None:
            return
        process, stream = self.cpu
        code = process.poll()
        if code is None:
            return
        stream.close()
        self.cpu = None
        good = not code and (self.results / 'stage2_cache/manifest.json').exists()
        self.state['data_preflight'] = 'DONE' if good else 'FAILED'
        if not good:
            for run in ('P4', 'P5', 'P6'):
                if self.jobs[run]['prepare'] == 'PENDING':
                    self.fail(run, 'prepare', f'Stage2 data preflight failed (exit={code}); see control/stage2_data.log')

    def pending_evaluations(self):
        if self.training_first and any(v['prepare'] in ('PENDING', 'RUNNING') for v in self.jobs.values()):
            return []
        return [r for r in RUNS if self.jobs[r]['prepare'] == 'DONE' and self.jobs[r]['evaluate'] == 'PENDING']

    def pending_preparations(self):
        if self.state['data_preflight'] != 'DONE':
            return []
        return [r for r in ('P4', 'P6', 'P5') if self.jobs[r]['prepare'] == 'PENDING'
                and (r == 'P5' or self.jobs['P3']['prepare'] == 'DONE')]

    def schedule(self):
        free = [g for g in range(4) if g not in self.active and g != self.retrieval_gpu]
        evaluations = self.pending_evaluations()
        evaluating = any(job[1] == 'evaluate' for job in self.active.values())
        if self.retrieval_gpu is not None and not evaluations and not evaluating:
            retrieval_service.stop()
            free.append(self.retrieval_gpu)
            self.retrieval_gpu = None
        if self.retrieval_gpu is None and evaluations and len(free) >= 2:
            gpu = max(free)
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
            try:
                retrieval_service.ready()
                self.retrieval_gpu = gpu
                free.remove(gpu)
            except Exception as error:
                retrieval_service.stop()
                for run in evaluations:
                    self.fail(run, 'evaluate', f'Shared retrieval startup: {error}')
                evaluations = []
        # At least one evaluator while ready models exist. Fill other slots
        # with training; no global preparation barrier. P6 waits only for a slot.
        preparations = self.pending_preparations()
        while free:
            evaluating = any(job[1] == 'evaluate' for job in self.active.values())
            if evaluations and self.retrieval_gpu is not None and (not evaluating or not preparations):
                run, phase = evaluations.pop(0), 'evaluate'
            elif preparations:
                run, phase = preparations.pop(0), 'prepare'
            elif evaluations and self.retrieval_gpu is not None:
                run, phase = evaluations.pop(0), 'evaluate'
            else:
                break
            gpu = min(free)
            free.remove(gpu)
            try:
                self.start_job(run, phase, gpu)
            except Exception as error:
                self.fail(run, phase, error)
        self.state['retrieval_gpu'] = self.retrieval_gpu

    def run(self):
        for run in RUNS:
            out = self.results / 'formal' / run
            if (out / 'prepared.json').exists():
                prepared(out, run, self.digest)
                self.jobs[run]['prepare'] = 'DONE'
            if (out / 'receipt.json').exists():
                validate_receipt(load(out / 'receipt.json'), run, 'formal', self.digest)
                self.jobs[run]['evaluate'] = 'DONE'
        if any(self.jobs[run]['prepare'] != 'DONE' for run in ('P1', 'P2', 'P3')):
            raise RuntimeError('Audited P1/P2/P3 preparation imports required before this continuation')
        self.start_data()
        while True:
            if fingerprint() != self.digest:
                raise RuntimeError('Global code/config fingerprint changed; refusing mixed results')
            self.poll_jobs()
            self.poll_data()
            self.schedule()
            # A failed preparation does not block independent preparations or
            # evaluations; dependent jobs alone are marked blocked.
            for run in ('P4', 'P6'):
                if self.jobs['P3']['prepare'] == 'FAILED' and self.jobs[run]['prepare'] == 'PENDING':
                    self.fail(run, 'prepare', 'P3 preparation unavailable')
            self.record()
            pending = any(v[p] in ('PENDING', 'RUNNING') for v in self.jobs.values() for p in ('prepare', 'evaluate'))
            if not pending and not self.active and self.cpu is None:
                break
            time.sleep(10)
        self.state['status'] = 'PASS' if all(v['evaluate'] == 'DONE' for v in self.jobs.values()) else 'COMPLETED_WITH_FAILURES'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', type=Path, required=True)
    ap.add_argument('--smoke-waiver', type=Path, required=True)
    ap.add_argument('--schedule', choices=('training-first', 'mixed'), default='training-first')
    args = ap.parse_args()
    results = args.results.resolve()
    digest = fingerprint()
    formal_smoke_gate(results, digest, args.smoke_waiver)
    locks = []
    for name in ('realprover_autorun.lock', 'realprover_pipeline.lock'):
        lock = (ROOT / 'runs' / name).open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
    campaign = Adaptive(results, digest, training_first=args.schedule == 'training-first')
    def interrupted(signum, frame):
        raise RuntimeError(f'Controller signal {signum}')
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        retrieval_service.stop()
        campaign.run()
    except BaseException as error:
        campaign.state.update(status='CONTROLLER_FAILED', error=str(error))
        raise
    finally:
        # Only global interruption/termination reaches cancellation. A method
        # failure is handled by poll_jobs and never enters this cleanup path.
        for run, phase, process, stream in campaign.active.values():
            terminate_tree(process.pid)
            process.wait()
            stream.close()
        if campaign.cpu:
            process, stream = campaign.cpu
            terminate_tree(process.pid)
            process.wait()
            stream.close()
        retrieval_service.stop()
        campaign.record()


if __name__ == '__main__':
    main()
