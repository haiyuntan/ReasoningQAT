"""Persistent serial controller: all smoke receipts -> gate -> formal campaign."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', type=Path, required=True)
    ap.add_argument('--formal-only', action='store_true')
    ap.add_argument('--smoke-waiver', type=Path)
    a = ap.parse_args()
    if a.formal_only != bool(a.smoke_waiver):
        ap.error('--formal-only requires an explicit --smoke-waiver, and vice versa')
    results = a.results.resolve()
    results.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / 'runs/realprover_autorun.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    control = results / 'control'
    control.mkdir(exist_ok=False)
    state = {'pid': os.getpid(), 'results': str(results), 'status': 'STARTING',
             'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'formal_only': a.formal_only,
             'smoke_waiver': str(a.smoke_waiver.resolve()) if a.smoke_waiver else None,
             'commits': {repo: subprocess.check_output(['git', '-C', str(ROOT/repo), 'rev-parse', 'HEAD'],
                                                      text=True).strip() for repo in ('research', 'REALPROVER')}}
    def save(**updates):
        state.update(updates)
        temporary = control / 'status.tmp'
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(control / 'status.json')
    environment = dict(os.environ, USE_TF='0', USE_FLAX='0', HF_HUB_OFFLINE='1',
                       HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
    pipeline = ROOT / 'research/scripts/experiment_pipeline.py'
    save()
    try:
        runtime = json.loads((ROOT/'research/configs/runtime.json').read_text())
        if runtime['eval_protocol_status'] != 'FROZEN':
            raise RuntimeError('Freeze and review evaluation protocol before scheduling automatic formal execution')
        stages = [('SMOKE', ['campaign', '--mode', 'smoke']),
                                 ('GATE', ['gate']),
                                 ('FORMAL', ['campaign', '--mode', 'formal'])]
        if a.formal_only:
            stages = [('FORMAL', ['campaign', '--mode', 'formal', '--smoke-waiver', str(a.smoke_waiver.resolve())])]
        for stage, arguments in stages:
            command = [sys.executable, str(pipeline), *arguments, '--results', str(results)]
            with (control / (stage.lower()+'.log')).open('w') as log:
                child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
                save(status=stage+'_RUNNING', child_pid=child.pid, command=command)
                code = child.wait()
            if code:
                save(status=stage+'_FAILED', exit_code=code)
                return code
        save(status='COMPLETE', exit_code=0)
        return 0
    except Exception as exc:
        save(status='FAILED', error=repr(exc))
        raise
    finally:
        try:
            save(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        finally:
            lock.close()


if __name__ == '__main__':
    sys.exit(main())
