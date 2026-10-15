"""Run preparation checks with persistent logs; never launch a P0-P6 campaign."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

RESEARCH = Path(__file__).resolve().parents[1]
ROOT = RESEARCH.parent
sys.path.insert(0, str(RESEARCH / 'adapters'))
from common import python_for, save


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--full-model', action='store_true')
    ap.add_argument('--native-runtime', action='store_true', help='Live Interactive/Jixia and native vLLM checks')
    args = ap.parse_args()
    import fcntl
    lock_path = ROOT / 'runs/realprover_pipeline.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open('a')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        ap.error('Another REAL-Prover job holds the single-GPU lock')
    if args.output.exists() and any(args.output.iterdir()):
        ap.error('Use a new empty output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, USE_TF='0', USE_FLAX='0', HF_HUB_OFFLINE='1',
                       HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    checks = []
    jobs = [(n, python_for('train'), [str(RESEARCH / 'tests' / (n + '.py'))]) for n in
            ('check_experiment_gate', 'check_native_aggregation', 'check_verifier_contract', 'check_generator_contract',
             'check_qwen2_adapters', 'check_serial_resume')]
    if args.gpu:
        jobs += [('training_gpu', python_for('train'), [str(RESEARCH / 'tests/check_qwen2_training_gpu.py')]),
                 ('awq_gpu', python_for('train'), [str(RESEARCH / 'tests/check_ptq_qwen2_gpu.py'), '--method', 'awq']),
                 ('gptq_gpu', python_for('gptq'), [str(RESEARCH / 'tests/check_ptq_qwen2_gpu.py'), '--method', 'gptq'])]
    if args.full_model:
        jobs += [('full_model', python_for('train'), [str(RESEARCH / 'tests/check_full_model.py'),
                  '--output', str((args.output / 'full_model.json').resolve())])]
    for name, python, argv in jobs:
        started = time.monotonic()
        print('CHECK', name, flush=True)
        with (args.output / (name + '.log')).open('w') as log:
            result = subprocess.run([python, *argv], cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        checks.append({'name': name, 'exit_code': result.returncode,
                       'seconds': time.monotonic() - started,
                       'log': str((args.output / (name + '.log')).resolve())})
        save(args.output / 'checks.json', {'checks': checks, 'full_smoke_pass': False})
    # Live dependency verification is separate from component and model checks.
    preflight = args.output / 'native_preflight'
    with (args.output / 'native_preflight.log').open('w') as log:
        native = subprocess.run([python_for('eval'), str(RESEARCH / 'adapters/evaluation.py'),
            '--model', str(ROOT / 'models/REAL-Prover-v1-bf16'), '--mode', 'smoke',
            '--output', str(preflight.resolve()), '--preflight'], cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    if args.native_runtime and native.returncode == 0:
        for name in ('lean_runtime', 'vllm_runtime'):
            started = time.monotonic()
            print('CHECK', name, flush=True)
            with (args.output / (name + '.log')).open('w') as log:
                result = subprocess.run([python_for('eval'), str(RESEARCH / 'tests' / ('check_' + name + '.py')),
                    '--output', str((args.output / (name + '.json')).resolve())], cwd=ROOT, env=environment,
                    stdout=log, stderr=subprocess.STDOUT)
            checks.append({'name': name, 'exit_code': result.returncode,
                           'seconds': time.monotonic() - started,
                           'log': str((args.output / (name + '.log')).resolve())})
    passed = all(c['exit_code'] == 0 for c in checks)
    report = {'status': 'READY_FOR_SMOKE' if passed and native.returncode == 0 and args.gpu and args.full_model and args.native_runtime
              else 'BLOCKED_OR_INCOMPLETE', 'component_checks_pass': passed,
              'gpu_checks_requested': args.gpu, 'full_model_requested': args.full_model,
              'native_runtime_requested': args.native_runtime,
              'native_preflight_exit_code': native.returncode, 'checks': checks,
              'full_smoke_pass': False,
              'note': 'Preparation is not a P0-P6 PASS receipt; formal gate remains mandatory.'}
    save(args.output / 'preparation.json', report)
    print(json.dumps(report, indent=2))
    sys.exit(0 if report['status'] == 'READY_FOR_SMOKE' else 1)


if __name__ == '__main__':
    main()
