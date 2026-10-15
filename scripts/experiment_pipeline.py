"""REAL-Prover experiment planning and fail-closed smoke/formal gate.

No Qwen3 reproduction commands are reused implicitly. Runtime adapters must be
implemented and reviewed before binding commands in pipeline.json.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

RESEARCH = Path(__file__).resolve().parents[1]
ROOT = RESEARCH.parent
ORDER = [f'P{i}' for i in range(7)]
DEPENDENCIES = {'P0': [], 'P1': ['P0'], 'P2': ['P0'], 'P3': ['P0'],
                'P4': ['P0', 'P3'], 'P5': ['P0'], 'P6': ['P0', 'P3']}


def read(path):
    return json.loads(path.read_text())


def fingerprint(root=ROOT):
    """Include local training/evaluation code, configs and asset manifests.

    Download progress is deliberately excluded: immutable model identity belongs
    in pipeline.json; the adapter must verify complete model hashes at runtime.
    """
    selected = []
    for folder in ('research', 'REALPROVER', 'ReasoningQAT', 'qat_baselines/EfficientQAT'):
        base = root / folder
        for p in base.rglob('*'):
            if (not p.is_file() or any(x in p.parts for x in
                    ('.git', '__pycache__', 'repro_results', 'logs', 'outputs'))):
                continue
            if p.suffix not in ('.py', '.sh', '.toml', '.json', '.lean'):
                continue
            if p == root / 'research/manifests/model_revision.json':
                continue
            selected.append(p)
    h = hashlib.sha256()
    for p in sorted(selected):
        h.update(str(p.relative_to(root)).encode() + b'\0')
        h.update(p.read_bytes())
    return h.hexdigest()


def requirements(run):
    common = ['model_identity', 'tokenizer', 'native_lean_evaluation',
              'retrieval', 'fixed_denominator', 'saved_reload']
    if run != 'P0':
        common += ['w2_g128', 'export_logits_parity', 'same_eval_protocol']
    if run in ('P1', 'P2'):
        common += ['numinamath_same_128', 'algorithm_identity']
    if run in ('P3', 'P4', 'P5', 'P6'):
        common += ['general_data_only', 'finite_loss', 'finite_gradients',
                   'trainable_parameters', 'parameter_update', 'qwen2_modules']
    if run in ('P4', 'P6'):
        common += ['same_p3_initialization', 'same_stage2_tokens', 'scale_only']
    if run == 'P4':
        common += ['teacher_rr_loss', 'top20_kd_loss', 'supervision_mask']
    if run == 'P5':
        common += ['independent_block_ap', 'independent_e2e_qp', 'matched_token_budget']
    if run == 'P6':
        common += ['ce_only_loss', 'teacher_disabled']
    return common


def validate_receipt(receipt, run, mode, digest):
    if receipt.get('run') != run or receipt.get('mode') != mode:
        raise ValueError(f'{run}: receipt identity mismatch')
    if receipt.get('fingerprint') != digest or receipt.get('status') != 'PASS':
        raise ValueError(f'{run}: missing PASS or stale code/config fingerprint')
    checks = receipt.get('checks', {})
    missing = [key for key in requirements(run) if checks.get(key) is not True]
    if missing:
        raise ValueError(f'{run}: failed/missing checks: {missing}')
    if receipt.get('expected_problems') != (2 if mode == 'smoke' else 244):
        raise ValueError(f'{run}: incorrect fixed denominator')
    if receipt.get('completed_problems') != receipt['expected_problems']:
        raise ValueError(f'{run}: incomplete evaluation')
    if not receipt.get('evidence'):
        raise ValueError(f'{run}: evidence files required')
    for item in receipt['evidence']:
        p = Path(item['path'])
        if not p.is_absolute() or not p.is_file():
            raise ValueError(f'{run}: missing absolute evidence path: {p}')
        if hashlib.sha256(p.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError(f'{run}: changed evidence: {p}')


def gate(directory, mode, runs, digest):
    errors = []
    for run in runs:
        try:
            validate_receipt(read(directory / mode / run / 'receipt.json'), run, mode, digest)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f'{run}: {exc}')
    if errors:
        raise ValueError('\n'.join(errors))


def plan(config, mode):
    for run in ORDER:
        yield {'run': run, 'mode': mode, 'depends_on': DEPENDENCIES[run],
               'method': config['methods'][run], 'required_checks': requirements(run),
               'command': config['commands'][mode].get(run),
               'status': 'CONFIGURED' if config['commands'][mode].get(run) else 'BLOCKED_ADAPTER_NOT_BOUND'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=('plan', 'gate', 'run'))
    ap.add_argument('--mode', choices=('smoke', 'formal'), default='smoke')
    ap.add_argument('--run', choices=ORDER)
    ap.add_argument('--results', type=Path, default=ROOT / 'runs/realprover_pipeline')
    args = ap.parse_args()
    config = read(RESEARCH / 'configs/pipeline.json')
    digest = fingerprint()
    if args.action == 'plan':
        print(json.dumps({'fingerprint': digest, 'jobs': list(plan(config, args.mode))}, indent=2))
        return
    if args.action == 'gate':
        gate(args.results, 'smoke', ORDER, digest)
        print('PASS: all P0-P6 smoke receipts match current code/config and evidence')
        return
    if not args.run:
        ap.error('run requires --run')
    if args.mode == 'formal':
        gate(args.results, 'smoke', ORDER, digest)
    gate(args.results, args.mode, DEPENDENCIES[args.run], digest)
    command = config['commands'][args.mode].get(args.run)
    if not command or not isinstance(command, list) or not all(isinstance(x, str) for x in command):
        raise ValueError(f'{args.run}: BLOCKED_ADAPTER_NOT_BOUND; no fallback command')
    destination = args.results / args.mode / args.run
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / 'receipt.json').exists():
        raise ValueError('Refusing to overwrite an existing run; use a new --results directory')
    import os
    environment = dict(os.environ, REALPROVER_RUN=args.run, REALPROVER_MODE=args.mode,
                       REALPROVER_FINGERPRINT=digest, REALPROVER_OUTPUT=str(destination.resolve()))
    (destination / 'invocation.json').write_text(json.dumps({
        'command': command, 'cwd': str(ROOT), 'fingerprint': digest}, indent=2))
    with (destination / 'execution.log').open('w') as stream:
        subprocess.run(command, cwd=ROOT, env=environment, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)
    if fingerprint() != digest:
        raise ValueError('Code/config changed during execution; result cannot pass')
    validate_receipt(read(destination / 'receipt.json'), args.run, args.mode, digest)
    print(f'PASS: {args.mode} {args.run}')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f'BLOCKED: {exc}', file=sys.stderr)
        sys.exit(1)
