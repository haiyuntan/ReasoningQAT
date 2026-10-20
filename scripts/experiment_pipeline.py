"""REAL-Prover experiment planning and fail-closed smoke/formal gate.

No Qwen3 reproduction commands are reused implicitly. Runtime adapters must be
implemented and reviewed before binding commands in pipeline.json.
"""
import argparse
import hashlib
import json
import os
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


def file_digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(root=ROOT):
    """Include local training/evaluation code, configs and asset manifests.

    Download progress is deliberately excluded: immutable model identity belongs
    in pipeline.json; the adapter must verify complete model hashes at runtime.
    """
    selected = []
    for folder in ('research', 'REALPROVER', 'ReasoningQAT', 'qat_baselines/EfficientQAT', 'ptq/llm-awq/awq', 'ptq/GPTQModel/gptqmodel'):
        base = root / folder
        for directory, folders, files in os.walk(base):
            folders[:] = [name for name in folders if name not in
                          ('.git', '.lake', '__pycache__', 'repro_results', 'logs', 'outputs', 'dependencies')]
            for name in files:
                p = Path(directory) / name
                if p.suffix not in ('.py', '.sh', '.toml', '.json', '.lean'):
                    continue
                if p == root / 'research/manifests/model_revision.json':
                    continue
                selected.append(p)
    # Compiled/cache output is excluded above; the pinned source inventory is
    # stable and the actual Lean runtime binaries/header are hashed below.
    inventory = root / 'REALPROVER/dependencies/source_restore.json'
    if inventory.is_file():
        selected.append(inventory)
    h = hashlib.sha256()
    for p in sorted(selected):
        h.update(str(p.relative_to(root)).encode() + b'\0')
        h.update(p.read_bytes())
    # Runtime choices participate too: switching Python, Lean binaries or the
    # retrieval endpoint invalidates old smoke evidence.
    runtime_path = root / 'research/configs/runtime.json'
    if runtime_path.exists():
        runtime = read(runtime_path)
        for role, value in runtime['python'].items():
            value = os.environ.get(f'REALPROVER_{role.upper()}_PYTHON', value)
            executable = Path(value)
            if not executable.is_absolute():
                executable = root / executable
            h.update(str(executable.resolve()).encode())
            if executable.exists():
                stat = executable.stat()
                h.update(str((stat.st_size, stat.st_mtime_ns)).encode())
            for metadata in sorted(executable.parent.parent.glob('lib/python*/site-packages/*.dist-info/METADATA')):
                if metadata.parent.name.lower().startswith(('torch-', 'transformers-', 'accelerate-', 'vllm-', 'gptqmodel-', 'trl-')):
                    h.update(metadata.read_bytes())
    for key in ('REALPROVER_LEAN_WORKSPACE', 'REALPROVER_JIXIA_BIN', 'REALPROVER_INTERACTIVE_ROOT', 'REALPROVER_LEAN_BIN', 'LEANSEARCH_PORT'):
        h.update((key + '=' + os.environ.get(key, '')).encode())
    lean = Path(os.environ.get('REALPROVER_LEAN_WORKSPACE', str(root / 'REALPROVER/dependencies/lean_test_v4160')))
    jixia = Path(os.environ.get('REALPROVER_JIXIA_BIN', str(root / 'REALPROVER/dependencies/jixia/.lake/build/bin/jixia')))
    interactive = Path(os.environ.get('REALPROVER_INTERACTIVE_ROOT', str(root / 'REALPROVER/dependencies/interactive')))
    for path in (lean / 'lean-toolchain', lean / 'Header.lean', lean / 'lake-manifest.json', jixia,
                 interactive / '.lake/build/bin/interactive', interactive / 'lake-manifest.json'):
        if path.is_file():
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    h.update(chunk)
    if os.environ.get('REALPROVER_LEAN_BIN'):
        for name in ('lean', 'lake'):
            h.update(file_digest(Path(os.environ['REALPROVER_LEAN_BIN']) / name).encode())
    for pattern in ('datasets/manifests/*.json', 'datasets/numinamath_128/calib.jsonl',
                    'datasets/eval/legacy_candidate/*.jsonl', 'REALPROVER/data/*.jsonl'):
        for path in sorted(root.glob(pattern)):
            h.update(str(path.relative_to(root)).encode() + b'\0' + path.read_bytes())
    # Large raw datasets: detect replacement without hashing 28GB on each
    # command. Actual selected token/mask hashes are saved by training adapters.
    for folder in ('datasets/openthoughts3', 'datasets/fineweb-edu'):
        for path in sorted((root / folder).rglob('*.parquet')):
            stat = path.stat()
            h.update(str((str(path.relative_to(root)), stat.st_size, stat.st_mtime_ns)).encode())
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
        if file_digest(p) != item['sha256']:
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


def formal_smoke_gate(directory, digest, waiver=None):
    if waiver is None:
        return gate(directory, 'smoke', ORDER, digest)
    authorization = read(waiver)
    if (authorization.get('fingerprint') != digest
            or authorization.get('results') != str(directory.resolve())
            or authorization.get('scope') != 'skip_smoke_for_direct_formal_restart'
            or not authorization.get('user_instruction')
            or not authorization.get('validation_report')):
        raise ValueError('Smoke waiver must bind user authorization and validation to this exact code and batch')
    report = read(Path(authorization['validation_report']))
    if report.get('status') != 'PASS':
        raise ValueError('Targeted repair validation must pass before direct restart')
    print('AUTHORIZED SMOKE WAIVER: direct formal restart; no new smoke PASS claimed', flush=True)


def plan(config, mode):
    for run in ORDER:
        yield {'run': run, 'mode': mode, 'depends_on': DEPENDENCIES[run],
               'method': config['methods'][run], 'required_checks': requirements(run),
               'command': config['commands'][mode].get(run),
               'status': 'CONFIGURED' if config['commands'][mode].get(run) else 'BLOCKED_ADAPTER_NOT_BOUND'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=('plan', 'gate', 'run', 'campaign'))
    ap.add_argument('--mode', choices=('smoke', 'formal'), default='smoke')
    ap.add_argument('--run', choices=ORDER)
    ap.add_argument('--results', type=Path, default=ROOT / 'runs/realprover_pipeline')
    ap.add_argument('--smoke-waiver', type=Path)
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
    if args.action == 'campaign':
        if args.mode == 'formal':
            formal_smoke_gate(args.results, digest, args.smoke_waiver)
        for run in ORDER:
            if (args.results / args.mode / run / 'receipt.json').exists():
                gate(args.results, args.mode, [run], digest)
                print(f'SKIP validated {args.mode} {run}', flush=True)
                continue
            child = [sys.executable, str(Path(__file__).resolve()), 'run', '--mode', args.mode,
                     '--run', run, '--results', str(args.results.resolve())]
            if args.smoke_waiver:
                child += ['--smoke-waiver', str(args.smoke_waiver.resolve())]
            subprocess.run(child, check=True)
        return
    if not args.run:
        ap.error('run requires --run')
    import fcntl
    lock_path = ROOT / 'runs/realprover_pipeline.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open('a')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise ValueError('Another REAL-Prover pipeline run holds the single-GPU lock')
    if args.mode == 'formal':
        formal_smoke_gate(args.results, digest, args.smoke_waiver)
    gate(args.results, args.mode, DEPENDENCIES[args.run], digest)
    command = config['commands'][args.mode].get(args.run)
    if not command or not isinstance(command, list) or not all(isinstance(x, str) for x in command):
        raise ValueError(f'{args.run}: BLOCKED_ADAPTER_NOT_BOUND; no fallback command')
    destination = args.results / args.mode / args.run
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError('Refusing to overwrite a completed/partial run; use a new --results directory')
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
