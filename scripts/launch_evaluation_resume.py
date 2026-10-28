"""Audit completed models/attempts into a new fingerprint and nohup evaluation only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from experiment_pipeline import ROOT, RESEARCH, fingerprint, validate_receipt
sys.path.insert(0, str(RESEARCH / 'adapters'))
sys.path.insert(0, str(ROOT / 'REALPROVER'))
from common import load, save, sha
from attempt_resume import read_attempts
from reuse_prepared import audit_source


def require_stopped(source, state):
    if state['status'] not in ('STOPPED_BY_USER', 'STOPPED_DISK_QUOTA'):
        raise RuntimeError('Stop source campaign before inheriting attempts')
    for command in Path('/proc').glob('[0-9]*/cmdline'):
        if command.parent.name == str(os.getpid()):
            continue
        try:
            argv = command.read_bytes().split(b'\0')
        except OSError:
            continue
        # Evaluator workers inherit their experiment.run argv. Do not mistake
        # this initializer's --source argument for a running evaluation.
        if (any(str(source).encode() in arg for arg in argv)
                and (b'experiment.run' in argv or any(b'run_experiment.py' in arg for arg in argv))):
            raise RuntimeError(f'Source evaluation is still running: PID {command.parent.name}')
    pid = state.get('pid')
    path = Path(f'/proc/{pid}/cmdline')
    if path.exists() and b'resume_four_gpu.py' in path.read_bytes():
        raise RuntimeError('Source controller is still running')


def audit_retrieval_identity(state):
    old = json.loads(subprocess.check_output(['git', '-C', str(RESEARCH), 'show',
                     f'{state["commits"]["research"]}:manifests/retrieval_assets.json']))
    new = load(RESEARCH / 'manifests/retrieval_assets.json')
    identity = lambda record: sorted((item['size'], item['sha256']) for item in record['files'])
    if old['service_source_revision'] != new['service_source_revision'] or identity(old) != identity(new):
        raise RuntimeError('Retrieval content changed; cannot merge old evaluations')
    return {'content_identical': True, 'source_manifest': old,
            'current_manifest_sha256': sha(RESEARCH / 'manifests/retrieval_assets.json')}


def import_prepared(source, results, digest):
    audit = audit_source(source)
    state = load(source / 'control/status.json')
    # All P4/P5/P6 data preparation and optimization code must also be identical.
    for name in ('adapters/training.py', 'adapters/completion_tokens.py', 'adapters/stage2_cache.py',
                 'vendor/reasoningqat/repro_scripts/baseline_training.py'):
        old = subprocess.check_output(['git', '-C', str(RESEARCH), 'show',
                                       f'{state["commits"]["research"]}:{name}'])
        if old != (RESEARCH / name).read_bytes():
            raise RuntimeError(f'Training changed; refusing checkpoint reuse: {name}')
    for run in ('P1', 'P2', 'P3', 'P4', 'P5', 'P6'):
        src, dst = source / 'formal' / run, results / 'formal' / run
        record = load(src / 'prepared.json')
        if record['run'] != run or record['fingerprint'] != state['fingerprint']:
            raise RuntimeError(f'Original preparation identity mismatch: {run}')
        for item in record['evidence']:
            if sha(Path(item['path'])) != item['sha256']:
                raise RuntimeError(f'Original prepared artifact changed: {item["path"]}')
        def copy(a, b):
            if Path(a).suffix in ('.pt', '.safetensors'):
                os.link(a, b)
            else:
                shutil.copy2(a, b)
            return b
        shutil.copytree(src, dst, copy_function=copy,
                        ignore=shutil.ignore_patterns('prepared.json', 'receipt.json', 'eval', 'preflight'))
        evidence = []
        for item in record['evidence']:
            old = Path(item['path'])
            if old == src / 'reuse_manifest.json':
                continue  # Its old hash is bound by source_prepared_sha256 below.
            target = dst / old.relative_to(src)
            if old.suffix in ('.pt', '.safetensors'):
                if not os.path.samefile(old, target):
                    raise RuntimeError('Immutable binary inheritance lost inode identity')
            elif sha(target) != item['sha256']:
                raise RuntimeError('Copied artifact hash mismatch')
            evidence.append({'path': str(target), 'sha256': item['sha256']})
        lineage = dict(audit, source=str(src), source_fingerprint=state['fingerprint'],
                       source_prepared_sha256=sha(src / 'prepared.json'), fingerprint=digest,
                       all_training_code_identical=True, immutable_binary_hardlinks=True,
                       no_old_receipt_imported=True)
        save(dst / 'reuse_manifest.json', lineage)
        evidence.append({'path': str(dst / 'reuse_manifest.json'), 'sha256': sha(dst / 'reuse_manifest.json')})
        save(dst / 'prepared.json', dict(record, fingerprint=digest, evidence=evidence, inherited_from=str(src)))
        print(f'REUSE {run}: original model hashes and unchanged training verified', flush=True)
    for name in ('shared_stage1.json', 'shared_stage2.json'):
        shutil.copy2(source / 'formal' / name, results / 'formal' / name)


def import_attempts(source, results, digest):
    dataset = ROOT / 'REALPROVER/data/minif2f_test.jsonl'
    rows = [json.loads(s) for s in dataset.read_text().splitlines() if s.strip()]
    summaries = {}
    state = load(source / 'control/status.json')
    for run in ('P1', 'P2', 'P3', 'P4', 'P5', 'P6'):
        src = source / 'formal' / run / 'eval'
        if not src.exists():
            summaries[run] = {'completed': 0, 'successes': 0, 'attempts': 0, 'timeouts': 0}
            continue
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib
        actual = tomllib.loads((src / 'config.toml').read_text())
        expected = tomllib.loads((ROOT / 'REALPROVER/configs/p0_candidate.toml').read_text())
        for config in (actual, expected):
            config.pop('info', None)
            config['model'].pop('prover_model_path')
            config['data'].pop('data_path')
        if actual != expected:
            raise RuntimeError(f'Old evaluation protocol differs: {run}')
        limit = actual['search']['max_retries']
        dst = results / 'formal' / run / 'eval'
        evidence = []
        def copy_row(row):
            old_dir = src / 'native/generated' / str(row['id'])
            records, success = read_attempts(old_dir, limit, row['formal_statement'])
            items = []
            for index in records:
                old = old_dir / f'{row["id"]}_{index}.json'
                new = dst / 'native/generated' / str(row['id']) / old.name
                new.parent.mkdir(parents=True, exist_ok=True)
                data = old.read_bytes()
                new.write_bytes(data)
                if sha(old) != sha(new):
                    raise RuntimeError('Imported attempt hash mismatch')
                items.append({'path': str(new), 'source_path': str(old), 'sha256': sha(new)})
            return {'id': str(row['id']), 'attempts': len(records), 'success': success,
                    'completed': success or len(records) == limit,
                    'timeouts': sum(d.get('pipeline_attempt', {}).get('status') == 'timeout' for d in records.values()),
                    'missing_slots': [i for i in range(limit) if i not in records]}, items
        with ThreadPoolExecutor(max_workers=8) as pool:
            copied = list(pool.map(copy_row, rows))
        problems = [p for p, _ in copied]
        for _, items in copied:
            evidence.extend(items)
        summary = {'completed': sum(p['completed'] for p in problems),
                   'successes': sum(p['success'] for p in problems),
                   'attempts': sum(p['attempts'] for p in problems),
                   'timeouts': sum(p['timeouts'] for p in problems)}
        telemetry = src / 'retrieval.jsonl'
        save(dst / 'evaluation_resume.json', {'source': str(src), 'source_fingerprint': state['fingerprint'],
             'fingerprint': digest, 'dataset_sha256': sha(dataset), 'max_retries': limit,
             'summary': summary, 'problems': problems, 'evidence': evidence,
             'source_retrieval': {'path': str(telemetry), 'sha256': sha(telemetry)},
             'unpublished_temporary_files': [str(p) for p in (src / 'native/generated').rglob('*.tmp')],
             'rng_note': 'New evaluator starts with seed 2; previous in-memory RNG state was not checkpointed. Slot budget is preserved; bitwise uninterrupted equivalence is not claimed.'})
        summaries[run] = summary
        print(f'RESUME {run}: {summary}', flush=True)
    save(results / 'control/previous_progress.json', summaries)
    return summaries


def inherit_completed(source, results, digest, summaries):
    """Audit completed receipts and rebind their evidence without new inference."""
    from types import SimpleNamespace
    from run_experiment import finish
    state = load(source / 'control/status.json')
    completed = []
    for run, progress in summaries.items():
        if progress['completed'] != 244:
            continue
        src, dst = source / 'formal' / run, results / 'formal' / run
        original = load(src / 'receipt.json')
        validate_receipt(original, run, 'formal', state['fingerprint'])
        stats = load(src / 'eval/evaluation.json')
        if (stats['expected_problems'] != 244 or stats['completed_problems'] != 244
                or stats['successes'] != progress['successes'] or not stats['retrieval_ok']):
            raise RuntimeError(f'Completed receipt/attempt summary mismatch: {run}')
        lineage = load(dst / 'eval/evaluation_resume.json')
        original_raw = {item['path']: item['sha256'] for item in stats['raw_evidence']}
        inherited_raw = {item['source_path']: item['sha256'] for item in lineage['evidence']}
        if original_raw != inherited_raw:
            raise RuntimeError(f'Completed evaluation evidence differs: {run}')
        shutil.copytree(src / 'preflight', dst / 'preflight')
        for name in ('config.toml', 'retrieval.jsonl'):
            shutil.copy2(src / 'eval' / name, dst / 'eval' / name)
        shutil.copy2(src / 'resource_lifecycle.json', dst / 'resource_lifecycle.json')
        stats.update(raw_evidence=[{'path': e['path'], 'sha256': e['sha256']} for e in lineage['evidence']],
                     evaluation_origin='inherited_completed_no_new_inference',
                     source_receipt_sha256=sha(src / 'receipt.json'),
                     source_fingerprint=state['fingerprint'], resumed_attempts=progress['attempts'],
                     evaluation_resume_sha256=sha(dst / 'eval/evaluation_resume.json'))
        save(dst / 'eval/evaluation.json', stats)
        # Use the normal method checks and receipt writer, not copied PASS flags.
        previous = {k: os.environ.get(k) for k in ('REALPROVER_MODE', 'REALPROVER_FINGERPRINT')}
        os.environ.update(REALPROVER_MODE='formal', REALPROVER_FINGERPRINT=digest)
        try:
            training = load(dst / 'train/training.json') if run in ('P3', 'P4', 'P5', 'P6') else None
            finish(SimpleNamespace(run=run, mode='formal'), dst, load(dst / 'model_identity.json'), training)
            receipt = load(dst / 'receipt.json')
            receipt.update(evaluation_origin=stats['evaluation_origin'],
                           source_receipt_sha256=stats['source_receipt_sha256'])
            save(dst / 'receipt.json', receipt)
            validate_receipt(receipt, run, 'formal', digest)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        completed.append(run)
        print(f'COMPLETE {run}: audited inherited 244/244; no GPU inference scheduled', flush=True)
    return completed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--results', type=Path, required=True)
    ap.add_argument('--validation-report', type=Path, default=ROOT / 'runs/evaluation_resume_20261026/validation.json')
    args = ap.parse_args()
    source, results = args.source.resolve(), args.results.resolve()
    state = load(source / 'control/status.json')
    require_stopped(source, state)
    retrieval_audit = audit_retrieval_identity(state)
    for key, value in state['lean_environment'].items():
        if not value or not Path(value).exists():
            raise RuntimeError(f'Missing audited Lean environment: {key}')
        os.environ[key] = value
    os.environ['PATH'] = state['lean_environment']['REALPROVER_LEAN_BIN'] + ':' + os.environ['PATH']
    os.environ.update(USE_TF='0', USE_FLAX='0', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                      OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    os.environ.pop('REALPROVER_SHARED_RETRIEVAL', None)
    for repo in ('research', 'REALPROVER'):
        if subprocess.check_output(['git', '-C', str(ROOT / repo), 'status', '--porcelain'], text=True):
            raise RuntimeError(f'Commit reviewed changes first: {repo}')
    validation = args.validation_report.resolve()
    if load(validation)['status'] != 'PASS' or load(ROOT / 'runs/memory_repair_20261026/live_gpu.json')['status'] != 'PASS':
        raise RuntimeError('Resume and real GPU memory regressions must pass')
    digest = fingerprint()
    results.mkdir(parents=True, exist_ok=False)
    (results / 'control').mkdir()
    save(results / 'control/initialization.json', {'status': 'IMPORTING', 'fingerprint': digest,
         'source': str(source), 'pid': os.getpid()})
    import_prepared(source, results, digest)
    summaries = import_attempts(source, results, digest)
    completed = inherit_completed(source, results, digest, summaries)
    (results / 'stage2_cache').symlink_to((source / 'stage2_cache').resolve(), target_is_directory=True)
    waiver = results / 'smoke_waiver.json'
    save(waiver, {'fingerprint': digest, 'results': str(results),
         'scope': 'skip_smoke_for_direct_formal_restart',
         'user_instruction': '从已经评估的题目继续评估的流水线，并且启动；保留已有题目和尝试预算，沿用已授权的不重复完整冒烟续跑。',
         'validation_report': str(validation)})
    save(results / 'campaign_plan.json', {'fingerprint': digest, 'source': str(source),
         'schedule': 'evaluation-only', 'retrieval_gpu': 3, 'evaluation_gpus': [0, 1, 2],
         'method_failures_isolated': True, 'previous_progress': summaries,
         'models_retrained': False, 'old_timeouts_preserved_as_failed_attempts': True})
    save(results / 'control/completed_inheritance.json', {'methods': completed,
         'retrieval_identity': retrieval_audit, 'no_new_inference_for_completed_methods': True})
    log = results.with_name(results.name + '.nohup.log')
    command = ['nohup', sys.executable, '-u', str(RESEARCH / 'scripts/resume_four_gpu.py'),
               '--results', str(results), '--smoke-waiver', str(waiver), '--schedule', 'mixed']
    with log.open('x') as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ), stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    save(results / 'launch.json', {'pid': process.pid, 'command': command, 'log': str(log), 'fingerprint': digest})
    results.with_name(results.name + '.pid').write_text(str(process.pid) + '\n')
    save(results / 'control/initialization.json', {'status': 'CONTROLLER_STARTED', 'pid': process.pid, 'fingerprint': digest})
    print(f'NOHUP controller {process.pid}; {log}', flush=True)


if __name__ == '__main__':
    main()
