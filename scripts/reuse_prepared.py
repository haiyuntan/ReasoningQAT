"""Verify unchanged P1/P2/P3 preparation math, then inherit immutable artifacts."""
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from experiment_pipeline import ROOT, RESEARCH
sys.path.insert(0, str(RESEARCH / 'adapters'))
from common import load, save, sha


def audit_source(source):
    state = load(source / 'control/status.json')
    commits = state['commits']
    def previous(repo, path):
        return subprocess.check_output(['git', '-C', str(ROOT / repo), 'show',
                                        f'{commits[repo]}:{path}'], text=True)
    files = {'research': ['adapters/quantization.py', 'configs/pipeline.json', 'configs/runtime.json',
                          'configs/training_general_data.json', 'vendor/reasoningqat/datautils_block.py',
                          'vendor/reasoningqat/repro_scripts/baseline_ops.py'],
             'REALPROVER': ['configs/vllm_runtime.json', 'configs/p0_candidate.toml', 'data/minif2f_test.jsonl']}
    for repo, paths in files.items():
        for path in paths:
            if previous(repo, path) != (ROOT / repo / path).read_text():
                raise RuntimeError(f'Preparation reuse needs review: {repo}/{path} changed')
    def nodes(text):
        return {n.name: n for n in ast.parse(text).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    for path, names in [('adapters/training.py', ['calibration', 'reasoning_block_ap', 'OptimizerAudit']),
                        ('vendor/reasoningqat/repro_scripts/baseline_training.py',
                         ['DiskSamples', 'capture_inputs', 'layer_output', 'to_device_kwargs'])]:
        old, new = nodes(previous('research', path)), nodes((RESEARCH / path).read_text())
        for name in names:
            if ast.dump(old[name]) != ast.dump(new[name]):
                raise RuntimeError(f'P3 math changed: {name}')
    old = nodes(previous('research', 'adapters/training.py'))['main']
    new = nodes((RESEARCH / 'adapters/training.py').read_text())['main']
    def p3_branch(function):
        return next(n.body for n in ast.walk(function) if isinstance(n, ast.If)
                    and ast.unparse(n.test) == "a.run == 'P3'")
    if [ast.dump(n) for n in p3_branch(old)] != [ast.dump(n) for n in p3_branch(new)]:
        raise RuntimeError('P3 training entrypoint changed')
    return {'source_commits': commits, 'status': 'PASS', 'review':
            'P1/P2 quantization and P3 stage1 math unchanged; stage2 data and checkpoint/scheduling repairs do not retrain P1/P2/P3'}


def inherit(source, results, digest):
    source, results = source.resolve(), results.resolve()
    audit = audit_source(source)
    for run in ('P1', 'P2', 'P3'):
        src, dst = source / 'formal' / run, results / 'formal' / run
        record = load(src / 'prepared.json')
        if record['fingerprint'] != load(source / 'control/status.json')['fingerprint']:
            raise RuntimeError('Source preparation fingerprint mismatch')
        for item in record['evidence']:
            if sha(Path(item['path'])) != item['sha256']:
                raise RuntimeError(f'Original artifact changed: {item["path"]}')
        def copy(a, b):
            if Path(a).suffix in ('.pt', '.safetensors'):
                os.link(a, b)
            else:
                shutil.copy2(a, b)
            return b
        shutil.copytree(src, dst, copy_function=copy,
                        ignore=shutil.ignore_patterns('prepared.json', 'eval', 'preflight'))
        evidence = []
        for item in record['evidence']:
            relative = Path(item['path']).relative_to(src)
            target = dst / relative
            if sha(target) != item['sha256']:
                raise RuntimeError('Inherited copy hash mismatch')
            evidence.append({'path': str(target), 'sha256': item['sha256']})
        lineage = dict(audit, source=str(src), source_fingerprint=record['fingerprint'],
                       source_prepared_sha256=sha(src / 'prepared.json'), fingerprint=digest,
                       immutable_binary_hardlinks=True, no_old_evaluation_receipts_imported=True)
        save(dst / 'reuse_manifest.json', lineage)
        evidence.append({'path': str(dst / 'reuse_manifest.json'), 'sha256': sha(dst / 'reuse_manifest.json')})
        save(dst / 'prepared.json', dict(record, fingerprint=digest, evidence=evidence,
                                         inherited_from=str(src)))
        print(f'REUSE {run}: source hashes and unchanged preparation math verified', flush=True)
    stage1 = load(results / 'formal/P3/train/training.json')['stage1_data']
    original = load(source / 'formal/shared_stage1.json')
    if stage1 != original:
        raise RuntimeError('Source shared stage1 data differs from P3 evidence')
    save(results / 'formal/shared_stage1.json', stage1)
