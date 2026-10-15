"""Read-only legacy audit; writes evidence only in this experiment workspace.

No imports of legacy code: some import paths build Lean dependencies in place.
No model downloads, package installs, GPU allocation or evaluation.
"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT.parent / 'FedProver'
OUT = ROOT / 'research/manifests'


def command(args, cwd=None, timeout=30):
    try:
        p = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return {'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {'error': str(e)}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    for name in ['research/configs', 'research/adapters', 'research/reports',
                 'research/patches', 'upstream', 'eval_runtime', 'checkpoints',
                 'runs', 'datasets/eval', 'datasets/calibration', 'datasets/finetune']:
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = {'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                 'root': str(ROOT), 'legacy': str(LEGACY),
                 'gpu': command(['nvidia-smi']), 'disk': command(['df', '-h', str(ROOT)]),
                 'legacy_sha': command(['git', 'rev-parse', 'HEAD'], LEGACY),
                 'legacy_status': command(['git', 'status', '--short'], LEGACY),
                 'environments': {}, 'eval_candidates': [], 'datasets': [], 'dependencies': {}}
    # Read installed metadata with each environment's interpreter; do not import vLLM.
    envroot = Path('/HOME/sysu_qling/sysu_qling_3/HDD_POOL/anaconda3/envs')
    for name in ['PROVER', 'REQAT_PROVER']:
        py = envroot / name / 'bin/python'
        code = ('import sys,json,importlib.metadata as m; '
                'names=["torch","vllm","transformers","trl","accelerate","flash-attn"]; '
                'installed={d.metadata["Name"].lower():d.version for d in m.distributions()}; '
                'print(json.dumps({"python":sys.version,"packages":{n:installed.get(n) for n in names}}))')
        inventory['environments'][name] = command([str(py), '-c', code])
    for name in ['lean_test_v4160', 'jixia', 'interactive']:
        p = LEGACY / name
        target = p.resolve()
        inventory['dependencies'][name] = {'path': str(p), 'resolved': str(target), 'exists': p.exists()}
        tc = target / 'lean-toolchain'
        if tc.exists():
            inventory['dependencies'][name]['toolchain'] = tc.read_text().strip()
    for p in sorted((LEGACY / 'experiment/logs').rglob('config.toml')):
        # Only whitelisted non-secret fields; never copy arbitrary configs into logs.
        try:
            import tomllib
            c = tomllib.loads(p.read_text())
            inventory['eval_candidates'].append({'config': str(p), 'sha256': digest(p),
                'model': c.get('model', {}).get('prover_model_path'),
                'data': c.get('data', {}).get('data_path'),
                'result_exists': (p.parent / 'result.txt').exists(),
                'generated_exists': (p.parent / 'generated').exists()})
        except Exception as e:
            inventory['eval_candidates'].append({'config': str(p), 'error': type(e).__name__})
    for p in sorted((LEGACY / 'data').glob('*.jsonl')):
        rows = [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
        ids = [str(r.get('id')) for r in rows]
        inventory['datasets'].append({'path': str(p), 'count': len(rows), 'sha256': digest(p),
            'unique_ids': len(ids) == len(set(ids)),
            'has_required_fields': all('id' in r and 'formal_statement' in r for r in rows)})
    for base in ['models', 'datasets/manifests', 'datasets/numinamath_128']:
        inventory[base] = [str(p.relative_to(ROOT)) for p in sorted((ROOT / base).rglob('*'))
                           if p.is_file() and len(p.relative_to(ROOT / base).parts) <= 3]
    model = ROOT / 'models/REAL-Prover-v1-bf16'
    inventory['bf16_model_present'] = (model / 'config.json').exists() and any(model.glob('*.safetensors'))
    (OUT / 'preparation_audit.json').write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'report': str(OUT / 'preparation_audit.json'),
        'eval_candidate_count': len(inventory['eval_candidates']),
        'bf16_model_present': inventory['bf16_model_present'],
        'dependencies': inventory['dependencies'], 'environments': inventory['environments']}, indent=2))


if __name__ == '__main__':
    main()
