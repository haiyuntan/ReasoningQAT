"""CPU-only checks; do not import the evaluator or invoke lake/build/model."""
import ast
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('local_config', ROOT / 'conf/config.py')
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)
errors = []
for directory, folders, files in os.walk(ROOT):
    folders[:] = [name for name in folders if name not in
                  ('dependencies', '.git', '.lake', '__pycache__', 'outputs', 'logs')]
    for name in files:
        if name.endswith('.py'):
            ast.parse((Path(directory) / name).read_text())
manifest = json.loads((ROOT / 'provenance/source_manifest.json').read_text())
for row in manifest['files']:
    p = ROOT / row['path']
    if hashlib.sha256(p.read_bytes()).hexdigest() != row['copied_sha256']:
        errors.append('Snapshot changed: ' + row['path'])
for name, count in [('minif2f_test', 244), ('proofnet_test', 186), ('fate_m', 141)]:
    rows = [json.loads(s) for s in (ROOT / f'data/{name}.jsonl').read_text().splitlines() if s.strip()]
    assert len(rows) == count and len({str(r['id']) for r in rows}) == count
    assert all('formal_statement' in r for r in rows)
packages = {}
for name in ['torch', 'vllm', 'transformers', 'requests', 'heapdict', 'matplotlib', 'networkx']:
    try:
        packages[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        packages[name] = None
        errors.append('Missing Python package: ' + name)
paths = {'lean_workspace': Path(config.LEAN_TEST_PATH) / 'lean-toolchain',
         'header': Path(config.LEAN_TEST_PATH) / 'Header.lean',
         'jixia': config.analyzer_path,
         'interactive': config.build_path / 'bin/interactive'}
for name, p in paths.items():
    if not p.exists():
        errors.append(f'Missing {name}: {p}')
toolchain = paths['lean_workspace']
if toolchain.exists() and toolchain.read_text().strip() != 'leanprover/lean4:v4.16.0':
    errors.append('Lean toolchain requires review: ' + toolchain.read_text().strip())
model = Path(config.PROVER_MODEL_PATH)
index = model / 'model.safetensors.index.json'
if not index.exists():
    errors.append('Missing model shard index')
else:
    shards = set(json.loads(index.read_text())['weight_map'].values())
    if any(not (model / shard).exists() for shard in shards):
        errors.append('Model weights download incomplete')
print(json.dumps({'code_and_data_checks': 'PASS', 'packages': packages,
    'retrieval': {'endpoint': config.API_CONFIG['lean_search'], 'status': 'NOT_PROBED'},
    'issues': errors, 'runtime_status': 'BLOCKED' if errors else 'REQUIRES_E2E_SMOKE'}, indent=2))
sys.exit(1 if errors else 0)
