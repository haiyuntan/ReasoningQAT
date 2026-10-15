"""Shared contracts for REAL-Prover migration adapters (stdlib only)."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

RESEARCH = Path(__file__).resolve().parents[1]
ROOT = RESEARCH.parent
MODEL = ROOT / 'models/REAL-Prover-v1-bf16'
VENDOR = RESEARCH / 'vendor/reasoningqat'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def python_for(role):
    config = load(RESEARCH / 'configs/runtime.json')
    value = os.environ.get(f'REALPROVER_{role.upper()}_PYTHON', config['python'][role])
    p = Path(value)
    if not p.is_absolute():
        p = ROOT / p
    if not p.is_file():
        raise RuntimeError(f'Missing {role} interpreter: {p}')
    return str(p)


def execute(argv, cwd=ROOT, env=None):
    print('EXECUTE', json.dumps([str(x) for x in argv]), flush=True)
    subprocess.run([str(x) for x in argv], cwd=cwd, env=env, check=True,
                   timeout=int(os.environ.get('REALPROVER_STAGE_TIMEOUT', '604800')))


def verify_model():
    metadata = load(RESEARCH / 'manifests/model_revision.json')
    expected = load(RESEARCH / 'configs/pipeline.json')['model']
    if metadata['revision'] != expected['revision'] or metadata['repo_id'] != expected['repo']:
        raise RuntimeError('Unexpected model repository/revision')
    if metadata['status'] not in ('DOWNLOADED_REQUIRES_LOAD_CHECK', 'VERIFIED'):
        raise RuntimeError(f'Model download is incomplete: {metadata["status"]}')
    index = load(MODEL / 'model.safetensors.index.json')
    shards = sorted(set(index['weight_map'].values()))
    if sum((MODEL / f).stat().st_size for f in shards) != index['metadata']['total_size']:
        # Safetensors files contain headers in addition to tensor bytes.
        if sum((MODEL / f).stat().st_size for f in shards) < index['metadata']['total_size']:
            raise RuntimeError('Model shards shorter than tensor metadata')
    config = load(MODEL / 'config.json')
    if config['model_type'] != 'qwen2':
        raise RuntimeError('REAL-Prover must use Qwen2 architecture')
    hashes = {f: sha(MODEL / f) for f in shards}
    for filename, digest in hashes.items():
        item = metadata.get('files', {}).get(filename, {})
        if item.get('sha256') != digest or item.get('size') != (MODEL / filename).stat().st_size:
            raise RuntimeError(f'Model manifest/hash mismatch: {filename}')
    return {'revision': expected['revision'], 'model_type': config['model_type'], 'shards': hashes}


def data_environment(mode):
    env = dict(os.environ, HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false', RQ_RUN_MODE=mode,
               PYTHONPATH=str(VENDOR) + os.pathsep + str(VENDOR / 'repro_scripts'),
               RQ_DATA=str(ROOT / 'datasets'))
    env['RQ_OPENTHOUGHTS_PARQUET_GLOB'] = str(ROOT / 'datasets/openthoughts3/**/*.parquet')
    env['RQ_FINEWEB_PARQUET'] = str(ROOT / 'datasets/fineweb-edu/sample/10BT/000_00000.parquet')
    # No implicit reuse of Qwen3 smoke token caches or chat fixtures.
    env.pop('RQ_SMOKE_OPENTHOUGHTS', None)
    return env


def install_vendor():
    sys.path.insert(0, str(VENDOR))
    sys.path.insert(0, str(VENDOR / 'repro_scripts'))
    from runtime_compat import ensure_dtensor_alias
    ensure_dtensor_alias()


def model_signature(path):
    path = Path(path)
    return {p.name: sha(p) for p in sorted(path.iterdir())
            if p.is_file() and (p.suffix in ('.safetensors', '.pt', '.json') or p.name == 'tokenizer.model')}
