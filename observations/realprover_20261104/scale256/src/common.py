"""Observation-only paths, atomic output, and content identities."""
import hashlib
import json
import os
from pathlib import Path

OBS = Path(__file__).resolve().parents[1]
ROOT = OBS.parent

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def atomic(path, value):
    path = Path(path).absolute()
    if OBS not in path.parents:
        raise ValueError('Outputs must stay in observation workspace')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.partial')
    with tmp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)

def read(path):
    return json.loads(Path(path).read_text())
