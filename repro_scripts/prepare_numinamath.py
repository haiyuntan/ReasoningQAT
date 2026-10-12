"""Freeze identical NuminaMath-128 for GPTQ/AWQ; scheme is an explicit assumption."""
import hashlib
import json
from pathlib import Path
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
files = sorted((ROOT/'datasets/numinamath_source').rglob('*.parquet'))
if not files:
    raise FileNotFoundError('NuminaMath source shard not downloaded')
rows = pq.ParquetFile(files[0]).iter_batches(batch_size=128)
batch = next(rows).to_pylist()
assert len(batch) == 128
dest = ROOT/'datasets/numinamath_128/calib.jsonl'
dest.parent.mkdir(parents=True, exist_ok=True)
with dest.open('w') as f:
    for i, row in enumerate(batch):
        text = row['problem']
        assert isinstance(text, str) and text.strip()
        f.write(json.dumps({'id':i, 'text':text}, ensure_ascii=False)+'\n')
manifest = {'count':128, 'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
            'source':str(files[0].relative_to(ROOT)),
            'selection':'First 128 original train rows, problem-only. Author indices/template unpublished.',
            'comparability':'IMPLEMENTATION_ASSUMPTION'}
(ROOT/'datasets/manifests/numinamath.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
