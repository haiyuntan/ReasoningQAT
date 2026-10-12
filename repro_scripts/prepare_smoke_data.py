"""Freeze small real records from a completed shard while downloads continue."""
import hashlib
import json
from pathlib import Path
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[2]
sources=json.loads((ROOT/'datasets/manifests/sources.json').read_text())
info=sources['open-thoughts/OpenThoughts3-1.2M']
target=ROOT/'datasets/smoke/openthoughts.jsonl'
manifest=ROOT/'datasets/manifests/smoke_openthoughts.json'
if target.exists() or manifest.exists():
    assert target.exists() and manifest.exists(), 'Incomplete fixture'
    evidence=json.loads(manifest.read_text())
    assert hashlib.sha256(target.read_bytes()).hexdigest()==evidence['sha256']
else:
    parts=[ROOT/info['destination']/name for name,meta in sorted(info['verified_files'].items())
           if (ROOT/info['destination']/name).is_file()
           and (ROOT/info['destination']/name).stat().st_size==meta['size']]
    assert parts, 'No completed shard available'
    source=parts[0]
    rows=next(pq.ParquetFile(source).iter_batches(batch_size=64)).to_pylist()
    assert len(rows)==64 and all(row.get('conversations') for row in rows)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
    evidence={'purpose':'Diagnostic smoke only; not the formal globally shuffled sample',
              'source':str(source.relative_to(ROOT)), 'source_rows':[0,64], 'records':64,
              'source_revision':info.get('revision'),
              'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
              'bytes':target.stat().st_size}
    manifest.write_text(json.dumps(evidence,indent=2))
print(json.dumps(evidence,indent=2))
