"""Report frozen download targets and resumable bytes without rehashing large files."""
import datetime
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sources=json.loads((ROOT/'datasets/manifests/sources.json').read_text())
report={'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'method':'Single-file streaming HTTP Range resume via hf-mirror; SHA256 before final rename',
        'repositories':{}}
for repo,info in sources.items():
    files=info.get('verified_files',{})
    total=downloaded=complete=0
    for relative,meta in files.items():
        size=meta['size']
        total+=size
        path=ROOT/info['destination']/relative
        if path.exists() and path.stat().st_size==size:
            complete+=1
            downloaded+=size
        else:
            partial=Path(str(path)+'.partial')
            if partial.exists():
                downloaded+=min(size,partial.stat().st_size)
    report['repositories'][repo]={'complete_files':complete,'required_files':len(files),
                                'downloaded_bytes':downloaded,'required_bytes':total,
                                'percent':round(100*downloaded/total,2) if total else 0}
(ROOT/'results/download-status.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
