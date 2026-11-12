import argparse, hashlib, json, os, time
from pathlib import Path
W=Path(__file__).resolve().parents[1]
ROOT=W.parent
def digest(x):
    if not isinstance(x,(str,bytes)): x=json.dumps(x,sort_keys=True,ensure_ascii=False)
    return hashlib.sha256(x.encode() if isinstance(x,str) else x).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
    return h.hexdigest()
def read(p): return json.loads(Path(p).read_text())
def rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def save(p,x):
    p=Path(p); assert p.resolve().is_relative_to(W)
    p.parent.mkdir(parents=True,exist_ok=True); q=p.with_name(p.name+f'.{os.getpid()}.partial')
    q.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)); q.replace(p)
def jsonl(p,x):
    p=Path(p);q=p.with_suffix('.partial');q.parent.mkdir(parents=True,exist_ok=True)
    q.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in x));q.replace(p)
def protocol(): return read(W/'configs/protocol.json')
def status(stage,state,**kw):
    x=dict(stage=stage,status=state,updated_at=time.time(),pid=os.getpid(),**kw)
    save(W/'manifests'/f'{stage}.json',x);print(stage,state,kw,flush=True)
def args():
    a=argparse.ArgumentParser();a.add_argument('--dry-run',action='store_true');a.add_argument('--verify-only',action='store_true');a.add_argument('--resume',action='store_true');a.add_argument('--protocol',default=str(W/'configs/protocol.json'));a.add_argument('--max-cases',type=int);a.add_argument('--phase',default='pilot',choices=['smoke','pilot','confirm']);return a.parse_args()
def identity(): return sha(W/'configs/protocol.json')
def check_resume(p,expected):
    if not p.exists():return None
    r=read(p)
    assert r.get('identity')==expected,'Receipt identity mismatch: '+str(p)
    return r
