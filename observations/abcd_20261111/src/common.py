import hashlib,json,os,time
from pathlib import Path
W=Path(__file__).resolve().parents[1]
ROOT=W.parent
P=json.loads((W/'configs/protocol.json').read_text())
def digest(x):return hashlib.sha256(x.encode() if isinstance(x,str) else x).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 p=Path(p);assert p.resolve().is_relative_to(W.resolve()),p
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False));q.replace(p)
def write_rows(p,rows):
 p=Path(p);assert p.resolve().is_relative_to(W.resolve())
 p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
def read_rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def record(stage,status,**kw):
 save(W/'manifests'/f'{stage}.json',dict(stage=stage,status=status,timestamp=time.time(),**kw));print(stage,status,kw,flush=True)

def resources():
 path=W/"configs/runtime_resources.json"
 return json.loads(path.read_text()) if path.exists() else {"allocator_cap_gib":P["allocator_cap_gib"],"duty_cycle":P["duty_cycle"],"candidate_batch_size":1,"gpu_free_reserve_mib":P["gpu_free_reserve_mib"],"batch_score_max_abs_tolerance":.02}
