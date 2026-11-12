import hashlib,json,os,time
from pathlib import Path
W=Path(__file__).resolve().parents[1];ROOT=W.parent;PILOT=ROOT/'observation_calibration_v3_20261111'
P=json.loads((W/'configs/protocol.json').read_text())
def digest(value):return hashlib.sha256(value.encode() if isinstance(value,str) else value).hexdigest()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,data):
 path=Path(path);assert path.resolve().is_relative_to(W.resolve()),path
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
 tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(path)
def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def jsonl(path,data):
 path=Path(path);assert path.resolve().is_relative_to(W.resolve())
 tmp=path.with_suffix('.tmp');tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data));tmp.replace(path)
def status(stage,state,**kw):save(W/'manifests'/f'{stage}.json',dict(stage=stage,status=state,time=time.time(),**kw));print(stage,state,kw,flush=True)
def protocol_sha():return sha(W/'configs/protocol.json')
def atomic_torch(path,value):
 import torch
 path=Path(path);assert path.resolve().is_relative_to(W.resolve());path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)
def builder():
 import importlib.util
 spec=importlib.util.spec_from_file_location('frozen_prompt',W/'runtime/source/manager/manage/prompt_manage.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m.PromptManage
