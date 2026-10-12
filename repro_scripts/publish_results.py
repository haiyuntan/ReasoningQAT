"""Collect compact verified evidence, commit it, optionally push the reproduction branch."""
import argparse
import json
from pathlib import Path
import re
import subprocess

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO.parent
DEST=REPO/'repro_results'
SECRET=re.compile(rb'(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]+|hf_[A-Za-z0-9]{20,})')
BANNED={'.safetensors','.bin','.pt','.pth','.parquet','.arrow','.ckpt','.pdf','.partial'}

def audit_paths(paths):
    for path in paths:
        if path.suffix in BANNED or path.stat().st_size>2*1024*1024:
            raise ValueError(f'Large/binary result is prohibited: {path.name}')
        if SECRET.search(path.read_bytes()):
            raise ValueError(f'Credential-like content found: {path.name}')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--commit',action='store_true')
    p.add_argument('--push',action='store_true')
    a=p.parse_args()
    DEST.mkdir(exist_ok=True)
    manifests=ROOT/'datasets/manifests'
    source=manifests/'sources.json'
    if source.exists():
        raw=json.loads(source.read_text())
        compact={repo:{k:v for k,v in data.items() if k in ('revision','repo_type','destination','verified_files')}
                 for repo,data in raw.items()}
        (DEST/'sources.json').write_text(json.dumps(compact,indent=2))
    for relative,name in [('results/cpu-loss-check.json','cpu_checks.json'),
                          ('results/cpu-pipeline-check.json','cpu_pipeline_checks.json'),
                          ('results/gpu-model-check.json','gpu_model_check.json'),
                          ('results/gpu_handoff.json','gpu_handoff.json'),
                          ('datasets/manifests/numinamath.json','numinamath.json'),
                          ('datasets/manifests/benchmarks.json','benchmarks.json'),
                          ('results/preparation_status.json','preparation_status.json')]:
        src=ROOT/relative
        if src.exists():
            (DEST/name).write_bytes(src.read_bytes())
    for src in (ROOT/'env').glob('*pip-freeze.txt'):
        (DEST/src.name).write_bytes(src.read_bytes())
    audit_paths([p for p in DEST.rglob('*') if p.is_file()])
    if a.commit:
        subprocess.run(['git','diff','--check'],cwd=REPO,check=True)
        subprocess.run(['git','add','repro_results'],cwd=REPO,check=True)
        staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=REPO,text=True).splitlines()
        if any(not path.startswith('repro_results/') for path in staged):
            raise RuntimeError('Other staged code changes require their own reviewed commit')
        audit_paths([REPO/path for path in staged])
        if staged:
            subprocess.run(['git','commit','-m','repro: publish compact experiment evidence'],cwd=REPO,check=True)
    if a.push:
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()
        if branch!='reproduce/qwen3-1p7b-w2g128':
            raise RuntimeError('Push is limited to the reproduction branch')
        subprocess.run(['git','push','origin',branch],cwd=REPO,check=True)

if __name__=='__main__':
    main()
