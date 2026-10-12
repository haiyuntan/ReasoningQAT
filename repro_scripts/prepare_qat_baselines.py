"""Pin baseline sources, create isolated entry environments, and save original papers."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import requests
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[2]
REPO=ROOT/'ReasoningQAT'
config=json.loads((REPO/'repro_configs/qat_baselines.json').read_text())
shared=ROOT/'env/train/lib/python3.11/site-packages'
manifest={}
papers={'R8':('efficientqat_acl2025','https://aclanthology.org/2025.acl-long.498.pdf'),
        'R9':('bitdistiller_acl2024','https://aclanthology.org/2024.acl-long.7.pdf')}
for run in ('R8','R9'):
    cfg=config[run]
    source=ROOT/'qat_baselines'/cfg['method']
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    assert commit==cfg['commit'], 'Baseline source revision changed'
    target=ROOT/'env'/cfg['environment']
    subprocess.run([str(ROOT/'env/train/bin/python'),'-m','venv',str(target)],check=True)
    site=target/'lib/python3.11/site-packages'
    (site/'pinned_reasoning_runtime.pth').write_text(str(shared)+'\n')
    python=target/'bin/python'
    checked=subprocess.run([str(python),'-m','pip','check'],capture_output=True,text=True)
    (ROOT/'env'/f'{cfg["environment"]}-pip-freeze.txt').write_text(
        subprocess.check_output([str(python),'-m','pip','freeze'],text=True))
    name,url=papers[run]
    pdf=ROOT/'papers'/f'{name}.pdf'
    if not pdf.exists():
        for attempt in range(4):
            try:
                response=requests.get(url,timeout=(30,120))
                response.raise_for_status()
                assert response.content.startswith(b'%PDF'), 'Invalid PDF'
                pdf.write_bytes(response.content)
                break
            except Exception:
                if attempt==3:
                    raise
                time.sleep(5)
    reader=PdfReader(pdf)
    pdf.with_suffix('.txt').write_text('\n\n'.join(page.extract_text() or '' for page in reader.pages))
    manifest[run]={'method':cfg['method'],'upstream':cfg['upstream'],'commit':commit,
                   'environment':str(target.relative_to(ROOT)),
                   'environment_layout':'Isolated venv; shares pinned train runtime read-only to avoid CUDA library duplication',
                   'dependency_check_passed':checked.returncode==0,
                   'dependency_check':(checked.stdout+checked.stderr).strip(),
                   'paper':{'url':url,'path':str(pdf.relative_to(ROOT)),
                            'sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'pages':len(reader.pages)},
                   'adapter_status':'BLOCKED_PENDING_IMPLEMENTATION','blockers':cfg['blockers']}
    (ROOT/'results/qat-baselines-preparation.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({run:manifest[run]},indent=2),flush=True)
