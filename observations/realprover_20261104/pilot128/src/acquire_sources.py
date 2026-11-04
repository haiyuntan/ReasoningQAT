"""Network preparation; no writes outside OBS, no production imports."""
import concurrent.futures
import json
import os
import shutil
import subprocess
import urllib.request
from common import OBS, ROOT, atomic, sha

def fetch(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + '.partial')
    with urllib.request.urlopen(url, timeout=90) as response, tmp.open('wb') as f:
        shutil.copyfileobj(response, f)
    tmp.replace(target)

def dataset():
    url = 'https://huggingface.co/api/datasets/FrenzyMath/state_tactic_pairs'
    info = json.load(urllib.request.urlopen(url, timeout=60))
    atomic(OBS/'manifests/state_tactic_hub.json', info)
    records = []
    for item in info['siblings']:
        name = item['rfilename']
        if name.endswith('.parquet') or name == 'README.md':
            path = OBS/'assets/state_tactic_pairs'/name
            if not path.exists():
                fetch(f'https://huggingface.co/datasets/FrenzyMath/state_tactic_pairs/resolve/{info["sha"]}/{name}', path)
            records.append({'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size})
    return {'revision':info['sha'], 'files':records}

REPOS = {
 'User-ReasoningQAT':('haiyuntan/ReasoningQAT','feat/realprover-reasoningqat-w2','research'),
 'User-REALPROVER':('haiyuntan/ReasoningQAT','feat/realprover-evaluation-adapter','REALPROVER'),
 'ReasoningQAT':('yasu0001/ReasoningQAT',None,'ReasoningQAT'),
 'EfficientQAT':('OpenGVLab/EfficientQAT',None,'qat_baselines/EfficientQAT'),
 'ReQAT':('aiha-lab/ReQAT',None,None),
 'REAL-Prover':('frenzymath/REAL-Prover',None,'upstream/REAL-Prover')}

def source(name, spec):
    repo, branch, local = spec
    target = OBS/'upstream'/name
    if not (target/'.git').exists():
        if local and (ROOT/local/'.git').exists():
            # --no-hardlinks prevents metadata/object coupling to production.
            cmd = ['git','clone','--no-hardlinks',str(ROOT/local),str(target)]
        else:
            cmd = ['git','clone','--depth','1','https://github.com/'+repo+'.git',str(target)]
            if branch:
                cmd[2:2] = ['--branch',branch]
        subprocess.run(cmd,check=True,timeout=180,capture_output=True)
    commit = subprocess.check_output(['git','-C',str(target),'rev-parse','HEAD'],text=True).strip()
    return {'name':name,'commit':commit,'remote':'https://github.com/'+repo,'source':local or 'remote'}

PAPERS = {
 'REAL-Prover':('https://arxiv.org/pdf/2505.20613',None),
 'ReasoningQAT':('https://proceedings.iclr.cc/paper_files/paper/2026/file/209423f076b6479ab3a4f45886e30306-Paper-Conference.pdf','papers/reasoningqat_iclr2026.pdf'),
 'EfficientQAT':('https://aclanthology.org/2025.acl-long.498.pdf','papers/efficientqat_acl2025.pdf'),
 'ReQAT':('https://raw.githubusercontent.com/mlresearch/v306/main/assets/lee26an/lee26an.pdf',None)}

def paper(name, spec):
    url, local = spec
    path = OBS/'papers'/(name+'.pdf')
    if not path.exists():
        if local and (ROOT/local).exists():
            shutil.copyfile(ROOT/local,path)
        else:
            fetch(url,path)
    assert path.open('rb').read(5) == b'%PDF-', 'Not a PDF'
    import fitz
    document = fitz.open(path)
    assert len(document) > 0
    info = {'pages':len(document),'metadata':document.metadata,'validation':'PyMuPDF parser; pdfinfo unavailable'}
    text = OBS/'papers'/(name+'.txt')
    text.write_text('\n'.join(p.get_text() for p in document))
    return {'name':name,'url':url,'sha256':sha(path),'pdfinfo':info}

def main():
    tasks = [('dataset', dataset)]
    tasks += [('source:'+k,lambda k=k,v=v:source(k,v)) for k,v in REPOS.items()]
    tasks += [('paper:'+k,lambda k=k,v=v:paper(k,v)) for k,v in PAPERS.items()]
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(fn):name for name,fn in tasks}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                results[name] = {'status':'DONE','result':future.result()}
            except Exception as error:
                results[name] = {'status':'SOURCE_BLOCKED','error':str(error)}
            atomic(OBS/'manifests/acquisition.json',results)
            print(name,results[name]['status'],flush=True)

if __name__ == '__main__':
    main()
