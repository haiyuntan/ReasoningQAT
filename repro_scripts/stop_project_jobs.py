"""Stop only this workspace's experiment/download processes and descendants."""
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess

ROOT=Path(__file__).resolve().parents[2]
NAMES={'pipeline.py','serve_model.py','eval_paper_5tasks.py','export_baseline.py',
       'export_qat.py','r8_efficientqat_qwen3_w2.py','r9_bitdistiller_qwen3_w2.py',
       'download_ranges.py','background_campaign.py'}
def snapshot():
    result={}
    for line in subprocess.check_output(['ps','-eo','pid,ppid,args'],text=True).splitlines()[1:]:
        pid,parent,command=line.strip().split(None,2)
        result[int(pid)]=(int(parent),command)
    return result

table=snapshot();roots=[]
for pid,(parent,command) in table.items():
    if pid==os.getpid(): continue
    try:
        args=shlex.split(command)
        if not args or not Path(args[0]).name.startswith('python'): continue
        cwd=Path(f'/proc/{pid}/cwd').resolve(strict=True)
        if cwd!=ROOT and ROOT not in cwd.parents: continue
        if any(Path(arg).name in NAMES for arg in args[1:]): roots.append(pid)
    except (ValueError,FileNotFoundError,PermissionError): continue
for pid in roots:
    try: os.kill(pid,signal.SIGSTOP)
    except ProcessLookupError: pass
table=snapshot();selected=set(roots)
while True:
    children={pid for pid,(parent,command) in table.items() if parent in selected}
    if children<=selected: break
    selected.update(children)
stopped=[]
for pid in sorted(selected,reverse=True):
    try: os.kill(pid,signal.SIGTERM);stopped.append(pid)
    except ProcessLookupError: pass
for pid in roots:
    try: os.kill(pid,signal.SIGCONT)
    except ProcessLookupError: pass
result={'stopped_project_pids':stopped,'download_files_deleted':False}
(ROOT/'results/background-restart.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
