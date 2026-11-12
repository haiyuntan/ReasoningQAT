from pathlib import Path
import shutil,concurrent.futures,json,time
s=Path('/XYFS01/HDD_POOL/sysu_qling/sysu_qling_3/tan/Reasoning/observation_abcd_confirm_v4_20261111/envs/forward');d=Path('/XYFS01/HDD_POOL/sysu_qling/sysu_qling_3/tan/Reasoning/observation_dtr_thinkdeep_20261112/envs/dtr_forward');d.mkdir()
packages=s/'lib/python3.11/site-packages'
tasks=[]
def copy(pair):
 a,b=pair;b.parent.mkdir(parents=True,exist_ok=True)
 if a.is_symlink():b.symlink_to(a.readlink(),target_is_directory=a.is_dir())
 elif a.is_dir():shutil.copytree(a,b,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 else:shutil.copy2(a,b)
for p in s.iterdir():
 if p.name!='lib':tasks.append((p,d/p.name))
for p in (s/'lib').rglob('*'):
 if p.parent==packages:tasks.append((p,d/p.relative_to(s)))
 elif packages not in p.parents and p!=packages and not p.is_dir():tasks.append((p,d/p.relative_to(s)))
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:list(pool.map(copy,tasks))
Path('/XYFS01/HDD_POOL/sysu_qling/sysu_qling_3/tan/Reasoning/observation_dtr_thinkdeep_20261112/manifests/environment_copy_done.json').write_text(json.dumps(dict(status='DONE',tasks=len(tasks),time=time.time())))
