"""Record tested independent prefixes and immutable dependency download identities."""
import json
import os
import subprocess
import urllib.request
from pathlib import Path
from common import OBS, atomic, read

def main():
    for name in ['data','forward']:
        python=OBS/'envs'/name/'bin/python'
        env=os.environ.copy();env['PYTHONNOUSERSITE']='1';env['CUDA_VISIBLE_DEVICES']=''
        check=subprocess.check_output([str(python),'-m','pip','check'],env=env,text=True)
        freeze=subprocess.check_output([str(python),'-m','pip','freeze'],env=env,text=True)
        (OBS/'manifests'/(name+'-pip-freeze.txt')).write_text(freeze)
        explicit=subprocess.check_output(['/HOME/sysu_qling/sysu_qling_3/HDD_POOL/anaconda3/bin/conda','list','--prefix',str(OBS/'envs'/name),'--explicit'],text=True)
        (OBS/'configs'/('conda-'+name+'-explicit.txt')).write_text(explicit)
        # Retain precise wheel URLs + published content SHA. Old cache is read only.
        install=[]
        for report in ([OBS/'manifests/torch-cache-install.json',OBS/'manifests/forward-install.json'] if name=='forward' else [OBS/'manifests/data-install.json']):
            install.extend(read(report)['install'])
        urls=[]
        for item in install:
            info=item['download_info'];url=info['url'];hashes=info.get('archive_info',{}).get('hashes',{})
            if url.startswith('file:'):
                package=item['metadata']['name'];version=item['metadata']['version']
                published=json.load(urllib.request.urlopen(f'https://pypi.org/pypi/{package}/{version}/json',timeout=30))
                matching=[x for x in published['urls'] if x['digests'].get('sha256')==hashes.get('sha256')]
                if not matching:raise RuntimeError(f'Cached wheel not matched to published PyPI hash: {package} {version}')
                url=matching[0]['url']
            urls.append({'name':item['metadata']['name'],'version':item['metadata']['version'],'url':url,'hashes':hashes})
        atomic(OBS/'manifests'/(name+'-wheel-origins.json'),urls)
        # JSON is valid YAML 1.2; keeps exact URLs explicit without an extra parser.
        spec={'name':'obs-'+name,'channels':['https://repo.anaconda.com/pkgs/main'],'dependencies':['python=3.11.18','pip',{'pip':freeze.splitlines()}],'rebuild_note':'Install conda explicit file, then exact wheel URLs/hashes in manifests/'+name+'-wheel-origins.json. Freeze contains transitive versions. No production prefix reused.','pip_check':check.strip()}
        (OBS/'configs'/('observation_env_'+name+'.yml')).write_text(json.dumps(spec,indent=2)+'\n')
        print(name,'locked',flush=True)

if __name__=='__main__':main()
