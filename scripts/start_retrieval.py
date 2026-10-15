"""Start the restored local retrieval service with its existing interpreter."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import urllib.error
import urllib.request

RESEARCH = Path(__file__).resolve().parents[1]
ROOT = RESEARCH.parent
sys.path.insert(0, str(RESEARCH / 'adapters'))
from common import python_for, save


def main():
    port = int(os.environ.get('LEANSEARCH_PORT', '8080'))
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=5) as response:
            health = json.load(response)
        expected = hashlib.sha256((RESEARCH / 'manifests/retrieval_assets.json').read_bytes()).hexdigest()
        if (not health.get('adapter_loaded')
                or health.get('source_revision') != '3e5987dc1009a7addc935bcd7b4ae777738a0848'
                or health.get('asset_manifest_sha256') != expected):
            raise RuntimeError('Port belongs to an unexpected service')
        print(json.dumps({'status': 'ALREADY_READY', 'health': health}))
        return
    except urllib.error.URLError:
        pass
    output = ROOT / 'runs/realprover_retrieval'
    output.mkdir(parents=True, exist_ok=True)
    process_file = output / 'process.json'
    if process_file.exists():
        pid = json.loads(process_file.read_text())['pid']
        command = Path(f'/proc/{pid}/cmdline')
        if command.exists() and b'REALPROVER/retrieval/server.py' in command.read_bytes():
            raise RuntimeError(f'Retrieval process {pid} is still starting; inspect server.log')
    env = dict(os.environ, HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
               USE_TF='0', USE_FLAX='0',
               TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
    env.pop('PYTHONPATH', None)
    server = ROOT / 'REALPROVER/retrieval/server.py'
    with (output / 'server.log').open('a') as stream:
        process = subprocess.Popen([python_for('eval'), '-u', str(server)], cwd=server.parent,
                                   env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    save(process_file, {'pid': process.pid, 'log': str(output / 'server.log'), 'port': port})
    print(json.dumps({'status': 'STARTING', 'pid': process.pid, 'log': str(output / 'server.log')}))


if __name__ == '__main__':
    main()
