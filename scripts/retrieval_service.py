"""Manage only this workspace's recorded retrieval process on the single GPU."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.error
import urllib.request

from start_retrieval import ROOT, RESEARCH, main as start
from common import load, sha, save

PROCESS = ROOT / 'runs/realprover_retrieval/process.json'


def wait_gpu_exit(pid):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                          '--format=csv,noheader,nounits'], text=True, timeout=10)
        if str(pid) not in {line.strip() for line in output.splitlines()}:
            return
        time.sleep(1)
    raise RuntimeError(f'Retrieval PID {pid} still has a GPU context; refusing training')


def alive(pid):
    path = Path(f'/proc/{pid}/stat')
    try:
        return path.read_text().split(') ', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def stop():
    if not PROCESS.exists():
        return
    state = load(PROCESS)
    pid = state['pid']
    if not alive(pid):
        wait_gpu_exit(pid)
        return
    command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    expected = ROOT / 'REALPROVER/retrieval/server.py'
    if not any(Path(os.fsdecode(arg)).resolve() == expected.resolve() for arg in command if arg):
        raise RuntimeError(f'Refusing to signal unrelated PID {pid}')
    print(f'RETRIEVAL stopping managed PID {pid}', flush=True)
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 60
    while alive(pid) and time.monotonic() < deadline:
        time.sleep(1)
    if alive(pid):
        raise RuntimeError(f'Retrieval PID {pid} did not stop; refusing training with retrieval resident')
    wait_gpu_exit(pid)
    save(PROCESS, dict(state, status='STOPPED'))
    print('RETRIEVAL stopped; GPU memory released by process exit', flush=True)


def ready(timeout=1800):
    shared = os.environ.get('REALPROVER_SHARED_RETRIEVAL') == '1'
    if not shared:
        start()
    deadline = time.monotonic() + timeout
    port = int(os.environ.get('LEANSEARCH_PORT', '8080'))
    expected = sha(RESEARCH / 'manifests/retrieval_assets.json')
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=5) as response:
                health = json.load(response)
            if (health.get('status') != 'ready' or health.get('adapter_loaded') is not True
                    or health.get('asset_manifest_sha256') != expected
                    or health.get('source_revision') != load(RESEARCH / 'manifests/retrieval_assets.json')['service_source_revision']):
                raise RuntimeError('Retrieval identity mismatch')
            print('RETRIEVAL ready with pinned asset identity', flush=True)
            return health
        except (urllib.error.URLError, TimeoutError):
            if PROCESS.exists() and not alive(load(PROCESS)['pid']):
                raise RuntimeError('Retrieval exited during startup; inspect runs/realprover_retrieval/server.log')
            # Shared clients may wait through a transient connection/health
            # timeout, but never start or stop the controller-owned service.
            time.sleep(2)
    raise TimeoutError('Retrieval readiness timeout')
