"""Real HTTP regression: a slow serialized retrieval must not block health."""
import concurrent.futures
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'research/scripts'), str(ROOT/'research/adapters')]
import retrieval_service
from common import python_for


class HealthTests(unittest.TestCase):
    def test_shared_client_retries_timeout_without_starting_service(self):
        identity = retrieval_service.load(ROOT/'research/manifests/retrieval_assets.json')
        health = {'status': 'ready', 'adapter_loaded': True,
                  'source_revision': identity['service_source_revision'],
                  'asset_manifest_sha256': retrieval_service.sha(ROOT/'research/manifests/retrieval_assets.json')}
        with tempfile.TemporaryDirectory() as folder, \
             patch.dict(os.environ, {'REALPROVER_SHARED_RETRIEVAL': '1'}), \
             patch.object(retrieval_service, 'PROCESS', Path(folder)/'absent.json'), \
             patch.object(retrieval_service, 'start') as start, \
             patch.object(retrieval_service.time, 'sleep'), \
             patch.object(retrieval_service.urllib.request, 'urlopen',
                          side_effect=[TimeoutError(), io.BytesIO(json.dumps(health).encode())]):
            self.assertEqual(retrieval_service.ready(timeout=5), health)
            start.assert_not_called()

    def test_production_server_keeps_health_responsive_and_queries_serial(self):
        with tempfile.TemporaryDirectory() as folder:
            events = Path(folder)/'events.jsonl'
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            # Execute the production __main__, substituting only the heavy model.
            code = '''
import runpy,sys,types,time,json,threading,os
from pathlib import Path
events=Path(sys.argv[1]); active=0
class Selector:
    def __init__(self):
        self.index=types.SimpleNamespace(ntotal=252444)
        self.model=types.SimpleNamespace(peft_config={'test':True})
        self.asset_manifest_sha256='test'
    def init_model(self): pass
    def retrieve(self,q,num):
        global active
        active+=1
        assert active==1,'Concurrent GPU retrieval would be unsafe'
        with events.open('a') as f: f.write(json.dumps({'event':'start','time':time.monotonic()})+'\\n')
        time.sleep(6)
        with events.open('a') as f: f.write(json.dumps({'event':'end','time':time.monotonic()})+'\\n')
        active-=1
        return [[q]]
worker=types.ModuleType('worker'); worker.PremiseSelector=Selector
util=types.ModuleType('util'); util.RequestUtil=types.SimpleNamespace(gen_fail_data=lambda x:{'error':x},gen_success_data=lambda x:{'data':x})
sys.modules.update(worker=worker,util=util)
runpy.run_path(sys.argv[2],run_name='__main__')
'''
            env = dict(os.environ, LEANSEARCH_PORT=str(port))
            with (Path(folder)/'server.log').open('w') as log:
                process = subprocess.Popen([python_for('eval'), '-u', '-c', code, str(events),
                    str(ROOT/'REALPROVER/retrieval/server.py')], env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    base = f'http://127.0.0.1:{port}'
                    deadline = time.monotonic()+30
                    while True:
                        try:
                            with urllib.request.urlopen(base+'/health', timeout=1) as response:
                                self.assertEqual(json.load(response)['status'], 'ready')
                            break
                        except (OSError, ValueError):
                            if process.poll() is not None or time.monotonic()>deadline:
                                self.fail((Path(folder)/'server.log').read_text())
                            time.sleep(.05)
                    def retrieve(q):
                        request = urllib.request.Request(base+'/retrieve_premises',
                            data=json.dumps({'query':q}).encode(), headers={'Content-Type':'application/json'})
                        with urllib.request.urlopen(request, timeout=20) as response:
                            return json.load(response)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                        first = pool.submit(retrieve, 'first')
                        while not events.exists(): time.sleep(.01)
                        second = pool.submit(retrieve, 'second')
                        began = time.monotonic()
                        with urllib.request.urlopen(base+'/health', timeout=2) as response:
                            self.assertEqual(json.load(response)['premises'], 252444)
                        self.assertLess(time.monotonic()-began, 2)
                        self.assertFalse(first.done())
                        self.assertEqual(first.result()['data'], [['first']])
                        self.assertEqual(second.result()['data'], [['second']])
                    self.assertEqual([json.loads(line)['event'] for line in events.read_text().splitlines()],
                                     ['start','end','start','end'])
                finally:
                    process.terminate()
                    process.wait(timeout=10)


if __name__ == '__main__':
    unittest.main()
