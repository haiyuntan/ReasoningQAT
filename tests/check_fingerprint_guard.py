"""A transient read cannot silently reset the frozen experiment identity."""
from pathlib import Path
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

RESEARCH=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(RESEARCH/'scripts'),str(RESEARCH/'adapters')]
from fingerprint_guard import FingerprintGuard
from experiment_pipeline import fingerprint


class GuardTests(unittest.TestCase):
    def reader(self, values):
        values=iter(values)
        def read(details=None):
            value=next(values)
            if details is not None:
                details['source/example.py']=value
            return value
        return read

    def test_transient_recovers_without_changing_baseline(self):
        with tempfile.TemporaryDirectory() as folder:
            guard=FingerprintGuard('original',folder,self.reader(['original','transient','original','original']))
            with patch('fingerprint_guard.time.sleep'):
                guard.check([])
            incident=json.loads(next(Path(folder).glob('fingerprint_incident.*.json')).read_text())
            self.assertEqual(incident['status'],'TRANSIENT_RECOVERED')
            self.assertEqual(incident['differences']['source/example.py'],{'expected':'original','observed':'transient'})
            self.assertEqual(guard.digest,'original')

    def test_persistent_change_still_aborts(self):
        with tempfile.TemporaryDirectory() as folder:
            guard=FingerprintGuard('original',folder,self.reader(['original','changed','changed']))
            with patch('fingerprint_guard.time.sleep'), self.assertRaisesRegex(RuntimeError,'Persistent'):
                guard.check([])
            incident=json.loads(next(Path(folder).glob('fingerprint_incident.*.json')).read_text())
            self.assertEqual(incident['status'],'PERSISTENT_CHANGE_ABORTED')

    def test_workers_freeze_and_unrelated_child_survives(self):
        with tempfile.TemporaryDirectory() as folder:
            owned=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True)
            unrelated=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True)
            try:
                guard=FingerprintGuard('original',folder,self.reader(['original','transient','original','original']))
                states=[]
                def verify_pause(_):
                    states.append(Path(f'/proc/{owned.pid}/stat').read_text().rsplit(') ',1)[1].split()[0])
                    self.assertNotEqual(Path(f'/proc/{unrelated.pid}/stat').read_text().rsplit(') ',1)[1].split()[0],'T')
                with patch('fingerprint_guard.time.sleep',side_effect=verify_pause):
                    guard.check([owned])
                self.assertTrue(all(state=='T' for state in states))
                self.assertIsNone(owned.poll())
                self.assertIsNone(unrelated.poll())
            finally:
                for child in (owned,unrelated):
                    child.kill(); child.wait()

    def test_component_recording_preserves_legacy_hash(self):
        old=subprocess.check_output(['git','-C',str(RESEARCH),'show','57d278f:scripts/experiment_pipeline.py'],text=True)
        namespace={'__file__':str(RESEARCH/'scripts/experiment_pipeline.py'),'__name__':'legacy_fingerprint'}
        exec(compile(old,'legacy_fingerprint','exec'),namespace)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'research').mkdir()
            (root/'research/example.py').write_text('print(1)\n')
            with patch.dict(os.environ,{},clear=True):
                details={}
                self.assertEqual(namespace['fingerprint'](root),fingerprint(root,details=details))
                self.assertIn('source/research/example.py',details)


if __name__=='__main__':
    unittest.main()
