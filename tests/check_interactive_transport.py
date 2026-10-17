"""Regression checks for stderr saturation, response deadlines and group cleanup."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
config = types.ModuleType('conf.config')
conf = types.ModuleType('conf')
conf.config = config
structs = types.ModuleType('manager.struct.structs')
for name in ('from_json', 'Goal', 'ProofGoal', 'ProofVariable'):
    setattr(structs, name, object)
spec = importlib.util.spec_from_file_location('transport_under_test', ROOT/'REALPROVER/manager/thirdparty/interactive.py')
transport = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {'conf': conf, 'conf.config': config, 'manager.struct.structs': structs}):
    spec.loader.exec_module(transport)


class TransportTest(unittest.TestCase):
    def session(self, code, stderr):
        real_popen = subprocess.Popen
        def launch(*args, **kwargs):
            self.assertIsNone(kwargs['stderr'])
            kwargs['stderr'] = stderr
            return real_popen([sys.executable, '-u', '-c', code], **kwargs)
        with patch.object(transport, 'get_project_toolchain', return_value='test'), \
             patch.object(transport, 'build_interactive', return_value=Path('/unused')), \
             patch.object(transport.subprocess, 'Popen', side_effect=launch):
            return transport.Interactive(Path('/tmp'), Path('Header.lean'))

    def test_stderr_flood_and_buffered_responses(self):
        with tempfile.TemporaryFile() as log:
            session = self.session('import os; os.write(2,b"x"*2000000); print(\'{"a":1}\\n{"b":2}\',flush=True)', log)
            try:
                session.response_timeout = 5
                self.assertEqual(session.read(), {'a': 1})
                self.assertEqual(session.read(), {'b': 2})
                session.proc.wait(timeout=5)
                self.assertEqual(log.seek(0, 2), 2000000)
            finally:
                session.close()

    def test_timeout_kills_launcher_and_child(self):
        with tempfile.TemporaryFile() as log:
            session = self.session('import subprocess,time; p=subprocess.Popen(["sleep","60"]); print(p.pid,flush=True); time.sleep(60)', log)
            child = session.read()
            session.response_timeout = .2
            with self.assertRaises(TimeoutError):
                session.read()
            self.assertIsNotNone(session.proc.poll())
            for _ in range(100):
                status = Path(f'/proc/{child}/stat')
                if not status.exists() or status.read_text().split(') ', 1)[1].startswith('Z'):
                    break
                time.sleep(.01)
            else:
                self.fail('Lean child remained alive after timeout')
            session.close()

    def test_eof_is_error(self):
        with tempfile.TemporaryFile() as log:
            session = self.session('print("partial",end="",flush=True)', log)
            with self.assertRaisesRegex(transport.InteractiveSessionError, 'stdout closed'):
                session.read()

    def test_lean_abort_is_not_swallowed_as_invalid_tactic(self):
        with tempfile.TemporaryFile() as log:
            session = self.session('import os,sys; print(\'{"declName":"test"}\',flush=True); sys.stdin.readline(); os._exit(134)', log)
            self.assertEqual(session.get_next_problem(), 'test')
            with self.assertRaises(transport.InteractiveSessionError) as caught:
                session.run_tactic(0, 'native_decide')
            self.assertNotIsInstance(caught.exception, RuntimeError)
            self.assertEqual(caught.exception.operation, 'runTactic')
            self.assertIsNotNone(session.proc.poll())

    def test_invalid_json_closes_session(self):
        with tempfile.TemporaryFile() as log:
            session = self.session('print("invalid-json",flush=True)', log)
            with self.assertRaises(transport.InteractiveSessionError):
                session.read()


if __name__ == '__main__':
    unittest.main()
