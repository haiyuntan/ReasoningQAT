"""The lifecycle manager must never signal an unrelated live process."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import retrieval_service


class LifecycleTests(unittest.TestCase):
    def test_unrelated_pid_is_rejected_without_termination(self):
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
        try:
            with tempfile.TemporaryDirectory() as temporary:
                record = Path(temporary) / 'process.json'
                record.write_text(json.dumps({'pid': child.pid}))
                with patch.object(retrieval_service, 'PROCESS', record):
                    with self.assertRaisesRegex(RuntimeError, 'unrelated PID'):
                        retrieval_service.stop()
                self.assertIsNone(child.poll())
        finally:
            child.terminate()
            child.wait(timeout=5)

    def test_exited_process_is_already_stopped(self):
        child = subprocess.Popen([sys.executable, '-c', 'pass'])
        child.wait(timeout=5)
        with tempfile.TemporaryDirectory() as temporary:
            record = Path(temporary) / 'process.json'
            record.write_text(json.dumps({'pid': child.pid}))
            with patch.object(retrieval_service, 'PROCESS', record):
                retrieval_service.stop()


if __name__ == '__main__':
    unittest.main()
