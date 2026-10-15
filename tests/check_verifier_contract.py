import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

runtime = Path(__file__).resolve().parents[2] / 'REALPROVER'
sys.path.insert(0, str(runtime))
spec = importlib.util.spec_from_file_location('verifier_contract', runtime / 'manager/thirdparty/verifier.py')
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


class VerifierTests(unittest.TestCase):
    def test_response_contract(self):
        for response, code, expected in [('{}', 0, False), ('{"env":0}', 1, False),
                ('{"env":0}', 0, True), ('{"env":0,"sorries":[{}]}', 0, False),
                ('{"env":0,"messages":[{"severity":"error","data":"bad proof"}]}', 0, False)]:
            with self.subTest(response=response, code=code), patch.object(verifier.subprocess, 'run',
                    return_value=SimpleNamespace(stdout=response, returncode=code)):
                self.assertEqual(verifier.verify_proof('theorem x : True := by trivial'), expected)


if __name__ == '__main__':
    unittest.main()
