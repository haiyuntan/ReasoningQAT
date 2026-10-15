"""CPU-only regression checks for the scientific campaign gate."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('pipeline', Path(__file__).resolve().parents[1] / 'scripts/experiment_pipeline.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


class GateTests(unittest.TestCase):
    def test_source_changes_and_build_output_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'research/adapter.py'
            source.parent.mkdir()
            source.write_text('version = 1\n')
            initial = p.fingerprint(root)
            source.write_text('version = 2\n')
            changed = p.fingerprint(root)
            self.assertNotEqual(initial, changed)
            cache = root / 'REALPROVER/dependencies/cache/generated.json'
            cache.parent.mkdir(parents=True)
            cache.write_text('{}')
            self.assertEqual(changed, p.fingerprint(root))
            inventory = root / 'REALPROVER/dependencies/source_restore.json'
            inventory.write_text('{"revision": "new"}')
            self.assertNotEqual(changed, p.fingerprint(root))

    def test_receipt_and_failures(self):
        with tempfile.TemporaryDirectory() as d:
            evidence = Path(d) / 'per_problem.json'
            evidence.write_text('verified evidence fixture')
            receipt = {'run': 'P4', 'mode': 'smoke', 'fingerprint': 'current',
                       'status': 'PASS', 'checks': dict.fromkeys(p.requirements('P4'), True),
                       'expected_problems': 2, 'completed_problems': 2,
                       'evidence': [{'path': str(evidence), 'sha256': hashlib.sha256(evidence.read_bytes()).hexdigest()}]}
            with patch.object(Path, 'read_bytes', side_effect=AssertionError('Evidence must be streamed')):
                p.validate_receipt(receipt, 'P4', 'smoke', 'current')
            with self.assertRaises(ValueError):
                p.validate_receipt(receipt, 'P4', 'smoke', 'changed')
            receipt['completed_problems'] = 1
            with self.assertRaises(ValueError):
                p.validate_receipt(receipt, 'P4', 'smoke', 'current')
            receipt['completed_problems'] = 2
            receipt['checks']['teacher_rr_loss'] = False
            with self.assertRaises(ValueError):
                p.validate_receipt(receipt, 'P4', 'smoke', 'current')
            receipt['checks']['teacher_rr_loss'] = True
            evidence.write_text('changed')
            with self.assertRaises(ValueError):
                p.validate_receipt(receipt, 'P4', 'smoke', 'current')

    def test_missing_run_blocks_campaign(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                p.gate(Path(d), 'smoke', p.ORDER, 'current')

    def test_p5_is_independent(self):
        self.assertNotIn('P3', p.DEPENDENCIES['P5'])
        self.assertIn('P3', p.DEPENDENCIES['P4'])
        self.assertIn('P3', p.DEPENDENCIES['P6'])


if __name__ == '__main__':
    unittest.main()
