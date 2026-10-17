"""A direct formal restart requires an explicit batch-bound authorization."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from experiment_pipeline import formal_smoke_gate


class WaiverTest(unittest.TestCase):
    def test_binding_and_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report = root/'validation.json'
            report.write_text(json.dumps({'status': 'PASS'}))
            authorization = {'fingerprint': 'current', 'results': str(root.resolve()),
                             'scope': 'skip_smoke_for_direct_formal_restart',
                             'user_instruction': '修复后清理然后不用再次冒烟直接重启正式实验',
                             'validation_report': str(report)}
            waiver = root/'waiver.json'
            waiver.write_text(json.dumps(authorization))
            formal_smoke_gate(root, 'current', waiver)
            with self.assertRaises(ValueError):
                formal_smoke_gate(root, 'changed', waiver)
            with self.assertRaises(ValueError):
                formal_smoke_gate(root/'other_batch', 'current', waiver)
            report.write_text(json.dumps({'status': 'FAIL'}))
            with self.assertRaises(ValueError):
                formal_smoke_gate(root, 'current', waiver)

    def test_default_still_requires_all_smoke(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                formal_smoke_gate(Path(folder), 'current')


if __name__ == '__main__':
    unittest.main()
