"""Exercise the real importer with immutable models and partial verified results."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import launch_evaluation_resume as launcher


class ImportTests(unittest.TestCase):
    def test_attempt_import_preserves_bytes_and_partial_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder)/'source', Path(folder)/'target'
            launcher.save(source/'control/status.json', {'fingerprint': 'old'})
            base = source/'formal/P1/eval'
            base.mkdir(parents=True)
            (base/'config.toml').write_text((launcher.ROOT/'REALPROVER/configs/p0_candidate.toml').read_text())
            (base/'retrieval.jsonl').write_text('')
            rows = [json.loads(s) for s in (launcher.ROOT/'REALPROVER/data/minif2f_test.jsonl').read_text().splitlines()]
            original = {}
            for row, indices, success in [(rows[0], [0], True), (rows[1], [0, 2], False)]:
                for index in indices:
                    path = base/'native/generated'/str(row['id'])/f'{row["id"]}_{index}.json'
                    data = {'formal_statement': row['formal_statement'], 'collect_results': [{'success': success}],
                            'pipeline_verifier': {'executed': success, 'success': success}}
                    if not success:
                        data['pipeline_attempt'] = {'status': 'timeout', 'consumes_attempt': True}
                    launcher.save(path, data)
                    original[path.relative_to(base)] = path.read_bytes()
            summaries = launcher.import_attempts(source, target, 'new')
            self.assertEqual(summaries['P1'], {'completed': 1, 'successes': 1, 'attempts': 3, 'timeouts': 2})
            for relative, data in original.items():
                self.assertEqual((base/relative).read_bytes(), data)
                self.assertEqual((target/'formal/P1/eval'/relative).read_bytes(), data)
            lineage = launcher.load(target/'formal/P1/eval/evaluation_resume.json')
            self.assertIn(1, lineage['problems'][1]['missing_slots'])
            self.assertNotIn(2, lineage['problems'][1]['missing_slots'])

    def test_prepared_import_rebinds_previous_lineage_without_conflicting_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder)/'source', Path(folder)/'target'
            launcher.save(source/'control/status.json', {'fingerprint': 'old', 'commits': {'research': 'old'}})
            for method in ('P1', 'P2', 'P3', 'P4', 'P5', 'P6'):
                base = source/'formal'/method
                launcher.save(base/'reuse_manifest.json', {'previous': True})
                weights = base/'train/native/weights.pt'
                weights.parent.mkdir(parents=True)
                weights.write_bytes(b'unchanged weights')
                launcher.save(base/'prepared.json', {'run': method, 'mode': 'formal', 'fingerprint': 'old',
                     'evidence': [{'path': str(p), 'sha256': launcher.sha(p)} for p in (weights, base/'reuse_manifest.json')]})
            for stage in ('stage1', 'stage2'):
                launcher.save(source/f'formal/shared_{stage}.json', {})
            with patch.object(launcher, 'audit_source', return_value={'status': 'PASS'}), \
                 patch.object(launcher.subprocess, 'check_output', side_effect=lambda cmd: (launcher.RESEARCH/cmd[-1].split(':', 1)[1]).read_bytes()):
                launcher.import_prepared(source, target, 'new')
            for method in ('P1', 'P2', 'P3', 'P4', 'P5', 'P6'):
                record = launcher.load(target/'formal'/method/'prepared.json')
                self.assertEqual(record['fingerprint'], 'new')
                self.assertEqual(len({p['path'] for p in record['evidence']}), len(record['evidence']))
                for item in record['evidence']:
                    self.assertEqual(launcher.sha(Path(item['path'])), item['sha256'])
                self.assertEqual(launcher.load(source/'formal'/method/'reuse_manifest.json'), {'previous': True})


if __name__ == '__main__':
    unittest.main()
