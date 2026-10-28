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
    def test_frozen_assets_match_migrated_runtime(self):
        protocol = launcher.load(launcher.RESEARCH/'manifests/evaluation_protocol.json')
        for item in protocol['pinned_files']:
            self.assertEqual(launcher.sha(launcher.ROOT/item['path']), item['sha256'], item['path'])

    def test_retrieval_content_change_is_rejected(self):
        original = launcher.load(launcher.RESEARCH/'manifests/retrieval_assets.json')
        state = {'commits': {'research': 'source'}}
        with patch.object(launcher.subprocess, 'check_output', return_value=json.dumps(original).encode()):
            self.assertTrue(launcher.audit_retrieval_identity(state)['content_identical'])
        original['files'][0]['sha256'] = 'changed-content'
        with patch.object(launcher.subprocess, 'check_output', return_value=json.dumps(original).encode()):
            with self.assertRaisesRegex(RuntimeError, 'Retrieval content changed'):
                launcher.audit_retrieval_identity(state)

    def test_stopped_state_rejects_running_source(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError, 'Stop source'):
                launcher.require_stopped(Path(folder), {'status': 'RUNNING'})
            launcher.require_stopped(Path(folder), {'status': 'STOPPED_DISK_QUOTA'})

    def test_completed_inheritance_checks_receipt_and_skips_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder)/'source', Path(folder)/'target'
            src, dst = source/'formal/P1', target/'formal/P1'
            launcher.save(source/'control/status.json', {'fingerprint': 'old'})
            rows = [json.loads(s) for s in (launcher.ROOT/'REALPROVER/data/minif2f_test.jsonl').read_text().splitlines()]
            evidence, imported = [], []
            for row in rows:
                relative = Path('eval/native/generated')/str(row['id'])/f'{row["id"]}_0.json'
                launcher.save(src/relative, {'formal_statement': row['formal_statement'],
                    'collect_results': [{'success': True}], 'pipeline_verifier': {'executed': True, 'success': True}})
                (dst/relative).parent.mkdir(parents=True, exist_ok=True)
                (dst/relative).write_bytes((src/relative).read_bytes())
                evidence.append({'path': str(src/relative), 'sha256': launcher.sha(src/relative)})
                imported.append({'path': str(dst/relative), 'source_path': str(src/relative), 'sha256': launcher.sha(src/relative)})
            launcher.save(dst/'eval/evaluation_resume.json', {'evidence': imported})
            for base in (src, dst):
                launcher.save(base/'model_identity.json', {'model_type': 'qwen2'})
                launcher.save(base/'eval_model/export.json', {'tokenizer_parity': True, 'saved_reload': True,
                    'bits': 2, 'group_size': 128, 'greedy_generation_parity': True, 'samples': 128,
                    'calibration_sha256': launcher.sha(launcher.ROOT/'datasets/numinamath_128/calib.jsonl'),
                    'gptq_reference_validation': {'exact_weight_checks': launcher.load(launcher.ROOT/'models/REAL-Prover-v1-bf16/config.json')['num_hidden_layers'] * 7}})
                (base/'native').mkdir()
            launcher.save(src/'preflight/preflight.json', {'status': 'PASS'})
            launcher.save(src/'resource_lifecycle.json', {'evaluation_finished': True})
            (src/'eval/config.toml').write_text('historical config')
            (src/'eval/retrieval.jsonl').write_text('{"success":true}\n')
            launcher.save(src/'eval/evaluation.json', {'expected_problems': 244, 'completed_problems': 244,
                'successes': 244, 'retrieval_ok': True, 'raw_evidence': evidence,
                'protocol_sha256': launcher.sha(launcher.ROOT/'REALPROVER/configs/p0_candidate.toml')})
            from experiment_pipeline import requirements
            launcher.save(src/'receipt.json', {'run': 'P1', 'mode': 'formal', 'fingerprint': 'old', 'status': 'PASS',
                'checks': {k: True for k in requirements('P1')}, 'expected_problems': 244,
                'completed_problems': 244, 'evidence': evidence})
            summary = {'P1': {'completed': 244, 'successes': 244, 'attempts': 244}}
            self.assertEqual(launcher.inherit_completed(source, target, 'new', summary), ['P1'])
            receipt = launcher.load(dst/'receipt.json')
            self.assertEqual(receipt['evaluation_origin'], 'inherited_completed_no_new_inference')
            launcher.validate_receipt(receipt, 'P1', 'formal', 'new')
            # Evidence mutation must fail closed rather than granting DONE.
            (src/'eval/native/generated'/str(rows[0]['id'])/f'{rows[0]["id"]}_0.json').write_text('{}')
            with self.assertRaises(ValueError):
                launcher.inherit_completed(source, target, 'new', summary)

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
