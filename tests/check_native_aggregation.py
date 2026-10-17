import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'adapters'))
from evaluation import aggregate, make_config


class AggregateTests(unittest.TestCase):
    def test_fixed_denominator_and_duplicate_retries(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            dataset = root / 'data.jsonl'
            rows = [{'id': 'a', 'formal_statement': 'theorem a : True := by sorry'},
                    {'id': 'b', 'formal_statement': 'theorem b : True := by sorry'}]
            dataset.write_text('\n'.join(json.dumps(x) for x in rows))
            output = root / 'generated/a'; output.mkdir(parents=True)
            result = {'formal_statement': rows[0]['formal_statement'], 'collect_results': [{'success': True}],
                      'pipeline_verifier': {'executed': True, 'success': True}}
            for name in ('a_0.json', 'a_1.json'):
                (output / name).write_text(json.dumps(result))
            stats = aggregate(dataset, root)
            self.assertEqual(stats['expected_problems'], 2)
            self.assertEqual(stats['completed_problems'], 1)
            self.assertEqual(stats['successes'], 1)
            self.assertEqual(stats['success_rate'], .5)
            result['pipeline_verifier']['success'] = False
            (output / 'a_1.json').write_text(json.dumps(result))
            with self.assertRaises(ValueError):
                aggregate(dataset, root)

    def test_config_only_changes_model_and_dataset(self):
        with tempfile.TemporaryDirectory() as d:
            dataset, digest = make_config(Path(d), 'smoke', Path(d) / 'test.toml')
            self.assertTrue(dataset.is_file())
            self.assertEqual(len(digest), 64)

    def test_timeout_attempt_is_failure_but_later_success_counts_once(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            dataset = root/'data.jsonl'
            statement = 'theorem a : True := by sorry'
            dataset.write_text(json.dumps({'id': 'a', 'formal_statement': statement}))
            output = root/'generated/a'
            output.mkdir(parents=True)
            failed = {'formal_statement': statement, 'collect_results': [{'success': False}],
                      'pipeline_attempt': {'status': 'timeout', 'consumes_attempt': True}}
            (output/'a_0.json').write_text(json.dumps(failed))
            stats = aggregate(dataset, root)
            self.assertEqual(stats['successes'], 0)
            self.assertEqual(stats['lean_timeout_attempts'], 1)
            successful = {'formal_statement': statement, 'collect_results': [{'success': True}],
                          'pipeline_verifier': {'executed': True, 'success': True}}
            (output/'a_1.json').write_text(json.dumps(successful))
            self.assertEqual(aggregate(dataset, root)['successes'], 1)
            failed['pipeline_attempt']['status'] = 'lean_error'
            (output/'a_2.json').write_text(json.dumps(failed))
            stats = aggregate(dataset, root)
            self.assertEqual(stats['lean_error_attempts'], 1)
            self.assertEqual(stats['successes'], 1)


if __name__ == '__main__':
    unittest.main()
