"""Exercise the real batch retry loop without loading a GPU model."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import os
import traceback

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('attempt_failure', ROOT/'REALPROVER/manager/service/attempt_failure.py')
failure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(failure)


class LeanTimeout(TimeoutError):
    seconds = 1200
    operation = 'runTactic'


class SearchError(Exception):
    def __init__(self, error):
        self.error_type = error
        self.error_data = []


source = ROOT/'REALPROVER/manager/service/batch_main_service.py'
tree = ast.parse(source.read_text())
nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in ('_process_run', 'search_check')]
namespace = {'os': os, 'json': json, 'traceback': traceback,
             'InteractiveResponseTimeout': LeanTimeout, 'SearchError': SearchError,
             'timeout_result': failure.timeout_result,
             'conf': SimpleNamespace(config=SimpleNamespace(LEAN_ENV_PATH='/tmp', LEAN_TEST_PATH='/tmp')),
             'profiler': SimpleNamespace(start=lambda *a: None, stop=lambda *a: None),
             'logging': SimpleNamespace(error=lambda *a, **k: None),
             'error_logging': lambda *a: None, 'verify_proof': lambda *a: True,
             'CommonUtil': SimpleNamespace(write_to_json_file=lambda p, d: Path(p).write_text(json.dumps(d)))}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)


class AttemptTests(unittest.TestCase):
    def run_attempts(self, directory, outcomes, limit=64):
        queue = iter([{'id': 'x', 'formal_statement': 'theorem x : True := by sorry'}, None])
        outcomes = iter(outcomes)
        def process(**kwargs):
            item = next(outcomes)
            if isinstance(item, Exception):
                raise item
            return item
        service = SimpleNamespace(gpus_list=[0], generator={0: SimpleNamespace(calls=[])},
            queue=SimpleNamespace(get=lambda: next(queue)), result_dir=str(directory),
            max_retries=limit, process_one=process, info={}, get_info=lambda: {},
            parse_result=lambda statement, result: {'formal_statement': statement,
                'collect_results': [{'success': True, 'calls': [1]}], 'formal_proof': 'by trivial'})
        namespace['_process_run'](service, 0)
        return sorted((directory/'generated/x').glob('*.json'))

    def test_timeout_then_success_continues_and_counts_once(self):
        with tempfile.TemporaryDirectory() as d:
            files = self.run_attempts(Path(d), [LeanTimeout('slow tactic'), 'success'])
            self.assertEqual(len(files), 2)
            first, second = [json.loads(p.read_text()) for p in files]
            self.assertFalse(first['collect_results'][0]['success'])
            self.assertTrue(first['pipeline_attempt']['consumes_attempt'])
            self.assertTrue(second['collect_results'][0]['success'])

    def test_all_64_timeouts_exhaust_budget_even_without_calls(self):
        with tempfile.TemporaryDirectory() as d:
            files = self.run_attempts(Path(d), [LeanTimeout('slow')]*64)
            self.assertEqual(len(files), 64)
            self.assertTrue(namespace['search_check'](str(Path(d)/'generated/x'), 64))

    def test_wrapped_search_timeout_is_counted(self):
        with tempfile.TemporaryDirectory() as d:
            files = self.run_attempts(Path(d), [SearchError(LeanTimeout('slow')), 'success'])
            self.assertEqual(len(files), 2)

    def test_infrastructure_error_is_not_model_failure(self):
        for error in [RuntimeError('broken protocol'), SearchError(ValueError('bad transport'))]:
            with tempfile.TemporaryDirectory() as d:
                with self.assertRaises(type(error)):
                    self.run_attempts(Path(d), [error])
                self.assertEqual(list((Path(d)/'generated/x').glob('*.json')), [])


if __name__ == '__main__':
    unittest.main()
