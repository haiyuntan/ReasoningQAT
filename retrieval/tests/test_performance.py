import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

path=Path(__file__).resolve().parents[2]/'manager/thirdparty/performance.py'
spec=importlib.util.spec_from_file_location('performance',path)
performance=importlib.util.module_from_spec(spec); spec.loader.exec_module(performance)


class PerformanceTests(unittest.TestCase):
    def setUp(self): performance.reset()

    def test_return_and_exception_are_preserved(self):
        @performance.timed('lean_tactic')
        def operation(value):
            if value: raise ValueError('original')
            return 42
        self.assertEqual(operation(False),42)
        with self.assertRaisesRegex(ValueError,'original'): operation(True)
        self.assertEqual(performance.snapshot()['phases']['lean_tactic']['calls'],2)

    def test_reset_does_not_keep_old_attempt_counters(self):
        with performance.measure('generation'): pass
        performance.reset()
        self.assertEqual(performance.snapshot()['phases'],{})

    def test_missing_output_directory_never_changes_search_result(self):
        with patch.dict(performance.os.environ,{'REALPROVER_OUTPUT':'/nonexistent/diagnostic-output'}):
            performance.emit()
        self.assertGreaterEqual(performance._io_errors,1)


if __name__=='__main__': unittest.main()
