"""Verify that the persistent controller never starts formal after smoke/gate failure."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/run_smoke_then_formal.py'
spec = importlib.util.spec_from_file_location('autorun', SOURCE)
autorun = importlib.util.module_from_spec(spec)
spec.loader.exec_module(autorun)


class AutorunTests(unittest.TestCase):
    def run_case(self, failure, frozen=True, formal_only=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'runs').mkdir()
            (root/'research/scripts').mkdir(parents=True)
            (root/'research/configs').mkdir()
            (root/'research/configs/runtime.json').write_text(json.dumps({
                'eval_protocol_status': 'FROZEN' if frozen else 'CANDIDATE'}))
            (root/'research/scripts/experiment_pipeline.py').write_text(
                'import pathlib,sys\n'
                'stage = "gate" if sys.argv[1] == "gate" else sys.argv[3]\n'
                'with pathlib.Path("calls").open("a") as stream: stream.write(stage+"\\n")\n'
                f'sys.exit(9 if stage == {failure!r} else 0)\n')
            args = ['autorun', '--results', str(root/'runs/result')]
            if formal_only:
                args += ['--formal-only', '--smoke-waiver', str(root/'authorization.json')]
            with patch.object(autorun, 'ROOT', root), patch.object(sys, 'argv', args), \
                    patch.object(autorun.subprocess, 'check_output', return_value='testcommit\n'):
                if frozen:
                    code = autorun.main()
                else:
                    with self.assertRaises(RuntimeError):
                        autorun.main()
                    code = 1
            calls = (root/'calls').read_text().splitlines() if (root/'calls').exists() else []
            state = json.loads((root/'runs/result/control/status.json').read_text())
            return code, calls, state['status']

    def test_transition_only_after_smoke_and_gate(self):
        self.assertEqual(self.run_case(None), (0, ['smoke', 'gate', 'formal'], 'COMPLETE'))

    def test_failures_prevent_later_stages(self):
        for failure, expected in [('smoke', ['smoke']), ('gate', ['smoke', 'gate']),
                                  ('formal', ['smoke', 'gate', 'formal'])]:
            with self.subTest(failure=failure):
                self.assertEqual(self.run_case(failure), (9, expected, failure.upper()+'_FAILED'))

    def test_unfrozen_protocol_starts_nothing(self):
        self.assertEqual(self.run_case(None, frozen=False), (1, [], 'FAILED'))

    def test_explicit_direct_formal_restart(self):
        self.assertEqual(self.run_case(None, formal_only=True), (0, ['formal'], 'COMPLETE'))
        self.assertEqual(self.run_case('formal', formal_only=True), (9, ['formal'], 'FORMAL_FAILED'))


if __name__ == '__main__':
    unittest.main()
