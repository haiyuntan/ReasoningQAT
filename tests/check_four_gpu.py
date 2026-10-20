"""Exercise the actual scheduler with simulated child processes, plus shared service ownership."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_four_gpu as scheduler
import retrieval_service


class FourGpuTests(unittest.TestCase):
    def run_phase(self, phase):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign = scheduler.Campaign(root, 'digest')
            events, processes = [], []

            class Child:
                def __init__(self, command, **kwargs):
                    self.run = command[command.index('--run') + 1]
                    self.pid, self.polls = 10000 + len(processes), 0
                    self.gpu = kwargs['env']['CUDA_VISIBLE_DEVICES']
                    self.done = False
                    self.out = Path(kwargs['env']['REALPROVER_OUTPUT'])
                    processes.append(self)
                    events.append(('start', self.run))
                    self_test.assertEqual(kwargs['env']['REALPROVER_SHARED_RETRIEVAL'], '1')
                    active = [p for p in processes if not p.done]
                    self_test.assertEqual(len(active), len({p.gpu for p in active}))
                    self_test.assertLessEqual(len(active), 4 if phase == 'prepare' else 3)

                def poll(self):
                    self.polls += 1
                    if self.polls < (3 if self.run == 'P3' else 2):
                        return None
                    if not self.done:
                        self.done = True
                        events.append(('done', self.run))
                        if phase == 'prepare':
                            (self.out / 'prepared.json').write_text(json.dumps({
                                'run': self.run, 'mode': 'formal', 'fingerprint': 'digest',
                                'evidence': [{'path': 'fake', 'sha256': 'fake'}]}))
                        else:
                            (self.out / 'receipt.json').write_text('{}')
                    return 0

            self_test = self
            with patch.object(scheduler, 'fingerprint', return_value='digest'), \
                 patch.object(scheduler.subprocess, 'Popen', Child), \
                 patch.object(scheduler.time, 'sleep'), \
                 patch.object(scheduler, 'validate_receipt') as validate:
                campaign.phase(phase, [0, 1, 2, 3] if phase == 'prepare' else [0, 1, 2])
            self.assertEqual({p.run for p in processes}, set(scheduler.RUNS))
            self.assertNotIn('P0', {p.run for p in processes})
            if phase == 'prepare':
                for run in ('P4', 'P6'):
                    self.assertLess(events.index(('done', 'P3')), events.index(('start', run)))
            else:
                self.assertEqual(validate.call_count, 6)
                self.assertNotIn('3', {p.gpu for p in processes})

    def test_preparation_dependencies_and_exclusive_gpu_assignment(self):
        self.run_phase('prepare')

    def test_evaluation_reserves_gpu3_and_validates_all_receipts(self):
        self.run_phase('evaluate')

    def test_shared_client_cannot_restart_service(self):
        with patch.dict(os.environ, {'REALPROVER_SHARED_RETRIEVAL': '1'}), \
             patch.object(retrieval_service, 'start') as start, \
             patch.object(retrieval_service.urllib.request, 'urlopen',
                          side_effect=urllib.error.URLError('offline')):
            with self.assertRaisesRegex(RuntimeError, 'clients must not start or stop'):
                retrieval_service.ready()
            start.assert_not_called()

    def test_fingerprint_change_prevents_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = scheduler.Campaign(Path(directory), 'before')
            with patch.object(scheduler, 'fingerprint', return_value='after'), \
                 patch.object(scheduler.subprocess, 'Popen') as launch:
                with self.assertRaisesRegex(RuntimeError, 'changed'):
                    campaign.phase('prepare', [0, 1, 2, 3])
                launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
