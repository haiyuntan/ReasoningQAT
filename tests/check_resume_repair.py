import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'adapters'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from completion_tokens import completion_window
import resume_four_gpu as scheduler


class RepairTests(unittest.TestCase):
    def test_normal_tokens_and_masks_unchanged(self):
        ids, labels, adjusted = completion_window([1, 2, 3, 4, 5], [1, 2], 4)
        self.assertEqual(ids, [1, 2, 3, 4])
        self.assertEqual(labels, [-100, -100, 3, 4])
        self.assertFalse(adjusted)

    def test_long_prompt_keeps_tail_and_answer_without_dropping_sample(self):
        prefix = list(range(30))
        ids, labels, adjusted = completion_window(prefix + [101, 102, 103, 104, 105], prefix, 8)
        self.assertEqual(ids, [26, 27, 28, 29, 101, 102, 103, 104])
        self.assertEqual(labels, [-100] * 4 + [101, 102, 103, 104])
        self.assertTrue(adjusted)

    def test_one_token_completion_is_supervised(self):
        ids, labels, _ = completion_window(list(range(30)) + [99], list(range(30)), 8)
        self.assertEqual(len(ids), 8)
        self.assertEqual(labels, [-100] * 7 + [99])

    def test_invalid_source_still_fails(self):
        with self.assertRaises(ValueError):
            completion_window([1, 2], [1, 2], 8)
        with self.assertRaises(ValueError):
            completion_window([1, 2], [3], 8)

    def test_failed_real_child_does_not_stop_other_gpu_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = scheduler.Adaptive(root, 'digest')
            failed = subprocess.Popen([sys.executable, '-c', 'raise SystemExit(1)'])
            marker = root / 'survived'
            survivor = subprocess.Popen([sys.executable, '-c',
                'import time,pathlib,sys; time.sleep(1); pathlib.Path(sys.argv[1]).write_text("done")', str(marker)])
            streams = [(root / f'{i}.log').open('w') for i in range(2)]
            campaign.active = {0: ('P4', 'prepare', failed, streams[0]),
                               1: ('P5', 'prepare', survivor, streams[1])}
            campaign.jobs['P4'].update(prepare='RUNNING', log='test')
            campaign.jobs['P5'].update(prepare='RUNNING', log='test')
            try:
                failed.wait(timeout=5)
                campaign.poll_jobs()
                self.assertEqual(campaign.jobs['P4']['prepare'], 'FAILED')
                self.assertEqual(campaign.jobs['P4']['evaluate'], 'BLOCKED')
                self.assertIsNone(survivor.poll())
                survivor.wait(timeout=5)
                with patch.object(scheduler, 'prepared'):
                    campaign.poll_jobs()
                self.assertEqual(marker.read_text(), 'done')
                self.assertEqual(campaign.jobs['P5']['prepare'], 'DONE')
            finally:
                if survivor.poll() is None:
                    survivor.kill()
                    survivor.wait()
                for stream in streams:
                    stream.close()

    def test_preparation_failure_does_not_invalidate_p3_initialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            campaign = scheduler.Adaptive(Path(temporary), 'digest')
            campaign.state['data_preflight'] = 'DONE'
            campaign.jobs['P3']['prepare'] = 'DONE'
            campaign.jobs['P4'].update(evaluate='PENDING', prepare='DONE')
            campaign.fail('P4', 'evaluate', 'Lean worker error')
            self.assertIn('P6', campaign.pending_preparations())
            self.assertEqual(campaign.jobs['P3']['prepare'], 'DONE')

    def test_mixed_schedule_uses_three_task_gpus_and_shared_retrieval(self):
        with tempfile.TemporaryDirectory() as temporary:
            campaign = scheduler.Adaptive(Path(temporary), 'digest')
            campaign.state['data_preflight'] = 'DONE'
            campaign.retrieval_gpu = 3
            for run in ('P1', 'P2', 'P3'):
                campaign.jobs[run]['prepare'] = 'DONE'
            launches = []
            def start(run, phase, gpu):
                self.assertNotIn(gpu, campaign.active)
                self.assertNotEqual(gpu, 3)
                launches.append((run, phase, gpu))
                campaign.active[gpu] = (run, phase, None, None)
                campaign.jobs[run][phase] = 'RUNNING'
            with patch.object(campaign, 'start_job', side_effect=start):
                campaign.schedule()
            self.assertEqual(launches, [('P1', 'evaluate', 0), ('P5', 'prepare', 1), ('P4', 'prepare', 2)])
            self.assertEqual(campaign.jobs['P6']['prepare'], 'PENDING')

    def test_failed_preparation_slot_is_reused_without_touching_survivors(self):
        with tempfile.TemporaryDirectory() as temporary:
            campaign = scheduler.Adaptive(Path(temporary), 'digest')
            campaign.state['data_preflight'] = 'DONE'
            campaign.retrieval_gpu = 3
            for run in ('P1', 'P2', 'P3', 'P5'):
                campaign.jobs[run]['prepare'] = 'DONE'
            campaign.jobs['P1']['evaluate'] = 'RUNNING'
            campaign.active[0] = ('P1', 'evaluate', None, None)
            campaign.active[1] = ('P5', 'prepare', None, None)
            campaign.jobs['P5']['prepare'] = 'RUNNING'
            campaign.fail('P4', 'prepare', 'simulated failure')
            with patch.object(campaign, 'start_job') as start:
                campaign.schedule()
            start.assert_called_once_with('P6', 'prepare', 2)
            self.assertEqual(set(campaign.active), {0, 1})


if __name__ == '__main__':
    unittest.main()
