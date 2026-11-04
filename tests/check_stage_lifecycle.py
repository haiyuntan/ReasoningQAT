"""Deadline removal, orphan cleanup and occupied-GPU regression checks."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

BASE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BASE / 'adapters'), str(BASE / 'scripts')]
import common
import resume_four_gpu as scheduler


class LifecycleTests(unittest.TestCase):
    def test_default_has_no_total_deadline(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(common.subprocess, 'Popen') as popen, patch.object(common, 'terminate_job'):
            popen.return_value.wait.return_value = 0
            common.execute(['unused'])
            self.assertEqual(popen.return_value.wait.call_args_list[0].kwargs, {'timeout': None})

    def test_explicit_timeout_cleans_real_grandchild(self):
        with tempfile.TemporaryDirectory() as folder:
            marker = Path(folder) / 'child'
            code = ('import subprocess,sys,time,pathlib; '
                    'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); '
                    'pathlib.Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(60)')
            with patch.dict(os.environ, {'REALPROVER_STAGE_TIMEOUT': '1'}):
                with self.assertRaises(subprocess.TimeoutExpired):
                    common.execute([sys.executable, '-c', code, str(marker)])
            pid = int(marker.read_text())
            stat = Path(f'/proc/{pid}/stat')
            self.assertTrue(not stat.exists() or stat.read_text().rsplit(') ', 1)[1].split()[0] == 'Z')

    def test_dead_group_leader_still_cleans_orphan(self):
        with tempfile.TemporaryDirectory() as folder:
            marker = Path(folder) / 'child'
            code = ('import subprocess,sys,pathlib; '
                    'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); '
                    'pathlib.Path(sys.argv[1]).write_text(str(p.pid))')
            process = subprocess.Popen([sys.executable, '-c', code, str(marker)], start_new_session=True)
            process.wait(timeout=10)
            common.terminate_job(process.pid)
            stat = Path(f'/proc/{int(marker.read_text())}/stat')
            self.assertTrue(not stat.exists() or stat.read_text().rsplit(') ', 1)[1].split()[0] == 'Z')

    def test_foreign_gpu_process_prevents_scheduling(self):
        with tempfile.TemporaryDirectory() as folder:
            campaign = scheduler.Adaptive(Path(folder), 'digest', training_first=False)
            campaign.retrieval_gpu = 3
            campaign.state['data_preflight'] = 'DONE'
            for job in campaign.jobs.values():
                job.update(prepare='DONE', evaluate='DONE')
            campaign.jobs['P6']['evaluate'] = 'PENDING'
            with patch.object(scheduler, 'occupied_gpus', return_value={0, 1, 2, 3}), patch.object(campaign, 'start_job') as start:
                campaign.schedule()
                start.assert_not_called()
            with patch.object(scheduler, 'occupied_gpus', return_value={0, 1, 3}), patch.object(campaign, 'start_job') as start:
                campaign.schedule()
                start.assert_called_once_with('P6', 'evaluate', 2)


if __name__ == '__main__':
    unittest.main()
