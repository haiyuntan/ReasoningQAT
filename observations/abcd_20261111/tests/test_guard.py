"""Fault injection: low reserve must reap observation child, not production."""
import sys,subprocess,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pipeline
class GuardTest(unittest.TestCase):
 def test_low_reserve_reaps_owned_child(self):
  real=subprocess.Popen;children=[]
  def spawn(*args,**kw):
   self.assertEqual(kw['env']['CUDA_VISIBLE_DEVICES'],pipeline.P['gpu_uuid'])
   child=real([sys.executable,'-c','import time; time.sleep(60)'],**kw);children.append(child);return child
  log=pipeline.W/'logs/test_guard_low_reserve.log'
  if log.exists():log.unlink()
  with patch.object(pipeline.subprocess,'Popen',side_effect=spawn),patch.object(pipeline,'request',return_value=({'status':'ready','adapter_loaded':True},.01)),patch.object(pipeline,'memory',return_value=(0,100)):
   with self.assertRaisesRegex(RuntimeError,'RESOURCE_RESERVE'):pipeline.run('test_guard_low_reserve','unused.py',gpu=True)
  self.assertIsNotNone(children[0].poll());self.assertIsNone(pipeline.active)
if __name__=='__main__':unittest.main()
