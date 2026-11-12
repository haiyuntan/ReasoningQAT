import sys,unittest,re,subprocess
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from prepare import make_rules
from collect import exact_token_span
from common import W,P
import pipeline
class RulesAndControls(unittest.TestCase):
 def test_rule_instances_unique(self):
  cases=make_rules();self.assertEqual(len(cases),128)
  originals=[re.sub(r'theorem \w+','theorem CANONICAL',r['proof_code']) for r in cases if r['variant']=='original']
  self.assertEqual(len(set(originals)),64)
  for r in cases:
   self.assertNotIn('sorry',r['proof_code']);self.assertNotIn('admit',r['proof_code'])
   self.assertEqual(r['formal_statement'].split('theorem ')[1].split(' ')[0],r['base_group'])
 def test_exact_span_preserves_outside_and_length(self):
  class Tokens:
   def __call__(self,text,**kw):return {'input_ids':[ord(x) for x in text],'offset_mapping':[(i,i+1) for i in range(len(text))]}
  text='before\nPREMISE\nafter';ids,a,b=exact_token_span(text,'PREMISE',Tokens());changed=ids[:a]+[99]*(b-a)+ids[b:]
  self.assertEqual(len(ids),len(changed));self.assertEqual(ids[:a],changed[:a]);self.assertEqual(ids[b:],changed[b:])
 def test_guard_only_kills_own_child(self):
  real=subprocess.Popen;children=[]
  def spawn(*args,**kw):
   self.assertEqual(kw['env']['CUDA_VISIBLE_DEVICES'],P['GPU']['uuid'])
   self.assertEqual(kw['env']['CUBLAS_WORKSPACE_CONFIG'],':4096:8')
   child=real([sys.executable,'-c','import time;time.sleep(60)'],**kw);children.append(child);return child
  with patch.object(pipeline,'write_controller'),patch.object(pipeline,'check_frozen'),patch.object(pipeline,'production_errors',return_value={}),patch.object(pipeline.subprocess,'Popen',side_effect=spawn),patch.object(pipeline,'request',return_value=({'status':'ready','adapter_loaded':True,'asset_manifest_sha256':'test'},.001)),patch.object(pipeline,'sha',return_value='test'),patch.object(pipeline,'resources',return_value=(0,200)):
   with self.assertRaises(pipeline.RetrievalPriorityStop):pipeline.run('unit_guard','unused.py',gpu=True)
  self.assertIsNotNone(children[0].poll());self.assertIsNone(pipeline.active)
 def test_cleanup_includes_separate_lean_like_session(self):
  import os,time
  child=subprocess.Popen([sys.executable,'-u','-c','import subprocess,time; p=subprocess.Popen(["sleep","60"],start_new_session=True); print(p.pid,flush=True); time.sleep(60)'],stdout=subprocess.PIPE,text=True,start_new_session=True)
  descendant=int(child.stdout.readline().strip());pipeline.active=child
  self.assertIn(descendant,pipeline.owned_processes(child.pid));pipeline.stop()
  stat=Path(f'/proc/{descendant}/stat')
  for _ in range(100):
   if not stat.exists() or stat.read_text().rsplit(') ',1)[1].startswith('Z'):break
   time.sleep(.01)
  else:self.fail('separate-session descendant survived cleanup')
  child.stdout.close()

class Geometry(unittest.TestCase):
 def test_orthogonal_errors_are_not_evidence_of_cancellation(self):
  from replay import geometry_from_norms
  r=geometry_from_norms(3,4,5,0)
  self.assertAlmostEqual(r['cancellation'],2/7)
  self.assertAlmostEqual(r['excess_over_orthogonal'],0)
  self.assertAlmostEqual(r['energy_relative_to_orthogonal'],1)
 def test_zero_norm_geometry_is_undefined(self):
  from replay import geometry_from_norms
  r=geometry_from_norms(0,0,0,0)
  self.assertIsNone(r['cosine']);self.assertIsNone(r['cancellation']);self.assertIsNone(r['energy_relative_to_orthogonal'])

class Statistics(unittest.TestCase):
 def test_identical_predictors_have_zero_paired_difference(self):
  from statistics_abcd import paired_correlation
  x=list(range(20));r=paired_correlation(x,x,list(reversed(x)),100)
  self.assertAlmostEqual(r['paired_difference'],0);self.assertEqual(r['ci95'],[0.,0.])
 def test_family_weights_not_token_or_replication_weights(self):
  from statistics_abcd import cluster_mean
  rows=[{'family':'A','v':0}]*100+[{'family':'B','v':10},{'family':'C','v':20}]
  r=cluster_mean(rows,'v','family',100);self.assertEqual(r['clusters'],3);self.assertAlmostEqual(r['mean'],10.)
if __name__=='__main__':unittest.main()
