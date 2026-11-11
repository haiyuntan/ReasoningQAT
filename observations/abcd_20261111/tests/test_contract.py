import sys,json,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import *
class Contract(unittest.TestCase):
 def test_disjoint(self):
  a=read_rows(W/'manifests/S_cal_train.jsonl');b=read_rows(W/'manifests/S_obs_disc.jsonl')
  self.assertFalse({r['state_id'] for r in a}&{r['state_id'] for r in b})
 def test_valid_only(self):
  cases=read_rows(W/'manifests/V_real_theorems.jsonl');self.assertEqual(len(cases),32)
  self.assertTrue(all(r['split']=='V-REAL' for r in cases))
 def test_rules(self):
  records=read_rows(W/'manifests/lean_runtime_receipts.jsonl');rules=[r for r in records if 'rule_id' in r]
  self.assertEqual(len(rules),16);self.assertTrue(all(r['proof_verified'] for r in rules))
 def test_frozen_candidates(self):
  for row in read_rows(W/'manifests/V_real_states.jsonl'):
   self.assertEqual(len(row['candidates']),len(row['lean_verdicts']))
   self.assertEqual([r['action'] for r in row['lean_verdicts']],row['candidates'])
   self.assertEqual(row['retrieval_sha'],digest(json.dumps(row['premises'],sort_keys=True)))
 def test_protocol(self):
  self.assertFalse(P['training']);self.assertEqual(P['gpu_uuid'],'GPU-74ffe9e9-634a-34f9-22fa-03120b4764f7')
  self.assertEqual(P['models'],['P0','P3','P4','P6'])
 def test_no_optimizer(self):
  source=(W/'src/score.py').read_text()+(W/'src/replay.py').read_text()
  self.assertNotIn('optimizer.step(',source);self.assertNotIn('save_pretrained(',source)
if __name__=='__main__':unittest.main()
