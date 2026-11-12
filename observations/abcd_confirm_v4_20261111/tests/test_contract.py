import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import *
class DataContract(unittest.TestCase):
 def test_heldout_ids(self):
  old={str(x['id']) for x in rows(PILOT/'manifests/V_real_theorems.jsonl')};new=rows(W/'manifests/selected_theorems.jsonl')
  self.assertEqual(len(new),128);self.assertFalse(old&{str(x['id']) for x in new})
 def test_fit_disjoint_from_pilot_and_valid(self):
  fit=rows(W/'manifests/A_fit_inputs.jsonl');old={x['state_id'] for x in rows(PILOT/'manifests/discovery_pilot.jsonl')}
  self.assertEqual(len(fit),128);self.assertEqual(len({x['state_id'] for x in fit}),128);self.assertFalse(old&{x['state_id'] for x in fit})
 def test_case_coverage(self):
  data=rows(W/'manifests/frozen_cases.jsonl');self.assertEqual(len(data),256)
  self.assertEqual(sum(x['D_selected'] for x in data),32);self.assertTrue(all(x['eligible'] for x in data))
  self.assertEqual(sum(x['split']=='V-REAL-CONFIRM' for x in data),128)
 def test_rule_kernel_receipt(self):self.assertEqual(read(W/'manifests/rule_elaboration.json')['status'],'PASS')
 def test_controls_keep_length(self):
  for case in rows(W/'manifests/frozen_cases.jsonl'):
   if case['split']!='V-REAL-CONFIRM':continue
   self.assertEqual(len(case['rendered_premises']),6)
   self.assertEqual(len(case['candidates']),11)
   self.assertEqual(len(case['lean_verdicts']),11)
   for j in range(3):self.assertEqual(len(case['input_ids']['full']),len(case['input_ids'][f'same_length_{j}']))
 def test_all_frozen_hashes(self):
  for name,expected in read(W/'manifests/frozen_identity.json').items():self.assertEqual(protocol_sha() if name=='protocol' else sha(W/'manifests'/name),expected)
if __name__=='__main__':unittest.main()
