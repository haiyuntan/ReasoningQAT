"""Verify the published ABCD artifacts using only Python's standard library."""
import hashlib,json
from pathlib import Path
W=Path(__file__).resolve().parent

def rows(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]

def main():
 expected=json.loads((W/'SHA256SUMS.json').read_text())
 for name,digest in expected.items():
  actual=hashlib.sha256((W/name).read_bytes()).hexdigest()
  if actual!=digest:raise ValueError('Hash mismatch: '+name)
 teacher={r['theorem_id']:r for r in rows(W/'metrics/P0_scores.jsonl')}
 assert len(teacher)==48
 for method in ['P0','P3','P4','P6']:
  data=rows(W/'metrics'/f'{method}_scores.jsonl')
  assert len(data)==48 and all(r['status']=='PILOT' for r in data)
  assert sum(r['split']=='V-REAL' for r in data)==32
  assert sum(r['split']=='V-RULE' for r in data)==16
  for r in data:
   t=teacher[r['theorem_id']]
   assert r['input_ids_sha']==t['input_ids_sha'] and r['candidates']==t['candidates']
  if method!='P0':
   d=rows(W/'metrics'/f'D_{method}.jsonl')
   assert len(d)==32 and len({r['theorem_id'] for r in d})==8
   assert all(r['replay_max_abs']==0 and r['algebra_residual_max_abs']<1e-5 for r in d)
   assert json.loads((W/'metrics'/f'{method}_batch_stress.json').read_text())['selected_batch']==1
 summary=json.loads((W/'metrics/ABCD_summary.json').read_text())
 assert set(summary)=={'P3','P4','P6'}
 assert all(set(r['A'])=={'7','14','21','28'} for r in summary.values())
 print(f'PASS: {len(expected)} artifact hashes; 48 cases/model; shared token/candidate identities; 96 D replay records; no missing/skipped cases.')
if __name__=='__main__':main()
