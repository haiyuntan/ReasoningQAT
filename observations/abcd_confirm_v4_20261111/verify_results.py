import hashlib,json
from pathlib import Path
W=Path(__file__).resolve().parent
expected=json.loads((W/'SHA256SUMS.json').read_text())
for name,h in expected.items():
 assert hashlib.sha256((W/name).read_bytes()).hexdigest()==h,name
assert json.loads((W/'manifests/controller.json').read_text())['status']=='CONFIRMATION_DONE'
for m in ['P0','P3','P4','P5','P6']:
 assert len(list((W/'metrics/scores'/m).glob('*.json')))==256,m
for m in ['P3','P4','P5','P6']:
 records=list((W/'metrics/D'/m).glob('*.json'))
 assert len(records)==32,m
 assert all(len(json.loads(p.read_text())['layers'])==28 for p in records),m
assert all((W/'plots'/('ABCD_confirmation.'+ext)).stat().st_size>1000 for ext in ['png','pdf'])
print(f'PASS: {len(expected)} hashes; 256 cases/model; 3584 D layer records; completed plots.')
