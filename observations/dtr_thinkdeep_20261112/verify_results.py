import json,hashlib
from pathlib import Path
W=Path(__file__).resolve().parent
def main():
    h=json.loads((W/'SHA256SUMS.json').read_text())
    for p,expected in h.items():assert hashlib.sha256((W/p).read_bytes()).hexdigest()==expected,p
    s=json.loads((W/'configs/split_registry.json').read_text());assert not set(s['reqat_reserved64'])&set(s['dtr_confirm20'])
    g=json.loads((W/'manifests/G2.json').read_text());assert g['status']=='PASS'
    for p in (W/'metrics/per_tactic').glob('*.json'):
        r=json.loads(p.read_text())
        if r.get('status')!='DONE':continue
        assert hashlib.sha256((W/r['token_file']).read_bytes()).hexdigest()==r['token_sha']
        for t in json.loads((W/r['token_file']).read_text()):assert len(t['jsd'])==28 and abs(t['jsd'][-1])<1e-5
        assert not r['kernel_verified']
    print('PASS: hashes, disjoint confirmation, full28-layer own-final JSD and Lean contract')
if __name__=='__main__':main()
