"""Validate immutable prepared evidence, without consulting live attempt outputs."""
import csv
import json
from common import OBS, ROOT, atomic, read, sha

def main():
    campaign = ROOT/'runs/realprover_formal_four_gpu_v12'
    rows = []
    identities = {}
    for method in ['P0','P1','P2','P3','P4','P5','P6']:
        print('Inventory',method,flush=True)
        if method == 'P0':
            model = ROOT/'models/REAL-Prover-v1-bf16'
            entries = [{'path':str(p),'sha256':sha(p)} for p in sorted(model.iterdir()) if p.is_file() and p.suffix in {'.json','.safetensors','.txt'}]
            ready = read(model/'config.json')['model_type'] == 'qwen2'
            export = None
            evidence = {'model_revision':'fe76f68d9a88f342cb7b546307c20292fea9cced','files':entries}
        else:
            prepared = read(campaign/'formal'/method/'prepared.json')
            model_candidates = {str(__import__('pathlib').Path(e['path']).parent) for e in prepared['evidence'] if e['path'].endswith('/eval_model/config.json')}
            if len(model_candidates) != 1:
                raise ValueError(f'Ambiguous eval model {method}: {model_candidates}')
            model = __import__('pathlib').Path(model_candidates.pop())
            entries = [e for e in prepared['evidence'] if __import__('pathlib').Path(e['path']).parent == model]
            ready = all(sha(e['path']) == e['sha256'] for e in entries)
            export = read(model/'export.json')
            evidence = {'prepared_sha256':sha(campaign/'formal'/method/'prepared.json'),'files':entries,'export':export}
        identity = __import__('hashlib').sha256(json.dumps(entries,sort_keys=True).encode()).hexdigest()
        row = {'method':method,'source_path':str(model),'checkpoint_type':'BF16' if method=='P0' else 'W2A16_g128_dense_BF16','evidence_sha256':identity,'export_path':str(model),'is_quantized_numeric':method!='P0','is_dequant_eval':method!='P0','ready':ready,'training_complete':True,'eval_complete':method in ['P1','P2'],'blockers':'' if ready else 'HASH_MISMATCH'}
        rows.append(row)
        identities[method] = {'inventory':row,'evidence':evidence}
        atomic(OBS/'manifests/checkpoint_identities.json',identities)
    with (OBS/'manifests/checkpoint_inventory.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    assert all(r['ready'] for r in rows)

if __name__ == '__main__':
    main()
