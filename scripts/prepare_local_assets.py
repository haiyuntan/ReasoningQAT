"""Freeze legacy candidate eval inputs and record reusable raw data, never tokens.

Candidate is deliberately not named eval_protocol_v1: official data parity,
runtime recovery and P0 smoke have not passed yet.
"""
import json
from pathlib import Path
import tomllib
from audit_preparation import ROOT, LEGACY, OUT, digest


def write_once(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f'Refusing to overwrite changed artifact: {path}')
    else:
        path.write_bytes(data)


def main():
    evidence = {'status': 'CANDIDATE_NOT_OFFICIAL_PARITY_VERIFIED', 'eval': []}
    for name in ['minif2f_test.jsonl', 'proofnet_test.jsonl', 'fate_m.jsonl']:
        source = LEGACY / 'data' / name
        raw = source.read_bytes()
        rows = [json.loads(s) for s in raw.decode().splitlines() if s.strip()]
        assert rows and all('id' in r and 'formal_statement' in r for r in rows)
        assert len({str(r['id']) for r in rows}) == len(rows)
        destination = ROOT / 'datasets/eval/legacy_candidate' / name
        write_once(destination, raw)
        evidence['eval'].append({'source': str(source), 'path': str(destination),
                                'count': len(rows), 'sha256': digest(destination)})
        if name == 'minif2f_test.jsonl':
            assert len(rows) == 244
            smoke = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows[:2]).encode()
            write_once(destination.with_name('minif2f_smoke2.jsonl'), smoke)
    template = (LEGACY / 'experiment/examples/default.toml').read_text()
    template = template.replace('bf_minif2f_test_pass64', 'P0_CANDIDATE_NOT_READY')
    template = template.replace('/path/to/your/model', str(ROOT / 'models/REAL-Prover-v1-bf16'))
    template = template.replace('data/minif2f_test.jsonl', str(ROOT / 'datasets/eval/legacy_candidate/minif2f_test.jsonl'))
    config = tomllib.loads(template)
    assert config['model']['use_retrieval'] and config['model']['template'] == 'qwen'
    write_once(ROOT / 'research/configs/p0_candidate.toml', template.encode())
    write_once(ROOT / 'research/configs/p0_smoke2_candidate.toml', template.replace(
        'P0_CANDIDATE_NOT_READY', 'P0_SMOKE2_NOT_A_FORMAL_SCORE').replace(
        '/minif2f_test.jsonl', '/minif2f_smoke2.jsonl').encode())
    sources = ROOT / 'datasets/manifests/sources.json'
    evidence['existing_sources_manifest'] = {'path': str(sources), 'sha256': digest(sources)}
    evidence['openthoughts_raw_parquet_count'] = len(list((ROOT / 'datasets/openthoughts3').rglob('*.parquet')))
    evidence['fineweb_raw_parquets'] = [str(p) for p in (ROOT / 'datasets/fineweb-edu').rglob('*.parquet')]
    evidence['reuse_rule'] = 'Reuse raw parquet and NuminaMath after hash checks; regenerate Qwen2 token caches. No Qwen3 token cache reuse.'
    write_once(OUT / 'local_assets.json', (json.dumps(evidence, indent=2) + '\n').encode())
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
