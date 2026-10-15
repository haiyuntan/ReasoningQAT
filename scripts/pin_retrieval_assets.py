"""Hash the reused local retrieval assets; do not infer their upstream origin."""
import argparse
from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parents[1]
ROOT = RESEARCH.parent
sys.path.insert(0, str(RESEARCH / 'adapters'))
from common import load, save, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=RESEARCH / 'manifests/retrieval_assets.json')
    args = ap.parse_args()
    legacy = ROOT.parent / 'FedProver/model'
    base = legacy / 'e5-mistral-7b-instruct'
    adapter = legacy / 'LeanSearch-PS'
    files = [base / name for name in sorted(set(load(base / 'model.safetensors.index.json')['weight_map'].values()))]
    files += [base / name for name in ('config.json', 'tokenizer_config.json', 'tokenizer.json',
                                       'special_tokens_map.json', 'model.safetensors.index.json')]
    files += [base / name for name in ('added_tokens.json', 'tokenizer.model') if (base / name).is_file()]
    files += [adapter / name for name in ('adapter_config.json', 'adapter_model.safetensors')]
    files += [legacy / 'LeanSearch-PS-faiss.index', Path.home() / 'LeanSearch-PS-faiss/answers.json']
    records = []
    for path in files:
        print('HASH', path.name, flush=True)
        records.append({'path': str(path.resolve()), 'size': path.stat().st_size, 'sha256': sha(path)})
    save(args.output, {'status': 'LOCAL_ASSETS_PINNED', 'files': records,
         'upstream_origin': 'Reused legacy assets; identity is pinned by content, not independently certified.',
         'service_source_revision': '3e5987dc1009a7addc935bcd7b4ae777738a0848'})


if __name__ == '__main__':
    main()
