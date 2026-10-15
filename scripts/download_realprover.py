"""Pin and download exactly one official BF16 base (also used as teacher).

Run with existing env/bootstrap/bin/python. --metadata-only resolves revision.
Never installs packages or modifies the two legacy evaluation environments.
"""
import argparse
import hashlib
import json
import os
import fnmatch
from pathlib import Path
from urllib.parse import quote
from serial_download import download

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata-only', action='store_true')
    parser.add_argument('--endpoint', default='https://hf-mirror.com',
                        choices=['https://hf-mirror.com', 'https://huggingface.co'])
    args = parser.parse_args()
    # Set before importing Hub constants; keep the official repository and SHA.
    os.environ['HF_ENDPOINT'] = args.endpoint
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    from huggingface_hub import HfApi
    manifest = ROOT / 'research/manifests/model_revision.json'
    if manifest.exists():
        info = json.loads(manifest.read_text())
        assert info['repo_id'] == 'FrenzyMath/REAL-Prover'
    else:
        remote = HfApi(endpoint=args.endpoint).model_info('FrenzyMath/REAL-Prover')
        info = {'repo_id': 'FrenzyMath/REAL-Prover', 'revision': remote.sha, 'status': 'REVISION_RESOLVED'}
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(info, indent=2) + '\n')
    if args.metadata_only:
        print(json.dumps(info, indent=2))
        return
    info.update(status='DOWNLOADING', download_endpoint=args.endpoint,
                download_mode='serial_http_range', max_concurrent_files=1)
    manifest.write_text(json.dumps(info, indent=2) + '\n')
    remote = HfApi(endpoint=args.endpoint).model_info(info['repo_id'],
        revision=info['revision'], files_metadata=True)
    assert remote.sha == info['revision'], 'Mirror revision mismatch'
    path = ROOT / 'models/REAL-Prover-v1-bf16'
    patterns = ['*.json', '*.safetensors', '*.model', 'merges.txt', 'vocab.*', '*.jinja']
    files = [s for s in remote.siblings if any(fnmatch.fnmatch(s.rfilename, p) for p in patterns)]
    # Small configs first, followed by one weight shard at a time.
    for s in sorted(files, key=lambda s: (s.rfilename.endswith('.safetensors'), s.rfilename)):
        assert s.size is not None, f'Missing remote size: {s.rfilename}'
        url = f'{args.endpoint}/{info["repo_id"]}/resolve/{info["revision"]}/{quote(s.rfilename)}'
        expected = s.lfs.sha256 if s.lfs else None
        download(url, path / s.rfilename, s.size, expected, path / '.cache')
    config = json.loads((path / 'config.json').read_text())
    assert config['model_type'] == 'qwen2', config['model_type']
    shards = sorted(path.glob('*.safetensors'))
    assert shards, 'Missing model shards'
    index = json.loads((path / 'model.safetensors.index.json').read_text())
    assert {p.name for p in shards} == set(index['weight_map'].values()), 'Incomplete shard set'
    info['files'] = {}
    for p in sorted(path.iterdir()):
        if p.is_file():
            h = hashlib.sha256()
            with p.open('rb') as f:
                for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
                    h.update(b)
            info['files'][p.name] = {'size': p.stat().st_size, 'sha256': h.hexdigest()}
    info.update(status='DOWNLOADED_REQUIRES_LOAD_CHECK', model_type=config['model_type'])
    manifest.write_text(json.dumps(info, indent=2) + '\n')
    print('Downloaded and hashed:', path)


if __name__ == '__main__':
    main()
