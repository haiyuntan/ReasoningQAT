"""Download pinned assets without initializing CUDA. Safe to resume."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download, hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
os.environ['HF_HOME'] = str(ROOT / 'hf_cache')
os.environ['HF_HUB_DISABLE_XET'] = '1'
MANIFEST = ROOT / 'datasets/manifests/sources.json'
SOURCES = [
    ('model', 'Qwen/Qwen3-1.7B', 'models/Qwen3-1.7B', ['*.json', '*.safetensors', '*.jinja', 'README.md', '*.txt']),
    ('dataset', 'open-thoughts/OpenThoughts3-1.2M', 'datasets/openthoughts3', ['data/*.parquet', 'README.md', '*.json']),
    ('dataset', 'HuggingFaceFW/fineweb-edu', 'datasets/fineweb-edu', ['sample/10BT/000_00000.parquet', 'README.md']),
    ('dataset', 'AI-MO/NuminaMath-CoT', 'datasets/numinamath_source', None),
]

def main():
    api = HfApi()
    frozen = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    for kind, repo, dest, patterns in SOURCES:
        if repo not in frozen:
            info = api.repo_info(repo, repo_type=kind, files_metadata=True)
            frozen[repo] = {'revision': info.sha, 'repo_type': kind, 'destination': dest,
                            'files': [{'path': s.rfilename, 'size': s.size} for s in info.siblings]}
            MANIFEST.write_text(json.dumps(frozen, indent=2))
    def fetch(item):
        kind, repo, dest, patterns = item
        revision = frozen[repo]['revision']
        if patterns is not None:
            snapshot_download(repo, repo_type=kind, revision=revision, local_dir=ROOT/dest,
                              allow_patterns=patterns, max_workers=3)
        else:
            # Fetch a single source shard, then freeze first 128 rows locally.
            parts = sorted(f['path'] for f in frozen[repo]['files'] if f['path'].endswith('.parquet') and 'train' in f['path'])
            if not parts:
                raise RuntimeError(f'No train parquet in {repo}')
            hf_hub_download(repo, parts[0], repo_type=kind, revision=revision, local_dir=ROOT/dest)
        print('DOWNLOAD_COMPLETE', repo, revision, flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for _ in pool.map(fetch, SOURCES):
            pass

if __name__ == '__main__':
    main()
