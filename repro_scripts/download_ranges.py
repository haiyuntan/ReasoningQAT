"""Serial streaming HTTP resume with official SHA256 validation."""
import concurrent.futures
import hashlib
import json
import os
import re
import time
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[2]
CHUNK = 1024 * 1024

def download(entry):
    repo, source, relative, size, expected = entry
    dest = ROOT / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == size:
        return verify(dest, expected)
    part = Path(str(dest) + '.partial')
    offset = part.stat().st_size if part.exists() else 0
    session = requests.Session()
    while offset < size:
        start_offset = offset
        for attempt in range(12):
            try:
                # One continuous response, not a new connection per 8MB.
                url = source + f'?download=true&rq_stream_offset={offset}'
                with session.get(url, headers={'Range': f'bytes={offset}-'}, stream=True, timeout=(30, 120)) as response:
                    response.raise_for_status()
                    content_range = response.headers.get('Content-Range', '')
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', content_range)
                    if response.status_code == 206:
                        if not match or int(match[1]) != offset or int(match[3]) != size:
                            raise ValueError(f'Invalid range: {response.status_code} {content_range}')
                    elif response.status_code != 200 or offset != 0:
                        raise ValueError(f'Server did not resume: {response.status_code} {content_range}')
                    last_report = offset
                    with part.open('ab') as f:
                        for data in response.iter_content(chunk_size=CHUNK):
                            if offset + len(data) > size:
                                raise ValueError('Response exceeds official file size')
                            f.write(data)
                            offset += len(data)
                            if offset-last_report >= 16*1024*1024 or offset == size:
                                f.flush()
                                print(f'PROGRESS {relative} {offset}/{size}', flush=True)
                                last_report = offset
                if offset == start_offset:
                    raise IOError('Empty download response')
                break
            except Exception as exc:
                print(f'RETRY {relative} offset={offset} attempt={attempt+1} {type(exc).__name__}: {str(exc)[:160]}', flush=True)
                if attempt == 11:
                    raise
                time.sleep(min(2**attempt, 30))
    verify(part, expected)
    part.replace(dest)
    print('VERIFIED', relative, flush=True)

def verify(path, expected):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(16*1024*1024), b''):
            digest.update(chunk)
    if expected and digest.hexdigest() != expected:
        raise ValueError(f'Checksum mismatch: {path}')
    return digest.hexdigest()

def main():
    manifest_path = ROOT / 'datasets/manifests/sources.json'
    manifest = json.loads(manifest_path.read_text())
    entries = []
    for repo, info in manifest.items():
        kind = info['repo_type']
        revision = info['revision']
        prefix = 'datasets/' if kind == 'dataset' else ''
        cache = ROOT / 'datasets/manifests' / (repo.replace('/', '__') + '.files.json')
        if cache.exists():
            metadata = json.loads(cache.read_text())
        else:
            for attempt in range(10):
                try:
                    response = requests.get(f'https://huggingface.co/api/{kind}s/{repo}/revision/{revision}?blobs=true', timeout=60)
                    response.raise_for_status()
                    metadata = response.json()
                    cache.write_text(json.dumps(metadata))
                    break
                except requests.RequestException:
                    if attempt == 9:
                        raise
                    time.sleep(3)
        if 'siblings' not in metadata:
            raise ValueError(f'Missing HF file metadata: {repo}')
        siblings = {s['rfilename']: s for s in metadata['siblings']}
        paths = [f['path'] for f in info['files']]
        if 'NuminaMath' in repo:
            paths = sorted(p for p in paths if p.endswith('.parquet') and 'train' in p)[:1]
        elif 'fineweb' in repo:
            paths = ['sample/10BT/000_00000.parquet']
        else:
            paths = [p for p in paths if p.endswith(('.safetensors', '.parquet'))]
        for path in paths:
            meta = siblings[path]
            size = meta.get('size') or meta.get('lfs', {}).get('size')
            sha = meta.get('lfs', {}).get('sha256')
            info.setdefault('verified_files', {})[path] = {'size': size, 'sha256': sha}
            url = f'https://hf-mirror.com/{prefix}{repo}/resolve/{revision}/{path}'
            entries.append((repo, url, info['destination']+'/'+path, size, sha))
    manifest_path.write_text(json.dumps(manifest, indent=2))
    # Prioritize the small mandatory sources before the full OpenThoughts corpus.
    entries.sort(key=lambda x: 3 if 'OpenThoughts' in x[0] else (0 if x[0].startswith('Qwen/') else 1))
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(os.environ.get('RQ_DOWNLOAD_WORKERS','1'))) as pool:
        for _ in pool.map(download, entries):
            pass
    (ROOT / 'results/download-complete.json').write_text(json.dumps({'status':'PASS', 'files':len(entries)}))

if __name__ == '__main__':
    main()
