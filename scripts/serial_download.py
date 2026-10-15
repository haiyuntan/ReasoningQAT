"""One HTTP request at a time, stable partial paths, verified Range resumes."""
import hashlib
from http.client import HTTPException
import os
from pathlib import Path
import time
from urllib.request import Request, urlopen


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def download(url, target, size, expected_sha=None, legacy_dir=None, attempts=100):
    target = Path(target)
    partial = target.with_name(target.name + '.part')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == size:
        if expected_sha and sha256(target) != expected_sha:
            raise RuntimeError(f'Existing file hash mismatch: {target}')
        print(f'VERIFIED/SKIP {target.name}', flush=True)
        return
    if not partial.exists() and expected_sha and legacy_dir:
        candidates = [p for p in Path(legacy_dir).rglob('*.incomplete')
                      if f'.{expected_sha}.' in p.name and 0 < p.stat().st_size <= size]
        if candidates:
            best = max(candidates, key=lambda p: p.stat().st_size)
            # The legacy filename contains the pinned LFS content hash. Also
            # check its prefix against the remote before adopting its bytes.
            n = min(best.stat().st_size, 65536)
            with urlopen(Request(url, headers={'Range': f'bytes=0-{n-1}',
                         'Accept-Encoding': 'identity'}), timeout=60) as response:
                prefix = response.read(n)
            with best.open('rb') as f:
                if prefix != f.read(n):
                    raise RuntimeError(f'Legacy partial prefix mismatch: {best}')
            os.replace(best, partial)
            print(f'ADOPT {target.name} bytes={partial.stat().st_size}', flush=True)
    for attempt in range(attempts):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > size:
            raise RuntimeError(f'Partial larger than remote file: {partial}')
        try:
            if offset < size:
                print(f'RESUME {target.name} offset={offset} total={size}', flush=True)
                headers = {'Accept-Encoding': 'identity'}
                if offset:
                    headers['Range'] = f'bytes={offset}-'
                with urlopen(Request(url, headers=headers), timeout=60) as response:
                    if offset and (response.status != 206 or not response.headers.get(
                            'Content-Range', '').startswith(f'bytes {offset}-')):
                        raise RuntimeError('Server did not honor Range; partial preserved')
                    with partial.open('ab') as f:
                        last = time.monotonic()
                        while True:
                            block = response.read(1024 * 1024)
                            if not block:
                                break
                            f.write(block)
                            if time.monotonic() - last > 30:
                                f.flush()
                                print(f'PROGRESS {target.name} {f.tell()}/{size}', flush=True)
                                last = time.monotonic()
                if partial.stat().st_size != size:
                    raise OSError('Incomplete response; retry from current byte offset')
            if expected_sha and sha256(partial) != expected_sha:
                raise RuntimeError(f'SHA256 mismatch; preserved for inspection: {partial}')
            os.replace(partial, target)
            print(f'COMPLETE {target.name} bytes={size}', flush=True)
            return
        except (OSError, TimeoutError, HTTPException) as e:
            print(f'RETRY {target.name} attempt={attempt+1} error={e}', flush=True)
            if attempt + 1 == attempts:
                raise
            time.sleep(min(5 * (attempt + 1), 30))
