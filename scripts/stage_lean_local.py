"""Copy pinned Lean dependencies onto node-local storage and verify every copied byte."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[2]


def stage(destination):
    if destination.exists():
        raise RuntimeError('Refusing to overwrite local Lean cache')
    destination.mkdir(parents=True)
    started = time.monotonic()
    source = ROOT / 'REALPROVER/dependencies'
    jobs, skipped = [], []
    sources = [(name, source / name) for name in ('lean_test_v4160', 'interactive', 'jixia')]
    sources.append(('toolchain', Path.home() / '.elan/toolchains/leanprover--lean4---v4.16.0'))
    for name, base in sources:
        for directory, folders, files in os.walk(base, followlinks=True):
            # Lake checks Git remotes even for `lake env`; retain metadata or it
            # deletes restored packages and clones them again.
            folders[:] = [f for f in folders if f != '__pycache__']
            relative = Path(directory).relative_to(base)
            target = destination / name / relative
            target.mkdir(parents=True, exist_ok=True)
            for filename in files:
                if filename.startswith(('TestOne_', 'pipeline_probe_')):
                    continue
                candidate = Path(directory) / filename
                try:
                    candidate.stat()
                except OSError:
                    if not candidate.is_symlink():
                        raise
                    skipped.append(str(candidate))
                    continue
                jobs.append((candidate, target / filename))

    def copy(pair):
        src, dst = pair
        shutil.copy2(src, dst)
        def digest(path):
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    h.update(chunk)
            return h.hexdigest()
        value = digest(dst)
        if digest(src) != value:
            raise RuntimeError(f'Local Lean cache differs from source: {src}')
        return {'path': str(dst.relative_to(destination)), 'size': dst.stat().st_size, 'sha256': value}

    with ThreadPoolExecutor(max_workers=8) as pool:
        files = list(pool.map(copy, jobs))
    report = {'status': 'PASS', 'source': str(source), 'destination': str(destination),
              'files': files, 'seconds': time.monotonic() - started,
              'bytes': sum(f['size'] for f in files), 'skipped_broken_symlinks': skipped}
    (destination / 'copy_manifest.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'files'}), flush=True)
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--destination', type=Path, required=True)
    stage(ap.parse_args().destination.resolve())
