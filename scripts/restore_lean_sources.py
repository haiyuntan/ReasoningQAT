"""Restore pinned official source archives serially, without touching FedProver.

GitHub codeload is tried before two archive mirrors. No credentials are sent.
Source archive SHA256 and every upstream lockfile are retained for auditing.
This downloads sources only; build and native preflight are separate steps.
"""
import hashlib
import json
import os
import posixpath
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
DEPS = ROOT / 'REALPROVER/dependencies'
CACHE = DEPS / 'source_cache'
ARCHIVES = DEPS / 'archives'
REPORT = DEPS / 'source_restore.json'
records = {}


def restore_git(directory, repo, revision):
    """Attach real upstream Git objects to an archive snapshot, preserving files.

    Per-command proxy overrides stale Git configuration; global settings and
    credentials are never changed. read-tree sets the index, not the worktree.
    """
    proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy') or ''
    command = ['git', '-c', 'http.proxy=' + proxy, '-C', str(directory)]
    if not (directory / '.git').exists():
        subprocess.run(command + ['init', '--initial-branch=restored-pinned'], check=True,
                       stdout=subprocess.DEVNULL)
        subprocess.run(command + ['remote', 'add', 'origin', f'https://github.com/{repo}'], check=True)
    head = subprocess.run(command + ['rev-parse', 'HEAD'], capture_output=True, text=True)
    if head.returncode or head.stdout.strip() != revision:
        if head.returncode == 0:
            raise RuntimeError(f'Refusing to replace existing Git HEAD: {directory}')
        subprocess.run(command + ['fetch', '--depth=1', 'origin', revision], check=True, timeout=180)
        subprocess.run(command + ['update-ref', 'HEAD', revision], check=True)
        subprocess.run(command + ['read-tree', revision], check=True)
    if repo == 'leanprover-community/ProofWidgets4':
        tag = subprocess.run(command + ['rev-parse', 'v0.0.50'], capture_output=True, text=True)
        if tag.returncode:
            subprocess.run(command + ['tag', 'v0.0.50', revision], check=True)


def persist():
    temporary = REPORT.with_suffix('.tmp')
    temporary.write_text(json.dumps(records, indent=2))
    temporary.replace(REPORT)


def obtain(repo, revision):
    if not re.fullmatch(r'[\w.-]+/[\w.-]+', repo) or not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('Only pinned GitHub source archives are accepted')
    key = repo.replace('/', '--') + '-' + revision
    directory = CACHE / key
    if directory.is_dir() and key in records:
        restore_git(directory, repo, revision)
        return directory
    archive = ARCHIVES / (key + '.tar.gz')
    if not archive.exists():
        urls = [f'https://codeload.github.com/{repo}/tar.gz/{revision}',
                f'https://gh-proxy.org/https://github.com/{repo}/archive/{revision}.tar.gz',
                f'https://ghfast.top/https://github.com/{repo}/archive/{revision}.tar.gz']
        for url in urls:
            print('DOWNLOAD', repo, revision, url, flush=True)
            part = archive.with_suffix('.part')
            try:
                with urllib.request.urlopen(url, timeout=60) as response, part.open('wb') as stream:
                    shutil.copyfileobj(response, stream, length=1024 * 1024)
                # Reject HTML error pages before preserving a completed archive.
                with tarfile.open(part, 'r:gz') as source:
                    source.getmembers()
                part.replace(archive)
                break
            except Exception as exc:
                print('FAILED', type(exc).__name__, str(exc)[:200], flush=True)
        else:
            raise RuntimeError(f'All archive endpoints failed for {repo}@{revision}')
    with tarfile.open(archive, 'r:gz') as source:
        members = source.getmembers()
        prefix = members[0].name.split('/')[0]
        staging = CACHE / (key + '.extracting')
        staging.mkdir(parents=True, exist_ok=True)
        for member in members:
            path = Path(member.name)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError(f'Unsafe source archive member: {member.name}')
            if member.issym() or member.islnk():
                target = posixpath.normpath(posixpath.join(str(path.parent), member.linkname)
                                           if member.issym() else member.linkname)
                if not target.startswith(prefix + '/'):
                    raise ValueError(f'Unsafe source archive link: {member.name}')
        source.extractall(staging, filter='data')
        if directory.exists():
            raise RuntimeError(f'Unrecorded source directory already exists: {directory}')
        (staging / prefix).rename(directory)
        staging.rmdir()
    records[key] = {'repository': repo, 'revision': revision, 'directory': str(directory),
                    'archive': str(archive), 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}
    persist()
    restore_git(directory, repo, revision)
    return directory


def link(path, target):
    if path.is_symlink() and path.resolve() == target.resolve():
        return
    if path.exists() or path.is_symlink():
        raise RuntimeError(f'Refusing to overwrite: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(os.path.relpath(target, path.parent), target_is_directory=True)


def dependencies(directory, visited):
    if directory in visited:
        return
    visited.add(directory)
    manifest = directory / 'lake-manifest.json'
    if (directory / 'lake-manifest.upstream.json').is_file():
        manifest = directory / 'lake-manifest.upstream.json'
    if not manifest.is_file():
        return
    data = json.loads(manifest.read_text())
    for package in data.get('packages', []):
        if package['type'] != 'git':
            raise ValueError(f'Unexpected upstream dependency type: {package}')
        repo = package['url'].removeprefix('https://github.com/').removesuffix('.git')
        child = obtain(repo, package['rev'])
        link(directory / data['packagesDir'] / package['name'], child)
        dependencies(child, visited)


def main():
    global records
    for path in (DEPS, CACHE, ARCHIVES):
        path.mkdir(parents=True, exist_ok=True)
    if REPORT.exists():
        records = json.loads(REPORT.read_text())
    roots = [('lean_test_v4160', 'frenzymath/lean_test_v4160', 'faf496a4bc025db8f13170ab85b8bdd5a0737a36'),
             ('jixia', 'frenzymath/jixia', '157c55dd5754db3795838e7c05104b91535ee5fc'),
             ('interactive', 'frenzymath/interactive', '9e7417953d80de8e4daff17c550ad872fde7f9bb')]
    visited = set()
    for name, repo, revision in roots:
        directory = obtain(repo, revision)
        link(DEPS / name, directory)
        dependencies(directory, visited)
    official = obtain('frenzymath/REAL-Prover', '3e5987dc1009a7addc935bcd7b4ae777738a0848')
    link(ROOT / 'upstream/REAL-Prover', official)
    print('SOURCE_DOWNLOAD_COMPLETE; compilation and native verification are still required', flush=True)


if __name__ == '__main__':
    main()
