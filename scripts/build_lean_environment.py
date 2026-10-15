"""Build pinned restored Lean sources; preserve original mathematical sources."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

RESEARCH = Path(__file__).resolve().parents[1]
ROOT = RESEARCH.parent
sys.path.insert(0, str(RESEARCH / 'adapters'))
from common import save, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--skip-download', action='store_true')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    deps = ROOT / 'REALPROVER/dependencies'
    if not args.skip_download:
        subprocess.run([sys.executable, str(RESEARCH / 'scripts/restore_lean_sources.py')], check=True)
    toolchain = Path.home() / '.elan/toolchains/leanprover-lean-v4.16.0/bin'
    env = dict(os.environ, PATH=str(toolchain) + os.pathsep + os.environ['PATH'],
               XDG_CACHE_HOME=str(deps / 'cache'))
    count = int(env.get('GIT_CONFIG_COUNT', '0'))
    env.update(GIT_CONFIG_COUNT=str(count + 1))
    env[f'GIT_CONFIG_KEY_{count}'] = 'http.proxy'
    env[f'GIT_CONFIG_VALUE_{count}'] = env.get('HTTPS_PROXY', env.get('https_proxy', ''))
    # Earlier archive-only bootstrap used path entries. Real upstream Git
    # objects now exist, so retain the upstream lockfiles byte for byte.
    for name in ('lean_test_v4160', 'jixia', 'interactive'):
        directory = (deps / name).resolve()
        original = subprocess.check_output(['git', '-C', str(directory), 'show', 'HEAD:lake-manifest.json'])
        (directory / 'lake-manifest.json').write_bytes(original)
    workspace = deps / 'lean_test_v4160'
    original = subprocess.check_output(['git', '-C', str(workspace), 'show', 'HEAD:lakefile.lean']).decode()
    # Official lockfile includes REPL, but lakefile omitted the dependency.
    # Declare it explicitly so `lake exe repl` is available to the verifier.
    (workspace / 'lakefile.lean').write_text(original + '\nrequire REPL from git\n'
        '  "https://github.com/leanprover-community/repl.git" @ "v4.16.0"\n')
    jobs = [('jixia', deps / 'jixia', ['build']),
            ('interactive', deps / 'interactive', ['build']),
            ('mathlib_cache', workspace, ['exe', 'cache', 'get']),
            ('repl', workspace, ['build', 'repl']),
            ('header', workspace, ['env', 'lean', '-o', 'Header.olean', 'Header.lean'])]
    checks = []
    for name, cwd, command in jobs:
        log = args.output / (name + '.log')
        print('BUILD', name, flush=True)
        with log.open('w') as stream:
            result = subprocess.run([str(toolchain / 'lake'), *command], cwd=cwd, env=env,
                                    stdout=stream, stderr=subprocess.STDOUT, timeout=7200)
        checks.append({'name': name, 'exit_code': result.returncode, 'log': str(log.resolve())})
        save(args.output / 'build.json', {'status': 'BUILDING' if result.returncode == 0 else 'FAIL', 'checks': checks})
        result.check_returncode()
    save(args.output / 'build.json', {'status': 'PASS', 'checks': checks,
         'jixia_sha256': sha(deps / 'jixia/.lake/build/bin/jixia'),
         'interactive_sha256': sha(deps / 'interactive/.lake/build/bin/interactive'),
         'note': 'Compilation success still requires live native environment validation.'})


if __name__ == '__main__':
    main()
