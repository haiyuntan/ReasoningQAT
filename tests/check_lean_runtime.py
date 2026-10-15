"""Live Lean/Mathlib/REPL/Jixia/Interactive and retrieval environment check."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

RESEARCH = Path(__file__).resolve().parents[1]
ROOT = RESEARCH.parent
sys.path.insert(0, str(ROOT / 'REALPROVER'))
sys.path.insert(0, str(RESEARCH / 'adapters'))
from common import save, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    from conf import config
    from manager.thirdparty.verifier import verify_proof
    from manager.thirdparty.interactive import Interactive
    from manager.thirdparty.lean_search import LeanSearch
    started = time.monotonic()
    workspace = Path(config.LEAN_TEST_PATH)
    lake = str(Path(config.LEAN_ENV_PATH) / 'lake')
    checks = {}
    interactive = None
    code = 'import Mathlib\ntheorem pipeline_env_nat (n : ℕ) : n + 0 = n := by simp\n'
    source = workspace / ('PipelineSanity_' + uuid.uuid4().hex + '.lean')
    try:
        source.write_text(code)
        assert verify_proof('theorem pipeline_true : True := by trivial', lake, str(workspace)), 'REPL rejected True'
        assert not verify_proof('theorem pipeline_sorry : False := by sorry', lake, str(workspace)), 'REPL accepted sorry'
        checks['repl_accept_and_reject'] = True
        print('PASS REPL accepts valid proof and rejects sorry', flush=True)
        subprocess.run([lake, 'env', 'lean', '-o', str(source.with_suffix('.olean')), str(source)],
                       cwd=workspace, check=True, timeout=1200)
        checks['mathlib_theorem_compiles'] = True
        declaration = source.with_suffix('.decl.json')
        subprocess.run([lake, 'env', str(config.analyzer_path), '-i', '-d', str(declaration), str(source)],
                       cwd=workspace, check=True, timeout=1200)
        parsed = json.loads(declaration.read_text())
        assert 'pipeline_env_nat' in json.dumps(parsed), 'Jixia failed to find the declaration'
        checks['jixia_declaration'] = True
        print('PASS Mathlib theorem and Jixia analysis', flush=True)

        def timeout_handler(*unused):
            raise TimeoutError('Interactive live probe timed out')

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(1200)
        interactive = Interactive(workspace, Path('Header.lean'))
        interactive.open_file(source, ['pipeline_env_nat'])
        declaration_name = interactive.get_next_problem()
        print('Interactive declaration:', repr(declaration_name), flush=True)
        assert declaration_name in ('pipeline_env_nat', ['pipeline_env_nat']), repr(declaration_name)
        assert interactive.get_state(0), 'Interactive initial goal missing'
        try:
            interactive.run_tactic(0, 'exact True.intro')
        except RuntimeError:
            checks['interactive_rejects_invalid_tactic'] = True
        else:
            raise AssertionError('Interactive accepted an invalid tactic')
        sid = interactive.run_tactic(0, 'simp')
        assert not interactive.get_state(sid), 'Tactic did not close the goal'
        interactive.commit(sid)
        checks['interactive_closes_goal'] = True
        signal.alarm(0)
        print('PASS Interactive tactic rejection and successful goal closure', flush=True)
        premises = LeanSearch.get_related_theorem('n : ℕ\n⊢ n = n')
        assert premises
        checks['retrieval'] = True
        save(args.output, {'status': 'PASS', 'checks': checks,
             'seconds': time.monotonic() - started, 'premise_count': len(premises),
             'jixia_sha256': sha(config.analyzer_path),
             'interactive_sha256': sha(config.interactive_path / '.lake/build/bin/interactive'),
             'note': 'Environment validation only; not a P0-P6 smoke receipt.'})
    except BaseException as exc:
        save(args.output, {'status': 'FAIL', 'checks': checks, 'error': str(exc),
                          'seconds': time.monotonic() - started})
        raise
    finally:
        signal.alarm(0)
        if interactive is not None:
            interactive.close()
        # Keep the source, declaration and olean as auditable evidence.


if __name__ == '__main__':
    main()
