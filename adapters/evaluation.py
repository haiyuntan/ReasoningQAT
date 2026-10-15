"""Native Lean invocation and fixed-dataset aggregation; no generic benchmark."""
import argparse
import json
import os
from pathlib import Path
import sys

from common import ROOT, RESEARCH, load, save, sha


def aggregate(dataset, directory):
    rows = [json.loads(s) for s in Path(dataset).read_text().splitlines() if s.strip()]
    ids = [str(r['id']) for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate evaluation IDs')
    states = []
    evidence = []
    for row in rows:
        identifier = str(row['id'])
        if Path(identifier).name != identifier or identifier in ('.', '..'):
            raise ValueError('Unsafe theorem ID')
        files = sorted((directory / 'generated' / identifier).glob('*.json'))
        success, valid = False, False
        for path in files:
            d = load(path)
            if d.get('formal_statement') != row['formal_statement']:
                raise ValueError(f'Result theorem mismatch: {identifier}')
            if not isinstance(d.get('collect_results'), list):
                raise ValueError('Malformed native result')
            valid = valid or bool(d['collect_results'])
            claimed = any(p.get('success') is True for p in d['collect_results'])
            verifier = d.get('pipeline_verifier', {})
            if claimed and verifier != {'executed': True, 'success': True}:
                raise ValueError(f'Success lacks full Lean verification: {identifier}')
            success = success or claimed
            evidence.append({'path': str(path.resolve()), 'sha256': sha(path)})
        states.append({'id': identifier, 'success': success, 'completed': valid,
                       'failure_type': None if success else ('search_exhausted' if valid else 'missing_or_runtime_failure')})
    return {'expected_problems': len(rows), 'completed_problems': sum(p['completed'] for p in states),
            'successes': sum(p['success'] for p in states),
            'success_rate': sum(p['success'] for p in states) / len(rows),
            'problems': states, 'raw_evidence': evidence}


def make_config(model, mode, destination):
    runtime = load(RESEARCH / 'configs/runtime.json')
    source = ROOT / runtime['eval_candidate']
    text = source.read_text()
    try:
        import tomllib
    except ModuleNotFoundError:
        from pip._vendor import tomli as tomllib
    base = tomllib.loads(text)
    dataset = ROOT / ('datasets/eval/legacy_candidate/minif2f_smoke2.jsonl' if mode == 'smoke'
                      else 'REALPROVER/data/minif2f_test.jsonl')
    text = text.replace(json.dumps(base['model']['prover_model_path']), json.dumps(str(model.resolve())))
    text = text.replace(json.dumps(base['data']['data_path']), json.dumps(str(dataset.resolve())))
    config = tomllib.loads(text)
    reference = json.loads(json.dumps(base))
    reference['model']['prover_model_path'] = str(model.resolve())
    reference['data']['data_path'] = str(dataset.resolve())
    if config != reference:
        raise ValueError('Evaluation config differs beyond model and dataset path')
    if mode == 'smoke':
        import re
        # One shared smoke budget across all methods; full protocol untouched.
        for key, value in runtime['smoke']['eval_search'].items():
            text, count = re.subn(rf'(?m)^{key} = \d+$', f'{key} = {value}', text)
            if count != 1:
                raise ValueError(f'Ambiguous smoke search setting: {key}')
    destination.write_text(text)
    return dataset, sha(source)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--mode', choices=['smoke', 'formal'], required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--preflight', action='store_true')
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if a.mode == 'formal' and load(RESEARCH / 'configs/runtime.json')['eval_protocol_status'] != 'FROZEN':
        raise RuntimeError('Formal evaluation protocol/dataset parity is not frozen')
    path = a.output / 'config.toml'
    dataset, protocol_sha = make_config(a.model, a.mode, path)
    # This script itself is launched using the existing PROVER interpreter.
    sys.path.insert(0, str(ROOT / 'REALPROVER'))
    from conf import config
    import subprocess
    toolchain = Path(config.LEAN_TEST_PATH) / 'lean-toolchain'
    if not toolchain.is_file() or toolchain.read_text().strip() != 'leanprover/lean4:v4.16.0':
        raise RuntimeError('Missing/mismatched new Lean 4.16.0 workspace')
    legacy = ROOT.parent / 'FedProver'
    if Path(config.LEAN_TEST_PATH).resolve().is_relative_to(legacy.resolve()):
        raise RuntimeError('Refusing to write into read-only FedProver Lean workspace')
    for asset in [Path(config.LEAN_TEST_PATH) / 'Header.lean', config.analyzer_path,
                  config.interactive_path / '.lake/build/bin/interactive']:
        if not asset.is_file():
            raise RuntimeError(f'Missing Lean dependency: {asset}')
    version = subprocess.check_output(['lake', 'env', 'lean', '--version'], cwd=config.LEAN_TEST_PATH, text=True)
    if '4.16.0' not in version:
        raise RuntimeError('Runtime Lean version differs from pinned toolchain')
    from manager.thirdparty.verifier import verify_proof
    if not verify_proof('theorem pipeline_sanity : True := by trivial',
                        str(Path(config.LEAN_ENV_PATH) / 'lake'), config.LEAN_TEST_PATH):
        raise RuntimeError('Native Lean verifier rejected the sanity proof')
    if verify_proof('theorem pipeline_invalid : False := by sorry',
                    str(Path(config.LEAN_ENV_PATH) / 'lake'), config.LEAN_TEST_PATH):
        raise RuntimeError('Native Lean verifier incorrectly accepted sorry')
    telemetry = a.output / 'retrieval.jsonl'
    if telemetry.exists():
        raise RuntimeError('Refusing to overwrite retrieval telemetry')
    os.environ['REALPROVER_RETRIEVAL_TELEMETRY'] = str(telemetry.resolve())
    os.environ['REALPROVER_SEED'] = '2'
    from manager.thirdparty.lean_search import LeanSearch
    LeanSearch.get_related_theorem('n : ℕ\n⊢ n = n')
    if a.preflight:
        save(a.output / 'preflight.json', {'status': 'PASS', 'lean_version': version.strip(),
             'lean_workspace': str(Path(config.LEAN_TEST_PATH).resolve()),
             'jixia_sha256': sha(config.analyzer_path),
             'interactive_sha256': sha(config.interactive_path / '.lake/build/bin/interactive')})
        return
    from common import execute
    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT / 'REALPROVER')
    native_exit = 0
    try:
        execute([sys.executable, '-m', 'experiment.run', path, '--result-dir', a.output / 'native'],
                cwd=ROOT / 'REALPROVER', env=env)
    except subprocess.CalledProcessError as exc:
        native_exit = exc.returncode
    stats = aggregate(dataset, a.output / 'native')
    events = [json.loads(s) for s in telemetry.read_text().splitlines()]
    stats.update(protocol_sha256=protocol_sha, dataset_sha256=sha(dataset), seed=2, native_exit_code=native_exit,
                 lean_version=version.strip(), retrieval_requests=len(events),
                 retrieval_ok=bool(events) and all(e['success'] for e in events))
    save(a.output / 'evaluation.json', stats)
    if native_exit or stats['completed_problems'] != stats['expected_problems'] or not stats['retrieval_ok']:
        raise RuntimeError('Incomplete evaluation or retrieval failure; full denominator retained')


if __name__ == '__main__':
    main()
