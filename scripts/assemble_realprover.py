"""Create an isolated, traceable evaluation-code subset from read-only FedProver."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT.parent / 'FedProver'
DEST = ROOT / 'REALPROVER'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    if DEST.exists():
        raise RuntimeError(f'Refusing to overwrite existing directory: {DEST}')
    sources = []
    for folder in ['conf', 'manager', 'util']:
        sources.extend(p for p in (SOURCE / folder).rglob('*.py')
                       if not any(n in p.parts for n in ['__pycache__', 'tests'])
                       and p.name != 'debug_util.py')
    sources.extend(SOURCE / 'experiment' / n for n in ['__init__.py', 'run.py', 'score.py'])
    sources.extend(SOURCE / 'data' / n for n in ['minif2f_test.jsonl', 'proofnet_test.jsonl', 'fate_m.jsonl'])
    # Scan before creating the target; never copy credential literals.
    for p in sources:
        if re.search(rb'(ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-ant-[A-Za-z0-9_-]{20,})', p.read_bytes()):
            raise RuntimeError(f'Credential-like content requires redaction: {p.relative_to(SOURCE)}')
    entries, patches = [], []
    for p in sorted(sources):
        rel = p.relative_to(SOURCE)
        original = p.read_bytes()
        data = original
        if str(rel) == 'conf/config.py':
            text = original.decode()
            replacements = {
                "PROVER_MODEL_PATH = '/XYFS01/HDD_POOL/sysu_qling/sysu_qling_3/tan/FedProver/model/bubble_merged'":
                    "PROVER_MODEL_PATH = os.environ.get('REALPROVER_MODEL_PATH', str(Path(__file__).resolve().parents[2] / 'models/REAL-Prover-v1-bf16'))",
                "LEAN_TEST_PATH = '/HOME/sysu_qling/sysu_qling_3/project/lean_test_v4160'":
                    "LEAN_TEST_PATH = os.environ.get('REALPROVER_LEAN_WORKSPACE', str(Path(__file__).resolve().parents[1] / 'dependencies/lean_test_v4160'))",
                'analyzer_path = Path("/HOME/sysu_qling/sysu_qling_3/project/lean_deps/jixia/.lake/build/bin/jixia")':
                    "analyzer_path = Path(os.environ.get('REALPROVER_JIXIA_BIN', str(base_path / 'dependencies/jixia/.lake/build/bin/jixia')))",
                'interactive_path = Path("/HOME/sysu_qling/sysu_qling_3/project/lean_deps/interactive")':
                    "interactive_path = Path(os.environ.get('REALPROVER_INTERACTIVE_ROOT', str(base_path / 'dependencies/interactive')))",
                'build_path = Path("/HOME/sysu_qling/sysu_qling_3/project/lean_deps/interactive/.lake/build")':
                    "build_path = interactive_path / '.lake/build'",
            }
            for old, new in replacements.items():
                assert old in text, old
                text = text.replace(old, new)
            data = text.encode()
        elif str(rel) == 'manager/thirdparty/verifier.py':
            text = original.decode().replace(
                'DEFAULT_LEAN_WORKSPACE = "/HOME/sysu_qling/sysu_qling_3/project/lean_test_v4160"',
                'from conf import config\nDEFAULT_LEAN_WORKSPACE = config.LEAN_TEST_PATH')
            data = text.encode()
        elif str(rel) == 'manager/thirdparty/claude.py':
            # Local evaluation does not need the optional remote client SDK.
            text = original.decode().replace('from anthropic import AsyncAnthropic\n', '')
            text = text.replace('    def __init__(self):\n',
                                '    def __init__(self):\n        from anthropic import AsyncAnthropic\n')
            data = text.encode()
        if rel.suffix == '.py':
            ast.parse(data.decode())
        target = DEST / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        assert p.read_bytes() == original, 'Source changed during extraction'
        entries.append({'path': str(rel), 'source_sha256': sha(original),
                        'copied_sha256': sha(data), 'modified': original != data})
        if original != data:
            patches.extend(difflib.unified_diff(original.decode().splitlines(True),
                data.decode().splitlines(True), fromfile='FedProver/' + str(rel),
                tofile='REALPROVER/' + str(rel)))
    (DEST / 'provenance').mkdir()
    (DEST / 'provenance/fedprover_adaptations.patch').write_text(''.join(patches))
    manifest = {'source': str(SOURCE), 'source_git_sha': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip(),
        'status': 'EXTRACTED_NOT_LEAN_RUNTIME_VALIDATED', 'files': entries}
    (DEST / 'provenance/source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (DEST / 'configs').mkdir()
    for name in ['p0_candidate.toml', 'p0_smoke2_candidate.toml']:
        text = (ROOT / 'research/configs' / name).read_text().replace(
            str(ROOT / 'datasets/eval/legacy_candidate/minif2f_test.jsonl'), str(DEST / 'data/minif2f_test.jsonl'))
        (DEST / 'configs' / name).write_text(text)
    (DEST / 'dependencies').mkdir()
    print(json.dumps({'target': str(DEST), 'files': len(entries),
                      'adapted': [e['path'] for e in entries if e['modified']]}, indent=2))


if __name__ == '__main__':
    main()
