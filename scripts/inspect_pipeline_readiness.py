"""Read-only readiness report; never installs dependencies or launches jobs."""
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'adapters'))
from common import ROOT, RESEARCH, MODEL, load, python_for


def main():
    runtime = load(RESEARCH / 'configs/runtime.json')
    issues = []
    download = load(RESEARCH / 'manifests/model_revision.json')
    total = load(MODEL / 'model.safetensors.index.json')['metadata']['total_size']
    received = sum(p.stat().st_size for p in MODEL.iterdir() if p.name.endswith(('.safetensors', '.part')))
    if download['status'] not in ('DOWNLOADED_REQUIRES_LOAD_CHECK', 'VERIFIED'):
        issues.append('MODEL_DOWNLOAD_INCOMPLETE')
    for role in runtime['python']:
        try:
            python_for(role)
        except RuntimeError as exc:
            issues.append(str(exc))
    dependencies = ROOT / 'REALPROVER/dependencies'
    lean = Path(os.environ.get('REALPROVER_LEAN_WORKSPACE', str(dependencies / 'lean_test_v4160')))
    jixia = Path(os.environ.get('REALPROVER_JIXIA_BIN', str(dependencies / 'jixia/.lake/build/bin/jixia')))
    interactive = Path(os.environ.get('REALPROVER_INTERACTIVE_ROOT', str(dependencies / 'interactive')))
    for name, path in [('LEAN_TOOLCHAIN', lean / 'lean-toolchain'), ('LEAN_HEADER', lean / 'Header.lean'),
                       ('JIXIA_BINARY', jixia), ('INTERACTIVE_BINARY', interactive / '.lake/build/bin/interactive')]:
        if not path.is_file():
            issues.append(f'MISSING_{name}: {path}')
    formal_issues = []
    if runtime['eval_protocol_status'] != 'FROZEN':
        formal_issues.append('FORMAL_EVAL_PROTOCOL_AND_OFFICIAL_DATA_PARITY_NOT_FROZEN')
    if not (ROOT / 'datasets/numinamath_128/calib.jsonl').is_file():
        issues.append('MISSING_NUMINAMATH128')
    print(json.dumps({'status': 'BLOCKED' if issues else 'REQUIRES_REAL_P0_P6_SMOKE',
        'download': {'status': download['status'], 'GB': received / 1e9, 'total_GB': total / 1e9,
                     'percent': min(100., received / total * 100)},
        'issues': issues,
        'formal_issues': formal_issues,
        'formal_status': 'BLOCKED' if issues or formal_issues else 'REQUIRES_ALL_P0_P6_SMOKE_PASS',
        'retrieval_status': 'NOT_PROBED_BY_READ_ONLY_INSPECTOR; execution preflight probes signed requests',
        'required_smoke': [f'P{i}' for i in range(7)]}, indent=2))


if __name__ == '__main__':
    main()
