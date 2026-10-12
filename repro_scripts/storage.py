"""Measure run artifacts and remove verified, reproducible model duplicates."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]

def inventory(paths):
    files = []
    for root in paths:
        if root.exists():
            for path in sorted(root.rglob('*')):
                if path.is_file() and not path.is_symlink():
                    files.append({'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size})
    return {'total_bytes': sum(row['bytes'] for row in files), 'file_count': len(files),
            'files': files[:200], 'file_listing_truncated': len(files) > 200}

def prune(path, reason):
    path = Path(path)
    allowed = [ROOT / 'outputs', ROOT / 'logs']
    if path.is_symlink() or not any(root in path.resolve().parents for root in allowed):
        raise ValueError(f'Unsafe cleanup path: {path}')
    # A cleanup target must be a single run artifact, never a mode/session root.
    if len(path.relative_to(ROOT).parts) < 4:
        raise ValueError(f'Cleanup target too broad: {path}')
    measured = inventory([path])
    record = {'path': str(path.relative_to(ROOT)), 'reason': reason,
              'removed_bytes': measured['total_bytes'], 'file_count': measured['file_count']}
    if path.exists():
        shutil.rmtree(path)
    return record

def cleanup_verified_run(run, mode, out, eval_model, logdir, evidence_dir):
    """Called only after export, reload, evaluation and run gates succeed."""
    records = []
    if run not in ('R0', 'R1', 'R2'):
        metadata = json.loads((eval_model / 'repro_quantization.json').read_text())
        if 'reload_logits_max_diff' not in metadata or metadata.get('replaced_linears', 0) <= 0:
            raise ValueError('Missing export verification evidence')
        manifest = []
        for path in sorted(out.rglob('*')):
            if path.is_file() and not path.is_symlink():
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                        digest.update(block)
                manifest.append({'path': str(path.relative_to(out)), 'bytes': path.stat().st_size,
                                 'sha256': digest.hexdigest()})
        if not any(row['path'].endswith(('.safetensors', '.bin', '.pt')) for row in manifest):
            raise ValueError('Native checkpoint weights missing; retain export')
        (evidence_dir / f'{run}.checkpoint.json').write_text(json.dumps(
            {'native_files': manifest, 'export_verification': metadata}, indent=2))
        records.append(prune(eval_model, 'Verified HF export can be rebuilt from native checkpoint'))
    for path in sorted(logdir.glob('checkpoint-*')):
        records.append(prune(path, 'Final checkpoint and completed evaluation supersede trainer snapshot'))
    if mode == 'smoke' and run not in ('R0', 'R3'):
        records.append(prune(out, 'Passed smoke artifact; minimal training is reproducible'))
    return records
