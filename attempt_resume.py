"""Validate immutable attempt slots and publish new results without overwriting."""
import json
import os
from pathlib import Path
import tempfile


def read_attempts(directory, max_retries, statement=None):
    directory = Path(directory)
    records = {}
    success = False
    for path in sorted(directory.glob('*.json')):
        prefix = directory.name + '_'
        suffix = path.stem.removeprefix(prefix)
        if not path.stem.startswith(prefix) or not suffix.isdigit():
            raise ValueError(f'Unexpected attempt filename: {path}')
        index = int(suffix)
        if suffix != str(index) or not 0 <= index < max_retries or index in records:
            raise ValueError(f'Invalid/duplicate attempt slot: {path}')
        data = json.loads(path.read_text())
        if not isinstance(data.get('formal_statement'), str):
            raise ValueError(f'Missing theorem: {path}')
        if statement is not None and data['formal_statement'] != statement:
            raise ValueError(f'Result theorem mismatch: {path}')
        results = data.get('collect_results')
        if not isinstance(results, list) or any(not isinstance(r, dict) or type(r.get('success')) is not bool for r in results):
            raise ValueError(f'Malformed attempt result: {path}')
        claimed = any(r['success'] for r in results)
        if claimed and data.get('pipeline_verifier') != {'executed': True, 'success': True}:
            raise ValueError(f'Success lacks complete Lean verification: {path}')
        success = success or claimed
        records[index] = data
    return records, success


def write_attempt(path, data):
    """Publish an entire JSON atomically; an existing attempt always wins."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Atomic, exclusive: never overwrite an old slot.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
