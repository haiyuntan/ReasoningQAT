"""Process-local aggregate timing; no prompts, RNG or model state changes."""
from contextlib import contextmanager
from functools import wraps
import json
import os
from pathlib import Path
import time
import sys

_values = {}
_started = time.perf_counter()
_io_errors = 0


def reset():
    global _started
    _values.clear()
    _started = time.perf_counter()


def snapshot():
    return {'elapsed_seconds': time.perf_counter()-_started,
            'phases': {key: dict(value) for key, value in _values.items()}}


def emit():
    global _io_errors
    output = os.environ.get('REALPROVER_OUTPUT')
    if not output:
        return
    path = Path(output) / 'eval/performance.jsonl'
    event = dict(snapshot(), timestamp=time.time(), pid=os.getpid())
    fd = None
    try:
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        os.write(fd, (json.dumps(event)+'\n').encode())
    except OSError as error:
        _io_errors += 1
        if _io_errors == 1:
            print(f'Performance telemetry unavailable: {error}', file=sys.stderr, flush=True)
    finally:
        if fd is not None:
            os.close(fd)


@contextmanager
def measure(name):
    started = time.perf_counter()
    try:
        yield
    finally:
        item = _values.setdefault(name, {'calls': 0, 'seconds': 0.0})
        item['calls'] += 1
        item['seconds'] += time.perf_counter()-started
        if name == 'generation' and item['calls'] % 16 == 0:
            emit()


def timed(name):
    def decorate(function):
        @wraps(function)
        def call(*args, **kwargs):
            with measure(name):
                return function(*args, **kwargs)
        return call
    return decorate
