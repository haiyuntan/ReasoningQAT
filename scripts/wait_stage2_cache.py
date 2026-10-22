"""Observe an already-running audited CPU preflight without running it twice."""
import argparse
import json
from pathlib import Path
import time

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pid', type=int, required=True)
    ap.add_argument('--cache', type=Path, required=True)
    args = ap.parse_args()
    cache = args.cache.resolve()
    proc = Path(f'/proc/{args.pid}')
    if proc.exists():
        command = [x.decode() for x in (proc / 'cmdline').read_bytes().split(b'\0') if x]
        if (not any(Path(x).name == 'prepare_stage2_cache.py' for x in command)
                or '--output' not in command
                or Path(command[command.index('--output') + 1]).resolve() != cache):
            raise RuntimeError('Refusing unrelated external preflight PID')
    while not (cache / 'manifest.json').exists():
        try:
            state = (proc / 'stat').read_text().split(') ', 1)[1].split()[0]
        except FileNotFoundError:
            state = 'Z'
        if state == 'Z':
            raise RuntimeError('CPU preflight ended without publishing a complete token cache')
        time.sleep(5)
    if json.loads((cache / 'manifest.json').read_text())['status'] != 'PASS':
        raise RuntimeError('CPU preflight did not pass')
    print('PASS: complete external stage2 cache published', flush=True)
