"""Validate all selected formal examples before scheduling GPU stage2 training."""
import argparse
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'adapters'))
from common import MODEL, install_vendor

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    os.environ['REALPROVER_BUILD_STAGE2_CACHE'] = str(args.output.resolve())
    os.environ.pop('REALPROVER_STAGE2_CACHE', None)
    install_vendor()
    from transformers import AutoTokenizer
    from training import stage2_data
    _, evidence = stage2_data(AutoTokenizer.from_pretrained(MODEL, local_files_only=True), 'formal')
    print('PASS stage2 token/mask preflight', evidence, flush=True)
