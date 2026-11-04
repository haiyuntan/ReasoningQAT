"""Check saved real-query vectors against the immutable CPU FAISS reference."""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import time
import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vectors', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shards', type=int, default=8)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('exact_search',
        ROOT / 'REALPROVER/retrieval/worker/exact_search.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = faiss.read_index(str(ROOT / 'models/retrieval/LeanSearch-PS-faiss.index'))
    search = module.ExactSearch(original, args.shards)
    vectors, reference = np.load(args.vectors), np.load(args.reference)
    if len(vectors) != len(reference['ids']) or len(vectors) == 0:
        raise ValueError('Query/reference count mismatch or empty validation')
    timings, mismatches = [], []
    for i, row in enumerate(vectors):
        started = time.perf_counter()
        scores, ids = search.search(row[None, :], reference['ids'].shape[1])
        timings.append(time.perf_counter() - started)
        if (not np.array_equal(scores, reference['scores'][i:i+1])
                or not np.array_equal(ids, reference['ids'][i:i+1])):
            mismatches.append(i)
    result = {'status': 'PASS' if not mismatches else 'FAIL', 'queries': len(vectors),
              'shards': args.shards, 'mean_seconds': statistics.mean(timings),
              'max_seconds': max(timings), 'mismatched_queries': mismatches,
              'tie_fallbacks': search.tie_fallbacks}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    if mismatches:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
