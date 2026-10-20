"""Targeted live checks; these are not P0-P6 smoke receipts or accuracy results."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'research/adapters'))
from common import MODEL, load, save


def lean():
    sys.path.insert(0, str(ROOT / 'REALPROVER'))
    from conf import config
    from manager.thirdparty.interactive import Interactive
    from manager.thirdparty.verifier import verify_proof
    root = Path(config.LEAN_TEST_PATH)
    path = root / f'pipeline_probe_{os.getpid()}.lean'
    path.write_text('theorem pipeline_local_probe (n : ℕ) : n = n := by sorry\n')
    started = time.monotonic()
    session = Interactive(root, Path('Header.lean'))
    try:
        session.open_file(path, [None])
        declaration = session.get_next_problem()
        assert declaration
        loaded = time.monotonic() - started
        sid = session.run_tactic(0, 'rfl')
        assert not session.get_state(sid)
        session.commit(sid)
        assert session.get_next_problem() is None
    finally:
        session.close()
        path.unlink(missing_ok=True)
    assert verify_proof('theorem pipeline_valid : True := by trivial',
                        str(Path(config.LEAN_ENV_PATH) / 'lake'), root)
    assert not verify_proof('theorem pipeline_invalid : False := by sorry',
                           str(Path(config.LEAN_ENV_PATH) / 'lake'), root)
    return {'status': 'PASS', 'load_seconds': loaded, 'total_seconds': time.monotonic() - started,
            'lean_workspace': str(root), 'lean_bin': config.LEAN_ENV_PATH}


def vllm():
    from vllm import LLM, SamplingParams
    profile = load(ROOT / 'REALPROVER/configs/vllm_runtime.json')
    started = time.monotonic()
    engine = LLM(model=str(MODEL), tokenizer=str(MODEL), dtype='bfloat16', seed=2,
                 trust_remote_code=True, **profile)
    initialized = time.monotonic() - started
    # Same 64 candidates / 256-token upper bound as formal inference.
    outputs = engine.generate(['Prove the Lean theorem: theorem test (n : Nat) : n = n := by'],
                              SamplingParams(n=64, temperature=1.5, top_p=.9, max_tokens=256,
                                             min_tokens=256, ignore_eos=True))
    assert len(outputs) == 1 and len(outputs[0].outputs) == 64
    assert all(len(o.token_ids) == 256 for o in outputs[0].outputs)
    return {'status': 'PASS', 'profile': profile, 'initialization_seconds': initialized,
            'total_seconds': time.monotonic() - started, 'candidates': 64,
            'generated_tokens': sum(len(o.token_ids) for o in outputs[0].outputs),
            'cache_blocks': engine.llm_engine.cache_config.num_gpu_blocks}


def retrieval():
    import urllib.request
    port = os.environ.get('LEANSEARCH_PORT', '8080')
    query = {'query': 'n : ℕ\n⊢ n = n', 'num': 10}
    def request(_):
        started = time.monotonic()
        req = urllib.request.Request(f'http://127.0.0.1:{port}/retrieve_premises',
                                     json.dumps(query).encode(), {'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=180) as response:
            result = json.load(response)
        assert len(result['data'][0]) == 10
        return result['data'], time.monotonic() - started
    with ThreadPoolExecutor(max_workers=3) as pool:
        replies = list(pool.map(request, range(3)))
    assert replies[0][0] == replies[1][0] == replies[2][0]
    return {'status': 'PASS', 'simultaneous_clients': 3, 'seconds': [r[1] for r in replies]}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('check', choices=['lean', 'vllm', 'retrieval'])
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    result = globals()[args.check]()
    save(args.output, result)
    print(json.dumps(result), flush=True)
