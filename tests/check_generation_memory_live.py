"""GPU regression: reproduce retained vLLM groups, then verify stable cleanup/output."""
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('generation_memory', ROOT / 'REALPROVER/manager/thirdparty/generation_memory.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
from vllm import LLM, SamplingParams


def rss():
    return int(next(s.split()[1] for s in Path('/proc/self/status').read_text().splitlines() if s.startswith('VmRSS:'))) / 1024


def digest(outputs):
    values = [[(o.text, list(o.token_ids), o.cumulative_logprob,
                [[(k, v.logprob) for k, v in token.items()] for token in o.logprobs])
               for o in r.outputs] for r in outputs]
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def main():
    report = ROOT / 'runs/memory_repair_20261026/live_gpu.json'
    options = json.loads((ROOT / 'REALPROVER/configs/vllm_runtime.json').read_text())
    llm = LLM(model=str(ROOT / 'runs/realprover_formal_four_gpu_v9/formal/P1/eval_model'),
              seed=2, trust_remote_code=True, **options)
    prompt = '<|im_start|user\nProve in Lean 4: x : ℕ\n⊢ x = x<|im_end|>\n<|im_start|assistant\n'
    records = []
    for phase, repeats, tokens in [('unpatched', 8, 64), ('patched', 24, 64), ('formal_length', 2, 256)]:
        for index in range(repeats):
            params = SamplingParams(n=64, temperature=1.5, top_p=.9, max_tokens=tokens,
                                    logprobs=1, ignore_eos=True)
            outputs = llm.generate([prompt], params, use_tqdm=False)
            before = len(llm.llm_engine.seq_id_to_seq_group)
            hash_before = digest(outputs)
            removed = 0
            if phase != 'unpatched':
                removed = module.release_completed_parallel_groups(llm)
                assert len(llm.llm_engine.seq_id_to_seq_group) == 0
            assert digest(outputs) == hash_before, 'Cleanup changed generated text/tokens/logprobs'
            records.append({'phase': phase, 'index': index, 'tokens': tokens,
                            'map_before': before, 'removed': removed,
                            'map_after': len(llm.llm_engine.seq_id_to_seq_group),
                            'rss_mib': rss(), 'output_sha256': hash_before})
            print(json.dumps(records[-1]), flush=True)
            report.write_text(json.dumps({'status': 'RUNNING', 'records': records}, indent=2))
            del outputs
        if phase == 'unpatched':
            assert records[-1]['map_after'] == 8 * 64, 'Installed version did not reproduce expected leak'
            module.release_completed_parallel_groups(llm)
            gc.collect()
    patched = [r['rss_mib'] for r in records if r['phase'] == 'patched']
    growth = patched[-1] - patched[4]
    assert growth < 128, f'Patched RSS continues growing: {growth} MiB'
    report.write_text(json.dumps({'status': 'PASS', 'records': records,
        'patched_growth_mib_after_warmup': growth, 'engine_options': options,
        'sampling': {'n': 64, 'temperature': 1.5, 'top_p': .9, 'logprobs': 1},
        'output_parity': 'Every request text/token IDs/cumulative and per-token logprobs identical before/after cleanup'}, indent=2))
    print('PASS: reproduced leak, bounded completed ownership and RSS, exact output preservation', flush=True)


if __name__ == '__main__':
    main()
