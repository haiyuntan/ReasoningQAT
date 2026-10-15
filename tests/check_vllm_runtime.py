"""Exercise the actual native generator with restored retrieval and BF16 model."""
import argparse
import math
from pathlib import Path
import sys
import time

RESEARCH = Path(__file__).resolve().parents[1]
ROOT = RESEARCH.parent
sys.path.insert(0, str(ROOT / 'REALPROVER'))
sys.path.insert(0, str(RESEARCH / 'adapters'))
from common import save


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    started = time.monotonic()
    from conf import config
    from manager.thirdparty.generator import TacticGenerator
    params = dict(config.PROVER_MODEL_PARAMS, max_tokens=8)
    generator = TacticGenerator(['local'], 0, local_model_path=config.PROVER_MODEL_PATH,
                                sampling_params=params, max_calls=1)
    generator._init_model()
    raw = []
    original = generator.llm.generate

    def capture(*args, **kwargs):
        outputs = original(*args, **kwargs)
        raw.extend({'text': item.text, 'tokens': len(item.token_ids),
                    'cumulative_logprob': item.cumulative_logprob}
                   for request in outputs for item in request.outputs)
        return outputs

    generator.llm.generate = capture
    tactics, scores, prompt = generator.get_lean_tactics('n : ℕ\n⊢ n = n', 1,
        incontext='theorem pipeline_env_nat (n : ℕ) : n = n := by\n', template='qwen', use_retrieval=True)
    save(args.output, {'status': 'DIAGNOSTIC', 'raw_outputs': raw,
                      'tactics': tactics, 'scores': scores, 'prompt': prompt})
    assert len(raw) == 1 and raw[0]['tokens'] > 0 and raw[0]['text'].strip()
    assert math.isfinite(raw[0]['cumulative_logprob'])
    assert len(tactics) == len(scores)
    assert all(math.isfinite(score) for score in scores)
    save(args.output, {'status': 'PASS', 'seconds': time.monotonic() - started,
         'tactics': tactics, 'scores': scores, 'raw_outputs': raw, 'prompt': prompt,
         'note': 'Native vLLM generator runtime check; not a verified Lean proof or P0 smoke receipt.'})
    print('PASS native REAL-Prover BF16 vLLM generation with local retrieval', flush=True)


if __name__ == '__main__':
    main()
