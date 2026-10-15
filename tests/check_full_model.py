"""Verify downloaded REAL-Prover and exercise full BF16 load/reload on GPU."""
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'adapters'))
from common import MODEL, install_vendor, save, verify_model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from quantization import export
    install_vendor()
    torch.manual_seed(2)
    started = time.monotonic()
    identity = verify_model()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    assert tokenizer.chat_template, 'Missing REAL-Prover chat template'
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
        attn_implementation='eager', local_files_only=True).cuda().eval()
    assert all(torch.isfinite(p).all().item() for p in model.parameters()), 'Nonfinite model weights'
    report = export(model, tokenizer, args.output.parent / 'p0_export', 'P0')
    save(args.output, {'status': 'PASS', 'identity': identity, 'export': report,
        'seconds': time.monotonic() - started,
        'peak_allocated_GB': torch.cuda.max_memory_allocated() / 1e9,
        'note': 'Full BF16 model validation; excludes Lean and is not a P0 smoke receipt.'})
    print('PASS full REAL-Prover BF16 load, finite weights, 16 prompts, generation and reload parity')


if __name__ == '__main__':
    main()
