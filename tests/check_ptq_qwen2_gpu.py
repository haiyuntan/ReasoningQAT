"""Real PTQ operators on a tiny random Qwen2, not REAL-Prover accuracy."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'adapters'))
from common import ROOT, MODEL, install_vendor
install_vendor()
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', choices=['gptq', 'awq'], required=True)
    args = ap.parse_args()
    torch.set_num_threads(2); torch.manual_seed(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    config = Qwen2Config(vocab_size=len(tokenizer), hidden_size=128, intermediate_size=256,
                        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                        pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
                        use_cache=False)
    model = Qwen2ForCausalLM(config).half().eval()
    if args.method == 'awq':
        import os
        os.environ['RQ_DATA'] = str(ROOT / 'datasets')
        sys.path.insert(0, str(ROOT / 'ptq/llm-awq'))
        from awq.quantize.pre_quant import run_awq
        from awq.quantize.quantizer import pseudo_quantize_model_weight
        result = run_awq(model, tokenizer, w_bit=2, q_config={'zero_point': True, 'q_group_size': 128},
                         n_samples=2, seqlen=512, calib_data='numinamath_local')
        assert result['scale'] and result['clip']
        pseudo_quantize_model_weight(model, w_bit=2, q_config={'zero_point': True, 'q_group_size': 128})
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and name.startswith('model.layers.'):
                assert all(group.unique().numel() <= 4 for group in module.weight.reshape(-1, 128))
        print('PASS AWQ Qwen2 W2g128 search + clipping + quantization', flush=True)
    else:
        from gptqmodel import GPTQModel, QuantizeConfig, BACKEND
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / 'source'
            model.save_pretrained(source)
            tokenizer.save_pretrained(source)
            del model
            model = GPTQModel.load(str(source), QuantizeConfig(bits=2, group_size=128, desc_act=False, sym=True), backend=BACKEND.TORCH)
            data = [{'input_ids': torch.randint(0, 64, (1, 64)), 'attention_mask': torch.ones(1, 64, dtype=torch.long)} for _ in range(2)]
            model.quantize(data, batch_size=1)
            target = Path(d) / 'packed'
            model.save(str(target))
            del model
            torch.cuda.empty_cache()
            restored = GPTQModel.load(str(target), backend=BACKEND.TORCH)
            modules = [m for m in restored.model.modules() if hasattr(m, 'qweight')]
            assert len(modules) == 7 and all(m.bits == 2 and m.group_size == 128 for m in modules)
            with torch.no_grad():
                values = restored.model.cuda()(input_ids=torch.ones(1, 8, dtype=torch.long, device='cuda')).logits
                assert torch.isfinite(values).all()
            print('PASS GPTQ Qwen2 W2g128 quantize + save/reload + torch backend', flush=True)


if __name__ == '__main__':
    main()
