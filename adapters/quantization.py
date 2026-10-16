"""PTQ and common export/parity checks for Qwen2 REAL-Prover."""
import argparse
from pathlib import Path

from common import ROOT, RESEARCH, MODEL, load, save, sha, install_vendor

PROMPTS = [
    '⊢ True', 'n : ℕ\n⊢ n = n', 'a b : ℕ\nh : a = b\n⊢ b = a',
    '⊢ (0 : ℕ) + 1 = 1', 'p : Prop\nh : p\n⊢ p', '⊢ (2 : ℕ) ≤ 3',
    'x : ℝ\n⊢ x + 0 = x', 'a b : ℝ\n⊢ a + b = b + a',
    '⊢ ∀ n : ℕ, n = n', 'p q : Prop\nh : p ∧ q\n⊢ q',
    's : Set ℕ\n⊢ s = s', 'n : ℤ\n⊢ n - n = 0',
    'a : ℕ\n⊢ 1 * a = a', 'p : Prop\n⊢ p → p',
    'x : ℚ\n⊢ x * 0 = 0', 'n : ℕ\n⊢ n ≤ n'
]


def probe(model, tokenizer):
    import torch
    outputs = []
    generated = []
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for text in PROMPTS:
            inputs = tokenizer(text, return_tensors='pt').to(device)
            logits = model(**inputs).logits[:, -1].float().cpu()
            if not torch.isfinite(logits).all():
                raise FloatingPointError('Nonfinite export probe logits')
            outputs.append(logits)
            generated.append(model.generate(**inputs, max_new_tokens=4, do_sample=False,
                                           pad_token_id=tokenizer.eos_token_id).cpu().tolist())
    return outputs, generated


def export(model, tokenizer, out, method):
    import torch
    from transformers import AutoModelForCausalLM
    from quantize.int_linear_fake import QuantLinear
    from quantize.utils import set_op_by_name
    gptq_reference = None
    fp32_buffers = {}
    if method == 'P1':
        from types import MethodType
        # A packed TorchQuantLinear rounds matmul and bias addition separately;
        # nn.Linear may fuse them. BF16 outputs therefore need not be bitwise
        # equal even when every dequantized weight and bias is identical.
        # Validate the conversion in FP64 using the EXACT original BF16
        # dequantization result, not recomputed higher-precision scale arithmetic.
        def reference_weight(module, num_itr=1):
            if num_itr != 1:
                raise ValueError('Unsupported GPTQ export grouping')
            return module._export_reference_weight
        # RoPE inv_freq is constructed in FP32 even in a BF16 model; a blanket
        # dtype round trip must not quantize these nonpersistent buffers.
        fp32_buffers = {name: value.detach().clone() for name, value in model.named_buffers()
                        if value.dtype == torch.float32}
        with torch.no_grad():
            for module in model.modules():
                if hasattr(module, 'qweight') and hasattr(module, 'scales'):
                    module._export_reference_weight = module.dequantize_weight().detach().float()
                    module.dequantize_weight = MethodType(reference_weight, module)
        model.double()
        model.config._attn_implementation = 'eager'
        gptq_reference = {'arithmetic': 'FP64; exact native BF16 dequantized weights and biases',
                          'deployment_dtype': 'bfloat16', 'exact_weight_checks': 0,
                          'preserved_fp32_buffers': sorted(fp32_buffers)}
    model.config.use_cache = True
    before, generation = probe(model, tokenizer)
    tolerance = load(RESEARCH / 'configs/runtime.json')['export']
    native_probe = out.parent / 'train/native_probe.pt'
    native_error = None
    if method in ('P3', 'P4', 'P5', 'P6'):
        reference = torch.load(native_probe, map_location='cpu', weights_only=True)
        if not all(torch.allclose(a, b, atol=tolerance['atol'], rtol=tolerance['rtol'])
                   for a, b in zip(before, reference['logits'])) or generation != reference['generated']:
            raise ValueError('Native checkpoint reload parity failed')
        native_error = max(float((a - b).abs().max()) for a, b in zip(before, reference['logits']))
    count = 0
    quantized_names = []
    with torch.no_grad():
        for name, module in list(model.named_modules()):
            if isinstance(module, QuantLinear):
                if module.weight_quantizer.n_bits != 2 or module.weight_quantizer.group_size != 128:
                    raise ValueError('QAT module violates W2g128')
                weight = module.weight_quantizer(module.weight)
                in_features, out_features = module.in_features, module.out_features
            elif hasattr(module, 'qweight') and hasattr(module, 'scales'):
                if method == 'P5':
                    if module.bits != 2 or module.group_size != 128:
                        raise ValueError('EfficientQAT module violates W2g128')
                    from baseline_ops import packed_weight
                    weight = packed_weight(module).t().contiguous()
                    in_features, out_features = module.infeatures, module.outfeatures
                elif method == 'P1':
                    if module.bits != 2 or module.group_size != 128:
                        raise ValueError('GPTQ module violates W2g128')
                    weight = module.dequantize_weight().t().contiguous()
                    in_features, out_features = module.in_features, module.out_features
                else:
                    raise RuntimeError('Unknown packed module')
            else:
                continue
            layer = torch.nn.Linear(in_features, out_features, module.bias is not None,
                                    device=weight.device, dtype=next(model.parameters()).dtype)
            layer.weight.copy_(weight)
            if method == 'P1':
                if not torch.equal(layer.weight, module._export_reference_weight.t()):
                    raise ValueError('GPTQ materialization changed a dequantized weight')
                gptq_reference['exact_weight_checks'] += 1
                # Do not retain both cached native and dense weights for the
                # entire model: FP64 validation must fit the single 80GB GPU.
                del module._export_reference_weight
            if module.bias is not None:
                layer.bias.copy_(module.bias)
                if method == 'P1' and not torch.equal(layer.bias, module.bias):
                    raise ValueError('GPTQ materialization changed a bias')
            set_op_by_name(model, name, layer)
            count += 1
            quantized_names.append(name)
    if method not in ('P0', 'P2') and count != model.config.num_hidden_layers * 7:
        raise ValueError(f'Expected 7 quantized projections per Qwen2 block, found {count}')
    if hasattr(model.config, 'quantization_config'):
        del model.config.quantization_config
    model.config.use_cache = True
    materialized, generated = probe(model, tokenizer)
    tolerance = load(RESEARCH / 'configs/runtime.json')['export']
    def compare(other, stage):
        if not all(torch.allclose(a, b, atol=tolerance['atol'], rtol=tolerance['rtol']) for a, b in zip(before, other)):
            save(out / 'parity_failure.json', {
                'method': method, 'stage': stage, 'tolerance': tolerance,
                'max_abs_errors': [float((a-b).abs().max()) for a, b in zip(before, other)],
                'gptq_reference_validation': gptq_reference})
            raise ValueError('Export logits parity failed')
        return max(float((a - b).abs().max()) for a, b in zip(before, other))
    materialized_error = compare(materialized, 'native_to_dense_reference')
    if generated != generation:
        raise ValueError('Export greedy-generation parity failed')
    if method == 'P1':
        # The deployment remains BF16 for every method. Reload parity is checked
        # separately against this canonical dense BF16 model, at unchanged tolerances.
        model.bfloat16()
        for name, value in fp32_buffers.items():
            set_op_by_name(model, name, value)
        before, generation = probe(model, tokenizer)
    out.mkdir(parents=True, exist_ok=True)
    if method != 'P0':
        model.save_pretrained(out, safe_serialization=True)
        tokenizer.save_pretrained(out)
    dtype = next(model.parameters()).dtype
    # Reload on CPU so original + reloaded checkpoints fit one 80GB card even
    # if a future model is larger. Move original off GPU before the reload.
    model.cpu()
    torch.cuda.empty_cache()
    reloaded = AutoModelForCausalLM.from_pretrained(MODEL if method == 'P0' else out, torch_dtype=dtype,
        attn_implementation='eager', local_files_only=True).cuda().eval()
    after, regenerated = probe(reloaded, tokenizer)
    error = compare(after, 'deployment_save_reload')
    if regenerated != generation:
        raise ValueError('Reload generation parity failed')
    base_tokenizer_files = ['tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt']
    # save_pretrained may normalize tokenizer_config; compare actual vocabulary,
    # special IDs and template instead of assuming byte-identical serialization.
    from transformers import AutoTokenizer
    base = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    if tokenizer.get_vocab() != base.get_vocab() or tokenizer.chat_template != base.chat_template:
        raise ValueError('Tokenizer/template changed')
    base_config = load(MODEL / 'config.json')
    for key in ('model_type', 'hidden_size', 'num_hidden_layers', 'vocab_size', 'num_attention_heads', 'num_key_value_heads'):
        if getattr(reloaded.config, key) != base_config[key]:
            raise ValueError(f'Changed architecture: {key}')
    meta = {'method': method, 'bits': 16 if method == 'P0' else 2, 'group_size': None if method == 'P0' else 128,
            'storage': 'dequantized-for-evaluation; no INT2 performance claim',
            'model_type': 'qwen2', 'replaced_modules': quantized_names,
            'materialized_logits_max_error': materialized_error, 'reload_logits_max_error': error,
            'native_reload_logits_max_error': native_error,
            'gptq_reference_validation': gptq_reference,
            'prompts': len(PROMPTS), 'greedy_generation_parity': True,
            'tokenizer_parity': True, 'saved_reload': True}
    save(out / 'export.json', meta)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', choices=['P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6'], required=True)
    ap.add_argument('--mode', choices=['smoke', 'formal'], required=True)
    ap.add_argument('--input', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    install_vendor()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from quantize.utils import set_op_by_name, set_quant_state
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    if a.run in ('P0', 'P2'):
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                attn_implementation='eager', local_files_only=True)
        if a.run == 'P2':
            import sys
            sys.path.insert(0, str(ROOT / 'ptq/llm-awq'))
            from awq.quantize.pre_quant import run_awq
            from awq.quantize.quantizer import pseudo_quantize_model_weight
            model.config.use_cache = False
            result = run_awq(model, tokenizer, w_bit=2, q_config={'zero_point': True, 'q_group_size': 128},
                             n_samples=128, seqlen=512, calib_data='numinamath_local')
            a.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(result, a.output.parent / 'awq_search.pt')
            if not result.get('scale') or not result.get('clip'):
                raise RuntimeError('AWQ scale search or clipping evidence missing')
            pseudo_quantize_model_weight(model, w_bit=2, q_config={'zero_point': True, 'q_group_size': 128})
            observed = []
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear) and name.startswith('model.layers.'):
                    if module.in_features % 128:
                        raise RuntimeError('AWQ projection is not divisible into g128')
                    groups = module.weight.detach().reshape(-1, 128)
                    # Check distributed groups in every projection. This is a
                    # numerical smoke of the actual quantized result, not a
                    # substitute for running AWQ search/clipping above.
                    indices = torch.linspace(0, len(groups)-1, min(32, len(groups))).long()
                    if any(g.unique().numel() > 4 for g in groups[indices]):
                        raise RuntimeError('AWQ weights have more than four levels per sampled group')
                    observed.append(name)
            if len(observed) != model.config.num_hidden_layers * 7:
                raise RuntimeError('AWQ target module coverage mismatch')
            save(a.output.parent / 'awq_modules.json', {'names': observed, 'bits': 2, 'group_size': 128})
        model.cuda()
    elif a.run == 'P1':
        from gptqmodel import GPTQModel, QuantizeConfig, BACKEND
        rows = [__import__('json').loads(x) for x in (ROOT / 'datasets/numinamath_128/calib.jsonl').read_text().splitlines()]
        if len(rows) != 128:
            raise ValueError('Expected exactly NuminaMath128')
        calibration = []
        for row in rows:
            ids = tokenizer.encode(row['text'], add_special_tokens=True)[:512]
            ids += [tokenizer.eos_token_id] * (512 - len(ids))
            calibration.append({'input_ids': torch.tensor([ids]), 'attention_mask': torch.ones(1, 512, dtype=torch.long)})
        native = a.output.parent / 'native'
        wrapper = GPTQModel.load(str(MODEL), QuantizeConfig(bits=2, group_size=128, desc_act=False, sym=True), backend=BACKEND.TORCH)
        wrapper.quantize(calibration, batch_size=1)
        wrapper.save(str(native))
        del wrapper
        torch.cuda.empty_cache()
        wrapper = GPTQModel.load(str(native), backend=BACKEND.TORCH)
        model = wrapper.model.cuda().eval()
    elif a.run == 'P5':
        from baseline_ops import portable_packed_class
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                attn_implementation='eager', local_files_only=True)
        Packed = portable_packed_class()
        for name, module in list(model.named_modules()):
            if isinstance(module, torch.nn.Linear) and name.startswith('model.layers.'):
                packed = Packed(2, 128, module.in_features, module.out_features, module.bias is not None)
                packed.scales.data = packed.scales.data.float()
                set_op_by_name(model, name, packed)
        state = torch.load(a.input / 'efficientqat_state.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
        model.cuda()
    else:
        from quantize.int_linear_fake import load_quantized_model
        model, _ = load_quantized_model(str(a.input), 2, 128, replace=True, strict=True, use_flash_attn=False, device='cpu')
        set_quant_state(model, True)
        model.cuda()
    result = export(model, tokenizer, a.output, a.run)
    if a.run in ('P1', 'P2'):
        result['calibration_sha256'] = sha(ROOT / 'datasets/numinamath_128/calib.jsonl')
        result['samples'] = 128
        result['sequence_length'] = 512
        save(a.output / 'export.json', result)


if __name__ == '__main__':
    main()
