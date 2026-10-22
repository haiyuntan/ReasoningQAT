"""Qwen2 block-QAT, independent EfficientQAT, and masked scale-only E2E."""
import argparse
import gc
import hashlib
import math
import os
from pathlib import Path
import random
import sys

from common import ROOT, RESEARCH, MODEL, VENDOR, load, save, install_vendor, model_signature


def calibration(tokenizer, mode):
    import torch
    from datautils_block import get_loaders
    formal = load(RESEARCH / 'configs/training_general_data.json')['stage1']
    tiny = load(RESEARCH / 'configs/runtime.json')['smoke']
    size, val, length = ((tiny['block_train'], tiny['block_val'], tiny['length']) if mode == 'smoke'
                         else (formal['train_size'], formal['val_size'], formal['sequence_length']))
    train, validation = get_loaders('sweep_0.8', tokenizer, size, val, seed=2, seqlen=length)
    h = hashlib.sha256()
    for sample in train + validation:
        h.update(sample[0].numpy().tobytes())
    evidence = {'train': len(train), 'val': len(validation), 'length': length,
                'token_sha256': h.hexdigest(), 'seed': 2, 'reasoning_ratio': .8,
                'sources': ['OpenThoughts3-1.2M', 'FineWeb-Edu'], 'tokenizer': str(MODEL)}
    return train, validation, evidence


class OptimizerAudit:
    """Observe actual optimizer gradients and updates, without modifying math."""
    def __enter__(self):
        import torch
        self.original = torch.optim.AdamW.step
        self.steps = []
        owner = self

        def step(optimizer, *args, **kwargs):
            params = [p for g in optimizer.param_groups for p in g['params'] if p.grad is not None]
            if not params:
                return owner.original(optimizer, *args, **kwargs)
            if not all(torch.isfinite(p.grad).all() for p in params):
                raise FloatingPointError('Nonfinite optimizer gradients')
            # Full tensor comparison avoids missing a sparse or tiny update.
            before = [p.detach().clone() for p in params]
            norm = float(torch.linalg.vector_norm(torch.stack([p.grad.float().norm() for p in params])))
            result = owner.original(optimizer, *args, **kwargs)
            changed = sum(int((p.detach() != old).sum()) for p, old in zip(params, before))
            owner.steps.append({'grad_norm': norm, 'updated_values': changed})
            return result
        torch.optim.AdamW.step = step
        return self

    def __exit__(self, *args):
        import torch
        torch.optim.AdamW.step = self.original


def stage2_data(tokenizer, mode):
    import torch
    from completion_tokens import completion_window
    cache = os.environ.get('REALPROVER_STAGE2_CACHE')
    if cache and (Path(cache) / 'manifest.json').exists():
        from stage2_cache import cached_batches
        return cached_batches(Path(cache), mode)
    from repro_scripts.local_datasets import load_local_or_hf_openthoughts
    config = load(RESEARCH / 'configs/training_general_data.json')['stage2']
    tiny = load(RESEARCH / 'configs/runtime.json')['smoke']
    count = tiny['stage2_samples'] if mode == 'smoke' else config['dataset_size']
    maximum = tiny['length'] if mode == 'smoke' else config['max_length']
    ds = load_local_or_hf_openthoughts().shuffle(seed=2).select(range(count))

    class Batches:
        def __init__(self):
            self.adjusted_indices = set()

        def __len__(self):
            return len(ds)

        def __getitem__(self, index):
            if index >= len(self):
                raise IndexError(index)
            messages = []
            for m in ds[index]['conversations']:
                role = {'human': 'user', 'gpt': 'assistant'}[m['from']]
                text = m['value']
                if mode == 'smoke':
                    text = tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)[:32 if role == 'user' else 96])
                messages.append({'role': role, 'content': text})
            if not messages or messages[-1]['role'] != 'assistant':
                raise ValueError('Expected final assistant completion')
            prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
            full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            full_ids = tokenizer.encode(full, add_special_tokens=False)
            prefix = tokenizer.encode(prompt, add_special_tokens=False)
            ids, labels, adjusted = completion_window(full_ids, prefix, maximum)
            if adjusted:
                self.adjusted_indices.add(index)
            return {'input_ids': torch.tensor([ids]), 'attention_mask': torch.ones(1, len(ids), dtype=torch.long),
                    'labels': torch.tensor([labels])}

    batches = Batches()
    # Freeze exact token/mask budget before optimizer updates. P4/P5/P6 all use
    # this function and must produce identical fingerprints/counts per mode.
    digest = hashlib.sha256()
    tokens = 0
    destination = os.environ.get('REALPROVER_BUILD_STAGE2_CACHE')
    if destination:
        from stage2_cache import Writer
        writer = Writer(destination, count, maximum, mode)
    for batch in batches:
        tokens += int((batch['labels'][:, 1:] != -100).sum())
        for key in ('input_ids', 'attention_mask', 'labels'):
            digest.update(batch[key].numpy().tobytes())
        if destination:
            writer.append(batch)
    evidence = {'samples': count, 'dataset_fingerprint': ds._fingerprint,
                     'tensor_sha256': digest.hexdigest(), 'causal_tokens': tokens,
                     'max_length': maximum, 'truncation': 'right normally; long prompt tail plus answer head, up to half length reserved for answer; no samples dropped',
                     'mask': 'final assistant completion only; same as pinned ChatML', 'seed': 2,
                     'long_prompt_samples': len(batches.adjusted_indices)}
    if destination:
        writer.finish(evidence)
    return batches, evidence


def reasoning_block_ap(model, train, validation, config, epochs, batch_size):
    """Port block-wise reconstruction with per-sample Qwen2 masks/rotary data.

    The old driver silently ignored validation and repeated batched attention
    masks. This driver retains the original quantizer, parameter grouping, MSE,
    cosine schedules and sequential FP/quantized activation propagation.
    """
    import copy
    import torch
    from baseline_training import capture_inputs, DiskSamples, to_device_kwargs, layer_output
    from quantize.int_linear_fake import QuantLinear, cast_qat_dtype
    from quantize.utils import set_op_by_name, set_quant_state, quant_inplace
    from quantize.utils import set_quant_parameters, set_weight_parameters, quant_parameters, weight_parameters
    device = torch.device('cuda')
    fp = capture_inputs(model, [x[0] for x in train + validation], device)
    quant = DiskSamples((hidden.clone(), kwargs) for hidden, kwargs in fp)
    model.requires_grad_(False)
    history = []
    for index, original in enumerate(model.model.layers):
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            targets = DiskSamples(layer_output(original, hidden.cuda(), to_device_kwargs(kwargs, device)).float().cpu()
                                  for hidden, kwargs in fp)
        layer = copy.deepcopy(original).float()
        for name, module in list(layer.named_modules()):
            if isinstance(module, torch.nn.Linear):
                set_op_by_name(layer, name, QuantLinear(module, 2, 128, scale=.7))
        set_quant_state(layer, True)
        set_quant_parameters(layer, True); set_weight_parameters(layer, True)
        groups = [{'params': list(quant_parameters(layer)), 'lr': config['quant_lr']},
                  {'params': list(weight_parameters(layer)), 'lr': config['weight_lr']}]
        optimizer = torch.optim.AdamW(groups, weight_decay=0.)
        total = epochs * math.ceil(len(train) / batch_size)
        schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(
            torch.optim.AdamW([torch.tensor(0.)], lr=lr), T_max=total, eta_min=lr / 20)
            for lr in (config['quant_lr'], config['weight_lr'])]
        for epoch in range(epochs):
            losses = []
            for start in range(0, len(train), batch_size):
                end = min(start + batch_size, len(train))
                optimizer.zero_grad(set_to_none=True)
                for sample in range(start, end):
                    hidden, kwargs = quant[sample]
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        prediction = layer_output(layer, hidden.cuda(), to_device_kwargs(kwargs, device))
                        loss = torch.nn.functional.mse_loss(prediction.float(), targets[sample].cuda())
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite ReasoningQAT reconstruction')
                    (loss / (end - start)).backward()
                    losses.append(float(loss.detach()))
                optimizer.step()
                for group, scheduler in zip(optimizer.param_groups, schedulers):
                    scheduler.step(); group['lr'] = scheduler.get_last_lr()[0]
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                val_loss = sum(float(torch.nn.functional.mse_loss(
                    layer_output(layer, hidden.cuda(), to_device_kwargs(kwargs, device)).float(), targets[i].cuda()))
                    for i, (hidden, kwargs) in enumerate(quant) if i >= len(train)) / len(validation)
            if not math.isfinite(val_loss):
                raise FloatingPointError('Nonfinite validation reconstruction')
            history.append({'block': index, 'epoch': epoch, 'mse': sum(losses) / len(losses), 'val_mse': val_loss})
            print(history[-1], flush=True)
        del optimizer
        cast_qat_dtype(layer, torch.bfloat16)
        quant_inplace(layer)
        # Canonical saved/reloaded fake-quant forward also used by Stage2.
        set_quant_state(layer, True)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            quant = DiskSamples((layer_output(layer, hidden.cuda(), to_device_kwargs(kwargs, device)).cpu(), kwargs)
                                for hidden, kwargs in quant)
        fp = DiskSamples((targets[i].to(hidden.dtype), kwargs) for i, (hidden, kwargs) in enumerate(fp))
        model.model.layers[index] = layer
        del original
    return history


def e2e(model, teacher, batches, run, mode, checkpoint=None, audit_steps=None, data_signature=None):
    import torch
    from losses import objective
    config = load(RESEARCH / 'configs/training_general_data.json')['stage2']
    tiny = load(RESEARCH / 'configs/runtime.json')['smoke']
    model.requires_grad_(False)
    suffix = '.scales' if run == 'P5' else '.weight_quantizer.scale'
    for name, p in model.named_parameters():
        if name.endswith(suffix):
            p.data = p.data.float()
            p.requires_grad_(True)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    if not names or not all(n.endswith(suffix) for n in names):
        raise RuntimeError('Scale-only parameter contract failed')
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train()
    if teacher is not None:
        teacher.requires_grad_(False).eval()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config['learning_rate'], betas=(.9, .95), eps=1e-6, weight_decay=0.)
    accumulation = tiny['accumulation'] if mode == 'smoke' else config['gradient_accumulation']
    epochs = 1 if mode == 'smoke' else config['epochs']
    steps = epochs * math.ceil(len(batches) / accumulation)
    if mode == 'smoke' and steps != tiny['stage2_steps']:
        raise ValueError('Smoke optimizer step budget mismatch')
    warmup = math.ceil(steps * .2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step:
        (step + 1) / max(1, warmup) if step < warmup else
        .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup))))
    history = []
    resume_epoch, resume_sample = 0, 0
    audit_start = len(audit_steps) if audit_steps is not None else 0
    identity = {'run': run, 'mode': mode, 'config': config, 'data': data_signature, 'names': names}
    if checkpoint is not None and checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if state['identity'] != identity:
            raise RuntimeError('Stage2 resume contract changed')
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.requires_grad:
                    p.copy_(state['scales'][name].to(p.device))
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        history = state['history']
        resume_epoch, resume_sample = state['epoch'], state['next_sample']
        if audit_steps is not None:
            audit_steps.extend(state['audit'])
        random.setstate(state['random'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        print(f'RESUME {run} epoch={resume_epoch} sample={resume_sample}', flush=True)
    for epoch in range(epochs):
        for start in range(0, len(batches), accumulation):
            if epoch < resume_epoch or (epoch == resume_epoch and start < resume_sample):
                continue
            optimizer.zero_grad(set_to_none=True)
            end = min(start + accumulation, len(batches))
            for i in range(start, end):
                batch = {k: v.cuda() for k, v in batches[i].items()}
                inputs = {k: v for k, v in batch.items() if k != 'labels'}
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    with torch.no_grad():
                        target = teacher(**inputs).logits if teacher is not None else None
                    logits = model(**inputs).logits
                    loss, metrics = objective(logits, batch['labels'], target,
                        config['teacher_rr_alpha'], config['kd_beta'], config['teacher_top_k'])
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite E2E loss')
                (loss / (end - start)).backward()
                history.append({'epoch': epoch, 'sample': i, 'loss': float(loss.detach()), **metrics})
                del loss, logits, target, batch, inputs
            torch.nn.utils.clip_grad_norm_(parameters, .5, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            print({'run': run, 'epoch': epoch, 'next_sample': end, 'samples': len(batches),
                   'last_loss': history[-1]['loss']}, flush=True)
            if checkpoint is not None:
                temporary = checkpoint.with_suffix('.tmp')
                torch.save({'identity': identity, 'scales': {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
                            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                            'history': history, 'epoch': epoch, 'next_sample': end,
                            'audit': audit_steps[audit_start:] if audit_steps is not None else [],
                            'random': random.getstate(), 'torch_rng': torch.get_rng_state(),
                            'cuda_rng': torch.cuda.get_rng_state_all()}, temporary)
                temporary.replace(checkpoint)
    return {'trainable_names': names, 'steps': history, 'optimizer_steps': steps,
            'causal_tokens_processed': sum(s['tokens'] for s in history), 'epochs': epochs,
            'teacher_used': teacher is not None, 'objective': 'Teacher-RR+top20 KL' if run == 'P4' else 'ordinary CE',
            'engineering_change': 'Explicit torch loop replaces TRL; same loss/mask, specified cosine warmup; no on-policy generation'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', choices=['P3', 'P4', 'P5', 'P6'], required=True)
    ap.add_argument('--mode', choices=['smoke', 'formal'], required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--p3', type=Path)
    a = ap.parse_args()
    install_vendor()
    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    random.seed(2); np.random.seed(2); torch.manual_seed(2); torch.cuda.manual_seed_all(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    out = a.output
    out.mkdir(parents=True, exist_ok=True)
    evidence = {'run': a.run, 'mode': a.mode, 'model_type': 'qwen2'}
    with OptimizerAudit() as audit:
        if a.run == 'P3':
            train, val, data = calibration(tokenizer, a.mode)
            cfg = load(RESEARCH / 'configs/training_general_data.json')['stage1']
            tiny = load(RESEARCH / 'configs/runtime.json')['smoke']
            model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                    attn_implementation='eager', local_files_only=True).cuda()
            model.config.use_cache = False
            evidence['block_ap'] = reasoning_block_ap(model, train, val, cfg,
                tiny['block_epochs'] if a.mode == 'smoke' else cfg['epochs'], 1 if a.mode == 'smoke' else cfg['batch_size'])
            from quantization import probe
            model.config.use_cache = True
            logits, generated = probe(model, tokenizer)
            torch.save({'logits': logits, 'generated': generated}, out / 'native_probe.pt')
            model.save_pretrained(out / 'native', safe_serialization=True)
            tokenizer.save_pretrained(out / 'native')
            del model
            evidence['stage1_data'] = data
        else:
            if a.run == 'P5':
                from baseline_training import efficient_block_ap
                from baseline_ops import packed_weight
                model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                    attn_implementation='eager', local_files_only=True).cuda()
                model.config.use_cache = False
                train, val, data = calibration(tokenizer, a.mode)
                cfg = load(RESEARCH / 'configs/training_general_data.json')['stage1']
                tiny = load(RESEARCH / 'configs/runtime.json')['smoke']
                block_config = {'block_epochs': tiny['block_epochs'] if a.mode == 'smoke' else cfg['epochs'],
                    'block_batch_size': 1 if a.mode == 'smoke' else cfg['batch_size'],
                    'block_weight_lr': cfg['weight_lr'], 'block_quant_lr': cfg['quant_lr']}
                evidence['block_ap'] = efficient_block_ap(model, [x[0] for x in train], [x[0] for x in val],
                    torch.device('cuda'), block_config)
                evidence['stage1_data'] = data
            else:
                if a.p3 is None:
                    raise ValueError('P4/P6 require the same P3 native checkpoint')
                from quantize.int_linear_fake import load_quantized_model, cast_qat_dtype
                from quantize.utils import set_quant_state
                evidence['p3_signature'] = model_signature(a.p3)
                model, _ = load_quantized_model(str(a.p3), 2, 128, replace=True, strict=True,
                                               use_flash_attn=False, device='cpu')
                set_quant_state(model, True)
                model = cast_qat_dtype(model, torch.bfloat16).cuda()
            batches, data = stage2_data(tokenizer, a.mode)
            teacher = (AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                       attn_implementation='eager', local_files_only=True).cuda() if a.run == 'P4' else None)
            evidence['stage2'] = e2e(model, teacher, batches, a.run, a.mode,
                checkpoint=out / 'stage2_resume.pt', audit_steps=audit.steps, data_signature=data)
            evidence['stage2_data'] = data
            if teacher is not None:
                del teacher
            native = out / 'native'
            native.mkdir(exist_ok=True)
            from quantization import probe
            model.config.use_cache = True
            logits, generated = probe(model, tokenizer)
            torch.save({'logits': logits, 'generated': generated}, out / 'native_probe.pt')
            if a.run == 'P5':
                model.config.save_pretrained(native)
                tokenizer.save_pretrained(native)
                torch.save(model.cpu().state_dict(), native / 'efficientqat_state.pt')
            else:
                model.save_pretrained(native, safe_serialization=True)
                tokenizer.save_pretrained(native)
            del model
            gc.collect(); torch.cuda.empty_cache()
    if not audit.steps or not any(s['updated_values'] for s in audit.steps):
        raise RuntimeError('No observed finite optimizer updates')
    evidence['optimizer_audit'] = audit.steps
    save(out / 'training.json', evidence)


if __name__ == '__main__':
    main()
