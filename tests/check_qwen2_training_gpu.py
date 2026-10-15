"""Synthetic one-block GPU integration; does not certify REAL-Prover smoke."""
from pathlib import Path
import copy
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'adapters'))
from common import install_vendor, load, RESEARCH, save
install_vendor()
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from training import reasoning_block_ap, OptimizerAudit, e2e
from baseline_training import efficient_block_ap


def main():
    torch.set_num_threads(2)
    torch.manual_seed(2)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; integration cannot pass')
    config = Qwen2Config(vocab_size=64, hidden_size=128, intermediate_size=256,
                        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                        use_cache=False)
    model = Qwen2ForCausalLM(config).half().cuda().eval()
    train = [(torch.randint(0, 64, (1, 16)), None) for _ in range(2)]
    val = [(torch.randint(0, 64, (1, 16)), None)]
    cfg = load(RESEARCH / 'configs/training_general_data.json')['stage1']
    with OptimizerAudit() as audit:
        history = reasoning_block_ap(model, train, val, cfg, 1, 1)
    assert len(history) == 1 and len(audit.steps) == 2
    p3 = copy.deepcopy(model)
    batches = [{'input_ids': x[0], 'attention_mask': torch.ones_like(x[0]),
                'labels': torch.cat([torch.full((1, 4), -100), x[0][:, 4:]], dim=1)} for x in train]
    for run in ('P4', 'P6'):
        student = copy.deepcopy(p3).bfloat16()
        teacher = Qwen2ForCausalLM(config).bfloat16().cuda().eval() if run == 'P4' else None
        with OptimizerAudit() as audit:
            report = e2e(student, teacher, batches, run, 'smoke')
        assert len(audit.steps) == 2 and report['optimizer_steps'] == 2
        assert any(s['updated_values'] > 0 for s in audit.steps)
        print('PASS', run, report['objective'], flush=True)
        del student, teacher
    independent = Qwen2ForCausalLM(config).bfloat16().cuda().eval()
    with OptimizerAudit() as audit:
        report = efficient_block_ap(independent, [x[0] for x in train], [x[0] for x in val],
                                   torch.device('cuda'), {'block_epochs': 1, 'block_batch_size': 1,
                                   'block_weight_lr': cfg['weight_lr'], 'block_quant_lr': cfg['quant_lr']})
        qp = e2e(independent, None, batches, 'P5', 'smoke')
    assert len(report) == 1 and qp['optimizer_steps'] == 2
    assert any(s['updated_values'] > 0 for s in audit.steps)
    print('PASS P3 reconstruction and P5 independent Block-AP/E2E-QP', flush=True)


if __name__ == '__main__':
    main()
