"""Small random Qwen2/operator checks; not a REAL-Prover smoke receipt."""
import ast
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'adapters'))
from common import install_vendor, VENDOR
install_vendor()
import torch
import torch.nn.functional as F
from losses import objective


class LossTests(unittest.TestCase):
    def test_matches_upstream_rr_and_topk_kl(self):
        source = ast.parse((VENDOR / 'reference_e2e_distill.py').read_text())
        rr = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'dft_cross_entropy')
        cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'PolicyGKDTrainer')
        kd = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'generalized_jsd_loss')
        kd.decorator_list = []
        from typing import Optional
        namespace = {'torch': torch, 'F': F, 'Optional': Optional}
        exec(compile(ast.Module(body=[rr, kd], type_ignores=[]), 'pinned_loss_reference', 'exec'), namespace)
        torch.manual_seed(2)
        student = torch.randn(2, 8, 32, requires_grad=True)
        teacher = torch.randn(2, 8, 32)
        labels = torch.randint(0, 32, (2, 8)); labels[:, :3] = -100
        actual, _ = objective(student, labels, teacher)
        s, t, y = student[:, :-1].contiguous(), teacher[:, :-1].contiguous(), labels[:, 1:].contiguous()
        reference = .2 * namespace['dft_cross_entropy'](s.reshape(-1, 32), y.reshape(-1), t.reshape(-1, 32))
        reference += namespace['generalized_jsd_loss'](s, t, labels=y, beta=1., top_k=20)
        torch.testing.assert_close(actual, reference)
        expected_grad = torch.autograd.grad(reference, student, retain_graph=True)[0]
        actual_grad = torch.autograd.grad(actual, student)[0]
        torch.testing.assert_close(actual_grad, expected_grad)

    def test_ce_mask_and_teacher_disabled(self):
        s = torch.randn(1, 6, 32, requires_grad=True)
        labels = torch.tensor([[-100, -100, -100, 3, 4, 5]])
        loss, metrics = objective(s, labels)
        reference = F.cross_entropy(s[:, :-1].reshape(-1, 32), labels[:, 1:].reshape(-1))
        torch.testing.assert_close(loss, reference)
        loss.backward()
        self.assertEqual(float(s.grad[:, :2].abs().sum()), 0.)
        self.assertIsNone(metrics['rr'])
        with self.assertRaises(ValueError):
            objective(s, torch.full_like(labels, -100))


class Qwen2Tests(unittest.TestCase):
    def test_bf16_cast_and_reload_preserve_quantizers_and_rope(self):
        from transformers import Qwen2Config, Qwen2ForCausalLM, AutoTokenizer
        from quantize.int_linear_fake import QuantLinear, cast_qat_dtype, load_quantized_model
        from quantize.utils import set_op_by_name, set_quant_state
        from common import MODEL
        config = Qwen2Config(vocab_size=64, hidden_size=128, intermediate_size=256,
                            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
        config._attn_implementation = 'eager'
        model = Qwen2ForCausalLM(config).eval()
        for name, module in list(model.named_modules()):
            if isinstance(module, torch.nn.Linear) and name.startswith('model.layers.'):
                set_op_by_name(model, name, QuantLinear(module, 2, 128))
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name.endswith('.weight_quantizer.scale'):
                    p.add_(1.234e-7)
                elif name.endswith('.weight_quantizer.zero_point'):
                    p.add_(0.1234567)
        original = {n: p.clone() for n, p in model.named_parameters() if '.weight_quantizer.' in n}
        frequencies = model.model.rotary_emb.inv_freq.clone()
        cast_qat_dtype(model, torch.bfloat16)
        for name, expected in original.items():
            self.assertEqual(model.get_parameter(name).dtype, torch.float32)
            torch.testing.assert_close(model.get_parameter(name), expected, atol=0, rtol=0)
        self.assertEqual(model.model.rotary_emb.inv_freq.dtype, torch.float32)
        torch.testing.assert_close(model.model.rotary_emb.inv_freq, frequencies, atol=0, rtol=0)
        set_quant_state(model, True)
        ids = torch.ones(1, 8, dtype=torch.long)
        with torch.no_grad():
            before = model(ids).logits
        self.assertTrue(torch.isfinite(before).all())
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            AutoTokenizer.from_pretrained(MODEL, local_files_only=True).save_pretrained(directory)
            restored, _ = load_quantized_model(directory, 2, 128, replace=True, strict=True,
                                              use_flash_attn=False, device='cpu')
            set_quant_state(restored, True)
            restored.eval()
            for name, expected in original.items():
                torch.testing.assert_close(restored.get_parameter(name), expected, atol=0, rtol=0)
            with torch.no_grad():
                torch.testing.assert_close(before, restored(ids).logits, atol=0, rtol=0)

    def test_qwen2_forward_scale_gradient_and_materialization(self):
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from quantize.int_linear_fake import QuantLinear
        from quantize.utils import set_op_by_name, set_quant_state
        from training import OptimizerAudit
        torch.manual_seed(2)
        config = Qwen2Config(vocab_size=64, hidden_size=128, intermediate_size=256,
                            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
        model = Qwen2ForCausalLM(config).eval()
        names = []
        for name, module in list(model.named_modules()):
            if isinstance(module, torch.nn.Linear) and name.startswith('model.layers.'):
                set_op_by_name(model, name, QuantLinear(module, 2, 128))
                names.append(name)
        self.assertEqual(len(names), 7)
        set_quant_state(model, True)
        model.requires_grad_(False)
        for name, p in model.named_parameters():
            if name.endswith('.weight_quantizer.scale'):
                p.requires_grad_(True)
        ids = torch.randint(0, 64, (1, 8))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        with OptimizerAudit() as audit:
            loss, _ = objective(model(ids).logits, ids)
            loss.backward(); optimizer.step()
        self.assertGreater(audit.steps[0]['updated_values'], 0)
        from quantize.int_linear_fake import load_quantized_model
        with tempfile.TemporaryDirectory() as d:
            # Reuse the already downloaded tokenizer files; no weights needed.
            from transformers import AutoTokenizer
            from common import MODEL
            tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
            model.save_pretrained(d)
            tokenizer.save_pretrained(d)
            restored, _ = load_quantized_model(d, 2, 128, replace=True, strict=True,
                                               use_flash_attn=False, device='cpu')
            set_quant_state(restored, True)
            restored.eval()
            for name, p in model.named_parameters():
                if name.endswith('.weight_quantizer.scale'):
                    self.assertEqual(restored.get_parameter(name).dtype, torch.float32)
                    torch.testing.assert_close(p, restored.get_parameter(name))
            with torch.no_grad():
                torch.testing.assert_close(model(ids).logits, restored(ids).logits)
        with torch.no_grad():
            before = model(ids).logits
            for name in names:
                module = model.get_submodule(name)
                linear = torch.nn.Linear(module.in_features, module.out_features, module.bias is not None)
                linear.weight.copy_(module.weight_quantizer(module.weight))
                if module.bias is not None:
                    linear.bias.copy_(module.bias)
                set_op_by_name(model, name, linear)
            torch.testing.assert_close(before, model(ids).logits)
            with tempfile.TemporaryDirectory() as d:
                model.save_pretrained(d)
                reloaded = Qwen2ForCausalLM.from_pretrained(d).eval()
                torch.testing.assert_close(before, reloaded(ids).logits)

    def test_independent_efficientqat_pack_roundtrip(self):
        from baseline_ops import efficient_ops, portable_packed_class, packed_weight
        Fake, _, _ = efficient_ops()
        original = torch.nn.Linear(128, 128).half()
        fake = Fake(original, 2, 128)
        fake.set_quant_state(True)
        packed = portable_packed_class()(2, 128, 128, 128, True)
        with torch.no_grad():
            original.weight.copy_(fake.weight_quantizer(fake.weight))
            scales = fake.weight_quantizer.scale.clamp(1e-4, 1e4).reshape(128, -1).t().contiguous()
            zeros = fake.weight_quantizer.zero_point.round().reshape(128, -1).t().contiguous()
            packed.pack(original, scales.float(), zeros.float())
            actual = packed_weight(packed).t().contiguous()
            torch.testing.assert_close(actual.half(), original.weight, atol=1e-3, rtol=1e-3)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
