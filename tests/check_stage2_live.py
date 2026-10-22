"""Real tokenizer regression, cache round-trip and interrupted GPU optimizer resume."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'adapters'))
from common import MODEL, install_vendor
install_vendor()
import torch
from completion_tokens import completion_window
from training import e2e, OptimizerAudit


class Fake(torch.nn.Module):
    def __init__(self, fail_second=False):
        super().__init__()
        self.layer = torch.nn.Module()
        self.layer.weight_quantizer = torch.nn.Module()
        self.layer.weight_quantizer.scale = torch.nn.Parameter(torch.linspace(.1, .8, 8))
        self.config = SimpleNamespace(use_cache=True)
        self.calls, self.fail_second = 0, fail_second
    def gradient_checkpointing_enable(self, **kwargs):
        pass
    def forward(self, input_ids, **kwargs):
        self.calls += 1
        if self.fail_second and self.calls == 2:
            raise RuntimeError('simulated task interruption after saved optimizer step')
        scale = self.layer.weight_quantizer.scale
        logits = scale[None, None, :].expand(input_ids.shape[0], input_ids.shape[1], -1)
        return SimpleNamespace(logits=logits)


class LiveTests(unittest.TestCase):
    def test_interrupted_e2e_resumes_without_repeating_updates(self):
        batches = [{'input_ids': torch.tensor([[0, 1, 2, 3]]),
                    'attention_mask': torch.ones((1, 4), dtype=torch.long),
                    'labels': torch.tensor([[-100, 1, 2, 3]])} for _ in range(2)]
        reference = Fake().cuda()
        with OptimizerAudit() as audit:
            expected = e2e(reference, None, batches, 'P6', 'smoke', audit_steps=audit.steps,
                           data_signature={'case': 'tiny'})
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / 'latest.pt'
            interrupted = Fake(fail_second=True).cuda()
            with OptimizerAudit() as audit:
                with self.assertRaisesRegex(RuntimeError, 'simulated task interruption'):
                    e2e(interrupted, None, batches, 'P6', 'smoke', checkpoint, audit.steps, {'case': 'tiny'})
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            self.assertEqual(state['next_sample'], 1)
            resumed = Fake().cuda()
            with OptimizerAudit() as audit:
                actual = e2e(resumed, None, batches, 'P6', 'smoke', checkpoint, audit.steps, {'case': 'tiny'})
                self.assertEqual(len(audit.steps), 2)
            self.assertEqual(resumed.calls, 1)
            torch.testing.assert_close(reference.layer.weight_quantizer.scale,
                                       resumed.layer.weight_quantizer.scale, rtol=0, atol=0)
            self.assertEqual(expected['steps'], actual['steps'])

    def test_cache_round_trip_and_hash_rejection(self):
        from stage2_cache import Writer, cached_batches
        batches = [{'input_ids': torch.tensor([[1, 2, 3]]),
                    'attention_mask': torch.ones((1, 3), dtype=torch.long),
                    'labels': torch.tensor([[-100, 2, 3]])} for _ in range(2)]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'cache'
            writer = Writer(path, 2, 256, 'smoke')
            for batch in batches:
                writer.append(batch)
            writer.finish({'samples': 2, 'causal_tokens': 4})
            loaded, evidence = cached_batches(path, 'smoke')
            for i in range(2):
                for key in batches[i]:
                    torch.testing.assert_close(loaded[i][key], batches[i][key], rtol=0, atol=0)
            with (path / 'ids.npy').open('r+b') as stream:
                stream.seek(-1, 2)
                stream.write(b'x')
            with self.assertRaisesRegex(RuntimeError, 'hash mismatch'):
                cached_batches(path, 'smoke')

    def test_original_formal_sample_138_retains_completion(self):
        from transformers import AutoTokenizer
        from repro_scripts.local_datasets import load_local_or_hf_openthoughts
        tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        ds = load_local_or_hf_openthoughts().shuffle(seed=2).select([138])
        row = ds[0]
        messages = [{'role': {'human': 'user', 'gpt': 'assistant'}[m['from']], 'content': m['value']}
                    for m in row['conversations']]
        prefix = tokenizer.encode(tokenizer.apply_chat_template(messages[:-1], tokenize=False,
                                  add_generation_prompt=True), add_special_tokens=False)
        full = tokenizer.encode(tokenizer.apply_chat_template(messages, tokenize=False,
                                add_generation_prompt=False), add_special_tokens=False)
        ids, labels, adjusted = completion_window(full, prefix, 2048)
        self.assertGreaterEqual(len(prefix), 2048)
        self.assertTrue(adjusted)
        self.assertLessEqual(len(ids), 2048)
        self.assertGreater(sum(x != -100 for x in labels[1:]), 0)
        print('REAL_SAMPLE_138', {'prompt_tokens': len(prefix), 'length': len(ids),
              'supervised_tokens': sum(x != -100 for x in labels[1:])}, flush=True)


if __name__ == '__main__':
    unittest.main()
