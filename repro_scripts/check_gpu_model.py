"""Validate downloaded BF16 model loading and generation; not a benchmark score."""
import json
from pathlib import Path
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

root = Path(__file__).resolve().parents[2]
start = time.monotonic()
tokenizer = AutoTokenizer.from_pretrained(root / 'models/Qwen3-1.7B', local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    root / 'models/Qwen3-1.7B', torch_dtype=torch.bfloat16,
    attn_implementation='sdpa', local_files_only=True).cuda().eval()
prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': 'What is 2 + 2?'}],
                                      tokenize=False, add_generation_prompt=True, enable_thinking=True)
inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
torch.manual_seed(42)
with torch.no_grad():
    logits = model(**inputs).logits
    assert torch.isfinite(logits).all(), 'Non-finite BF16 logits'
    generated = model.generate(**inputs, max_new_tokens=16, do_sample=True, temperature=0.6,
                               top_k=20, top_p=1., pad_token_id=tokenizer.eos_token_id)
new = generated[0, inputs['input_ids'].shape[1]:]
assert len(new) > 0
result = {'status': 'PASS', 'meaning': 'GPU model loading and finite generation only; not R0 evaluation',
          'device': torch.cuda.get_device_name(), 'dtype': str(next(model.parameters()).dtype),
          'generated_tokens': len(new), 'text': tokenizer.decode(new, skip_special_tokens=True),
          'peak_memory_bytes': torch.cuda.max_memory_allocated(), 'wall_seconds': time.monotonic()-start}
(root / 'results/gpu-model-check.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
