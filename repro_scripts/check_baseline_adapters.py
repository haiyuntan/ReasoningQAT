"""CPU integration checks on a real tiny Qwen3, without claiming full smoke PASS."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import torch
from transformers import Qwen3Config,AutoModelForCausalLM,AutoTokenizer
from baseline_ops import efficient_ops,portable_packed_class,packed_weight,bitdistiller_ops
from baseline_training import efficient_block_ap,bitdistiller_clip,stage2_train,save_native,stage2_batches
from export_baseline import restore_native,materialize

ROOT=Path(__file__).resolve().parents[2]
torch.set_num_threads(2)
torch.manual_seed(2)
device=torch.device('cpu')
tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/Qwen3-1.7B',local_files_only=True)
cfg=Qwen3Config(vocab_size=151936,hidden_size=128,intermediate_size=256,num_hidden_layers=1,
               num_attention_heads=4,num_key_value_heads=2,head_dim=32,tie_word_embeddings=True)
cfg._attn_implementation='eager'
base=AutoModelForCausalLM.from_config(cfg,torch_dtype=torch.bfloat16,attn_implementation='eager').eval()
base.config.use_cache=False
tokens=torch.randint(5,100,(1,8))
labels=tokens.clone();labels[:,:3]=-100
batches=[{'input_ids':tokens,'attention_mask':torch.ones_like(tokens),'labels':labels}]
train_cfg={'block_epochs':1,'block_weight_lr':2e-5,'block_quant_lr':1e-4,'e2e_learning_rate':1e-3}
results={'status':'PASS','device':'CPU','meaning':'Tiny real Qwen3 adapter integration; full-model GPU smoke still required','runs':{}}
for run in ('R8','R9'):
    model=copy.deepcopy(base)
    if run=='R8':
        initialization=efficient_block_ap(model,[tokens],[tokens.flip(-1)],device,train_cfg)
        names=initialization[0]['weight_parameter_names']
        assert {'input_layernorm.weight','post_attention_layernorm.weight',
                'self_attn.q_norm.weight','self_attn.k_norm.weight'}<=set(names)
        assert len(names)==11
        teacher=None
        # Native upstream pack and CPU dequant parity, including scale gradients.
        Fake,Original,quantizer=efficient_ops()
        assert Fake.__init__.__globals__['UniformAffineQuantizer'] is quantizer.UniformAffineQuantizer
        linear=torch.nn.Linear(128,64,bias=False).half()
        fake=Fake(linear,2,128)
        with torch.no_grad(): linear.weight.copy_(fake.weight_quantizer(linear.weight))
        scales=fake.weight_quantizer.scale.detach().reshape(64,-1).t().contiguous().float()
        zeros=fake.weight_quantizer.zero_point.detach().reshape(64,-1).t().contiguous().float()
        packed=portable_packed_class()(2,128,128,64,False)
        packed.pack(linear,scales,zeros)
        codes=((linear.weight.t().float()/packed.scales.detach())+zeros).round()
        expected=(codes-zeros)*packed.scales
        assert torch.equal(packed_weight(packed),expected)
        x=torch.randn(2,128).half()
        output=packed(x)
        loss=output.float().square().mean();loss.backward()
        assert packed.scales.grad is not None and torch.isfinite(packed.scales.grad).all()
    else:
        teacher=copy.deepcopy(base).eval();teacher.requires_grad_(False)
        initialization=bitdistiller_clip(model,[tokens],device)
        linear,_,_,_=bitdistiller_ops()
        model,replaced=linear.convertModelToQuant(model,quant_type='int2-asym',q_group_size=128)
        assert replaced
    with tempfile.TemporaryDirectory(prefix='rq-stage1-',dir='/tmp') as folder:
        model.eval()
        with torch.no_grad(): stage1_before=model(input_ids=tokens).logits
        save_native(model,tokenizer,folder,run,{'mode':'cpu_integration','initialization':initialization})
        model,_=restore_native(folder)
        model.config.use_cache=False
        with torch.no_grad(): stage1_after=model(input_ids=tokens).logits
        assert torch.equal(stage1_before,stage1_after),f'{run} Stage 1 reload changed logits'
    stage2=stage2_train(model,teacher,batches,run,device,train_cfg)
    model.eval();model.config.use_cache=True
    with torch.no_grad(): before=model(input_ids=tokens).logits
    with tempfile.TemporaryDirectory(prefix='rq-baseline-',dir='/tmp') as folder:
        save_native(model,tokenizer,folder,run,{'mode':'cpu_integration','initialization':initialization,'stage2':stage2})
        restored,_=restore_native(folder)
        with torch.no_grad(): after=restored(input_ids=tokens).logits
        assert torch.equal(before,after),f'{run} native reload differs: {(before-after).abs().max().item()}'
        count=materialize(restored,run)
        with torch.no_grad(): exported=restored(input_ids=tokens).logits
        assert torch.equal(before,exported),f'{run} materialization differs'
    results['runs'][run]={'status':'PASS','native_reload_max_diff':float((before-after).abs().max()),
                          'materialization_max_diff':float((before-exported).abs().max()),
                          'quantized_linears':count,'train_steps':stage2['steps'],
                          'updated_parameter_probe_values':stage2['updated_parameter_probe_values'],
                          'stage1_strict_reload':True,
                          'confidence_beta':stage2['confidence_beta'],'teacher_used':stage2['teacher_used']}
    if teacher is not None: del teacher
    del model,restored
# Compare the production baseline collator tensors to R4's actual collator,
# extracted with its original format function, avoiding GPU-heavy main imports.
import ast
from dataclasses import dataclass
from trl.trainer.utils import DataCollatorForChatML
tree=ast.parse((ROOT/'ReasoningQAT/main_e2e_distill.py').read_text())
nodes=[node for node in tree.body if (isinstance(node,ast.FunctionDef) and node.name=='format_openthoughts_sample')
       or (isinstance(node,ast.ClassDef) and node.name=='CustomDataCollatorForChatML')]
namespace={'dataclass':dataclass,'DataCollatorForChatML':DataCollatorForChatML}
exec(compile(ast.Module(body=nodes,type_ignores=[]),'R4_original_collator','exec'),namespace)
os.environ['RQ_RUN_MODE']='smoke'
os.environ['RQ_SMOKE_OPENTHOUGHTS']=str(ROOT/'datasets/smoke/openthoughts.jsonl')
from repro_scripts.local_datasets import load_local_or_hf_openthoughts
ds=load_local_or_hf_openthoughts().shuffle(seed=2).select(range(2))
actual,budget=stage2_batches(tokenizer,'smoke')
collator=namespace['CustomDataCollatorForChatML'](tokenizer=tokenizer,max_length=256)
for i,sample in enumerate(ds):
    conversations=[]
    for entry in sample['conversations']:
        limit=32 if entry['from']=='human' else 96
        text=tokenizer.decode(tokenizer.encode(entry['value'],add_special_tokens=False)[:limit],skip_special_tokens=True)
        conversations.append({**entry,'value':text})
    reference=collator([{'conversations':conversations}])
    for key in ('input_ids','attention_mask','labels'):
        assert torch.equal(actual[i][key],reference[key]),f'R4 token mismatch: {key}'
results['R4_collator_parity']=budget
os.environ['RQ_RUN_MODE']='formal'
try: load_local_or_hf_openthoughts()
except ValueError: results['formal_rejects_smoke_fixture']=True
else: raise AssertionError('Formal accepted partial fixture')
(ROOT/'results/qat-baseline-adapters.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
