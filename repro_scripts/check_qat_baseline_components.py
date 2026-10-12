"""Test original BitDistiller CAKLD and INT2 operators without training or GPU use."""
import ast
import importlib.util
import json
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]
upstream=ROOT/'qat_baselines/BitDistiller'
tree=ast.parse((upstream/'train/mytrainer.py').read_text())
trainer=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='KDTrainer')
method=next(node for node in trainer.body if isinstance(node,ast.FunctionDef) and node.name=='cakld_loss')
namespace={'F':F}
exec(compile(ast.Module(body=[method],type_ignores=[]),'upstream_cakld','exec'),namespace)
cakld=namespace['cakld_loss']
torch.manual_seed(2)
student=torch.randn(2,4,7,dtype=torch.float64,requires_grad=True)
teacher=torch.randn(2,4,7,dtype=torch.float64,requires_grad=True)
labels=torch.tensor([[1,2,-100,-100],[3,4,5,-100]])
mask=(labels!=-100)
for beta in (0.,.37,1.):
    observed=cakld(None,labels,student,teacher.detach(),beta)
    log_s=F.log_softmax(student,dim=-1)
    log_t=F.log_softmax(teacher.detach(),dim=-1)
    reverse=(log_s.exp()*(log_s-log_t)).sum(-1)
    forward=(log_t.exp()*(log_t-log_s)).sum(-1)
    expected=((beta*reverse+(1-beta)*forward)*mask).sum(-1).mean()
    assert torch.allclose(observed,expected,atol=1e-12,rtol=1e-12)
    grad=torch.autograd.grad(observed,student,retain_graph=True)[0]
    reference_grad=torch.autograd.grad(expected,student,retain_graph=True)[0]
    assert torch.allclose(grad,reference_grad,atol=1e-12,rtol=1e-12)
    assert torch.isfinite(grad).all() and torch.count_nonzero(grad[~mask])==0
assert teacher.grad is None
spec=importlib.util.spec_from_file_location('bitdistiller_quantizer',upstream/'quantization/quantizer.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
weights=torch.randn(4,256,requires_grad=True)
quantized,scales,zeros=module.pseudo_quantize_tensor(weights,n_bit=2,zero_point=True,
                                                    q_group_size=128,get_scale_zp=True)
codes=(quantized.reshape(-1,128)/scales.reshape(-1,1)+zeros.reshape(-1,1)).round()
assert codes.min()>=0 and codes.max()<=3
assert all(torch.unique(row).numel()<=4 for row in quantized.reshape(-1,128))
ste=module.SteInt2AsymQuantizer(q_group_size=128)
loss=ste(weights).square().mean()
loss.backward()
assert weights.grad is not None and torch.isfinite(weights.grad).all() and weights.grad.abs().sum()>0
result={'status':'PASS','device':'CPU','meaning':'Original operator tests only; not a Qwen3 adapter or benchmark result',
        'checks':['CAKLD beta=0/0.37/1 value and student-gradient parity',
                  'Both KL directions match original sequence-sum/example-mean normalization',
                  'Padding mask has zero gradient; teacher logits detached',
                  'Asymmetric INT2 group128 has <=4 levels and codes 0..3',
                  'Original INT2 STE produces finite nonzero weight gradients'],
        'R8_adapter_status':'BLOCKED_PENDING_IMPLEMENTATION','R9_adapter_status':'BLOCKED_PENDING_IMPLEMENTATION'}
(ROOT/'results/qat-baseline-components.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
