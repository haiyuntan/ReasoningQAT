"""CPU-only tests of actual upstream loss functions (AST extraction avoids GPU imports)."""
import ast
import json
from pathlib import Path
from typing import Optional
import torch
import torch.nn.functional as F
from quantize.int_linear_fake import QuantLinear

REPO = Path(__file__).resolve().parents[1]
tree = ast.parse((REPO/'main_e2e_distill.py').read_text())
ce = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name=='dft_cross_entropy')
trainer = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name=='PolicyGKDTrainer')
kl = next(x for x in trainer.body if isinstance(x, ast.FunctionDef) and x.name=='generalized_jsd_loss')
kl.decorator_list = []
namespace = {'torch':torch, 'F':F, 'Optional':Optional}
exec(compile(ast.Module(body=[ce,kl],type_ignores=[]), str(REPO/'main_e2e_distill.py'), 'exec'), namespace)
torch.manual_seed(2)
s = torch.randn(1, 3, 8, requires_grad=True)
t = torch.randn(1, 3, 8, requires_grad=True)
y = torch.tensor([[2, 5, -100]])
actual = namespace['dft_cross_entropy'](s.reshape(-1,8),y.reshape(-1),t.reshape(-1,8))
manual = -(t.softmax(-1).detach()[0,:2].gather(1,y[0,:2,None]).squeeze()*s.log_softmax(-1)[0,:2].gather(1,y[0,:2,None]).squeeze()).mean()
assert torch.allclose(actual,manual,atol=1e-7)
g1 = torch.autograd.grad(actual,s,retain_graph=True)[0]
g2 = torch.autograd.grad(manual,s,retain_graph=True)[0]
assert torch.allclose(g1,g2,atol=1e-7)
for top_k in [None,3]:
    actual = namespace['generalized_jsd_loss'](s,t,y,beta=1.,top_k=top_k)
    lp,lt = s.log_softmax(-1),t.detach().log_softmax(-1)
    if top_k:
        indices=lt.topk(top_k,-1).indices
        lp,lt=lp.gather(-1,indices),lt.gather(-1,indices)
    expected=(lt.exp()*(lt-lp)).sum(-1)[y!=-100].mean()
    assert torch.allclose(actual,expected,atol=1e-7)
    gs,gt=torch.autograd.grad(actual,(s,t),allow_unused=True,retain_graph=True)
    assert gt is None and torch.isfinite(gs).all() and gs[:,2].eq(0).all()
layer=QuantLinear(torch.nn.Linear(128,8),wbits=2,group_size=128)
for name,p in layer.named_parameters():
    p.requires_grad_(name.endswith('weight_quantizer.scale'))
layer.set_quant_state(True)
before=layer.weight.detach().clone()
scale_before=layer.weight_quantizer.scale.detach().clone()
optim=torch.optim.AdamW([p for p in layer.parameters() if p.requires_grad],lr=1e-4)
layer(torch.randn(2,128)).square().mean().backward()
assert layer.weight.grad is None
assert layer.weight_quantizer.scale.grad.abs().sum()>0
optim.step()
assert torch.equal(before,layer.weight) and not torch.equal(scale_before,layer.weight_quantizer.scale)
result={'status':'PASS','device':'cpu','checks':['Teacher-RR value/gradient/detached teacher','beta=1 full and top-k truncated KL value/mask/gradient','scale-only update; frozen weight unchanged']}
print(json.dumps(result,indent=2))
(REPO.parent/'results/cpu-loss-check.json').write_text(json.dumps(result,indent=2))
