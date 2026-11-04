import sys
from pathlib import Path
import math
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

def test_tiny_tail_survives_float32_cancellation():
    torch=pytest.importorskip('torch')
    from token_metrics import distribution
    z=torch.tensor([[0.,-40.,-41.]])
    r=distribution(z,torch.tensor([0]))
    expected=math.log(math.exp(-40)+math.exp(-41))
    assert r['legacy_own_tail'].item()==0
    assert r['own_tail'].item()>0
    assert r['own_log_tail'].item()==pytest.approx(expected,abs=1e-12)
    assert r['entropy'].item()>0
    # Top-p correctly removes the true tiny tail; sampled tail is exactly zero.
    assert r['sample_fixed_tail'].item()==0

def test_log_ratio_needs_no_epsilon_even_at_underflow():
    torch=pytest.importorskip('torch')
    from token_metrics import stable_log_probs
    _,_,t=stable_log_probs(torch.tensor([[0.,-1000.]]))
    _,_,q=stable_log_probs(torch.tensor([[0.,-998.]]))
    assert torch.isfinite(t).all() and (q-t).item()==pytest.approx(2)

def test_stable_probability_normalization_and_fixed_tail():
    torch=pytest.importorskip('torch')
    from token_metrics import distribution,stable_log_probs
    torch.manual_seed(2)
    z=torch.randn(8,37,dtype=torch.float64)*5
    lp,idx,tail=stable_log_probs(z)
    assert torch.allclose(lp.exp().sum(-1),torch.ones(8,dtype=torch.float64),atol=1e-14)
    mask=torch.ones_like(z,dtype=torch.bool).scatter(-1,idx[:,None],False)
    assert torch.allclose(tail.exp(),torch.softmax(z,-1)[mask].reshape(8,-1).sum(-1),atol=1e-14)
    teacher=distribution(torch.tensor([[0.,-40.,-41.]]),torch.tensor([0]))
    student=distribution(torch.tensor([[-40.,0.,-41.]]),torch.tensor([0]),teacher)
    assert student['fixed_tail'].item()>0.99

def test_shift_invariance():
    torch=pytest.importorskip('torch')
    from token_metrics import distribution
    z=torch.tensor([[2.,-20.,-21.]],dtype=torch.float64)
    a=distribution(z,torch.tensor([0]));b=distribution(z+100,torch.tensor([0]))
    for key in ['entropy','own_log_tail','gold_logp']:
        assert torch.allclose(a[key],b[key],atol=1e-13)
