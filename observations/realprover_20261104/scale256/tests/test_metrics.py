import math
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

def test_write_boundary(tmp_path):
    from common import atomic, OBS
    # TMPDIR may intentionally be inside OBS; test an actual production path.
    with pytest.raises(ValueError):atomic(OBS.parent/'prepared-must-not-write.json',{})

def test_schema_aliases():
    pytest.importorskip('pyarrow')
    from audit_state_tactic_parquet import schema_columns
    assert schema_columns(['input','tactic'])==('input','tactic')
    assert schema_columns(['state','tactic'])==('state','tactic')
    with pytest.raises(ValueError):schema_columns(['state','input','tactic'])

def test_closed_form_and_teacher_wrong():
    torch=pytest.importorskip('torch')
    from token_metrics import distribution
    t=distribution(torch.tensor([[math.log(.99),math.log(.01)],[0.,0.]]),torch.tensor([1,0]))
    assert t['entropy'][1].item()==pytest.approx(math.log(2))
    assert t['gold_logp'][0].item()==pytest.approx(math.log(.01))
    assert t['top1'][0]!=1 # low entropy teacher can be wrong
    q=distribution(torch.tensor([[math.log(.9),math.log(.1)],[0.,0.]]),torch.tensor([1,0]),t)
    assert q['top1'][0]==t['top1'][0]
    assert q['fixed_tail'][0].item()==pytest.approx(.1)
    assert q['delta_nll'][0].item()==pytest.approx(math.log(.01/.1))
    assert torch.isfinite(q['rr']).all()

def test_shift_mask_and_layer_index():
    torch=pytest.importorskip('torch')
    from transformers import Qwen2Config,Qwen2ForCausalLM
    model=Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=14,num_attention_heads=2,num_key_value_heads=2))
    ids=torch.tensor([[1,2,3,4]]);labels=torch.tensor([[-100,-100,3,4]])
    outputs=[]
    handle=model.model.layers[13].register_forward_hook(lambda m,a,o:outputs.append(o[0].detach()))
    with torch.no_grad():out=model(ids,output_hidden_states=True)
    handle.remove()
    # Last hidden state is final norm; hook is actual 14th block pre-final-norm.
    assert torch.allclose(model.model.norm(outputs[0]),out.hidden_states[14])
    assert (labels[:,1:]!=-100).nonzero().tolist()==[[0,1],[0,2]]
    assert ids[:,1:][labels[:,1:]!=-100].tolist()==[3,4]
