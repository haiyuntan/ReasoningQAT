"""Strict all-28-block replay, prefix-only aggregation, explicit degenerate cases."""
import argparse,fcntl,math
from common import *
from engine import Engine

def geometry_from_norms(pn,qn,an,dot):
 normsum=pn+qn;energy=pn*pn+qn*qn
 cancellation=1-an/normsum if normsum else None
 orthogonal=1-math.sqrt(energy)/normsum if normsum else None
 return {'cosine':dot/(pn*qn) if pn and qn else None,
  'cancellation':cancellation,'orthogonal_baseline_cancellation':orthogonal,
  'excess_over_orthogonal':cancellation-orthogonal if normsum else None,
  'energy_relative_to_orthogonal':an*an/energy if energy else None}

def main():
 import torch
 ap=argparse.ArgumentParser();ap.add_argument('--method',required=True);args=ap.parse_args();method=args.method
 lock=(W/'tmp/gpu3.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
 engine=Engine(read(W/'manifests/models.json')['P0']);basis=torch.load(W/'caches/A_basis.pt',weights_only=True)
 cases=[c for c in rows(W/'manifests/frozen_cases.jsonl') if c['D_selected']];assert len(cases)==32
 def cuda(value):
  if torch.is_tensor(value):return value.to('cuda')
  if isinstance(value,dict):return {k:cuda(v) for k,v in value.items()}
  if isinstance(value,tuple):return tuple(cuda(v) for v in value)
  if isinstance(value,list):return [cuda(v) for v in value]
  return value
 def geometry(prop,inj,actual):
  pn=float(prop.norm());qn=float(inj.norm());an=float(actual.norm())
  return {'propagation_squared':float(prop.square().sum()),'injection_squared':float(inj.square().sum()),'actual_squared':float(actual.square().sum()),
   **geometry_from_norms(pn,qn,an,float((prop*inj).sum()))}
 for index,case in enumerate(cases):
  identifier=case['theorem_id'];output=W/'metrics/D'/method/(identifier+'.json')
  if output.exists():
   old=read(output);assert old['protocol_sha']==protocol_sha() and len(old['layers'])==28;continue
  tp=W/'caches/D/P0'/(identifier+'.pt');qp=W/'caches/D'/method/(identifier+'.pt')
  assert tp.exists() and qp.exists(),'MISSING_REPLAY_CACHE; never silently skip a theorem'
  T=torch.load(tp,map_location='cpu',weights_only=True);Q=torch.load(qp,map_location='cpu',weights_only=True)
  assert T['input_ids_sha']==Q['input_ids_sha'] and T['position']==Q['position']
  assert set(T['boundaries'])==set(Q['boundaries'])==set(range(29))
  records=[];position=T['position']
  for li in range(28):
   start=time.monotonic()
   with torch.inference_mode():
    layer=engine.model.model.layers[li];kw=cuda(T['kwargs'][li])
    t=layer(T['boundaries'][li].to('cuda'),**kw);t=t[0] if isinstance(t,tuple) else t
    expected=T['boundaries'][li+1];maxabs=float((t.cpu().float()-expected.float()).abs().max())
    assert maxabs<=1e-5,'REPLAY_PARITY_BLOCKED'
    tq=layer(Q['boundaries'][li].to('cuda'),**kw);tq=tq[0] if isinstance(tq,tuple) else tq
    teacher=expected[0,:position+1].double();student=Q['boundaries'][li+1][0,:position+1].double();middle=tq.cpu()[0,:position+1].double()
    prop=middle-teacher;inj=student-middle;actual=student-teacher;residual=actual-prop-inj
    error=float(residual.abs().max());assert error<1e-10,'ALGEBRA_RESIDUAL_BLOCKED'
    norms=geometry(prop,inj,actual);last=geometry(prop[-1],inj[-1],actual[-1]);projected=None
    if li+1 in basis:
     U=basis[li+1]['functional'].double();projected={name:float((v[-1]@U).square().sum()) for name,v in [('propagation',prop),('injection',inj),('actual',actual)]}
    records.append({'layer':li+1,'replay_max_abs':maxabs,'algebra_max_abs':error,'prefix':norms,'last_prompt':last,
     'prefix_nmse':norms['actual_squared']/(float(teacher.square().sum())+1e-30),'last_prompt_nmse':last['actual_squared']/(float(teacher[-1].square().sum())+1e-30),
     'functional_projection':projected,'functional_basis_available':li+1 in basis})
   engine.pace(start)
  save(output,{'theorem_id':identifier,'method':method,'protocol_sha':protocol_sha(),'status':'EXACT_REPLAY_PASS','input_ids_sha':T['input_ids_sha'],'layers':records,'teacher_cache_sha':sha(tp),'student_cache_sha':sha(qp),'scope':'prefix through last prompt token; no future action aggregation'})
  # Only this new workspace's explicitly temporary student cache is reclaimed.
  qp.unlink();status('replay_'+method+'_progress','RUNNING',completed=index+1,total=32)
 status('replay_'+method,'DONE',theorems=32,layers=28,temporary_student_cache_pruned=True)
if __name__=='__main__':main()
