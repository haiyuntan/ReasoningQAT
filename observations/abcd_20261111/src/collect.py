"""Independent ROOT_ONLY Lean observations and frozen low-rate retrieval snapshots."""
import importlib.util,json,os,sys,time,urllib.request,subprocess
from common import *

def setup():
 os.environ['REALPROVER_LEAN_WORKSPACE']=str(W/'runtime/lean')
 os.environ['REALPROVER_INTERACTIVE_ROOT']='/tmp/realprover_lean_v10/interactive'
 os.environ['REALPROVER_JIXIA_BIN']='/tmp/realprover_lean_v10/jixia/.lake/build/bin/jixia'
 os.environ['REALPROVER_LEAN_BIN']='/tmp/realprover_lean_v10/toolchain/bin'
 os.environ['PATH']='/tmp/realprover_lean_v10/toolchain/bin:'+os.environ['PATH']
 sys.path.insert(0,str(W/'runtime/source'))
 from manager.thirdparty.interactive import Interactive
 from manager.struct.structs import state_repr
 spec=importlib.util.spec_from_file_location('frozen_prompt',W/'runtime/source/manager/manage/prompt_manage.py')
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 return Interactive,state_repr,module.PromptManage

def health():
 with urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=5) as r:return json.load(r)

def query(state,index):
 payload=json.dumps({'query':state,'num':10,'request_id':f'ABCD-v3-{index}'}).encode()
 req=urllib.request.Request('http://127.0.0.1:8080/retrieve_premises',data=payload,headers={'Content-Type':'application/json','X-Observation-ID':f'ABCD-v3-{index}'})
 start=time.monotonic()
 with urllib.request.urlopen(req,timeout=20) as response:out=json.load(response)
 elapsed=time.monotonic()-start
 if elapsed>=20:raise RuntimeError('WAITING_RETRIEVAL_RESOURCE')
 time.sleep(5)
 # Official endpoint returns list of lists for batch-shaped FAISS results.
 data=out['data'];return data[0] if data and isinstance(data[0],list) else data

def root_state(row,Interactive,state_repr):
 path=W/'runtime/lean'/f"Observation_{digest(row['theorem_id'])[:16]}.lean"
 path.write_text(row['formal_statement'])
 session=Interactive(W/'runtime/lean',Path('Header.lean'));session.response_timeout=60
 try:
  session.open_file(path,[None]);decl=session.get_next_problem()
  if decl is None:raise RuntimeError('No proof obligation')
  state=state_repr(session.get_state(0));verdicts=[]
  for action in row['candidates']:
   try:
    sid=session.run_tactic(0,action);goals=session.get_state(sid)
    verdicts.append({'action':action,'lean_status':'VALID_TRANSITION','goals_remaining':len(goals),'proof_verified':False,'note':'tactic transition only, not full-kernel proof verification'})
   except RuntimeError as e:verdicts.append({'action':action,'lean_status':'INVALID_TACTIC','proof_verified':False,'error':str(e)[:500]})
  return state,verdicts
 finally:session.close();path.unlink(missing_ok=True)

def main():
 Interactive,state_repr,builder=setup()
 h=health();assert h['status']=='ready' and h['adapter_loaded']
 save(W/'manifests/retriever_snapshot.json',{'health':h,'assets_manifest_sha':sha(ROOT/'research/manifests/retrieval_assets.json'),'runtime_sha':sha(ROOT/'REALPROVER/configs/retrieval_runtime.json'),'production_pid':json.loads((ROOT/'runs/realprover_retrieval/process.json').read_text())['pid']})
 cases=[];receipts=[];rules=[]
 # Capture-avoiding alpha-renaming constructed over an explicit typed syntax
 # template (fresh names, bijective binder map, synchronized action references).
 for i in range(P['rule_pairs']):
  typ='Nat' if i%2==0 else 'Int';n=f'abcd_rule_{i}'
  names=(f'a{i}',f'b{i}',f'h{i}');renamed=(f'x{i}',f'y{i}',f'heq{i}')
  for variant,(a,b,hh) in [('original',names),('renamed',renamed)]:
   family=i//2
   goal=[f'{b} = {a}',f'{a} + 1 = {b} + 1',f'{a} * {a} = {b} * {b}',f'{b} = {a} ∧ {a} = {b}'][family]
   action=[f'exact {hh}.symm',f'simpa only [{hh}]',f'simpa only [{hh}]',f'exact ⟨{hh}.symm, {hh}⟩'][family]
   statement=f'import Mathlib\ntheorem {n}_{variant} ({a} {b} : {typ}) ({hh} : {a} = {b}) : {goal} := by\n  {action}\n'
   file=W/'runtime/lean'/f'{n}_{variant}.lean';file.write_text(statement)
   result=subprocess.run(['lake','env','lean',str(file)],cwd=W/'runtime/lean',capture_output=True,text=True,timeout=60)
   receipt={'rule_id':n,'variant':variant,'source_sha':digest(statement),'exit_code':result.returncode,'proof_verified':result.returncode==0,'stderr':result.stderr[-1000:]}
   receipts.append(receipt)
   if result.returncode!=0:raise RuntimeError('RULE_ELABORATION_BLOCKED')
   rule={'theorem_id':f'{n}_{variant}','base_group':n,'split':'V-RULE','rule_id':'R1_ALPHA','variant':variant,'formal_statement':statement.rsplit(':=',1)[0]+':= sorry\n','candidates':[action,'rfl','simp','aesop'],'alpha_map':dict(zip(names,renamed)),'proof_verified':True}
   rules.append(rule)
 write_rows(W/'manifests/V_rule_pairs.jsonl',rules)
 write_rows(W/'manifests/lean_runtime_receipts.jsonl',receipts)
 roots=[dict(row,theorem_id='valid_'+str(row['id']),base_group='valid_'+str(row['id']),candidates=P['candidates'],rule_id=None,proof_verified=False) for row in read_rows(W/'manifests/V_real_theorems.jsonl')]
 for i,row in enumerate(roots+rules):
  try:
   state,verdicts=root_state(row,Interactive,state_repr)
   premises=query(state,i) if row['split']=='V-REAL' else []
   history=row['formal_statement'].replace('sorry','').replace('by','').strip()+' by\n'
   prompt=builder.build_local_incontext_prompt_str(history,state,premises,'qwen')
   if premises:
    # Same token-count distractor is created after tokenizer loading in score.py.
    variations={'full':premises,'remove_first':premises[1:],'order_reverse':list(reversed(premises))}
   else:variations={'full':premises}
   prompts={key:builder.build_local_incontext_prompt_str(history,state,value,'qwen') for key,value in variations.items()}
   case=dict(row,state=state,state_id=digest(state),history=history,premises=premises,prompts=prompts,
    input_sha=digest(prompt),retrieval_id=f'ABCD-v3-{i}',retrieval_sha=digest(json.dumps(premises,sort_keys=True)),
    candidate_origin='STATIC_PREREGISTERED_TEMPLATE',lean_verdicts=verdicts,status='FROZEN',state_origin='ROOT_ONLY',
    prompt_builder_sha=sha(W/'runtime/source/manager/manage/prompt_manage.py'))
   cases.append(case);record('collect_'+str(i),'FROZEN',theorem=row['theorem_id'])
  except Exception as e:receipts.append({'theorem_id':row['theorem_id'],'status':'BLOCKED','error':repr(e)})
  write_rows(W/'manifests/V_real_states.jsonl',cases)
  write_rows(W/'manifests/lean_runtime_receipts.jsonl',receipts)
 assert sum(c['split']=='V-REAL' for c in cases)>=8, 'Not enough independent valid roots'
 save(W/'manifests/lean_toolchain.json',{'lean_version':subprocess.check_output(['lean','--version'],text=True).strip(),
  'toolchain_sha':sha(W/'runtime/lean/lean-toolchain'),'lake_manifest_sha':sha(W/'runtime/lean/lake-manifest.json'),
  'interactive_sha':sha('/tmp/realprover_lean_v10/interactive/.lake/build/bin/interactive'),
  'write_workspace':str(W/'runtime/lean'),'packages':'read-only shared dependency assets; independent generated files'})
 (W/'reports/04_lean_retrieval_toolchain_and_snapshot.md').write_text('# Independent evaluation-only runtime\nRoot states and all static candidate transitions are recorded. V-RULE original and renamed theorems were compiled without sorry. Valid transitions are not mislabeled as full proof verification. Retrieval snapshots preserve ordered full records; low-rate queries only, production service never restarted.\n')
 frozen={str(p.relative_to(W)):sha(p) for p in [W/'configs/protocol.json',W/'manifests/V_real_states.jsonl',W/'manifests/discovery_pilot.jsonl',W/'manifests/models.json']}
 save(W/'manifests/frozen_inputs.json',frozen)
 (W/'manifests/frozen_protocol.sha256').write_text(digest(json.dumps(frozen,sort_keys=True))+'\n')
 record('G3','PILOT_READY',cases=len(cases),valid=sum(c['split']=='V-REAL' for c in cases))
if __name__=='__main__':main()
