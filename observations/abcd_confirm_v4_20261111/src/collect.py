"""Independent Lean collection; exact rendered-prefix controls, never model outcomes."""
import os,sys,subprocess,urllib.request,re,time
from common import *

def setup():
 os.environ.update(REALPROVER_LEAN_WORKSPACE=str(W/'runtime/lean'),REALPROVER_INTERACTIVE_ROOT='/tmp/realprover_lean_v10/interactive',REALPROVER_JIXIA_BIN='/tmp/realprover_lean_v10/jixia/.lake/build/bin/jixia',REALPROVER_LEAN_BIN='/tmp/realprover_lean_v10/toolchain/bin')
 os.environ['PATH']='/tmp/realprover_lean_v10/toolchain/bin:'+os.environ['PATH'];sys.path.insert(0,str(W/'runtime/source'))
 from manager.thirdparty.interactive import Interactive,InteractiveSessionError,InteractiveResponseTimeout
 from manager.struct.structs import state_repr
 return Interactive,state_repr

def request(path,payload=None):
 req=urllib.request.Request('http://127.0.0.1:8080/'+path,data=None if payload is None else json.dumps(payload).encode(),headers={'Content-Type':'application/json','X-Observation-ID':'ABCD-confirm-v4'})
 start=time.monotonic()
 with urllib.request.urlopen(req,timeout=20) as response:value=json.load(response)
 elapsed=time.monotonic()-start
 if elapsed>=P['GPU']['latency_hard_s']:raise RuntimeError('WAITING_RETRIEVAL_RESOURCE')
 return value

def root_and_actions(row,actions,Interactive,state_repr):
 path=W/'runtime/lean'/f"Case_{row['theorem_id']}.lean";path.write_text(row['formal_statement']);session=None
 def open_session():
  s=Interactive(W/'runtime/lean',Path('Header.lean'));s.response_timeout=30
  s.open_file(path,[None]);assert s.get_next_problem() is not None
  return s,state_repr(s.get_state(0))
 try:
  session,state=open_session();verdicts=[]
  for action in actions:
   try:
    session.response_timeout=10;sid=session.run_tactic(0,action,heartbeats=2000000);goals=session.get_state(sid)
    verdicts.append({'action':action,'status':'VALID_TRANSITION','goals_remaining':len(goals),'proof_verified':False})
   except RuntimeError as e:verdicts.append({'action':action,'status':'INVALID_TACTIC','error':str(e)[:700],'proof_verified':False})
   except Exception as e:
    verdicts.append({'action':action,'status':'UNKNOWN_TRANSPORT','error':repr(e),'proof_verified':False})
    session.close();session,repeated=open_session();assert repeated==state
  return state,verdicts
 finally:
  if session is not None:session.close()
  path.unlink(missing_ok=True)

def terms(text):return set(re.findall(r'[A-Za-z_][A-Za-z_0-9.]*',text)) - {'theorem','Nat','Int','Real','Rat','Type','Prop'}

def exact_token_span(prompt,unit,tokenizer):
 start=prompt.index(unit);end=start+len(unit);encoded=tokenizer(prompt,add_special_tokens=False,return_offsets_mapping=True)
 indexes=[i for i,(a,b) in enumerate(encoded['offset_mapping']) if a>=start and b<=end and b>a]
 assert indexes and indexes==list(range(indexes[0],indexes[-1]+1))
 return encoded['input_ids'],indexes[0],indexes[-1]+1

def main():
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/REAL-Prover-v1-bf16',local_files_only=True)
 I,S=setup();B=builder();h=request('health')
 assert h['adapter_loaded'] and h['asset_manifest_sha256']==sha(ROOT/'research/manifests/retrieval_assets.json')
 save(W/'manifests/retrieval_identity.json',h)
 # Batch compile independent namespaces; no sorry/admit in any validation proof.
 planned=rows(W/'manifests/rules_planned.jsonl');validation='import Mathlib\n'+''.join(f"namespace validate_{i}\n{r['proof_code']}\nend validate_{i}\n" for i,r in enumerate(planned))
 assert ':= sorry' not in validation and 'admit' not in validation
 path=W/'runtime/lean/ValidateRules.lean';path.write_text(validation)
 result=subprocess.run(['lake','env','lean',str(path)],cwd=W/'runtime/lean',capture_output=True,text=True,timeout=180)
 save(W/'manifests/rule_elaboration.json',{'status':'PASS' if result.returncode==0 else 'BLOCKED','exit_code':result.returncode,'source_sha':digest(validation),'instances':len(planned),'stdout':result.stdout[-4000:],'stderr':result.stderr[-2000:]})
 assert result.returncode==0,'Rule elaboration failed; inspect receipt'
 roots=rows(W/'manifests/selected_theorems.jsonl');cache=W/'caches/collected';cache.mkdir(exist_ok=True)
 allcases=[]
 for index,row in enumerate(roots+planned):
  cached=cache/(row['theorem_id']+'.json')
  if cached.exists():
   case=read(cached);assert case['formal_statement']==row['formal_statement'];allcases.append(case);continue
  actions=row.get('candidates',P['static_actions'])
  state,verdicts=root_and_actions(row,actions,I,S)
  returned=[]
  if row['split']=='V-REAL-CONFIRM':
   data=request('retrieve_premises',{'query':state,'num':10,'request_id':'confirm-v4-'+str(index)})['data'];returned=data[0] if data and isinstance(data[0],list) else data
   time.sleep(5) # Sparse queries; never a retrieval load test.
   actions=list(actions)+['apply '+r['Formal name'] for r in returned[:P['C_premises']]]
   _,applications=root_and_actions(row,actions[len(P['static_actions']):],I,S);verdicts+=applications
  history=row['formal_statement'].replace('sorry','').replace('by','').strip()+' by\n'
  case=dict(row,state=state,state_id=digest(state),history=history,returned_premises=returned,rendered_premises=returned[:6],candidates=actions,lean_verdicts=verdicts,
   candidate_origin=row.get('candidate_origin',['STATIC']*8+['FROZEN_PREMISE_APPLICATION']*min(3,len(returned))),proof_verified=row['split']=='V-RULE-CONFIRM',root_origin='ROOT_ONLY',retrieval_sha=digest(json.dumps(returned,sort_keys=True)),semantic_overlap='UNKNOWN',protocol_sha=protocol_sha())
  save(cached,case);allcases.append(case);status('collect_progress','RUNNING',cases=len(allcases),target=len(roots)+len(planned))
 # Control pool selected by text only. Semantic irrelevance remains unknown.
 pool={r['Formal name']:r for c in allcases for r in c['rendered_premises']}
 frozen=[]
 for case in allcases:
  render=case['rendered_premises'];prompts={'full':B.build_local_incontext_prompt_str(case['history'],case['state'],render,'qwen')}
  controls=[]
  if case['split']=='V-REAL-CONFIRM':
   prompts['order_reverse']=B.build_local_incontext_prompt_str(case['history'],case['state'],list(reversed(render)),'qwen')
   prompts['empty_list']=B.build_local_incontext_prompt_str(case['history'],case['state'],[],'qwen')
   for j,target in enumerate(render[:P['C_premises']]):
    minus=render[:j]+render[j+1:] # Do not refill from FAISS result7.
    prompts[f'remove_{j}']=B.build_local_incontext_prompt_str(case['history'],case['state'],minus,'qwen')
    target_terms=terms(case['state']);existing={r['Formal name'] for r in render}
    alternatives=[r for name,r in pool.items() if name not in existing]
    alternatives.sort(key=lambda r:(len(terms(r['Formal statement'])&target_terms),len(tokenizer.encode(r['Formal statement'])),digest(r['Formal name'])))
    assert alternatives;distractor=alternatives[0]
    replaced=render.copy();replaced[j]=distractor
    prompts[f'record_replace_{j}']=B.build_local_incontext_prompt_str(case['history'],case['state'],replaced,'qwen')
    unit=f"ID:{j}\nFormal name: {target['Formal name']}\nInformal name: {target['Informal name']}\nFormal statement: {target['Formal statement']}"
    ids,a,b=exact_token_span(prompts['full'],unit,tokenizer)
    fill=tokenizer.encode(distractor['Formal name']+' '+distractor['Formal statement'],add_special_tokens=False);assert fill
    pattern=(fill*((b-a+len(fill)-1)//len(fill)))[:b-a]
    prompts[f'same_length_{j}']=ids[:a]+pattern+ids[b:]
    assert len(prompts[f'same_length_{j}'])==len(ids)
    applied=case['lean_verdicts'][8+j]
    controls.append({'premise_index':j,'source_name':target['Formal name'],'valid_application':applied['status']=='VALID_TRANSITION','relevance_label':'LEAN_VALID_APPLICATION' if applied['status']=='VALID_TRANSITION' else 'NOT_VALIDATED',
     'semantic_necessity':'UNKNOWN','distractor_source':distractor,'distractor_semantic_irrelevance':'UNKNOWN; lexical control only','token_span':[a,b],'source_faiss_numeric_id':'UNKNOWN'})
  encoded={key:(value if isinstance(value,list) else tokenizer.encode(value,add_special_tokens=False)) for key,value in prompts.items()}
  lengths={key:len(value) for key,value in encoded.items()};max_action=max(len(tokenizer.encode(a,add_special_tokens=False)) for a in case['candidates'])
  case.update(prompts=prompts,input_ids=encoded,input_ids_sha={k:digest(__import__('array').array('q',v).tobytes()) for k,v in encoded.items()},prompt_token_lengths=lengths,C_controls=controls,status='FROZEN',eligible=max(lengths.values())+max_action<=P['max_sequence_length'],gold_actions='NONE; constructed rules excepted')
  frozen.append(case)
 assert len(frozen)==256 and all(c['eligible'] for c in frozen),'Static overlength cases: retain failures, do not replace by outcomes'
 for case in frozen:
  if case['split']=='V-RULE-CONFIRM' and case['variant']=='original':
   other=next(r for r in frozen if r['base_group']==case['base_group'] and r.get('variant')=='renamed')
   matched=case['prompt_token_lengths']['full']==other['prompt_token_lengths']['full'] and [len(tokenizer.encode(a,add_special_tokens=False)) for a in case['candidates']]==[len(tokenizer.encode(a,add_special_tokens=False)) for a in other['candidates']]
   case['B_length_matched']=other['B_length_matched']=matched
 jsonl(W/'manifests/frozen_cases.jsonl',frozen)
 identity={name:sha(W/'manifests'/name) for name in ['selected_theorems.jsonl','A_fit_inputs.jsonl','rules_planned.jsonl','frozen_cases.jsonl','models.json','numeric_development.jsonl']}
 identity['protocol']=protocol_sha();save(W/'manifests/frozen_identity.json',identity)
 save(W/'manifests/lean_runtime.json',{'version':subprocess.check_output(['lean','--version'],text=True).strip(),'workspace':str(W/'runtime/lean'),'lake_manifest_sha':sha(W/'runtime/lean/lake-manifest.json'),'interactive_sha':sha('/tmp/realprover_lean_v10/interactive/.lake/build/bin/interactive')})
 status('collect','DONE',cases=256,confirm=128,rules=128,B_length_matched_pairs=sum(r.get('B_length_matched',False) for r in frozen if r.get('variant')=='original'))
if __name__=='__main__':main()
