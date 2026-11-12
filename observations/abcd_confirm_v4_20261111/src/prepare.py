"""Static selection and preregistration only; no model score/outcome access."""
import re,subprocess
from common import *

def make_rules():
 rules=[]
 # Typed syntax templates with a bijective fresh-binder map, no global replacement.
 for i in range(P['B_pairs']):
  family=i//8;typ=['Nat','Int','Rat','Real'][(i%8)//2];k=[3,7][i%2];base=f'rule_{i:03d}'
  for variant,(a,b,h) in [('original',('a','b','h')),('renamed',('x','y','g'))]:
   relation=f'{a} < {b}' if family==5 else (f'{a} ≤ {b}' if family==4 else (f'{a} + {k} = {b}' if family in (0,3) else f'{a} = {b}'))
   goals=[f'{b} = {a} + {k}',f'{a} + {k} = {b} + {k}',f'({a} + {k}) * ({a} + {k}) = ({b} + {k}) * ({b} + {k})',f'{b} = {a} + {k} ∧ {a} + {k} = {b}',f'{a} + {k} ≤ {b} + {k}',f'{a} + {k} < {b} + {k}',f'({a} + {k}) ^ 2 = ({b} + {k}) ^ 2',f'{a} = {b} ∧ {b} = {a} ∧ {a} + {k} = {b} + {k}']
   proofs=[f'exact {h}.symm',f'simpa only [{h}]',f'simpa only [{h}]',f'exact ⟨{h}.symm, {h}⟩',f'exact add_le_add_right {h} {k}',f'exact add_lt_add_right {h} {k}',f'simpa only [{h}]',f'exact ⟨{h}, {h}.symm, by simpa only [{h}]⟩']
   code=f'theorem {base} ({a} {b} : {typ}) ({h} : {relation}) : {goals[family]} := by\n  {proofs[family]}\n'
   formal='import Mathlib\n'+code.rsplit(':=',1)[0]+':= sorry\n' if ':=' not in proofs[family] else None
   assert formal
   rules.append({'theorem_id':base+'_'+variant,'base_group':base,'split':'V-RULE-CONFIRM','family_id':f'family_{family}','variant':variant,'formal_statement':formal,'proof_code':code,'candidates':[proofs[family],'rfl','simp','aesop'],'candidate_origin':['CONSTRUCTED_VERIFIED_REFERENCE','STATIC','STATIC','STATIC'],'alpha_map':{'a':'x','b':'y','h':'g'},'D_selected':False})
 assert len(rules)==128
 originals=[re.sub(r'theorem \w+', 'theorem CANONICAL',r['proof_code']) for r in rules if r['variant']=='original']
 assert len(set(originals))==64,'Duplicate rule instances'
 return rules

def main():
 from transformers import AutoTokenizer
 token=AutoTokenizer.from_pretrained(ROOT/'models/REAL-Prover-v1-bf16',local_files_only=True)
 official=ROOT/'observation_20261104_all_layers/upstream/REAL-Prover/Realprover/data/minif2f_valid.jsonl'
 data=rows(official)
 for r in data:
  if isinstance(r['formal_statement'],list):r['formal_statement']=r['formal_statement'][-1]
 old={str(r['id']) for r in rows(PILOT/'manifests/V_real_theorems.jsonl')}
 remaining=sorted([r for r in data if str(r['id']) not in old],key=lambda r:digest('confirm-v4|seed2|'+r['formal_statement']))
 selected=remaining[:P['confirm_theorems']];assert len(selected)==128 and len(old)==32
 assert not {str(r['id']) for r in selected}&old
 for i,r in enumerate(selected):r.update(split='V-REAL-CONFIRM',theorem_id='confirm_'+str(r['id']),base_group='confirm_'+str(r['id']),D_selected=i<P['D_theorems'])
 jsonl(W/'manifests/selected_theorems.jsonl',selected)
 # Fit set from the exact-state-disjoint upstream training-distribution pool.
 pool=rows(PILOT/'manifests/S_obs_disc.jsonl');oldfit={r['state_id'] for r in rows(PILOT/'manifests/discovery_pilot.jsonl')};B=builder()
 candidates=[]
 for r in sorted(pool,key=lambda r:digest('confirm-fit|seed2|'+r['state_id'])):
  if r['state_id'] in oldfit:continue
  prompt=r['input'] if '<|im_start|>' in r['input'] else B.chat_template_to_prompt([{'role':'user','content':r['input']}],'qwen')
  ids=token.encode(prompt,add_special_tokens=False);action=token.encode(r['reference_or_candidate_action'],add_special_tokens=False)
  if len(ids)+len(action)<=P['fit_max_sequence_length']:
   candidates.append(dict(r,prompt=prompt,prompt_tokens=len(ids),target_tokens=len(action)))
  if len(candidates)==P['A_fit_groups']:break
 assert len(candidates)==128
 jsonl(W/'manifests/A_fit_inputs.jsonl',candidates)
 # Numeric controls are development data only. Do not choose based on pilot scores.
 development=sorted(rows(PILOT/'manifests/V_real_states.jsonl'),key=lambda r:digest('numeric|'+r['theorem_id']))
 jsonl(W/'manifests/numeric_development.jsonl',[r for r in development if r['split']=='V-REAL'][:P['numeric_development_cases']])
 models=read(PILOT/'manifests/models.json')
 path=ROOT/'runs/realprover_formal_four_gpu_v16/formal/P5/eval_model';idx=read(path/'model.safetensors.index.json')
 files=[path/n for n in sorted(set(idx['weight_map'].values()))]+[path/'config.json',path/'tokenizer.json',path/'tokenizer_config.json']
 inv=[{'path':str(f),'sha256':sha(f),'size':f.stat().st_size} for f in files]
 models['P5']={'path':str(path),'files':inv,'checkpoint_sha':digest(json.dumps(inv,sort_keys=True))}
 # Reuse previously SHA-validated identities, verifying current length/config/tokenizer.
 for m in models.values():
  for f in m['files']:
   p=Path(f['path']);assert p.stat().st_size==f['size']
   if p.suffix=='.json':assert sha(p)==f['sha256']
 save(W/'manifests/models.json',models)
 rules=make_rules()
 jsonl(W/'manifests/rules_planned.jsonl',rules)
 save(W/'manifests/preregistration.json',{'protocol_sha':protocol_sha(),'valid_source_sha':sha(official),'pilot_valid_ids_excluded':sorted(old),'confirm_ids':[r['id'] for r in selected],'A_fit_group_count':128,'rule_pairs':64,'rule_families':8,'same_checkpoint_asset_identities':True,'no_confirm_scores_read':True,'primary_layer':28,'D_theorems':32,'source_theorem_overlap':'SEMANTIC_OVERLAP_UNKNOWN'})
 status('prepare','DONE',confirm=128,A_fit=128,B_pairs=64,D_theorems=32)
if __name__=='__main__':main()
