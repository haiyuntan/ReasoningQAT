"""CPU-only source, split, length, and immutable checkpoint audit."""
import json,re,subprocess,sys
from pathlib import Path
from common import *

def statement(row):
 s=row['formal_statement'];return s[-1] if isinstance(s,list) else s

def main():
 import numpy as np
 import pyarrow.parquet as pq
 from tokenizers import Tokenizer
 source=ROOT/'observation_20261104_all_layers/upstream/REAL-Prover'
 valid=source/'Realprover/data/minif2f_valid.jsonl'
 test=ROOT/'REALPROVER/data/minif2f_test.jsonl'
 vv=[dict(row,formal_statement=statement(row)) for row in read_rows(valid)]
 tt=read_rows(test)
 assert len(vv)==len(tt)==244
 assert len({r['formal_statement'] for r in vv})==244
 assert not ({re.sub(r'\s+',' ',r['formal_statement']).strip() for r in vv}&{re.sub(r'\s+',' ',r['formal_statement']).strip() for r in tt})
 selected=sorted(vv,key=lambda r:digest('seed2|'+r['formal_statement']))[:P['valid_theorems']]
 for r in selected:r.update(split='V-REAL',selection='STATIC_HASH_BEFORE_MODEL_SCORES',status='SELECTED',semantic_overlap='SEMANTIC_OVERLAP_UNKNOWN')
 write_rows(W/'manifests/V_real_theorems.jsonl',selected)
 save(W/'manifests/T_final_readonly_index.json',{'path':str(test),'sha256':sha(test),'rows':len(tt),'access':'theorem text for overlap screen only; no results read'})
 models={}
 for method in P['models']:
  path=ROOT/'models/REAL-Prover-v1-bf16' if method=='P0' else ROOT/'runs/realprover_formal_four_gpu_v16/formal'/method/'eval_model'
  index=json.loads((path/'model.safetensors.index.json').read_text())
  files=[path/n for n in sorted(set(index['weight_map'].values()))]+[path/'config.json',path/'tokenizer.json',path/'tokenizer_config.json']
  inventory=[{'path':str(p),'sha256':sha(p),'size':p.stat().st_size} for p in files]
  models[method]={'path':str(path),'files':inventory,'checkpoint_sha':digest(json.dumps(inventory,sort_keys=True))}
  record('model_'+method,'HASHED',shards=len(index['weight_map']))
 save(W/'manifests/models.json',models)
 data=ROOT/'observation_20261104_all_layers/assets/state_tactic_pairs/data/train-00000-of-00001.parquet'
 frame=pq.read_table(data).to_pandas();assert len(frame)==56407
 src='input' if 'input' in frame else 'state'
 tokenizer=Tokenizer.from_file(str(ROOT/'models/REAL-Prover-v1-bf16/tokenizer.json'))
 tokenizer.no_padding();tokenizer.no_truncation()
 names=set(re.findall(r'\b(?:theorem|lemma)\s+([^\s(:]+)','\n'.join(r['formal_statement'] for r in vv+tt)))
 screen=re.compile(r'(?<![\w.])(?:'+'|'.join(re.escape(n) for n in sorted(names))+r')(?![\w.])')
 lengths=[];seen={};excluded=0;template={};fit=[];cal=[];window_state=0;eligible=0
 for index,row in frame.iterrows():
  text=str(row[src]);action=str(row['tactic'])
  if 'Here is the current STATE:' in text:
   state=text.rsplit('Here is the current STATE:',1)[1].split('**Please generate a TACTIC',1)[0].strip()
  elif 'STATE:' in text:
   state=text.rsplit('STATE:',1)[1].split('TACTIC:',1)[0].strip()
  else:state=text
  group=digest(state);tokens=tokenizer.encode(text,add_special_tokens=False).ids;lengths.append(len(tokens))
  kind='native_incontext' if 'Here is the current STATE:' in text else ('chatml' if '<|im_start|>' in text else 'other')
  template[kind]=template.get(kind,0)+1
  if len(tokens)>=2048:
   eligible+=1
   # Offsets check whether the final current-state marker survives a suffix window.
   marker=max(text.rfind('STATE:'),text.rfind('⊢'))
   start=tokenizer.encode(text,add_special_tokens=False).offsets[-2048][0]
   window_state+=marker>=start
  if screen.search(text):excluded+=1;continue
  if group in seen:continue
  seen[group]=index
  record_data={'row_id':int(index),'source_group':group,'group_basis':'exact extracted state; source theorem UNKNOWN',
   'state_id':group,'state':state,'input':text,'input_sha':digest(text),'reference_or_candidate_action':action,
   'candidate_origin':'official expert iteration reference','distribution':'BF16_TRAIN_DISTRIBUTION',
   'semantic_overlap':'SEMANTIC_OVERLAP_UNKNOWN','tokens':len(tokens),'template':kind,'lean_status':'UNKNOWN',
   'proof_verified':None,'rule_id':None,'retrieval_id':None,'retrieval_sha':None}
  if int(group[:8],16)%5==0:
   record_data['split']='S-OBS-DISC';fit.append(record_data)
  else:record_data['split']='S-CAL-TRAIN';cal.append(record_data)
 # Never claim exact-state separation is proven source-theorem separation.
 fit=sorted(fit,key=lambda r:digest('seed2|'+r['state_id']))
 cal=sorted(cal,key=lambda r:digest('seed2|'+r['state_id']))
 write_rows(W/'manifests/S_obs_disc.jsonl',fit)
 write_rows(W/'manifests/S_cal_train.jsonl',cal)
 eligible_fit=[r for r in fit if r['tokens']+len(tokenizer.encode(r['reference_or_candidate_action']).ids)+64<=P['max_sequence_length']]
 assert len(eligible_fit)>=P['discovery_groups']
 write_rows(W/'manifests/discovery_pilot.jsonl',eligible_fit[:P['discovery_groups']])
 save(W/'metrics/data_audit.json',{'rows':len(frame),'schema':str(pq.read_schema(data)),'data_sha256':sha(data),'source_revision':json.loads((ROOT/'observation_20261102/manifests/acquisition.json').read_text()).get('dataset',{}).get('result',{}).get('revision'),
  'length_quantiles':{str(q):float(np.quantile(lengths,q)) for q in [0,.25,.5,.75,.9,.95,.99,1]},
  'templates':template,'long_2048':eligible,'short_2048':len(frame)-eligible,'suffix_marker_retained':window_state,
  'explicit_overlap_excluded':excluded,'unique_states_screened':len(seen),'cal_groups':len(cal),'obs_groups':len(fit),
  'state_group_disjoint':not(set(x['state_id'] for x in fit)&set(x['state_id'] for x in cal)),
  'source_theorem_disjoint':'UNKNOWN; dataset has no certified source theorem IDs',
  'M_training':'NOT_STARTED; G2 full span parity not implemented'})
 # Inventory configs only; never read test outcomes to select cases/layers.
 config_files=list((ROOT/'runs').glob('*/formal/*/eval/config.toml'))
 inventory=[{'path':str(p),'valid_split': 'minif2f_valid' in p.read_text(), 'sha256':sha(p)} for p in config_files]
 save(W/'manifests/log_inventory.json',inventory)
 sourcefiles=[ROOT/'REALPROVER/manager/manage/prompt_manage.py',ROOT/'REALPROVER/manager/thirdparty/interactive.py',ROOT/'REALPROVER/manager/thirdparty/generator.py',ROOT/'REALPROVER/manager/search/best_first.py',ROOT/'research/adapters/training.py',ROOT/'research/vendor/reasoningqat/datautils_block.py',ROOT/'REALProver_Calibration_and_ABCD_DataProtocol_v3_20261111.md']
 save(W/'manifests/sources.json',{'files':[{'path':str(p),'sha256':sha(p)} for p in sourcefiles],
  'official_realprover_revision':subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
  'valid_sha256':sha(valid),'tokenizer_sha256':sha(ROOT/'models/REAL-Prover-v1-bf16/tokenizer.json')})
 reports={
 '00_scope_and_source_audit.md':'# Scope\nABCD observation only; P0/P3/P4/P6 immutable. No M training. Referenced ABCD SOP missing; v3 instructions and preregistered small pilot apply. Source SHA and model inventories are in manifests.\n',
 '01_data_split_and_leakage.md':f'# Data splits\nS-CAL {len(cal)}, S-OBS {len(fit)}; exact-state disjoint. {excluded} explicit valid/test name matches excluded. Theorem provenance and semantic overlap UNKNOWN; BF16 SFT may have seen this dataset. V-REAL 32 static-hash-selected miniF2F-valid ROOT_ONLY cases; no test results accessed.\n',
 '02_calibration_loader_2048_feasibility.md':f'# 2048 feasibility\n{eligible}/{len(frame)} raw inputs are >=2048 tokens. Length/marker metrics in data_audit.json. This is a preliminary marker audit, not full STATE/retrieval/tactic span parity. CALIBRATION_FORMAT_BLOCKED for M until full correctness tests. No baseline loader modified.\n',
 '03_valid_search_log_inventory.md':f'# Existing logs\nInventoried {len(inventory)} formal configs; see manifest. New frozen root-state observations are created independently, not substituted from test logs. No search/proposer rollout.\n',
 '05_M_stage1_mixture_pilot.md':'# M\nNOT_STARTED; not requested in this observation task; no optimizer or weight writes.\n'}
 for name,text in reports.items():(W/'reports'/name).write_text(text)
 record('G0_G1','PASS_WITH_PROVENANCE_LIMITATIONS',selected_valid=32,discovery=P['discovery_groups'])
 record('G2','CALIBRATION_FORMAT_BLOCKED',scope='M only; ABCD observation continues')
if __name__=='__main__':main()
