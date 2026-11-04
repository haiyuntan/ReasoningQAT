"""D0 physical schema, duplicates, exact test overlap, and frozen grouped sample.

No test outcomes are read. Statement normalization is only a conservative screen;
not a semantic or provenance proof. Unknown source theorem identity stays unknown.
"""
import argparse
import collections
import hashlib
import json
import re
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from common import OBS, ROOT, atomic, read, sha

def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()

def normalize(text):
    return re.sub(r'\s+',' ',text).strip()

def schema_columns(names):
    if 'tactic' not in names:raise ValueError('No tactic column')
    if 'state' in names and 'input' in names:raise ValueError('Ambiguous state/input: inspect equality first')
    if 'state' in names:return 'state','tactic'
    if 'input' in names:return 'input','tactic'
    raise ValueError('No state/input column')

def family(text):
    match=re.search(r'[A-Za-z_][\w\u2080-\u209f]*',text)
    return match.group() if match else '<no_keyword>'

def raw_state(text):
    marker='Here is the current STATE:\n'
    ending='\n\n**Please generate a TACTIC in lean4 to solve the state.**'
    if marker in text:
        if ending not in text:raise ValueError('Unrecognized native prompt ending')
        return text.rsplit(marker,1)[1].split(ending,1)[0]
    if '\nSTATE:\n' in text and '\nTACTIC:' in text:
        return text.rsplit('\nSTATE:\n',1)[1].split('\nTACTIC:',1)[0].rstrip()
    if '⊢' in text and 'FORMAL PROOF' not in text:return text
    raise ValueError('Unknown input/state semantics')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args()
    acquisition=read(OBS/'manifests/acquisition.json')['dataset']['result']
    parquet=[x for x in acquisition['files'] if x['path'].endswith('.parquet')]
    assert parquet
    frames=[];schemas=[]
    for item in parquet:
        assert sha(item['path'])==item['sha256']
        p=pq.ParquetFile(item['path']);schemas.append(str(p.schema_arrow))
        source,target=schema_columns(p.schema_arrow.names)
        frames.append(p.read().to_pandas().rename(columns={source:'state',target:'tactic'}))
    frame=pd.concat(frames,ignore_index=True)
    nulls=frame[['state','tactic']].isna().sum().to_dict()
    if sum(nulls.values()):raise ValueError(f'Null inputs: {nulls}')
    if not all(isinstance(x,str) and x.strip() for x in frame.state):raise ValueError('Empty or non-string state')
    if not all(isinstance(x,str) and x.strip() for x in frame.tactic):raise ValueError('Empty or non-string tactic')
    frame['input_hash']=frame.state.map(digest)
    frame['raw_state']=frame.state.map(raw_state)
    frame['state_hash']=frame.raw_state.map(digest)
    frame['tactic_family']=frame.tactic.map(family)
    tokenizer=Tokenizer.from_file(str(ROOT/'models/REAL-Prover-v1-bf16/tokenizer.json'))
    tokenizer.no_padding();tokenizer.no_truncation()
    lengths={}
    for col in ['state','raw_state','tactic']:
        char=frame[col].map(len).to_numpy()
        tok=[]
        for start in range(0,len(frame),256):
            tok.extend(len(x.ids) for x in tokenizer.encode_batch(frame[col].iloc[start:start+256].tolist(),add_special_tokens=False))
        frame[col+'_tokens']=tok
        lengths[col]={'characters':{str(q):float(np.quantile(char,q)) for q in [.5,.9,.99]},'tokens':{str(q):float(np.quantile(tok,q)) for q in [.5,.9,.99]}}
    # Derive official test location from frozen evaluation inputs, never attempt results.
    test_candidates=list((ROOT/'REALPROVER').rglob('*minif2f*test*.json*'))
    # Manifest records which test file is used; choose only the native frozen source.
    test_candidates += list((ROOT/'datasets/eval/legacy_candidate').glob('minif2f_test.jsonl'))
    test_strings=set();test_names=set();test_sources=[]
    for path in sorted(set(test_candidates)):
        if 'generated' in path.parts:continue
        try:
            raw=path.read_text();data=json.loads(raw) if path.suffix=='.json' else [json.loads(s) for s in raw.splitlines() if s.strip()]
        except (ValueError,UnicodeError):continue
        records=list(data.values()) if isinstance(data,dict) else data
        count=0
        for record in records:
            if not isinstance(record,dict):continue
            for key in ['formal_statement','statement','problem','state','input']:
                if isinstance(record.get(key),str):
                    test_strings.add(normalize(record[key]));count+=1
                    test_names.update(re.findall(r'\b(?:theorem|lemma)\s+([\w.]+)',record[key]))
            for key in ['name','theorem_name']:
                if isinstance(record.get(key),str):test_names.add(record[key])
        test_sources.append({'path':str(path),'sha256':sha(path),'text_fields':count})
    if not test_strings:raise ValueError('No test statements found: leakage gate cannot pass')
    overlap=[]
    name_pattern=re.compile(r'(?<!\w)(?:'+'|'.join(re.escape(s) for s in sorted(test_names))+r')(?!\w)') if test_names else None
    for index,row in frame.iterrows():
        state=normalize(row.state)
        if state in test_strings or normalize(row.raw_state) in test_strings or (name_pattern and name_pattern.search(row.state)):overlap.append(int(index))
    atomic(OBS/'manifests/leakage_report.json',{'status':'FAIL' if overlap else 'EXACT_SCREEN_PASS_PROVENANCE_UNKNOWN','matched_rows':overlap,'test_sources':test_sources,'limitations':'No source-theorem IDs in parquet: absence of exact overlap does not establish semantic disjointness. Train-distribution diagnostic only.'})
    # Persist physical examples locally for schema audit, not a plot or public proof dump.
    atomic(OBS/'metrics/state_tactic_schema.json',{'rows':len(frame),'schemas':schemas,'nulls':nulls,'unique_states':int(frame.state_hash.nunique()),'duplicate_pairs':int(frame.duplicated(['state','tactic']).sum()),'multi_tactic_states':int((frame.groupby('state_hash').tactic.nunique()>1).sum()),'revision':acquisition['revision'],'tokenizer_sha256':sha(ROOT/'models/REAL-Prover-v1-bf16/tokenizer.json'),'input_template_counts':{key:int(frame.state.str.contains(key,regex=False).sum()) for key in ['STATE:','TACTIC:','<|im_start|>','Here is the current STATE:','⊢']}})
    atomic(OBS/'metrics/state_tactic_length_quantiles.json',lengths)
    frame.tactic_family.value_counts().rename_axis('tactic_family').to_csv(OBS/'metrics/tactic_family_counts.csv',header=['count'])
    frame.tactic.value_counts().rename_axis('tactic').to_csv(OBS/'metrics/state_tactic_frequency.csv',header=['count'])
    # Stratify by length quartile, group by exact state, seed fixed before GPU.
    groups=frame.loc[~frame.index.isin(overlap)].drop_duplicates('state_hash').copy()
    groups['stratum']=pd.qcut(groups.raw_state_tokens,4,labels=False,duplicates='drop')
    chosen=pd.concat([group.sample(n=min(128,len(group)),random_state=2) for _,group in groups.groupby('stratum')]).sample(frac=1,random_state=2)
    records=[]
    for rank,(index,row) in enumerate(chosen.iterrows()):
        records.append({'row':int(index),'state_hash':row.state_hash,'input_hash':row.input_hash,'input':row.state,'state':row.raw_state,'tactic':row.tactic,'tactic_family':row.tactic_family,'state_tokens':int(row.raw_state_tokens),'input_tokens':int(row.state_tokens),'tactic_tokens':int(row.tactic_tokens),'subset':'threshold' if rank<128 else 'analysis','distribution':'in_training_distribution','dataset_revision':acquisition['revision']})
    path=OBS/'manifests/state_tactic_sample_manifest.jsonl'
    with path.open('w') as f:
        for row in records:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    report=f'''# D0 official state–tactic audit\n\nRows: {len(frame)}; unique exact states: {frame.state_hash.nunique()}; duplicate pairs: {frame.duplicated(['state','tactic']).sum()}.\n\nPhysical schema: {schemas}. Source revision: {acquisition['revision']}.\n\n512 state groups stratified on token-length quartiles, seed 2; 128 threshold groups and 384 paired-analysis groups. Only training-distribution diagnostic, no optimizer updates.\n\nExact test screen matched {len(overlap)} rows. Source-theorem identity and semantic overlap remain UNKNOWN. Test success files are never read.\n\nFrequency and length statistics are not predictive entropy. Prompt semantics must pass the separate schema mapping gate before any GPU analysis.\n'''
    (OBS/'reports/03a_state_tactic_dataset_audit.md').write_text(report)
    print(report,flush=True)
    if overlap:raise RuntimeError('Explicit test overlap: freeze blocked pending review')

if __name__=='__main__':main()
