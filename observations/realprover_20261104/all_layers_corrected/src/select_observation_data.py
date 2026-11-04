"""Freeze inputs before model execution; no theorem success information read."""
import argparse
import importlib.util
import json
import random
import re
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from common import OBS, ROOT, read, atomic, sha

def save_jsonl(path, rows):
    tmp=path.with_suffix('.partial')
    with tmp.open('w') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    tmp.replace(path)

def classify(text,offsets):
    spans=[]
    keywords={'simp','simp_all','rw','rwa','apply','exact','intro','intros','constructor','linarith','nlinarith','norm_num','aesop','omega','ring','rfl','decide','have','refine','rcases','cases','induction','assumption','simpa','ext'}
    for match in re.finditer(r'[\w\u2080-\u209f]+(?:\.[\w\u2080-\u209f]+)*|[:=<>≤≥≠→←↔+*/^∧∨¬-]+|[^\w\s]',text):
        word=match.group()
        kind='tactic_keyword' if word in keywords else 'numeric' if word.isnumeric() else 'theorem_reference' if '.' in word and word[0].isalpha() else 'identifier' if word[0].isalpha() or word[0]=='_' else 'operator' if re.fullmatch(r'[:=<>≤≥≠→←↔+*/^∧∨¬-]+',word) else 'syntax_punctuation'
        spans.append((match.start(),match.end(),kind))
    categories=[]
    for start,end in offsets:
        overlap=[(min(end,b)-max(start,a),kind) for a,b,kind in spans if min(end,b)>max(start,a)]
        categories.append(max(overlap)[1] if overlap else 'other')
    return categories

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args()
    leakage=read(OBS/'manifests/leakage_report.json')
    if leakage['matched_rows']:raise RuntimeError('DATA_BLOCKED: explicit test overlap, no observation selection')
    samples=[json.loads(line) for line in (OBS/'manifests/state_tactic_sample_manifest.jsonl').read_text().splitlines()]
    tokenizer=Tokenizer.from_file(str(ROOT/'models/REAL-Prover-v1-bf16/tokenizer.json'))
    tokenizer.no_padding();tokenizer.no_truncation()
    source=OBS/'upstream/User-REALPROVER/manager/manage/prompt_manage.py'
    spec=importlib.util.spec_from_file_location('pinned_prompt',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    entropy=[];rejected=[];mapping={}
    for row in samples:
        content=row['input']
        variant='native_incontext' if 'Here is the current STATE:' in content else 'deepseek_wrapped'
        if content.startswith('User: ') and content.rstrip().endswith('Assistant:'):
            content=content[len('User: '):].rstrip()[:-len('Assistant:')].rstrip()
        elif content.startswith('User: '):raise ValueError('Unrecognized wrapping')
        prompt=module.PromptManage.chat_template_to_prompt([{'role':'user','content':content}],'qwen')
        encoded=tokenizer.encode(prompt+row['tactic'],add_special_tokens=False)
        eligible=[i for i,(a,b) in enumerate(encoded.offsets) if a>=len(prompt) and b>len(prompt)]
        boundary=[(a,b) for a,b in encoded.offsets if a<len(prompt)<b]
        if boundary:raise ValueError('Token crosses prompt/completion boundary')
        if len(encoded.ids)>2048 or len(eligible)>128 or not eligible:
            rejected.append({'row':row['row'],'reason':'length','tokens':len(encoded.ids),'subset':row['subset']});continue
        assert min(eligible)>0
        offsets=[(encoded.offsets[i][0]-len(prompt),encoded.offsets[i][1]-len(prompt)) for i in eligible]
        entropy.append({'sample_id':row['state_hash'],'row':row['row'],'state_hash':row['state_hash'],'subset':row['subset'],'distribution':'in_training_distribution','input_ids':encoded.ids,'prediction_positions':[i-1 for i in eligible],'gold_ids':[encoded.ids[i] for i in eligible],'categories':classify(row['tactic'],offsets),'offsets':offsets,'tactic_family':row['tactic_family'],'prompt_variant':variant})
        mapping[variant]=mapping.get(variant,0)+1
    assert set(r['state_hash'] for r in entropy if r['subset']=='threshold').isdisjoint(r['state_hash'] for r in entropy if r['subset']=='analysis')
    if not any(r['subset']=='threshold' for r in entropy) or not any(r['subset']=='analysis' for r in entropy):raise RuntimeError('No full-prompt eligible groups')
    save_jsonl(OBS/'manifests/entropy_inputs.jsonl',entropy)
    atomic(OBS/'manifests/entropy_selection.json',{'selected':len(entropy),'rejected':rejected,'variants':mapping,'prompt_source_sha256':sha(source),'source':'Full provided input, preserving proof and retrieval; normalize legacy User:/Assistant: wrapper to frozen formal qwen wrapper. No extra retrieval or tactic leakage.'})
    (OBS/'reports/schema_mapping.md').write_text('# Physical input mapping\n\n`input` includes full proof/state/retrieval content, with two observed template families. Current STATE is extracted only for exact-state grouping and raw-domain control. Entropy preserves full content, unwraps a legacy deepseek User/Assistant wrapper when present, and applies the pinned formal runtime qwen wrapper exactly once. The gold tactic is appended once and only its token positions are eligible. No EOS; overlength pairs rejected, not spliced.\n\n'+json.dumps({'selected':len(entropy),'variants':mapping,'rejected':len(rejected)},indent=2)+'\n')
    # Figure pilot: exact 128-token raw text windows, same qwen wrapper.
    # Lean raw states eliminate the shared long native instruction as a clustering shortcut.
    rng=random.Random(2);sources={'Lean':[],'General':[],'Math':[],'Code':[]};seen={k:set() for k in sources};windows=set()
    def add(domain,text,group,origin):
        if len(sources[domain])>=256 or group in seen[domain]:return
        tokens=tokenizer.encode(text,add_special_tokens=False).ids
        if len(tokens)<128:return
        window=tuple(tokens[:128])
        if window in windows:return
        windows.add(window)
        seen[domain].add(group)
        sources[domain].append({'raw_ids':tokens[:128],'group':group,'origin':origin,'original_tokens':len(tokens)})
    shuffled=samples.copy();rng.shuffle(shuffled)
    for row in shuffled:add('Lean',row['state'],row['state_hash'],'state_tactic_pairs:'+str(row['row']))
    if len(sources['Lean'])<256:
        from audit_state_tactic_parquet import raw_state,digest
        lean_path=next((OBS/'assets/state_tactic_pairs').rglob('*.parquet'))
        lean=pq.read_table(lean_path,columns=['input']).column('input').to_pylist()
        order=list(range(len(lean)));rng.shuffle(order)
        for index in order:
            state=raw_state(lean[index]);add('Lean',state,digest(state),'state_tactic_pairs:'+str(index))
            if len(sources['Lean'])>=256:break
    for path in sorted((ROOT/'datasets/fineweb-edu').rglob('*.parquet')):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128,columns=['text']):
            for i,row in enumerate(batch.to_pylist()):add('General',row['text'],__import__('hashlib').sha256(row['text'].encode()).hexdigest(),str(path))
            if len(sources['General'])>=256:break
        if len(sources['General'])>=256:break
    for path in sorted((ROOT/'datasets/openthoughts3').rglob('*.parquet')):
        parquet=pq.ParquetFile(path)
        available=set(str(x).lower() for x in parquet.read(columns=['domain']).column('domain').to_pylist())
        needed={domain.lower() for domain in ['Math','Code'] if len(sources[domain])<256}
        if not available.intersection(needed):continue
        print('Domain source',path.name,available,flush=True)
        for batch in parquet.iter_batches(batch_size=64,columns=['domain','source','conversations']):
            for row in batch.to_pylist():
                domain={'math':'Math','code':'Code'}.get(str(row['domain']).lower())
                if domain is None:continue
                text=next((x['value'] for x in row['conversations'] if x['from']=='human'),'')
                group=__import__('hashlib').sha256(text.encode()).hexdigest()
                add(domain,text,group,str(path)+':'+str(row['source']))
            if all(len(sources[d])>=256 for d in ['Math','Code']):break
        if all(len(sources[d])>=256 for d in ['Math','Code']):break
    counts={k:len(v) for k,v in sources.items()}
    atomic(OBS/'manifests/domain_selection.json',{'counts':counts,'source_groups':'SHA256 exact raw text/state, no shared group','selection':'Deterministic expanded selection; first eligible sequential stream for non-Lean, seed2 shuffle for Lean. Selection-order limitation must be reported.','raw_token_window':128,'shared_wrapper':'qwen','main_256_per_domain':'NOT_RUN_PILOT_ONLY'})
    if min(counts.values())<256:raise RuntimeError(f'Domain source shortage: {counts}; do not silently replace labels')
    domains=[]
    for domain,rows in sources.items():
        for row in rows:
            prefix='<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
            suffix='<|im_end|>\n<|im_start|>assistant\n'
            ids=tokenizer.encode(prefix,add_special_tokens=False).ids+row['raw_ids']+tokenizer.encode(suffix,add_special_tokens=False).ids
            domains.append({'sample_id':domain+'_'+row['group'],'domain':domain,'source_group':row['group'],'source':row['origin'],'input_ids':ids,'raw_input_ids':row['raw_ids'],'original_tokens':row['original_tokens']})
    save_jsonl(OBS/'manifests/domain_inputs.jsonl',domains)
    atomic(OBS/'manifests/selection_gate.json',{'status':'PASS','entropy_samples':len(entropy),'domain_samples':len(domains),'prompt_mapping_sha256':sha(OBS/'reports/schema_mapping.md'),'leakage_status':leakage['status']})

if __name__=='__main__':main()
