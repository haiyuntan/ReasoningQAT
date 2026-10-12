"""Prepare official block-QAT token sequences on CPU after all training shards arrive."""
import hashlib
import argparse
import json
import os
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from datautils_block import get_loaders

ROOT=Path(__file__).resolve().parents[2]
REPO=ROOT/'ReasoningQAT'
os.environ.setdefault('RQ_OPENTHOUGHTS_PARQUET_GLOB',str(ROOT/'datasets/openthoughts3/data/**/*.parquet'))
os.environ.setdefault('RQ_FINEWEB_PARQUET',str(ROOT/'datasets/fineweb-edu/sample/10BT/000_00000.parquet'))
matrix=json.loads((REPO/'repro_configs/matrix.json').read_text())
tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/Qwen3-1.7B',local_files_only=True)
manifest={}
parser=argparse.ArgumentParser()
parser.add_argument('--mode',choices=['smoke','formal','both'],default='both')
args=parser.parse_args()
manifest_path=ROOT/'datasets/manifests/block_calibration.json'
manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
for mode in (('smoke','formal') if args.mode=='both' else (args.mode,)):
    os.environ['RQ_RUN_MODE']=mode
    os.environ.pop('RQ_SMOKE_OPENTHOUGHTS',None)
    if mode=='smoke':
        os.environ['RQ_SMOKE_OPENTHOUGHTS']=str(ROOT/'datasets/smoke/openthoughts.jsonl')
    cfg={**matrix['formal'],**(matrix['smoke'] if mode=='smoke' else {})}
    train_size,val_size,seqlen=[cfg[key] for key in ('block_train_size','block_val_size','block_sequence_length')]
    train,val=get_loaders('sweep_0.8',tokenizer,train_size,val_size,seed=2,seqlen=seqlen)
    cache=ROOT/'hf_cache/block_qat'/mode
    cache.mkdir(parents=True,exist_ok=True)
    evidence={}
    for name,loader,count in [('train',train,train_size),('val',val,val_size)]:
        assert len(loader)==count
        canonical=[(inp.clone(),target.clone()) for inp,target in loader]
        assert all(inp.shape==(1,seqlen) and target.shape==(1,seqlen) for inp,target in canonical)
        digest=hashlib.sha256()
        for inp,target in canonical:
            digest.update(inp.numpy().tobytes())
            digest.update(target.numpy().tobytes())
        path=cache/f'dataloader_Qwen3-1.7B_sweep_0.8_{train_size}_{val_size}_{seqlen}_{name}.cache'
        torch.save(canonical,path)
        evidence[name]={'samples':count,'sequence_length':seqlen,'token_sha256':digest.hexdigest(),
                        'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,
                        'reasoning_samples':int(count*.8),'fineweb_samples':count-int(count*.8)}
    manifest[mode]={'seed':2,'sampler':'Unmodified official sweep_0.8 token selection and shuffle',
                    'training_source_requires_all_120_shards':mode=='formal','splits':evidence,
                    'diagnostic_partial_source':mode=='smoke'}
    (ROOT/'datasets/manifests/block_calibration.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({mode:manifest[mode]},indent=2),flush=True)
