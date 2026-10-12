"""CPU-only formal execution audit; never infer missing author configurations."""
import ast
import datetime
import json
from pathlib import Path
from pipeline import MATRIX,REQUIRED,train_command

ROOT=Path(__file__).resolve().parents[2]
REPO=ROOT/'ReasoningQAT'

def audit():
    checks={};blockers=[]
    for name in ('main_block_qat.py','main_e2e_distill.py',
                 'repro_scripts/baseline_training.py','repro_scripts/export_baseline.py',
                 'repro_scripts/pipeline.py','repro_scripts/local_datasets.py'):
        ast.parse((REPO/name).read_text(),filename=name)
    checks['python_syntax']=True
    formal=MATRIX['formal']
    checks['paper_known_training_budget']=(formal['block_train_size']==4096 and
        formal['block_val_size']==64 and formal['block_sequence_length']==2048 and
        formal['e2e_dataset_size']==32768 and formal['e2e_accumulation']*formal['e2e_micro_batch']==64 and
        formal['runs']['R4']['epochs']==3 and formal['e2e_learning_rate']==5e-6)
    plans={run:train_command(run,'formal',ROOT/'outputs/formal/audit'/run,'audit') for run in REQUIRED}
    checks['formal_has_no_smoke_flags']=all(not cmd or not any(flag in cmd for flag in ('--max_steps','--smoke_short_text')) for cmd in plans.values())
    loader=ast.parse((REPO/'repro_scripts/local_datasets.py').read_text())
    checks['full_corpus_size_guard_present']=any(isinstance(node,ast.Constant) and node.value==1200000 for node in ast.walk(loader))
    protocol=json.loads((REPO/'repro_configs/evaluation.json').read_text())
    checks['three_eval_seeds']=protocol['seeds']==[42,43,44]
    checks['author_protocol_confirmed']=protocol['confirmed_paper_protocol'] and not protocol['unresolved']
    baseline_tree=ast.parse((REPO/'repro_scripts/baseline_training.py').read_text())
    stage2=next(node for node in baseline_tree.body if isinstance(node,ast.FunctionDef) and node.name=='stage2_train')
    hardcoded_two_steps=any(isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
        and node.func.id=='range' and len(node.args)==1 and isinstance(node.args[0],ast.Constant)
        and node.args[0].value==2 for node in ast.walk(stage2))
    checks['baseline_formal_epoch_loop_implemented']=not hardcoded_two_steps
    checks['baseline_formal_accumulation_implemented']=any(isinstance(node,ast.Constant) and
        node.value=='e2e_accumulation' for node in ast.walk(stage2))
    if not checks['author_protocol_confirmed']:
        blockers.extend('UNVERIFIED_AUTHOR_CONFIG: '+reason for reason in protocol['unresolved'])
    cfg=json.loads((REPO/'repro_configs/qat_baselines.json').read_text())
    for run in ('R8','R9'):
        checks[run+'_formal_supported']=bool(cfg[run].get('formal_supported'))
        if not checks[run+'_formal_supported']:
            blockers.append(run+': implemented adapter is smoke-only; full-epoch/accumulation formal trainer is not implemented')
        blockers.extend(run+': '+reason for reason in cfg[run]['blockers'] if 'pending' not in reason.lower())
    for name,passed in checks.items():
        if not passed and name not in ('author_protocol_confirmed','R8_formal_supported','R9_formal_supported'):
            blockers.append('CODE_CHECK_FAILED: '+name)
    result={'status':'READY' if not blockers else 'BLOCKED','checks':checks,'blockers':blockers,
            'formal_commands':plans,'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'meaning':'Static formal code/protocol audit, not formal execution or paper reproduction'}
    (ROOT/'results/formal-code-audit.json').write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    print(json.dumps(audit(),indent=2))
