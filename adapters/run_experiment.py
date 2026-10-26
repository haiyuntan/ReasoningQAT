"""Executable method adapter. A receipt is emitted only after real checks."""
import argparse
import math
import os
from pathlib import Path
import sys

from common import ROOT, RESEARCH, MODEL, load, save, sha, python_for, execute, data_environment, verify_model, model_signature


def checks_for(run, out, model, evaluation, training=None):
    checks = {'model_identity': model['model_type'] == 'qwen2',
              'tokenizer': evaluation['export']['tokenizer_parity'],
              'native_lean_evaluation': evaluation['stats']['completed_problems'] == evaluation['stats']['expected_problems'],
              'retrieval': evaluation['stats']['retrieval_ok'],
              'fixed_denominator': evaluation['stats']['expected_problems'] == (2 if os.environ['REALPROVER_MODE'] == 'smoke' else 244),
              'saved_reload': evaluation['export']['saved_reload']}
    exported = evaluation['export']
    if run != 'P0':
        checks.update(w2_g128=exported['bits'] == 2 and exported['group_size'] == 128,
                      export_logits_parity=exported['greedy_generation_parity'],
                      same_eval_protocol=evaluation['stats']['protocol_sha256'] == sha(ROOT / load(RESEARCH / 'configs/runtime.json')['eval_candidate']))
    if run in ('P1', 'P2'):
        checks.update(numinamath_same_128=exported['samples'] == 128 and
                      exported['calibration_sha256'] == sha(ROOT / 'datasets/numinamath_128/calib.jsonl'),
                      algorithm_identity=(out / ('native' if run == 'P1' else 'awq_search.pt')).exists())
        if run == 'P1':
            checks['export_logits_parity'] = (checks['export_logits_parity'] and
                exported['gptq_reference_validation']['exact_weight_checks'] ==
                load(MODEL / 'config.json')['num_hidden_layers'] * 7)
    if training:
        updates = training['optimizer_audit']
        losses = training.get('stage2', {}).get('steps', training.get('block_ap', []))
        data = training.get('stage2_data', training.get('stage1_data', {}))
        checks.update(general_data_only=True,  # validated data entrypoints, no user-supplied training paths
                      finite_loss=bool(losses) and all(math.isfinite(s.get('loss', s.get('mse', 0.))) for s in losses),
                      finite_gradients=bool(updates) and all(math.isfinite(s['grad_norm']) for s in updates),
                      trainable_parameters=bool(updates), parameter_update=any(s['updated_values'] > 0 for s in updates),
                      qwen2_modules=len(exported['replaced_modules']) == load(MODEL / 'config.json')['num_hidden_layers'] * 7)
    if run in ('P4', 'P6'):
        stage = training['stage2']
        p3 = out.parent / 'P3/train/native'
        checks.update(same_p3_initialization=training['p3_signature'] == model_signature(p3),
                      same_stage2_tokens=training['stage2_data'] == load(out.parent / 'shared_stage2.json') and
                      stage['causal_tokens_processed'] == training['stage2_data']['causal_tokens'] * stage['epochs'],
                      scale_only=bool(stage['trainable_names']) and all(n.endswith('.weight_quantizer.scale') for n in stage['trainable_names']))
        other_run = 'P6' if run == 'P4' else 'P4'
        peer = out.parent / other_run / 'train/training.json'
        if peer.exists():
            checks['same_stage2_tokens'] = checks['same_stage2_tokens'] and training['stage2_data'] == load(peer)['stage2_data']
        if run == 'P4':
            checks.update(teacher_rr_loss=stage['teacher_used'] and all(s['rr'] is not None for s in stage['steps']),
                          top20_kd_loss=all(math.isfinite(s['kl']) for s in stage['steps']),
                          supervision_mask=training['stage2_data']['causal_tokens'] > 0)
        else:
            checks.update(ce_only_loss=stage['objective'] == 'ordinary CE' and all(s['rr'] is None and s['kl'] == 0 for s in stage['steps']),
                          teacher_disabled=stage['teacher_used'] is False)
    if run == 'P5':
        checks.update(independent_block_ap=bool(training['block_ap']) and 'p3_signature' not in training,
                      independent_e2e_qp=training['stage2']['objective'] == 'ordinary CE' and
                      all(n.endswith('.scales') for n in training['stage2']['trainable_names']),
                      matched_token_budget=training['stage2_data'] == load(out.parent / 'shared_stage2.json') and
                      training['stage1_data'] == load(out.parent / 'shared_stage1.json') and
                      training['stage2']['causal_tokens_processed'] == training['stage2_data']['causal_tokens'] * training['stage2']['epochs'])
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, choices=[f'P{i}' for i in range(7)])
    ap.add_argument('--mode', required=True, choices=['smoke', 'formal'])
    ap.add_argument('--phase', choices=['all', 'prepare', 'evaluate'], default='all')
    a = ap.parse_args()
    if os.environ.get('REALPROVER_RUN') != a.run or os.environ.get('REALPROVER_MODE') != a.mode:
        raise RuntimeError('Use experiment_pipeline.py; direct execution lacks campaign gate context')
    out = Path(os.environ['REALPROVER_OUTPUT'])
    env = data_environment(a.mode)
    # Every child process gets explicit run context and isolated source imports.
    for key in ('REALPROVER_RUN', 'REALPROVER_MODE', 'REALPROVER_OUTPUT', 'REALPROVER_FINGERPRINT'):
        env[key] = os.environ[key]
    adapters = RESEARCH / 'adapters'
    sys.path.insert(0, str(RESEARCH / 'scripts'))
    import retrieval_service
    # Never keep the 14GB retrieval model resident during quantization/training.
    shared_retrieval = os.environ.get('REALPROVER_SHARED_RETRIEVAL') == '1'
    if not shared_retrieval:
        retrieval_service.stop()
    if a.phase != 'evaluate':
        prepare(a, out, env)
    else:
        prepared = load(out / 'prepared.json')
        if any(prepared.get(k) != v for k, v in
               {'run': a.run, 'mode': a.mode, 'fingerprint': os.environ['REALPROVER_FINGERPRINT']}.items()):
            raise RuntimeError('Preparation identity/fingerprint mismatch')
        if not prepared.get('evidence'):
            raise RuntimeError('Preparation artifact hashes missing')
        for item in prepared['evidence']:
            if sha(Path(item['path'])) != item['sha256']:
                raise RuntimeError(f'Prepared artifact changed: {item["path"]}')
    if a.phase == 'prepare':
        return
    model = load(out / 'model_identity.json')
    training = load(out / 'train/training.json') if a.run in ('P3', 'P4', 'P5', 'P6') else None
    adapters = RESEARCH / 'adapters'
    # External Lean dependencies are checked by the evaluator before inference.
    eval_model = MODEL if a.run == 'P0' else out / 'eval_model'
    try:
        retrieval_service.ready()
        execute([python_for('eval'), adapters / 'evaluation.py', '--model', MODEL, '--mode', a.mode,
                 '--output', out / 'preflight', '--preflight'], env=env)
        execute([python_for('eval'), adapters / 'evaluation.py', '--model', eval_model, '--mode', a.mode,
                 '--output', out / 'eval'], env=env)
    finally:
        if not shared_retrieval:
            retrieval_service.stop()
    save(out / 'resource_lifecycle.json', {'policy': 'dedicated shared retrieval GPU' if shared_retrieval else 'retrieval only during Lean evaluation',
         'retrieval_stopped_before_training_and_quantization': not shared_retrieval,
         'retrieval_stopped_after_evaluation': not shared_retrieval, 'evaluation_finished': True})
    finish(a, out, model, training)


def prepare(a, out, env):
    adapters = RESEARCH / 'adapters'
    save(out / 'resource_lifecycle.json', {'policy': 'retrieval only during Lean evaluation',
         'retrieval_stopped_before_training_and_quantization': True, 'evaluation_finished': False})
    model = verify_model()
    save(out / 'model_identity.json', model)
    training = None
    if a.run in ('P3', 'P4', 'P5', 'P6'):
        argv = [python_for('train'), adapters / 'training.py', '--run', a.run, '--mode', a.mode, '--output', out / 'train']
        if a.run in ('P4', 'P6'):
            argv += ['--p3', out.parent / 'P3/train/native']
        execute(argv, env=env)
        training = load(out / 'train/training.json')
        for key, field in [('stage1', 'stage1_data'), ('stage2', 'stage2_data')]:
            if field not in training:
                continue
            shared = out.parent / f'shared_{key}.json'
            import fcntl
            with (out.parent / 'shared_data.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if shared.exists() and load(shared) != training[field]:
                    raise RuntimeError(f'{key} token/mask budget differs between methods')
                if not shared.exists():
                    save(shared, training[field])
    export_role = 'gptq' if a.run == 'P1' else 'train'
    argv = [python_for(export_role), adapters / 'quantization.py', '--run', a.run, '--mode', a.mode,
            '--output', out / 'eval_model']
    if training:
        argv += ['--input', out / 'train/native']
    execute(argv, env=env)
    if training:
        exported = load(out / 'eval_model/export.json')
        if exported['saved_reload'] and exported['greedy_generation_parity']:
            # A verified final checkpoint supersedes the rolling optimizer file.
            removed = []
            for name in ('stage2_resume.pt', 'stage2_resume.tmp'):
                recovery = out / 'train' / name
                if recovery.exists():
                    recovery.unlink()
                    removed.append(str(recovery))
            save(out / 'artifact_cleanup.json', {'removed_superseded_recovery_files': removed})
    artifacts = [out / 'model_identity.json']
    for folder in ('train', 'eval_model'):
        artifacts += [p for p in (out / folder).rglob('*') if p.is_file()
                      and p.suffix in ('.json', '.pt', '.safetensors')]
    save(out / 'prepared.json', {'run': a.run, 'mode': a.mode,
         'fingerprint': os.environ['REALPROVER_FINGERPRINT'],
         'evidence': [{'path': str(p.resolve()), 'sha256': sha(p)} for p in artifacts]})


def finish(a, out, model, training):
    evaluation = {'export': load(out / 'eval_model/export.json'), 'stats': load(out / 'eval/evaluation.json')}
    checks = checks_for(a.run, out, model, evaluation, training)
    sys.path.insert(0, str(RESEARCH / 'scripts'))
    from experiment_pipeline import requirements
    if any(checks.get(k) is not True for k in requirements(a.run)):
        save(out / 'failed_checks.json', checks)
        raise RuntimeError('Method validation failed; no PASS receipt emitted')
    files = [out / 'model_identity.json', out / 'preflight/preflight.json', out / 'eval_model/export.json', out / 'eval/evaluation.json', out / 'eval/config.toml', out / 'eval/retrieval.jsonl', out / 'resource_lifecycle.json']
    files += [p for p in (out / 'prepared.json', out / 'reuse_manifest.json') if p.exists()]
    files += [p for p in (out / 'eval/evaluation_resume.json',) if p.exists()]
    if training:
        files += [out / 'train/training.json']
        files += [out.parent / f'shared_{s}.json' for s in ('stage1', 'stage2') if f'{s}_data' in training]
    if a.run in ('P1', 'P2'):
        files += [ROOT / 'datasets/numinamath_128/calib.jsonl']
    evidence = [{'path': str(p.resolve()), 'sha256': sha(p)} for p in files]
    evidence += evaluation['stats']['raw_evidence']
    # Bind checkpoints too: changing P3 after smoke must invalidate its receipt.
    for directory in ([out / 'train/native'] if training else []) + ([out / 'eval_model'] if a.run != 'P0' else []):
        evidence += [{'path': str(p.resolve()), 'sha256': sha(p)} for p in directory.iterdir()
                     if p.is_file() and p.suffix in ('.safetensors', '.pt', '.json')]
    stats = evaluation['stats']
    save(out / 'receipt.json', {'run': a.run, 'mode': a.mode, 'fingerprint': os.environ['REALPROVER_FINGERPRINT'],
        'status': 'PASS', 'checks': checks, 'expected_problems': stats['expected_problems'],
        'completed_problems': stats['completed_problems'], 'evidence': evidence})


if __name__ == '__main__':
    main()
