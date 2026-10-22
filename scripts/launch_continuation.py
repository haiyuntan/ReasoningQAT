"""Create an audited fresh batch and nohup the independent four-GPU controller."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from experiment_pipeline import ROOT, RESEARCH, fingerprint
sys.path.insert(0, str(RESEARCH / 'adapters'))
from common import load, save
from reuse_prepared import inherit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--results', type=Path, required=True)
    ap.add_argument('--validation-report', type=Path, required=True)
    ap.add_argument('--authorized-direct-formal', action='store_true', required=True)
    ap.add_argument('--stage2-cache', type=Path)
    ap.add_argument('--preflight-process', type=Path)
    ap.add_argument('--schedule', choices=('training-first', 'mixed'), default='training-first')
    args = ap.parse_args()
    source, results = args.source.resolve(), args.results.resolve()
    report = args.validation_report.resolve()
    if load(report)['status'] != 'PASS':
        raise RuntimeError('Targeted validation must pass')
    # Restore the audited node-local Lean runtime used by the source campaign.
    lean_env = load(source / 'control/status.json')['lean_environment']
    for key, value in lean_env.items():
        if not value or not Path(value).exists():
            raise RuntimeError(f'Missing audited Lean runtime {key}: {value}')
        os.environ[key] = value
    os.environ['PATH'] = lean_env['REALPROVER_LEAN_BIN'] + ':' + os.environ['PATH']
    os.environ.update(USE_TF='0', USE_FLAX='0', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                      OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    os.environ.pop('REALPROVER_SHARED_RETRIEVAL', None)
    if args.preflight_process:
        os.environ['REALPROVER_EXISTING_STAGE2_PREFLIGHT_PID'] = str(load(args.preflight_process)['pid'])
    else:
        os.environ.pop('REALPROVER_EXISTING_STAGE2_PREFLIGHT_PID', None)
    for repo in ('research', 'REALPROVER'):
        status = subprocess.check_output(['git', '-C', str(ROOT / repo), 'status', '--porcelain', '-uno'], text=True)
        if status:
            raise RuntimeError(f'Commit changes before formal launch: {repo}')
    digest = fingerprint()
    results.mkdir(parents=True, exist_ok=False)
    inherit(source, results, digest)
    if args.stage2_cache:
        (results / 'stage2_cache').symlink_to(args.stage2_cache.resolve(), target_is_directory=True)
    waiver = results / 'smoke_waiver.json'
    save(waiver, {'fingerprint': digest, 'results': str(results),
         'scope': 'skip_smoke_for_direct_formal_restart',
         'user_instruction': '修复后继续正式实验；单任务失败隔离；P5不保存逐block检查点；先完成训练再一卡检索三卡连续评估。沿用已授权的不重复完整冒烟重启。',
         'validation_report': str(report)})
    save(results / 'campaign_plan.json', {'fingerprint': digest, 'source': str(source),
         'schedule': args.schedule, 'reuse': ['P1', 'P2', 'P3'],
         'training_gpus': {'0': 'P4', '1': 'P6', '2': 'P5 independent restart'},
         'evaluation_gpus': [0, 1, 2], 'retrieval_gpu': 3, 'method_failures_isolated': True,
         'P5_per_block_checkpoints': False,
         'stage2_cache': str((results / 'stage2_cache').resolve()),
         'engine': load(ROOT / 'REALPROVER/configs/vllm_runtime.json')})
    log = results.with_name(results.name + '.nohup.log')
    command = ['nohup', sys.executable, '-u', str(RESEARCH / 'scripts/resume_four_gpu.py'),
               '--results', str(results), '--smoke-waiver', str(waiver), '--schedule', args.schedule]
    with log.open('x') as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ), stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    save(results / 'launch.json', {'pid': process.pid, 'command': command, 'log': str(log), 'fingerprint': digest})
    results.with_name(results.name + '.pid').write_text(str(process.pid) + '\n')
    print(json.dumps({'pid': process.pid, 'results': str(results), 'log': str(log), 'fingerprint': digest}), flush=True)


if __name__ == '__main__':
    main()
