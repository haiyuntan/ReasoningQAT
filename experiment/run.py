import sys
import datetime
try: import tomllib
except ModuleNotFoundError: import pip._vendor.tomli as tomllib
import torch
from manager.service import BatchMainService
from manager.manage import ProofParseManage
from util import profiler
from manager.search.exception import single_error_test
import shutil
import os


def main(config_path: str, resume_from_id: int = None, result_dir: str = None):
    # with open(f'experiment/examples/{config_path}', 'rb') as fp:
    #     config = tomllib.load(fp)
    with open(config_path, 'rb') as fp:
        config = tomllib.load(fp)
    if result_dir is None:
        result_dir = 'experiment/logs/{timestamp:%Y-%m-%d}/{info}-{source_id}-{num_samples}-{max_calls}-{timestamp:%H%M}'.format(
            timestamp=datetime.datetime.now(),
            info = config["info"],
            source_id=config['data']['data_id'],
            num_samples=config['search']['num_samples'],
            max_calls=config['search']['max_calls'])  # 结果文件目录
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(os.path.join(result_dir, 'config.toml')):
        shutil.copy(config_path, os.path.join(result_dir,'config.toml'))
    profiler.start("run_batch")
    # 默认使用可见的全部gpu, 也可以自己配置
    #gpus = 1
    gpus = torch.cuda.device_count()
    assert gpus >= 1
    print(f"gpus = {gpus}")
    gpus_list = list(range(gpus))

    main_service = BatchMainService(source_file=config['data']['data_path'],
                                    result_dir=result_dir,
                                    gpus_list=gpus_list,
                                    max_nodes=config['search']['max_nodes'],
                                    max_depth=config['search']['max_depth'],
                                    num_samples=config['search']['num_samples'],
                                    use_beam_search=config['beam_search_params']['use_beam_search'],
                                    use_mcts_search=config['mcts_params']['use_mcts_search'],
                                    beam_width=config['beam_search_params']['beam_width'],
                                    local_model_path=config['model']['prover_model_path'],
                                    sampling_params=config['model']['prover_model_params'],
                                    max_retries=config['search']['max_retries'],
                                    simulation_depth=config['mcts_params']['sim_depth'],
                                    c_puct=config['mcts_params']['c_puct'],
                                    c_score=config['mcts_params']['c_score'],
                                    c_expansion_fail_penalty=config['mcts_params']['c_expansion_fail_penalty'],
                                    max_root_expansion=config['mcts_params']['max_root_expansion'],
                                    max_calls=config['search']['max_calls'],
                                    abandon_if_contain=config['search']['abandon_if_contain'],
                                    is_incontext=config['model'].get('is_incontext', False),
                                    template=config['model'].get('template', 'deepseek'),
                                    use_retrieval = config['model'].get('use_retrieval', True),
                                    resume_from_id=resume_from_id,
                                    )
    main_service.batch_run()
    print(main_service.info)
    profiler.stop("run_batch")
    ProofParseManage.get_stats(result_dir, main_service.info)
    # shutil.copy(f'experiment/examples/{config_path}', f'{result_dir}/{config_path}')
    # ProofParseManage.get_stats("/AI4M/users/ytwang/auto-proof/stepprover-v2/experiment/logs/2025-01-21/alg-test-v2-32-256-2156")
    # single_error_test("experiment/logs/2025-02-16/test_new-example_20_index-8-64-2037/error/8.json")
    # ProofParseManage.get_stats("experiment/logs/2025-02-14/mcts_test-example_20_index-6-128-0209")


if __name__ == '__main__':
    import argparse
    _ap = argparse.ArgumentParser()
    _ap.add_argument("config_path")
    _ap.add_argument("--resume-from", type=int, default=None,
                     help="续跑起始题目 source_index（含）：只评估 source_index >= 该值的题目。用于断点续跑")
    _ap.add_argument("--result-dir", type=str, default=None,
                     help="指定结果目录（续跑时传上次的结果目录，继续写入而非新建 timestamp 目录）")
    _args = _ap.parse_args()
    main(_args.config_path, resume_from_id=_args.resume_from, result_dir=_args.result_dir)
