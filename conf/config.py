# -*- coding: utf-8 -*-
################################################################################

################################################################################
"""
This module provide configure for REAL-Prover evaluation.
"""

import os
import json
from pathlib import Path

# 支持的几个模型
MODEL_TYPE_LOCAL = 'local'
MODEL_TYPE_GEMINI = 'gemini'
MODEL_TYPE_CLAUDE = 'claude'

# 默认使用的模型List
DEFAULT_MODEL_LIST = [MODEL_TYPE_LOCAL]

# Model paths
PROVER_MODEL_ID = 'realprover'
PROVER_MODEL_PATH = os.environ.get('REALPROVER_MODEL_PATH', str(Path(__file__).resolve().parents[2] / 'models/REAL-Prover-v1-bf16'))
LEAN_TEST_PATH = os.environ.get('REALPROVER_LEAN_WORKSPACE', str(Path(__file__).resolve().parents[1] / 'dependencies/lean_test_v4160'))
LEAN_ENV_PATH = os.environ.get('REALPROVER_LEAN_BIN', os.path.expanduser("~/.elan/bin"))

# Search parameters - aligned with paper for pass@64
NUM_SAMPLES = 64
MAX_DEPTH = 128
MAX_NODES = 1024
MAX_CALLS = 1024

MAX_RETRIES = 64
USE_BEAM_SEARCH = False
USE_MCTS_SEARCH = False
BEAM_WIDTH = 3
IS_INCONTEXT = True
USE_RETRIEVAL = True

SIM_DEPTH = 5
C_PUCT = 1.0
C_SCORE = 1.0
C_EXPANSION_FAIL_PENALTY = 30.0
MAX_ROOT_EXPANSION = 5

# One fingerprinted engine profile for every P0-P6 method. Retrieval stays resident.
VLLM_ENGINE_OPTIONS = json.loads(
    (Path(__file__).resolve().parents[1] / 'configs/vllm_runtime.json').read_text())
GPU_MEMORY_UTILIZATION = VLLM_ENGINE_OPTIONS['gpu_memory_utilization']
VLLM_MAX_MODEL_LEN = VLLM_ENGINE_OPTIONS['max_model_len']

# params for model other than n
PROVER_MODEL_PARAMS = {
    "temperature": 1.5,
    "max_tokens": 256,
    "top_p": 0.9,
    "logprobs": 1
}

# Port configurable via env var LEANSEARCH_PORT so multiple retrieval
# servers can run on different ports (one per GPU) during parallel eval.
API_CONFIG = {
    "lean_search": f'http://localhost:{os.environ.get("LEANSEARCH_PORT", "8080")}/retrieve_premises'
}
NUM_QUERYS = 10


CLAUDE_CONFIG = {
    'base_url': 'url path',
    'api_key': 'your api key',
}

OTHER_MODELS = ["gemini", "claude"]

# 此为需要过滤的不合法tactic列表
ABANDON_IF_CONTAIN = ["sorry", "admit", "apply?"]

# interactive 用到的一些配置
base_path = Path(__file__).parent.parent
analyzer_path = Path(os.environ.get('REALPROVER_JIXIA_BIN', str(base_path / 'dependencies/jixia/.lake/build/bin/jixia')))
interactive_path = Path(os.environ.get('REALPROVER_INTERACTIVE_ROOT', str(base_path / 'dependencies/interactive')))
build_path = interactive_path / '.lake/build'

# 接口验权参数
SALT = "ai-for-math"
EXPIRE_TIME = 5 * 60
