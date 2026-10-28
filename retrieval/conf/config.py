"""
This module provide configurations.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RETRIEVAL_ASSETS = ROOT / 'models/retrieval'
INDEX_PATH = os.environ.get('REALPROVER_RETRIEVAL_INDEX', str(RETRIEVAL_ASSETS / 'LeanSearch-PS-faiss.index'))
TOKENIZER_PATH = os.environ.get('REALPROVER_RETRIEVAL_BASE', str(RETRIEVAL_ASSETS / 'e5-mistral-7b-instruct'))
MODEL_PATH = TOKENIZER_PATH
ADAPTER_PATH = os.environ.get('REALPROVER_RETRIEVAL_ADAPTER', str(RETRIEVAL_ASSETS / 'LeanSearch-PS'))
ANSWER_PATH = os.environ.get('REALPROVER_RETRIEVAL_ANSWERS', str(RETRIEVAL_ASSETS / 'answers.json'))
ASSET_MANIFEST = ROOT / 'research/manifests/retrieval_assets.json'


EXPIRE_TIME = 5 * 60

TEST_QUERY = '''G : Type u_1
      inst✝ : Group G
      a b c : G
      ⊢ (fun x => a * x * b = c) (a⁻¹ * c * b⁻¹) ∧ ∀ (y : G), (fun x => a * x * b = c) y → y = a⁻¹ * c * b⁻¹'''
