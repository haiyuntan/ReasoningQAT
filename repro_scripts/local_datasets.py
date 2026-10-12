"""Local parquet loaders; preserve upstream ordering and shuffle semantics."""
import glob
import os
from datasets import load_dataset


def load_local_or_hf_openthoughts():
    pattern = os.environ.get('RQ_OPENTHOUGHTS_PARQUET_GLOB')
    if not pattern:
        return load_dataset('open-thoughts/OpenThoughts3-1.2M', split='train')
    parts = sorted(glob.glob(pattern, recursive=True))
    if not parts:
        raise FileNotFoundError(pattern)
    ds = load_dataset('parquet', data_files={'train': parts}, split='train')
    if len(ds) != 1200000:
        raise ValueError(f'Expected full OpenThoughts 1.2M, found {len(ds)}')
    return ds


def load_local_or_hf_fineweb():
    path = os.environ.get('RQ_FINEWEB_PARQUET')
    if not path:
        return load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT', split='train', streaming=True)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return load_dataset('parquet', data_files={'train': [path]}, split='train', streaming=True)
