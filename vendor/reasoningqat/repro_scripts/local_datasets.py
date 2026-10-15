"""Local parquet loaders; preserve upstream ordering and shuffle semantics."""
import glob
import os
from datasets import load_dataset, IterableDataset


def load_local_or_hf_openthoughts():
    fixture = os.environ.get('RQ_SMOKE_OPENTHOUGHTS')
    if fixture:
        if os.environ.get('RQ_RUN_MODE') != 'smoke':
            raise ValueError('Partial OpenThoughts fixtures are restricted to smoke mode')
        return load_dataset('json', data_files={'train': fixture}, split='train')
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
    def rows():
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64,use_threads=False):
            yield from batch.to_pylist()
    return IterableDataset.from_generator(rows)
