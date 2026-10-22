"""One audited token cache shared by P4/P5/P6, published only when complete."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from common import ROOT, RESEARCH, MODEL, load, save, sha


def contract(mode):
    return {'mode': mode, 'stage2': load(RESEARCH / 'configs/training_general_data.json')['stage2'],
            'smoke': load(RESEARCH / 'configs/runtime.json')['smoke'] if mode == 'smoke' else None,
            'policy': sha(RESEARCH / 'adapters/completion_tokens.py'),
            'tokenizer': {p.name: sha(p) for p in MODEL.glob('tokenizer*') if p.is_file()},
            'source': [(str(p), p.stat().st_size, p.stat().st_mtime_ns)
                       for p in sorted((ROOT / 'datasets/openthoughts3').rglob('*.parquet'))]}


class Writer:
    def __init__(self, destination, count, maximum, mode):
        self.destination, self.mode = Path(destination), mode
        self.tmp = self.destination.with_name(self.destination.name + f'.building-{os.getpid()}')
        self.tmp.mkdir(parents=True, exist_ok=False)
        self.arrays = {name: np.lib.format.open_memmap(self.tmp / f'{name}.npy', mode='w+',
                        dtype=np.int32, shape=(count, maximum)) for name in ('ids', 'labels')}
        self.lengths = np.lib.format.open_memmap(self.tmp / 'lengths.npy', mode='w+', dtype=np.int32, shape=(count,))
        self.count = 0

    def append(self, batch):
        n = batch['input_ids'].shape[1]
        self.arrays['ids'][self.count, :n] = batch['input_ids'].numpy()[0]
        self.arrays['labels'][self.count, :n] = batch['labels'].numpy()[0]
        self.lengths[self.count] = n
        self.count += 1
        if self.count % 256 == 0:
            print(f'STAGE2_CACHE {self.count}/{len(self.lengths)}', flush=True)

    def finish(self, evidence):
        if self.count != len(self.lengths):
            raise RuntimeError('Incomplete stage2 cache')
        for array in [*self.arrays.values(), self.lengths]:
            array.flush()
        files = {p.name: sha(p) for p in self.tmp.glob('*.npy')}
        save(self.tmp / 'manifest.json', {'status': 'PASS', 'contract': contract(self.mode),
             'evidence': evidence, 'files': files})
        self.tmp.rename(self.destination)


def cached_batches(path, mode):
    import torch
    manifest = load(path / 'manifest.json')
    # JSON canonicalization also normalizes tuple inventories.
    if (manifest['status'] != 'PASS' or
            manifest['contract'] != json.loads(json.dumps(contract(mode)))):
        raise RuntimeError('Stage2 cache contract changed')
    for filename, digest in manifest['files'].items():
        if sha(path / filename) != digest:
            raise RuntimeError(f'Stage2 cache hash mismatch: {filename}')
    arrays = {name: np.load(path / f'{name}.npy', mmap_mode='r') for name in ('ids', 'labels', 'lengths')}
    class Batches:
        def __len__(self):
            return len(arrays['lengths'])
        def __getitem__(self, index):
            if index >= len(self):
                raise IndexError(index)
            n = int(arrays['lengths'][index])
            return {'input_ids': torch.tensor(arrays['ids'][index, :n].astype(np.int64)).unsqueeze(0),
                    'labels': torch.tensor(arrays['labels'][index, :n].astype(np.int64)).unsqueeze(0),
                    'attention_mask': torch.ones((1, n), dtype=torch.long)}
    return Batches(), manifest['evidence']
