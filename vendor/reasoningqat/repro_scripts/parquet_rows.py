"""Global-index Parquet views without a second full Arrow dataset on disk."""
from bisect import bisect_right
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


class ParquetRows:
    def __init__(self, parts, expected_rows=None):
        self.parts = [str(Path(p).resolve()) for p in parts]
        self.groups = []
        self.ends = []
        inventory = []
        total = 0
        for path in self.parts:
            metadata = pq.ParquetFile(path).metadata
            stat = Path(path).stat()
            inventory.append((path, stat.st_size, stat.st_mtime_ns, metadata.num_rows))
            for group in range(metadata.num_row_groups):
                self.groups.append((path, group, total))
                total += metadata.row_group(group).num_rows
                self.ends.append(total)
        if expected_rows is not None and total != expected_rows:
            raise ValueError(f'Expected {expected_rows} Parquet rows, found {total}')
        self._indices = np.arange(total, dtype=np.int64)
        self._rows = None
        self._fingerprint = hashlib.sha256(json.dumps(inventory).encode()).hexdigest()

    def __len__(self):
        return len(self._indices)

    def _view(self, indices, rows=None):
        view = copy.copy(self)
        view._indices = indices
        view._rows = rows
        view._fingerprint = hashlib.sha256(
            self._fingerprint.encode() + indices.tobytes()).hexdigest()
        return view

    def shuffle(self, seed):
        # datasets.Dataset.shuffle uses this exact NumPy global permutation.
        order = np.random.default_rng(seed).permutation(len(self))
        rows = None if self._rows is None else [self._rows[int(i)] for i in order]
        return self._view(self._indices[order], rows)

    def select(self, indices):
        indices = np.asarray(list(indices), dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= len(self)):
            raise IndexError('Parquet selection outside dataset')
        selected = self._indices[indices]
        rows = (self._read_many(selected) if self._rows is None
                else [self._rows[int(i)] for i in indices])
        return self._view(selected, rows)

    def _read_many(self, indices):
        # Group random global requests before I/O; read each needed row group
        # once, then restore the original requested order (including duplicates).
        requested = {}
        for position, index in enumerate(indices):
            group = bisect_right(self.ends, int(index))
            requested.setdefault(group, []).append((position, int(index)-self.groups[group][2]))
        result = [None] * len(indices)
        reader, current_path = None, None
        for group in sorted(requested):
            path, number, _ = self.groups[group]
            if path != current_path:
                reader = pq.ParquetFile(path)
                current_path = path
            entries = requested[group]
            table = reader.read_row_group(number, use_threads=False)
            rows = table.take([offset for _, offset in entries]).to_pylist()
            for (position, _), row in zip(entries, rows):
                result[position] = row
        return result

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return self._rows[index] if self._rows is not None else self._read_many([self._indices[index]])[0]

    def iter_rows(self, batch_size=64):
        if self._rows is not None:
            yield from self._rows
            return
        for start in range(0, len(self), batch_size):
            yield from self._read_many(self._indices[start:start+batch_size])

    def __iter__(self):
        return self.iter_rows()
