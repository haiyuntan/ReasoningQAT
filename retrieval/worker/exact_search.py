"""Parallel exact search with original CPU ordering for tied neighbors."""
from collections import OrderedDict
import faiss
import numpy as np


class ExactSearch:
    def __init__(self, index, shards=8):
        self.original = index
        self.parts = []
        self.parallel = None
        if shards < 1:
            raise ValueError('shards must be positive')
        if shards > 1:
            if not isinstance(index, faiss.IndexFlat):
                raise ValueError('Only exact flat indexes support sharding')
            self.parallel = faiss.IndexShards(index.d, True, True)
            width = (index.ntotal + shards - 1) // shards
            for start in range(0, index.ntotal, width):
                part = faiss.IndexFlat(index.d, index.metric_type)
                part.add(index.reconstruct_n(start, min(width, index.ntotal-start)))
                self.parts.append(part)
                self.parallel.add_shard(part)
        self.tie_fallbacks = 0

    def search(self, vectors, count):
        if self.parallel is None or len(vectors) != 1 or count >= self.original.ntotal:
            return self.original.search(vectors, count)
        scores, ids = self.parallel.search(vectors, count+1)
        if np.any(scores[:, :-1] == scores[:, 1:]):
            self.tie_fallbacks += 1
            return self.original.search(vectors, count)
        return scores[:, :count], ids[:, :count]


class QueryCache:
    """LRU bounded by both entries and accounted key/result bytes."""
    def __init__(self, entries=2048, max_bytes=16*1024*1024):
        self.entries, self.max_bytes = entries, max_bytes
        self.values = OrderedDict()
        self.bytes = 0
        self.hits = self.misses = 0

    def get(self, key):
        item = self.values.get(key)
        if item is None:
            self.misses += 1
            return None
        self.values.move_to_end(key)
        self.hits += 1
        return item[0]

    def put(self, key, ids):
        value = tuple(tuple(int(i) for i in row) for row in ids)
        cost = sum(len(q.encode('utf-8')) for q in key[0]) + sum(len(row) for row in value)*8 + 128
        if self.entries <= 0 or cost > self.max_bytes:
            return
        previous = self.values.pop(key, None)
        if previous:
            self.bytes -= previous[1]
        self.values[key] = value, cost
        self.bytes += cost
        while len(self.values) > self.entries or self.bytes > self.max_bytes:
            _, (_, released) = self.values.popitem(last=False)
            self.bytes -= released
