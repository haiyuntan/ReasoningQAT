import importlib.util
from pathlib import Path
import unittest
import faiss
import numpy as np

spec=importlib.util.spec_from_file_location('exact_search',Path(__file__).resolve().parents[1]/'worker/exact_search.py')
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class ExactTests(unittest.TestCase):
    def test_random_neighbors_and_scores_are_bitwise_identical(self):
        rng=np.random.default_rng(42)
        values=rng.normal(size=(257,32)).astype('float32')
        for metric in (faiss.METRIC_INNER_PRODUCT,faiss.METRIC_L2):
            original=faiss.IndexFlat(32,metric); original.add(values)
            search=module.ExactSearch(original,4)
            for query in rng.normal(size=(24,32)).astype('float32'):
                expected=original.search(query[None,:],10)
                actual=search.search(query[None,:],10)
                for a,b in zip(expected,actual): np.testing.assert_array_equal(a,b)

    def test_ties_use_original_ordering_including_cutoff(self):
        original=faiss.IndexFlatIP(16)
        original.add(np.ones((64,16),dtype='float32'))
        search=module.ExactSearch(original,8)
        query=np.ones((1,16),dtype='float32')
        for a,b in zip(original.search(query,10),search.search(query,10)):
            np.testing.assert_array_equal(a,b)
        self.assertEqual(search.tie_fallbacks,1)

    def test_batch_keeps_original_kernel(self):
        rng=np.random.default_rng(2)
        original=faiss.IndexFlatIP(16); original.add(rng.normal(size=(64,16)).astype('float32'))
        search=module.ExactSearch(original,4)
        query=rng.normal(size=(32,16)).astype('float32')
        for a,b in zip(original.search(query,10),search.search(query,10)):
            np.testing.assert_array_equal(a,b)

    def test_cache_is_bounded_and_lru_eviction_keeps_results_immutable(self):
        cache=module.QueryCache(2,1024)
        key=lambda value:((value,),10)
        source=[[1,2]]; cache.put(key('a'),source); source[0][0]=99
        self.assertEqual(cache.get(key('a')),((1,2),))
        cache.put(key('b'),[[3]]); cache.get(key('a')); cache.put(key('c'),[[4]])
        self.assertIsNone(cache.get(key('b')))
        for i in range(1000): cache.put(key(str(i)),[[i]])
        self.assertLessEqual(len(cache.values),2)
        self.assertLessEqual(cache.bytes,1024)
        cache.put(key('x'*2000),[[1]])
        self.assertLessEqual(cache.bytes,1024)


if __name__=='__main__': unittest.main()
