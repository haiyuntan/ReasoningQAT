"""Preserve HF global shuffle/selection and nested rows without a disk Arrow copy."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vendor/reasoningqat'))
from repro_scripts.parquet_rows import ParquetRows
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset


class ParquetRowsTests(unittest.TestCase):
    def test_global_order_shuffle_and_selection_match_hf(self):
        rows = [{'id': i, 'conversations': [{'from': 'human', 'value': str(i)}]} for i in range(23)]
        with tempfile.TemporaryDirectory() as temporary:
            paths = [Path(temporary)/f'{i}.parquet' for i in range(2)]
            pq.write_table(pa.Table.from_pylist(rows[:7]), paths[0], row_group_size=3)
            pq.write_table(pa.Table.from_pylist(rows[7:]), paths[1], row_group_size=5)
            data = ParquetRows(paths, expected_rows=23)
            self.assertEqual(list(data.iter_rows(batch_size=4)), rows)
            self.assertEqual(data[-1], rows[-1])
            for seed in [0, 2, 19]:
                reference = Dataset.from_list(rows).shuffle(seed=seed, keep_in_memory=True)
                shuffled = data.shuffle(seed=seed)
                self.assertEqual(list(shuffled), list(reference))
                selected = [22, 0, 7, 7, 2]
                self.assertEqual(list(shuffled.select(selected)), list(reference.select(selected)))
                self.assertEqual(list(shuffled.select(selected).shuffle(seed=2)),
                                 list(reference.select(selected).shuffle(seed=2, keep_in_memory=True)))
            self.assertEqual(data._fingerprint, ParquetRows(paths)._fingerprint)
            self.assertFalse(list(Path(temporary).rglob('*.arrow')))
            with self.assertRaises(ValueError):
                ParquetRows(paths, expected_rows=24)
            with self.assertRaises(IndexError):
                data.select([23])


if __name__ == '__main__':
    unittest.main()
