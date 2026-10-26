"""Completed parallel requests must not accumulate, including mixed/live groups."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

PATH = Path(__file__).resolve().parents[2] / 'REALPROVER/manager/thirdparty/generation_memory.py'
spec = importlib.util.spec_from_file_location('generation_memory', PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def group(done=True, consumed=True, pending=None):
    return NS(assembled_seq_group=NS(is_finished=lambda: done), output_produced=consumed,
              to_be_finished={} if pending is None else pending)


class MemoryTests(unittest.TestCase):
    def test_consumed_group_all_child_keys_removed(self):
        g = group()
        llm = NS(llm_engine=NS(seq_id_to_seq_group={str(i): g for i in range(64)},
                              has_unfinished_requests=lambda: False))
        self.assertEqual(module.release_completed_parallel_groups(llm), 64)
        self.assertEqual(llm.llm_engine.seq_id_to_seq_group, {})

    def test_live_unconsumed_and_incomplete_groups_survive(self):
        groups = {'done': group(), 'live': group(done=False),
                  'unconsumed': group(consumed=False), 'pending': group(pending={'x': 1})}
        llm = NS(llm_engine=NS(seq_id_to_seq_group=groups, has_unfinished_requests=lambda: False))
        self.assertEqual(module.release_completed_parallel_groups(llm), 1)
        self.assertEqual(set(groups), {'live', 'unconsumed', 'pending'})

    def test_no_cleanup_while_engine_has_unfinished_requests(self):
        llm = NS(llm_engine=NS(seq_id_to_seq_group={'x': group()}, has_unfinished_requests=lambda: True))
        self.assertEqual(module.release_completed_parallel_groups(llm), 0)
        self.assertEqual(len(llm.llm_engine.seq_id_to_seq_group), 1)

    def test_other_engine_versions_noop(self):
        self.assertEqual(module.release_completed_parallel_groups(NS()), 0)

    def test_output_identity_and_parameters_unchanged(self):
        result = [NS(text='exact h', score=-.25)]
        seen = []
        llm = NS(generate=lambda *a, **kw: (seen.append((a, kw)), result)[1],
                 llm_engine=NS(seq_id_to_seq_group={'x': group()}, has_unfinished_requests=lambda: False))
        actual = module.generate_with_cleanup(llm, ['prompt'], {'n': 64}, use_tqdm=False)
        self.assertIs(actual, result)
        self.assertEqual(seen, [((['prompt'], {'n': 64}), {'use_tqdm': False})])

    def test_repeated_requests_have_bounded_ownership(self):
        engine = NS(seq_id_to_seq_group={}, has_unfinished_requests=lambda: False)
        def generate():
            g = group()
            engine.seq_id_to_seq_group.update({str(i): g for i in range(64)})
            return ['rfl']
        llm = NS(llm_engine=engine, generate=generate)
        for _ in range(2000):
            self.assertEqual(module.generate_with_cleanup(llm), ['rfl'])
            self.assertEqual(len(engine.seq_id_to_seq_group), 0)


if __name__ == '__main__':
    unittest.main()
