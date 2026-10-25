"""Offline protocol/benefit tests; importing this code never imports GPU packages."""
import ast
import importlib.util
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('composition_worker_protocol', HERE/'worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def timing_rows(baseline=100., candidate=95.):
    return [dict(raw_us=dict(baseline=baseline, candidate=candidate))
            for _ in range(worker.MEASURED_SAMPLES)]


def families(target=None, control=None):
    return [dict(case=worker.TARGET, samples=timing_rows() if target is None else target),
            dict(case=worker.CONTROL, samples=timing_rows(100., 100.) if control is None else control)]


class RuntimeProtocol(unittest.TestCase):
    def test_frozen_sources_and_original_official_support(self):
        self.assertEqual(worker.source_gate(), {**worker.SOURCE_HASHES, **worker.SUPPORT_HASHES})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            with self.assertRaises(FileNotFoundError):
                worker.source_gate(path, worker.V2)
            for name in worker.SOURCE_HASHES:
                (path/name).write_bytes((worker.PROBE/name).read_bytes())
            (path/'candidate.py').write_bytes(b'changed candidate\n')
            with self.assertRaisesRegex(ValueError, 'Frozen source/support mismatch: candidate.py'):
                worker.source_gate(path, worker.V2)

    def test_five_distinct_cases_and_fresh_replication_seeds(self):
        self.assertEqual(worker.SEEDS, [135711, 135712])
        self.assertNotEqual(worker.SEEDS, [135701, 135702])
        self.assertEqual(len(worker.CASES), 5)
        self.assertEqual(len({x['name'] for x in worker.CASES}), 5)
        self.assertEqual(worker.TIMED_CASES, [worker.TARGET, worker.CONTROL])
        self.assertEqual((worker.TARGET['batch'], worker.TARGET['q_len'], worker.TARGET['layer']), (32, 1, 'full'))
        self.assertEqual((worker.TARGET['ctx'], worker.TARGET['page'], worker.TARGET['kv_scale']), (80000, 1, 1.))
        self.assertEqual((worker.TARGET['spread'], worker.TARGET['dist']), ('agentic', 'coherent'))
        self.assertEqual((worker.CONTROL['batch'], worker.CONTROL['q_len'], worker.CONTROL['layer']), (16, 8, 'full'))
        self.assertEqual((worker.FALLBACK_BOUNDARY['batch'], worker.FALLBACK_BOUNDARY['boundary']), (4, 'fallback'))
        self.assertEqual((worker.DP_SMOKE['heads'], worker.DP_SMOKE['ctx'], worker.DP_SMOKE['boundary']), ('dp', 65, 'dp'))
        for case in [worker.TARGET_BOUNDARY, worker.FALLBACK_BOUNDARY, worker.DP_SMOKE]:
            self.assertEqual((case['page'], case['kv_scale'], case['q_len']), (64, 1/64, 1))

    def test_page_tile_chunk_boundaries_and_dp_single_split_input_capacity(self):
        expected = {
            'target': (worker.TARGET_BOUNDARY, {0, 1, 63, 64, 65, 8191, 8192, 8193}),
            'fallback': (worker.FALLBACK_BOUNDARY, {0, 63, 64, 8193}),
            'dp': (worker.DP_SMOKE, {0, 63, 64, 65}),
        }
        for family, (case, lengths) in expected.items():
            a, b = [worker.boundary_lengths(case['batch'], version, family) for version in [0, 1]]
            self.assertEqual(len(a), case['batch'])
            self.assertEqual(set(a), lengths)
            self.assertEqual(b, list(reversed(a)))
            self.assertNotEqual(a, b)
            self.assertEqual(min(a), 0)
            self.assertLessEqual(max(a), case['ctx'])
        # These are source-contract capacities, not native launch/occupancy evidence.
        self.assertEqual((worker.DP_SMOKE['ctx']+1+63)//64*64, 128)
        self.assertEqual((128+8191)//8192, 1)
        with self.assertRaises(ValueError):
            worker.boundary_lengths(16, 0)

    def test_reverse_order_pairs_share_fixture_version(self):
        for i in range(0, worker.WARMUP_SAMPLES+worker.MEASURED_SAMPLES, 2):
            left, right = worker.order_and_version(i), worker.order_and_version(i+1)
            self.assertEqual(left[0], list(reversed(right[0])))
            self.assertEqual(left[1], right[1])
        measured_versions = [worker.order_and_version(i)[1]
                             for i in range(worker.WARMUP_SAMPLES, worker.WARMUP_SAMPLES+worker.MEASURED_SAMPLES)]
        self.assertEqual(set(measured_versions), {0, 1})

    def test_timing_summary_rejects_incomplete_and_invalid_samples(self):
        result = worker.timing_summary(timing_rows())
        self.assertAlmostEqual(result['median_reverse_order_pair_ratio'], 100/95)
        self.assertAlmostEqual(result['median_paired_latency_reduction_percent'], 5.)
        self.assertEqual(result['positive_reverse_order_pairs'], 6)
        for rows in [[], timing_rows()[:-1], timing_rows(float('nan'), 95.),
                     timing_rows(100., float('inf')), timing_rows(100., 0.)]:
            with self.assertRaises(ValueError):
                worker.timing_summary(rows)

    def test_new_replication_gate_does_not_relabel_the_historical_goal(self):
        result = worker.benefit_gate(families())
        self.assertTrue(result['admitted'])
        self.assertIn('10% goal was not met', result['historical_A2_15_goal'])
        self.assertFalse(worker.benefit_gate(families(target=timing_rows(100., 97.)))['admitted'])
        unfavorable = timing_rows()
        unfavorable[0] = unfavorable[1] = dict(raw_us=dict(baseline=100., candidate=101.))
        result = worker.benefit_gate(families(target=unfavorable))
        self.assertGreater(result['target']['median_paired_latency_reduction_percent'], 4.)
        self.assertEqual(result['target']['positive_reverse_order_pairs'], 5)
        self.assertFalse(result['admitted'])
        self.assertFalse(worker.benefit_gate(families(control=timing_rows(100., 106.)))['admitted'])
        with self.assertRaises(ValueError):
            worker.benefit_gate(families()[:1])

    def test_candidate_entrypoints_and_original_oracle_no_patching_or_profile(self):
        source = (HERE/'worker.py').read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                self.assertFalse(any(isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                                     and t.value.id in {'candidate', 'baseline'} for t in node.targets))
        self.assertIn('functions = dict(baseline=baseline.kernel_fn, candidate=candidate.kernel_fn)', source)
        self.assertIn('ref.check_implementation(self.work, raw)', source)
        self.assertIn("if case['boundary'] == 'target':", source)
        self.assertIn("if args.correctness_only:", source)
        self.assertNotIn('t.profiler', source)
        self.assertNotIn('num_splits=', source)
        compile(tree, str(HERE/'worker.py'), 'exec')


if __name__ == '__main__':
    unittest.main(verbosity=2)
