"""Focused stdlib checks for the frozen runtime protocol, without GPU imports."""
import ast
import importlib.util
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('fusion_worker_protocol', HERE / 'worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class RuntimeProtocol(unittest.TestCase):
    def test_frozen_sources_and_official_support(self):
        hashes = worker.source_gate()
        self.assertEqual(hashes, {**worker.SOURCE_HASHES, **worker.SUPPORT_HASHES})
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):
                worker.source_gate(Path(temp), worker.V2)
            for name in worker.SOURCE_HASHES:
                (Path(temp) / name).write_bytes((worker.PROBE / name).read_bytes())
            (Path(temp) / 'candidate.py').write_bytes(b'changed source\n')
            with self.assertRaisesRegex(ValueError, 'Frozen source/support mismatch: candidate.py'):
                worker.source_gate(Path(temp), worker.V2)

    def test_three_targets_have_two_distinct_required_preparation_families(self):
        self.assertEqual(worker.TARGET_PAIRS, [(16, 1), (16, 8), (32, 8)])
        for boundary, scored in zip(worker.BOUNDARIES, worker.TARGETS):
            self.assertEqual((boundary['batch'], boundary['q_len']),
                             (scored['batch'], scored['q_len']))
            self.assertEqual((boundary['page'], boundary['kv_scale']), (64, 1/64))
            self.assertEqual((scored['page'], scored['kv_scale']), (1, 1.))
            self.assertEqual(scored['ctx'], 80000)
            self.assertEqual(scored['spread'], 'agentic')
            self.assertEqual(scored['dist'], 'coherent')

    def test_boundary_lengths_change_live_values_keep_capacity_and_cover_append(self):
        for batch, length in worker.TARGET_PAIRS:
            a, b = [worker.boundary_lengths(batch, version) for version in [0, 1]]
            self.assertEqual(len(a), batch)
            self.assertEqual(b, list(reversed(a)))
            self.assertNotEqual(a, b)
            self.assertEqual(max(a), 129)
            self.assertEqual(set(a), {0, 1, 63, 64, 65, 127, 128, 129})
            self.assertTrue(any(seq // 64 != (seq + length - 1) // 64 for seq in a)
                            if length == 8 else True)

    def test_balanced_pairs_share_input_version(self):
        for index in range(0, worker.WARMUP_SAMPLES + worker.MEASURED_SAMPLES, 2):
            a, b = worker.order_and_version(index), worker.order_and_version(index + 1)
            self.assertEqual(a[0], list(reversed(b[0])))
            self.assertEqual(a[1], b[1])
        versions = [worker.order_and_version(i)[1]
                    for i in range(worker.WARMUP_SAMPLES, worker.WARMUP_SAMPLES + worker.MEASURED_SAMPLES)]
        self.assertEqual(set(versions), {0, 1})

    def test_summary_rejects_incomplete_nonfinite_and_nonpositive_samples(self):
        rows = [dict(raw_us=dict(baseline=200., candidate=195.))
                for _ in range(worker.MEASURED_SAMPLES)]
        result = worker.timing_summary(rows)
        self.assertAlmostEqual(result['median_reverse_order_pair_ratio'], 200/195)
        self.assertEqual(len(result['reverse_order_pair_geomeans']), 6)
        for bad in [rows[:-1], [],
                    [dict(raw_us=dict(baseline=float('nan'), candidate=195.))] * 12,
                    [dict(raw_us=dict(baseline=200., candidate=0.))] * 12]:
            with self.assertRaises(ValueError):
                worker.timing_summary(bad)

    def test_no_profiler_no_candidate_patching_and_call_entrypoints(self):
        tree = ast.parse((HERE / 'worker.py').read_text())
        assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)]
        for node in assignments:
            for target in node.targets:
                self.assertFalse(isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id in {'candidate', 'baseline'})
        self.assertNotIn('t.profiler', (HERE / 'worker.py').read_text())
        self.assertIn('functions = dict(baseline=baseline.kernel_fn, candidate=candidate.kernel_fn)',
                      (HERE / 'worker.py').read_text())
        self.assertIn('if args.correctness_only:', (HERE / 'worker.py').read_text())
        compile(tree, str(HERE / 'worker.py'), 'exec')


if __name__ == '__main__':
    unittest.main(verbosity=2)
