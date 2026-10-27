"""CPU protocol gates; no GPU packages or services are required."""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('dp_multichunk_worker_protocol', HERE/'worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class DPProtocol(unittest.TestCase):
    def test_frozen_source_and_support_gate_rejects_changes(self):
        self.assertEqual(worker.source_gate(), {**worker.SOURCE_HASHES, **worker.SUPPORT_HASHES})
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            for name in worker.SOURCE_HASHES:
                (source/name).write_bytes((worker.SOURCE_ROOT/name).read_bytes())
            (source/'candidate.py').write_bytes(b'changed\n')
            with self.assertRaisesRegex(ValueError, 'Frozen source/support mismatch: candidate.py'):
                worker.source_gate(source, worker.V2)

    def test_exact_cases_and_fresh_boundary_mutations(self):
        self.assertEqual(worker.SEEDS, [135721, 135722])
        self.assertEqual([(x['ctx'], x['page'], x['kv_scale'], x['dist']) for x in worker.CASES],
                         [(8193, 64, 1/64, 'soft'), (16385, 1, 1., 'coherent')])
        for case in worker.CASES:
            self.assertEqual((case['batch'], case['q_len'], case['heads'], case['layer']), (32, 1, 'dp', 'full'))
            a = worker.boundary_lengths(case['ctx'], 0)
            b = worker.boundary_lengths(case['ctx'], 1)
            self.assertEqual(len(a), 32)
            self.assertNotEqual(a, b)
            self.assertEqual(a, list(reversed(b)))
            self.assertTrue({0, 1, 63, 64, 65, 8191, 8192, 8193}.issubset(a))
            self.assertEqual(max(a), case['ctx'])
        self.assertTrue({127, 128, 129, 16320, 16321, 16383, 16384, 16385}.issubset(worker.boundary_lengths(16385, 0)))
        for values in [(65, 0), (8193, 2)]:
            with self.assertRaises(ValueError):
                worker.boundary_lengths(*values)

    def test_page_specific_suffix_includes_new_token_at_every_boundary(self):
        # A page64-specific mask would incorrectly discard nearly all page1 history.
        for page in (1, 64):
            for seq in (0, 1, 63, 64, 65, 127, 128, 129, 8191, 8192, 8193, 16383, 16384, 16385):
                count = worker.valid_pages(seq, 1, page)
                self.assertEqual(count, seq // page + 1)
                self.assertLess(seq // page, count)  # new token remains a legal page
                if seq:
                    self.assertLess((seq-1)//page, count)
        self.assertEqual(worker.valid_pages(8193, 1, 1), 8194)
        self.assertEqual(worker.valid_pages(8193, 1, 64), 129)
        for args in [(-1, 1, 64), (1, 0, 64), (1, 1, 16)]:
            with self.assertRaises(ValueError):
                worker.valid_pages(*args)

    def test_multichunk_geometry_and_non_power_of_two_merge_padding(self):
        a, b = [worker.geometry(case) for case in worker.CASES]
        self.assertEqual((a['page_table_shape'], a['splits'], a['merge_bs']), ([32, 129], 2, 2))
        self.assertEqual((b['page_table_shape'], b['splits'], b['merge_bs']), ([32, 16386], 3, 4))
        for geometry in [a, b]:
            n = 32*16*geometry['splits']*16
            self.assertEqual(geometry['partial_shapes'], {'o':[n,64], 'm':[n], 'l':[n]})
            self.assertEqual(geometry['source_derived_partial_bytes'], n*66*4)
            self.assertEqual(geometry['merge_grid'], [32, 128])
            self.assertLess(geometry['cache_bytes_per_tensor']*8, 5*1024**3)
        # The fourth reduction lane is masked by the original SPLITS stride.
        self.assertEqual([lane for lane in range(b['merge_bs']) if lane < b['splits']], [0, 1, 2])
        source = (worker.SOURCE_ROOT/'candidate.py').read_text()
        self.assertIn('if split == SPLITS - 1:', source)
        self.assertIn('split_ok = offs_s < SPLITS', source)

    def test_no_candidate_mutation_no_performance_branch_or_optional_tolerance(self):
        source = (HERE/'worker.py').read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    self.assertFalse(isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                                     and target.value.id in {'candidate', 'baseline', 'ref'})
        self.assertIn('functions = dict(baseline=baseline.kernel_fn, candidate=candidate.kernel_fn)', source)
        self.assertIn('ref.check_implementation(self.work, raw)', source)
        self.assertIn("helpers.restore(self.work, self.inputs[version])", source)
        for forbidden in ['helpers.timed(', 't.profiler', 'benefit_gate', 'num_splits=', 'CACHE_TOL =', 'REL_L2_TOL =']:
            self.assertNotIn(forbidden, source)
        compile(tree, str(HERE/'worker.py'), 'exec')

    def test_graph_pool_and_independent_output_reuse_evidence_is_explicit(self):
        source = (HERE/'worker.py').read_text()
        for required in ["getattr(self.graph, 'pool', None)", 'first.pool != second.pool',
                         'first.out_address != second.out_address', 'active.replay(version, False)',
                         'inactive.check(retained_version, False)',
                         'actual_hardware_overlap_measured=False', 'direct_scratch_pointers_observed=False']:
            self.assertIn(required, source)
        self.assertIn('extra_candidate_replays=2', source)
        self.assertIn('concurrent_original_oracle_calls=2', source)
        # Primary graph initialized A/B by ABA; independent graph initialized
        # A with its own setup oracle before B's concurrent original-oracle.
        # Every oracleFalse
        # replay/check must have that graph's own established expectation.
        expected = [{0, 1}, {0, 1}]
        for active, version, inactive, retained in worker.isolation_schedule():
            self.assertIn(version, expected[active])
            self.assertIn(retained, expected[inactive])
        self.assertEqual(worker.isolation_schedule(), [(0, 1, 1, 1), (1, 0, 0, 1)])
        self.assertLess(source.index('**second.replay(0, True)'),
                        source.index('helpers.restore(first.work, inputs[0])'))
        self.assertIn('independent_setup_oracle_calls=1', source)

    def test_source_failure_returns_bounded_partial_report_before_gpu_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'partial.json'
            with patch.object(worker, 'source_gate', side_effect=ValueError('test mismatch')):
                result = worker.main(argparse.Namespace(output=output, deadline=1))
            self.assertEqual(result, 1)
            report = json.loads(output.read_text())
            self.assertEqual(report['status'], 'incomplete_or_failed')
            self.assertFalse(report['passed'])
            self.assertFalse(report['correctness_complete'])
            self.assertFalse(report['official_check_admitted'])
            self.assertEqual(report['timings'], [])
            self.assertTrue(report['protocol']['correctness_only'])
            self.assertEqual(report['counts'], {})
            self.assertLess(output.stat().st_size, 400000)
            self.assertFalse(output.with_suffix('.tmp').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
