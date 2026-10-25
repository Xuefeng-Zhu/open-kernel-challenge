"""Offline boundary, frequency-accounting and paired-fixture runtime gates.

These tests import no GPU packages and make no native or performance claim.
"""
import argparse
import ast
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('common_pv_runtime_protocol', HERE/'worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def measured_rows(candidates=(97., 97.), baselines=(100., 100.)):
    result = []
    for index in range(worker.WARMUP_SAMPLES, worker.WARMUP_SAMPLES+worker.MEASURED_SAMPLES):
        order, version = worker.order_and_version(index)
        result.append(dict(index=index, order=order, input_version=version,
                           raw_us=dict(baseline=baselines[version], candidate=candidates[version])))
    return result


def families(target=None, control=None):
    return [dict(case=worker.TARGET, samples=measured_rows() if target is None else target),
            dict(case=worker.CONTROL, samples=measured_rows((100., 100.)) if control is None else control)]


def independent_tile_model(lengths, hkv, splits, placement):
    """Enumerate per-tile maxima, independent of the worker's count formula."""
    result = []
    for seq in lengths:
        tiles = (seq+63)//64
        if not seq:
            leader = None
        elif placement == 'first':
            leader = 0
        elif placement == 'middle':
            leader = ((tiles-1)//2)*64
        elif placement == 'last':
            leader = (tiles-1)*64
        else:
            raise ValueError('Invalid test placement')
        for _ in range(hkv):
            for split in range(splits):
                previous = -math.inf
                total = fast = slow = 0
                for start in range(split*8192, min((split+1)*8192, seq), 64):
                    tile_maximum = 1. if leader is not None and start <= leader < min(start+64, seq) else 0.
                    maximum = max(previous, tile_maximum)
                    # alpha==1 exactly when a finite previous maximum persists.
                    if previous == maximum:
                        fast += 1
                    else:
                        slow += 1
                    previous = maximum
                    total += 1
                result.append(dict(total=total, fast=fast, slow=slow))
    return result


class CommonPVRuntimeProtocol(unittest.TestCase):
    def test_exact_sources_diagnostic_and_original_oracle_are_hash_bound(self):
        observed = worker.source_gate()
        self.assertEqual(worker.SOURCE_HASHES, {
            'baseline.py':'3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8',
            'candidate.py':'14a8fa2e3d7b64fa299fe69c8da6569501c069425a62e07cc5f11b1e9b15f3e5'})
        for name, expected in {**worker.SOURCE_HASHES, **worker.SUPPORT_HASHES}.items():
            self.assertEqual(observed[name], expected)
        self.assertEqual(observed['diagnostic.py'], 'ee5488bd8c6b8ee1e4bb9b6fccc1184215949136008dd15df11859b876f76c60')
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            (source/'runtime').mkdir()
            for name in worker.SOURCE_HASHES:
                (source/name).write_bytes((worker.PROBE/name).read_bytes())
            (source/'runtime/diagnostic.py').write_bytes((HERE/'diagnostic.py').read_bytes())
            (source/'candidate.py').write_bytes(b'changed source\n')
            with self.assertRaises(ValueError):
                worker.source_gate(source, worker.V2)
            (source/'candidate.py').write_bytes((worker.PROBE/'candidate.py').read_bytes())
            altered_diagnostic = source/'runtime/diagnostic.py'
            altered_diagnostic.write_bytes(b'changed diagnostic\n')
            with self.assertRaises(ValueError):
                worker.source_gate(source, worker.V2, altered_diagnostic)

    def test_six_fixed_cases_and_fresh_seeds_cover_target_dp_and_control(self):
        self.assertEqual(worker.SEEDS, [135731, 135732])
        self.assertEqual(len(worker.CASES), 6)
        self.assertEqual(len({case['name'] for case in worker.CASES}), 6)
        self.assertEqual(worker.NAMES, ['baseline', 'candidate'])
        self.assertEqual(worker.TIMED_CASES, [worker.TARGET, worker.CONTROL])
        self.assertEqual((worker.TARGET['batch'], worker.TARGET['q_len'], worker.TARGET['ctx'], worker.TARGET['heads']), (32, 1, 80000, 'tp8'))
        self.assertEqual((worker.TARGET['page'], worker.TARGET['kv_scale'], worker.TARGET['spread'], worker.TARGET['dist']), (1, 1., 'agentic', 'coherent'))
        self.assertEqual((worker.CONTROL['batch'], worker.CONTROL['q_len'], worker.CONTROL['ctx'], worker.CONTROL['layer']), (16, 8, 80000, 'full'))
        boundaries = [case for case in worker.CASES if case is not worker.TARGET and case is not worker.CONTROL]
        self.assertEqual(len(boundaries), 4)
        self.assertEqual(sum(case['heads']=='dp' for case in boundaries), 1)
        for case in boundaries:
            self.assertEqual((case['batch'], case['q_len'], case['ctx'], case['page'], case['kv_scale'], case['dist']),
                             (32, 1, 8193, 64, 1/64, 'soft'))
        a, b = worker.boundary_lengths(0), worker.boundary_lengths(1)
        self.assertEqual(a, [0,1,63,64,65,8191,8192,8193]*4)
        self.assertEqual(b, list(reversed(a)))
        self.assertNotEqual(a, b)

    def test_leader_positions_are_inside_the_declared_first_middle_last_tile(self):
        for seq in [0,1,63,64,65,127,128,129,8191,8192,8193,16385]:
            positions = {force:worker.leader_token(seq, force) for force in ['first','middle','last']}
            if not seq:
                self.assertEqual(set(positions.values()), {None})
                continue
            self.assertEqual(positions['first'], 0)
            self.assertEqual(positions['middle'], (((seq+63)//64-1)//2)*64)
            self.assertEqual(positions['last'], ((seq+63)//64-1)*64)
            for position in positions.values():
                self.assertEqual(position%64, 0)
                self.assertLess(position, seq)

    def test_forced_stats_match_independent_running_maxima_tile_enumeration(self):
        lengths = [0,1,63,64,65,127,128,129,8191,8192,8193,16385]
        for hkv, splits in [(2,3),(16,3)]:
            for force in ['first','middle','last']:
                expected = independent_tile_model(lengths, hkv, splits, force)
                self.assertEqual(worker.expected_tile_rows(lengths, hkv, splits, force), expected)
                actual = [[row['total'],row['fast'],row['slow']] for row in expected]
                checked = worker.stats_check(actual, lengths, hkv, splits, force)
                self.assertTrue(checked['passed'])
                self.assertEqual(checked['total'], sum(row['total'] for row in expected))
                self.assertEqual(checked['fast']+checked['slow'], checked['total'])
                self.assertEqual(checked['expected_total_by_cta'], [row['total'] for row in expected])
                self.assertEqual(checked['expected_slow_by_cta'], [row['slow'] for row in expected])
        # A late leader in chunk0 adds a slow tile, but chunk1's first tile
        # was already slow and must not be counted twice.
        last = worker.expected_tile_rows([8193], 1, 2, 'last')
        self.assertEqual(last, [dict(total=128, fast=127, slow=1), dict(total=1, fast=0, slow=1)])

    def test_stats_check_rejects_wrong_cta_shape_inconsistent_or_forced_counts(self):
        lengths, hkv, splits = [65], 1, 1
        valid = [[2,1,1]]
        self.assertTrue(worker.stats_check(valid, lengths, hkv, splits, 'first')['passed'])
        invalid = [[], [[2,1]], [[2,1,1],[0,0,0]], [[2,-1,3]], [[2,0,1]], [[3,2,1]], [[2.,1,1]], [[True,0,1]], [[2,0,2]]]
        for actual in invalid:
            with self.subTest(actual=actual):
                with self.assertRaises(ValueError):
                    worker.stats_check(actual, lengths, hkv, splits, 'first')

    def test_unforced_nonempty_chunk_cannot_report_every_tile_fast(self):
        # A nonempty chunk starts with m_i=-inf, so its first historical tile
        # cannot have alpha==1. This invariant needs no distribution assumption.
        self.assertTrue(worker.stats_check([[2,1,1]], [65], 1, 1)['passed'])
        self.assertTrue(worker.stats_check([[2,0,2]], [65], 1, 1)['passed'])
        self.assertTrue(worker.stats_check([[0,0,0]], [0], 1, 1)['passed'])
        with self.assertRaises(ValueError):
            worker.stats_check([[2,2,0]], [65], 1, 1)
        with self.assertRaises(ValueError):
            worker.stats_check([[1,1,0]], [1], 1, 1)

    def test_diagnostic_immutability_exempts_only_designated_stats_output(self):
        values = {name:object() for name in ['qkv','seq_lens','page_table','alpha_stats','alpha_stats_backup','k_cache','config']}
        original_keys = set(values)
        plain = worker.without_stats(values)
        self.assertEqual(set(plain), original_keys-{'alpha_stats'})
        self.assertEqual(set(values), original_keys)
        for key in plain:
            self.assertIs(plain[key], values[key])
        # A normal uninstrumented dictionary keeps every field.
        self.assertEqual(worker.without_stats(plain), plain)

    def test_paired_orders_use_one_fixture_per_pair_and_three_pairs_per_fixture(self):
        for index in range(0, worker.WARMUP_SAMPLES+worker.MEASURED_SAMPLES, 2):
            left, right = worker.order_and_version(index), worker.order_and_version(index+1)
            self.assertEqual(left[0], list(reversed(right[0])))
            self.assertEqual(left[1], right[1])
        summary = worker.timing_summary(measured_rows())
        self.assertEqual(summary['positive_reverse_order_pairs'], 6)
        self.assertAlmostEqual(summary['median_reverse_order_pair_ratio'], 100/97)
        for version in ['0','1']:
            fixture = summary['per_fixture'][version]
            self.assertEqual(len(fixture['pair_ratios']), 3)
            self.assertEqual(fixture['positive_pairs'], 3)
            self.assertAlmostEqual(fixture['latency_reduction_percent'], 3.)

    def test_gate_requires_each_fixture_and_all_pairs_instead_of_pooled_gain(self):
        self.assertTrue(worker.benefit_gate(families())['admitted'])
        self.assertFalse(worker.benefit_gate(families(target=measured_rows((90.,100.))))['admitted'])
        self.assertFalse(worker.benefit_gate(families(control=measured_rows((97.,103.))))['admitted'])
        unfavorable = measured_rows()
        for row in unfavorable[:2]:
            row['raw_us']['candidate'] = 101.
        self.assertFalse(worker.benefit_gate(families(target=unfavorable))['admitted'])
        self.assertFalse(worker.benefit_gate(families(target=measured_rows((99.,99.))))['admitted'])
        with self.assertRaises(ValueError):
            worker.benefit_gate(families()[:1])

    def test_timing_rejects_nonfinite_missing_wrong_order_or_fixture_records(self):
        for samples in [[], measured_rows()[:-1], measured_rows((0.,97.)), measured_rows((float('inf'),97.)), measured_rows((float('nan'),97.))]:
            with self.assertRaises(ValueError):
                worker.timing_summary(samples)
        bad_order = measured_rows()
        bad_order[1]['order'] = bad_order[0]['order']
        with self.assertRaises(ValueError):
            worker.timing_summary(bad_order)
        bad_version = measured_rows()
        bad_version[1]['input_version'] = 1-bad_version[0]['input_version']
        with self.assertRaises(ValueError):
            worker.timing_summary(bad_version)

    def test_loaded_algorithms_and_oracle_are_not_mutated_or_tolerance_relaxed(self):
        source = (HERE/'worker.py').read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    self.assertFalse(isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                                     and target.value.id in {'candidate','baseline','diagnostic','ref'})
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr=='timed':
                self.assertFalse(any(isinstance(name, ast.Name) and 'diag' in name.id.lower() for name in ast.walk(node.func.value)))
        for forbidden in ['CACHE_TOL =', 'REL_L2_TOL =', 't.profiler', 'num_splits=']:
            self.assertNotIn(forbidden, source)
        wanted = ast.dump(ast.parse('ref.check_implementation(self.work, raw)', mode='eval').body, include_attributes=False)
        self.assertTrue(any(isinstance(node, ast.Call) and ast.dump(node, include_attributes=False)==wanted
                            for node in ast.walk(tree)))
        self.assertIn('diagnostic.kernel_fn', source)
        self.assertIn('helpers.timed', source)
        compile(tree, str(HERE/'worker.py'), 'exec')

    def test_source_failure_writes_bounded_partial_report_before_gpu_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'result.json'
            with patch.object(worker, 'source_gate', side_effect=ValueError('test source mismatch')):
                code = worker.main(argparse.Namespace(output=output, deadline=160.))
            self.assertEqual(code, 1)
            result = json.loads(output.read_text())
            self.assertFalse(result['passed'])
            self.assertFalse(result['correctness_complete'])
            self.assertFalse(result['diagnostic_complete'])
            self.assertFalse(result['timing_complete'])
            self.assertFalse(result['official_check_admitted'])
            self.assertEqual(result['timings'], [])
            self.assertLess(output.stat().st_size, 500000)
            self.assertFalse(output.with_suffix('.tmp').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
