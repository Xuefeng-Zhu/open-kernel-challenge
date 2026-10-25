"""Offline protocol guards: no Torch, CUDA, installed FA3 or network imports."""
import ast
import importlib.util
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('unsplit_worker_protocol', HERE/'worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def sample_rows(base, candidate):
    return [dict(raw_us=dict(baseline=base, candidate=candidate)) for _ in range(12)]


def fake_activity(candidate_combine=False, baseline_combine=True):
    rows, labels = [], []
    index = 0
    for case in worker.TARGETS:
        for name in worker.NAMES:
            label = worker.activity_label(case, name)
            labels.append(label)
            start = 100*index
            rows.append(dict(name=label, device_type='CPU', start_us=start, end_us=start+90))
            rows.append(dict(name='void FlashAttnFwdSm90<test>()', device_type='CUDA',
                             start_us=start+10, end_us=start+20))
            rows.append(dict(name='void prepare_varlen_num_blocks_kernel<1>()', device_type='CUDA',
                             start_us=start+22, end_us=start+24))
            if (name == 'baseline' and baseline_combine) or (name == 'candidate' and candidate_combine):
                rows.append(dict(name='void FlashAttnFwdCombine<test>()', device_type='CUDA',
                                 start_us=start+25, end_us=start+29))
            index += 1
    return rows, labels


class RuntimeProtocol(unittest.TestCase):
    def test_source_and_identity_helper_fail_closed(self):
        self.assertEqual(worker.source_gate(), {
            **worker.SOURCE_HASHES, **worker.SUPPORT_HASHES, 'identity.py': worker.IDENTITY_SHA256})
        with tempfile.TemporaryDirectory() as temp:
            probe = Path(temp)
            for name in worker.SOURCE_HASHES:
                (probe/name).write_bytes((worker.PROBE/name).read_bytes())
            (probe/'candidate.py').write_bytes(b'altered source\n')
            with self.assertRaisesRegex(ValueError, 'candidate.py'):
                worker.source_gate(probe, worker.V2)

    def test_cases_cover_scored_boundary_and_unmutated_edge_planting(self):
        self.assertEqual(worker.TARGET_PAIRS, [(16, 8), (32, 8)])
        self.assertEqual(len(worker.TARGETS + worker.BOUNDARIES + worker.EDGES), 6)
        for scored, boundary, edge in zip(worker.TARGETS, worker.BOUNDARIES, worker.EDGES):
            self.assertEqual((scored['ctx'], scored['page'], scored['kv_scale']), (80000, 1, 1.))
            self.assertEqual((boundary['ctx'], boundary['page'], boundary['kv_scale']), (2049, 64, 1/64))
            self.assertEqual((edge['ctx'], edge['spread'], edge['dist'], edge['boundary']),
                             (2048, 'fixed', 'edges', False))
        self.assertEqual(worker.CONTROL['layer'], 'full')
        self.assertEqual((worker.CONTROL['batch'], worker.CONTROL['q_len']), (16, 8))

    def test_boundary_lengths_change_live_values_and_cross_page_window_edges(self):
        for batch, _ in worker.TARGET_PAIRS:
            a, b = [worker.boundary_lengths(batch, version) for version in [0, 1]]
            self.assertEqual(len(a), batch)
            self.assertEqual(set(a), {0, 1, 63, 64, 65, 2047, 2048, 2049})
            self.assertEqual(b, a[::-1])
            self.assertNotEqual(a, b)
            self.assertTrue(any(value//64 != (value+7)//64 for value in a))

    def test_balanced_reverse_pairs_use_the_same_fixture_version(self):
        for index in range(0, worker.WARMUP_SAMPLES+worker.MEASURED_SAMPLES, 2):
            first, second = worker.order_and_version(index), worker.order_and_version(index+1)
            self.assertEqual(first[0], second[0][::-1])
            self.assertEqual(first[1], second[1])

    def test_activity_gate_requires_actual_combine_elimination_and_keeps_prepare(self):
        rows, labels = fake_activity()
        good = worker.activity_gate(rows, labels)
        self.assertTrue(good['passed'])
        self.assertTrue(all(row['candidate_counts']['schedule_prepare'] == 1 for row in good['mechanism']))
        self.assertTrue(all(not row['scheduler_prepare_removed'] for row in good['mechanism']))
        self.assertFalse(worker.activity_gate(*fake_activity(candidate_combine=True))['passed'])
        self.assertFalse(worker.activity_gate(*fake_activity(baseline_combine=False))['passed'])

    def test_activity_gate_fails_unknown_attribution_and_exact_transport_overflow(self):
        rows, labels = fake_activity()
        rows.append(dict(name='outside', device_type='CUDA', start_us=10000, end_us=10001))
        self.assertFalse(worker.activity_gate(rows, labels)['passed'])
        rows[0]['name'] = 'x'*(worker.ACTIVITY_EVENT_BYTES+1)
        with self.assertRaisesRegex(ValueError, '128KiB'):
            worker.activity_gate(rows, labels)

    def test_materiality_gate_rejects_tiny_savings_regression_and_missing_control(self):
        good = [dict(case=case, samples=sample_rows(20, 18)) for case in worker.TARGETS]
        good += [dict(case=worker.CONTROL, samples=sample_rows(200, 209))]
        self.assertTrue(worker.benefit_gate(good)['admitted'])
        tiny = [dict(case=case, samples=sample_rows(20, 18.1)) for case in worker.TARGETS]
        self.assertFalse(worker.benefit_gate(tiny+good[-1:])['admitted'])
        self.assertFalse(worker.benefit_gate(good[:-1]+[
            dict(case=worker.CONTROL, samples=sample_rows(200, 211))])['admitted'])
        with self.assertRaises(ValueError):
            worker.benefit_gate(good[:-1])

    def test_timing_rejects_incomplete_and_nonfinite_samples(self):
        summary = worker.timing_summary(sample_rows(20, 18))
        self.assertEqual(summary['median_reverse_order_pair_mean_saving_us'], 2)
        self.assertEqual(summary['positive_reverse_order_pairs'], 6)
        for rows in [[], sample_rows(20, 18)[:-1], sample_rows(float('nan'), 18), sample_rows(20, 0)]:
            with self.assertRaises(ValueError):
                worker.timing_summary(rows)

    def test_installed_api_gate_precedes_any_native_candidate_call_and_no_patch(self):
        source = (HERE/'worker.py').read_text()
        self.assertLess(source.index('identity.inspect_fa3('), source.index('if not t.cuda.is_available()'))
        self.assertLess(source.index("if not report['fa3_identity']['admitted']:"), source.index('functions = dict('))
        self.assertLess(source.index("if not report['fa3_identity']['exact_observed_extensions_verified']:"),
                        source.index('if not t.cuda.is_available()'))
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    self.assertFalse(isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                                     and target.value.id in {'baseline', 'candidate'})
        self.assertEqual(source.count('with t.profiler.profile('), 1)
        self.assertNotRegex(source.lower(), r'\bncu\b')
        compile(tree, str(HERE/'worker.py'), 'exec')


if __name__ == '__main__':
    unittest.main(verbosity=2)
