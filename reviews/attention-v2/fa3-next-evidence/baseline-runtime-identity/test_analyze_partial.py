import copy
import gzip
import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('baseline_partial_analyzer', HERE/'analyze_partial.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def fixture():
    label = 'PROFILE_SELECTED_FULL_VERIFY_B16'
    return {'traceEvents': [
        dict(ph='X', cat='user_annotation', name=label, ts=100, dur=100, pid=4, tid=4,
            args={'External id': 16}),
        dict(ph='X', cat='cuda_runtime', name='cudaGraphLaunch', ts=120, dur=10, pid=4, tid=4,
            args={'correlation': 113}),
        dict(ph='X', cat='gpu_user_annotation', name=label, ts=50, dur=100, pid=0, tid=7,
            args={'External id': 16}),
        dict(ph='X', cat='kernel', name='FlashAttnFwdSm90<opaque>', ts=60, dur=80, pid=0, tid=7,
            args={'correlation': 113, 'graph id': 2, 'graph node id': 9, 'stream': 7}),
        dict(ph='X', cat='gpu_memcpy', name='Unrelated restore', ts=150, dur=4, pid=0, tid=7,
            args={'correlation': 122, 'graph id': 0, 'stream': 7}),
    ]}


class OfflineAttributionTests(unittest.TestCase):
    def test_cross_domain_timestamp_offset_does_not_lose_graph(self):
        result = a.attribute(fixture(), 'full_verify_b16')
        self.assertEqual(result['kernel_count'], 1)
        self.assertEqual(result['correlation'], 113)
        self.assertEqual(result['graph_ids'], [2])
        self.assertFalse(result['cpu_gpu_timestamp_containment_valid'])
        self.assertEqual(result['original_cpu_containment_mainloop_count'], 0)

    def test_unrelated_restore_inside_host_range_excluded(self):
        result = a.attribute(fixture(), 'full_verify_b16')
        self.assertEqual(len(result['events']), 1)
        self.assertEqual(result['original_cpu_containment_foreign_correlations'], [122])

    def test_multiple_correlated_mainloops_reported_without_assuming_one(self):
        trace = fixture()
        second = copy.deepcopy(trace['traceEvents'][3])
        second.update(ts=70, dur=10)
        second['args']['graph node id'] = 10
        trace['traceEvents'].append(second)
        result = a.attribute(trace, 'full_verify_b16')
        self.assertEqual(len(result['full_mainloop_symbols']), 2)
        self.assertEqual(result['phase_kernel_counts']['fa3_mainloop'], 2)

    def test_ambiguous_host_launch_rejected(self):
        trace = fixture()
        trace['traceEvents'].append(copy.deepcopy(trace['traceEvents'][1]))
        with self.assertRaisesRegex(ValueError, 'one host graph launch'):
            a.attribute(trace, 'full_verify_b16')

    def test_external_id_mismatch_rejected(self):
        trace = fixture()
        trace['traceEvents'][2]['args']['External id'] = 99
        with self.assertRaisesRegex(ValueError, 'matching GPU annotation'):
            a.attribute(trace, 'full_verify_b16')

    def test_union_does_not_double_count_apparent_overlap(self):
        self.assertEqual(a.interval_union([(0, 10), (8, 15)]), 15)
        events = [dict(name='one', ts=0, dur=10, args={'stream': 7}),
            dict(name='two', ts=8, dur=7, args={'stream': 7})]
        self.assertEqual(a.overlaps(events)[0]['apparent_overlap_us'], 2)

    def test_real_evidence_keeps_partial_status_and_unreached_stages(self):
        result = json.loads((HERE/'result.json').read_text())
        trace = json.loads(gzip.decompress((HERE/'activity-trace.json.gz').read_bytes()))
        before = json.dumps(result, sort_keys=True)
        report = a.analyze(result, trace)
        self.assertEqual(json.dumps(result, sort_keys=True), before)
        self.assertEqual(report['raw_status'], 'incomplete_or_failed')
        self.assertFalse(report['raw_passed'])
        self.assertFalse(report['raw_status_changed'])
        self.assertEqual(report['completion']['initial_oracle_checks'], 2)
        self.assertEqual(report['completion']['checked_preprofile_replays'], 10)
        self.assertEqual(report['completion']['post_activity_oracle_checks'], 0)
        self.assertEqual(report['completion']['after_timing_samples'], 0)
        self.assertEqual(report['completion']['auxiliary_replays'], 0)
        self.assertEqual(report['completion']['hardware_counter_sessions'], 0)
        self.assertAlmostEqual(report['cases']['full_verify_b16']['before_median_us'], 199.5519995689392)
        self.assertAlmostEqual(report['cases']['full_verify_b32']['before_median_us'], 383.55201482772827)
        for case in report['cases'].values():
            self.assertEqual(case['activity']['kernel_count'], 5)
            self.assertEqual(case['activity']['mainloop_shape_tuple_literal'], [64, 128, 64])


if __name__ == '__main__':
    unittest.main()
