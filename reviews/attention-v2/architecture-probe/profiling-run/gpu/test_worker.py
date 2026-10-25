"""Local activity-protocol/parser guards; no torch, GPU, network or candidate import."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('activity_worker', HERE/'worker.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


def evt(name, cat, ts, dur, args=None):
    return dict(name=name, cat=cat, ph='X', ts=ts, dur=dur, pid=0, tid=7,
                args=args or {'stream': 7, 'correlation': 123, 'graph id': 4})


def trace():
    return {'traceEvents': [
        evt('RESTORE_EVICT_FA3', 'user_annotation', 0, 80),
        evt('copy', 'gpu_memcpy', 10, 20),
        evt(w.RANGES['fa3'], 'user_annotation', 100, 100),
        evt('cudaGraphLaunch', 'cuda_runtime', 110, 5),
        evt('_qk_rms_norm_fwd_kernel', 'kernel', 120, 15),
        evt('ConvertToFloat8E4M3fnOp', 'kernel', 120, 1),
        evt('store_kvcache_kernel', 'kernel', 121, 1),
        evt('prepare_varlen_num_blocks_kernel', 'kernel', 122, 1),
        evt('void FlashAttnFwdSm90<...>', 'kernel', 130, 40),
        evt('FlashAttnFwdCombine', 'kernel', 170, 5),
        evt('norm_gate_kernel', 'kernel', 174, 1),
        evt(w.RANGES['bf16_custom'], 'user_annotation', 300, 100),
        evt('cudaGraphLaunch', 'cuda_runtime', 310, 5),
        evt('[CUDA memset]', 'gpu_memset', 315, 5),
        evt('_fused', 'kernel', 325, 60)]}


class ProtocolTests(unittest.TestCase):
    def test_frozen_case_sources_and_artifact_bound(self):
        self.assertEqual(w.CASE, dict(batch=32, q_len=8, ctx=80000, heads='tp8',
            layer='full', page=1, kv_scale=1., spread='agentic', dist='coherent', seed=135701))
        base = HERE.parents[1]
        actual = {'candidate.py': base/'probe/candidate.py',
            'candidatecustomhelper.py': base/'paired/candidatecustomhelper.py',
            'modal_worker.py': base.parent/'modal_worker.py',
            'official/reference.py': base.parent/'official/reference.py',
            'official/checks.py': base.parent/'official/checks.py'}
        for name, path in actual.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), w.SOURCES[name], name)
        self.assertLessEqual(w.MAX_TRACE+w.MAX_SUMMARY, 8*1024*1024)

    def test_one_activity_session_no_benchmark_or_direct_counter_cli(self):
        tree = ast.parse((HERE/'worker.py').read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        profiles = [n for n in calls if isinstance(n.func, ast.Attribute)
                    and ast.unparse(n.func) == 't.profiler.profile']
        self.assertEqual(len(profiles), 1)
        kw = {k.arg: ast.unparse(k.value) for k in profiles[0].keywords}
        self.assertEqual(kw['activities'], '[t.profiler.ProfilerActivity.CPU, t.profiler.ProfilerActivity.CUDA]')
        for key in ['record_shapes', 'profile_memory', 'with_stack']:
            self.assertEqual(kw[key], 'False')
        text = ast.unparse(tree)
        self.assertNotIn('h.timed(', text)
        self.assertNotIn('t.cuda.Event(', text)
        literal_strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        self.assertNotIn('ncu', literal_strings)
        self.assertNotIn('nsys', literal_strings)
        self.assertNotIn('import modal', text)

    def test_profile_range_has_one_replay_and_fences(self):
        tree = ast.parse((HERE/'worker.py').read_text())
        profile_with = next(n for n in ast.walk(tree) if isinstance(n, ast.With)
            and any(isinstance(i.context_expr, ast.Call) and ast.unparse(i.context_expr.func) == 't.profiler.profile'
                    for i in n.items))
        calls = [ast.unparse(n.func) for n in ast.walk(profile_with) if isinstance(n, ast.Call)]
        self.assertEqual(calls.count("r['graph'].replay"), 1)
        self.assertEqual(calls.count('t.cuda.synchronize'), 2)
        self.assertEqual(calls.count('t.cuda.nvtx.range_push'), 1)
        self.assertEqual(calls.count('t.cuda.nvtx.range_pop'), 1)
        self.assertNotIn('h.capture', calls)
        self.assertNotIn('checked', calls)

    def test_union_does_not_sum_overlap(self):
        self.assertEqual(w.interval_union([(0, 10), (5, 20), (30, 40)]), 30)
        report = w.compact_trace(trace())
        fa = report['arms']['fa3']
        self.assertEqual(fa['gpu_activity_union_us'], 55)
        self.assertEqual(fa['gpu_activity_duration_sum_us'], 64)
        self.assertEqual(fa['gpu_activity_span_us'], 55)
        self.assertEqual(fa['events'][1]['args']['correlation'], 123)
        self.assertFalse(any(e['name'] == 'copy' for e in fa['events']))
        self.assertFalse(report['exact_cta_trace'])
        self.assertFalse(report['hardware_counters_collected'])

    def test_missing_or_ambiguous_profile_fails_closed(self):
        empty = trace();empty['traceEvents'] = [e for e in empty['traceEvents'] if e['name'] != '_fused']
        with self.assertRaises(ValueError):w.compact_trace(empty)
        duplicate = trace();duplicate['traceEvents'].append(evt('_fused', 'kernel', 390, 1))
        with self.assertRaises(ValueError):w.compact_trace(duplicate)
        mainloop_only = trace();mainloop_only['traceEvents'] = [e for e in mainloop_only['traceEvents'] if e['name'] not in {'store_kvcache_kernel', 'norm_gate_kernel'}]
        with self.assertRaises(ValueError):w.compact_trace(mainloop_only)
        no_range = trace();no_range['traceEvents'] = [e for e in no_range['traceEvents'] if e['name'] != w.RANGES['fa3']]
        with self.assertRaises(ValueError):w.compact_trace(no_range)

    def test_summary_overflow_preserves_explicit_failure(self):
        report = dict(status='complete', passed=True, activity={'events': 'a'*w.MAX_SUMMARY},
            artifacts={'activity-trace.json': {'sha256': 'unchanged_raw_hash'}})
        raw, overflow = w.bounded_summary(report)
        self.assertTrue(overflow)
        saved = json.loads(raw)
        self.assertLessEqual(len(raw.encode()), w.MAX_SUMMARY)
        self.assertEqual(saved['status'], 'incomplete_or_failed')
        self.assertFalse(saved['passed'])
        self.assertTrue(saved['compact_activity_omitted'])
        self.assertNotIn('activity', saved)
        self.assertEqual(saved['artifacts']['activity-trace.json']['sha256'], 'unchanged_raw_hash')

    def test_counter_overflow_cannot_invalidate_activity(self):
        activity = w.compact_trace(trace())
        report = dict(status='complete', passed=True, activity_passed=True, activity=activity,
            counter_branch=dict(status='counter_permission_denied', sessions_attempted=1,
                records=[dict(arm='fa3', status='permission_denied', stdout='x'*w.MAX_SUMMARY)]))
        raw, overflow = w.bounded_summary(report)
        saved = json.loads(raw)
        self.assertFalse(overflow)
        self.assertTrue(saved['activity_passed'])
        self.assertTrue(saved['passed'])
        self.assertEqual(saved['activity'], activity)
        self.assertTrue(saved['counter_branch']['summary_reduced'])
        self.assertEqual(saved['counter_branch']['status'], 'counter_permission_denied')

    def test_optional_counter_time_and_error_gate(self):
        with patch.object(w.time, 'monotonic', return_value=100), patch.object(w, 'load') as loader:
            result = w.optional_counters(HERE, HERE, 124)
            self.assertEqual(result['sessions_attempted'], 0)
            loader.assert_not_called()
        with patch.object(w.time, 'monotonic', return_value=100), patch.object(w, 'load', side_effect=PermissionError('denied')):
            result = w.optional_counters(HERE, HERE, 140)
            self.assertEqual(result['status'], 'unavailable_or_partial')
            self.assertIsNone(result['sessions_attempted'])
            self.assertEqual(result['error_type'], 'PermissionError')
        text = (HERE/'worker.py').read_text()
        self.assertLess(text.index("report['activity'] = compact_trace(trace)"), text.index("report['counter_branch'] = optional_counters"))
        self.assertIn("report['profiling_support_sha256']", text)

    def test_integrated_merge_not_invented_as_separate_stage(self):
        self.assertEqual(w.event_stage('bf16_custom', '_fused', 'kernel'), 'integrated_producer_and_merge')
        self.assertEqual(w.event_stage('bf16_custom', '[CUDA memset]', 'gpu_memset'),
                         'zero_or_memset_not_independently_attributed')
        self.assertEqual(w.event_stage('fa3', 'FlashAttnFwdCombine', 'kernel'), 'fa3_combine')

    def test_oracles_and_deadline_guards(self):
        text = (HERE/'worker.py').read_text()
        self.assertIn("checked(arm, 'pre_profile')", text)
        self.assertIn("checked(arm, 'post_profile')", text)
        self.assertIn('ref.check_implementation', text)
        self.assertIn('h.unchanged', text)
        self.assertIn('signal.alarm(math.ceil(deadline))', text)
        self.assertIn('1 <= args.deadline <= 165', text)
        self.assertIn('dest.mkdir(parents=True, exist_ok=False)', text)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ProtocolTests))
    (HERE/'local-verification.json').write_text(json.dumps(dict(status='passed' if result.wasSuccessful() else 'failed',
        tests_run=result.testsRun, worker_sha256=hashlib.sha256((HERE/'worker.py').read_bytes()).hexdigest(),
        gpu_launches=0, network_calls=0, candidate_imports=0), indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
