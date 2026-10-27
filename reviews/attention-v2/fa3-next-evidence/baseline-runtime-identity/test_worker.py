"""CPU-only contract/parser tests; no Torch imports, CUDA or services."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('baseline_identity_worker_tested', HERE/'worker.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)
REPO = HERE.parents[3]


def trace_fixture(symbol=None):
    symbol = symbol or 'void FlashAttnFwdSm90<cute::tuple<64,128,64>, unresolved_flags>()'
    events = []
    for i, label in enumerate(w.LABELS.values()):
        start = 1000*i
        events += [dict(name=label, cat='user_annotation', ph='X', ts=start, dur=900),
            dict(name='_fa3_prepare_fused', cat='kernel', ph='X', ts=start+10, dur=4),
            dict(name=symbol, cat='kernel', ph='X', ts=start+20, dur=100,
                args={'grid': [132, 1, 1], 'block': [384, 1, 1], 'registers per thread': 255}),
            dict(name='FlashAttnFwdCombine', cat='kernel', ph='X', ts=start+120, dur=5),
            dict(name='norm_gate', cat='kernel', ph='X', ts=start+125, dur=3)]
    return {'traceEvents': events}


class WorkerContract(unittest.TestCase):
    def test_source_hash_bindings_match_selected_support(self):
        actual_paths = {'selected.py': 'reviews/attention-v2/fusion-lookahead-probe/candidate.py',
            'reference.py': 'reviews/attention-v2/official/reference.py',
            'checks.py': 'reviews/attention-v2/official/checks.py',
            'modal_worker.py': 'reviews/attention-v2/modal_worker.py'}
        for name, relative in actual_paths.items():
            self.assertEqual(hashlib.sha256((REPO/relative).read_bytes()).hexdigest(), w.SOURCES[name])

    def test_fixed_two_case_scope(self):
        self.assertEqual({case['batch'] for case in w.CASES.values()}, {16, 32})
        for case in w.CASES.values():
            self.assertEqual((case['q_len'], case['ctx'], case['heads'], case['layer'], case['page'], case['seed']),
                (8, 80000, 'tp8', 'full', 1, 135741))
            self.assertEqual((case['dist'], case['spread'], case['kv_scale']), ('coherent', 'agentic', 1.))

    def test_interval_union_not_duration_sum(self):
        self.assertEqual(w.interval_union([(1, 3), (2, 6), (6, 8), (11, 12)]), 8)
        self.assertEqual(w.interval_union([]), 0)
        self.assertEqual(w.interval_union([(5, 1)]), 0)

    def test_full_symbols_and_resources_preserved_no_template_guess(self):
        symbol = 'FlashAttnFwdSm90<'+'opaque_parameter,'*200+'>()'
        report = w.compact_trace(trace_fixture(symbol))
        self.assertFalse(report['hardware_counters_collected'])
        for row in report['cases'].values():
            self.assertEqual(row['full_mainloop_symbols'], [symbol])
            main = next(event for event in row['events'] if event['stage'] == 'fa3_mainloop')
            self.assertEqual(main['args']['registers per thread'], 255)
            self.assertTrue(row['resolved_split_count'].startswith('UNKNOWN'))
            self.assertTrue(row['resolved_pack_gqa'].startswith('UNKNOWN'))

    def test_range_excludes_unrelated_restore_activity(self):
        trace = trace_fixture()
        trace['traceEvents'].append(dict(name='RESTORE', cat='gpu_memcpy', ph='X', ts=-100, dur=20))
        result = w.compact_trace(trace)
        for row in result['cases'].values():
            self.assertEqual(len(row['events']), 4)
            self.assertEqual(row['gpu_activity_union_us'], 112)

    def test_duplicate_range_rejected(self):
        trace = trace_fixture()
        trace['traceEvents'].append(dict(trace['traceEvents'][0]))
        with self.assertRaisesRegex(ValueError, 'one fenced range'):
            w.compact_trace(trace)

    def test_missing_mainloop_rejected(self):
        with self.assertRaisesRegex(ValueError, 'actual FA3 mainloop'):
            w.compact_trace(trace_fixture('UnknownKernel'))

    def test_missing_trace_not_silent_success(self):
        with self.assertRaisesRegex(ValueError, 'traceEvents'):
            w.compact_trace({})

    def test_adapter_is_separate_not_namespace_mutation(self):
        selected = (REPO/'reviews/attention-v2/fusion-lookahead-probe/candidate.py').read_text()
        namespace = {'flash_attn_with_kvcache': object(), 'sentinel': object()}
        original_keys = set(namespace)
        function, source = w.make_aux_adapter(selected, namespace)
        self.assertEqual(function.__name__, '_diagnostic_aux_full')
        self.assertEqual(set(namespace), original_keys)
        self.assertNotIn('_diagnostic_aux_full', namespace)
        tree = ast.parse(source)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
            and ast.unparse(node.func) == 'flash_attn_with_kvcache']
        self.assertEqual(len(calls), 1)
        self.assertEqual(sum(keyword.arg == 'return_softmax_lse' and ast.literal_eval(keyword.value) is True
            for keyword in calls[0].keywords), 1)

    def test_adapter_native_arguments_unchanged_except_return_flag(self):
        selected = (REPO/'reviews/attention-v2/fusion-lookahead-probe/candidate.py').read_text()
        original = next(node for node in ast.parse(selected).body
            if isinstance(node, ast.FunctionDef) and node.name == '_fa3_full_verify_fused')
        _, adapted = w.make_aux_adapter(selected, {})
        original_call = next(node for node in ast.walk(original) if isinstance(node, ast.Call)
            and ast.unparse(node.func) == 'flash_attn_with_kvcache')
        adapted_call = next(node for node in ast.walk(ast.parse(adapted)) if isinstance(node, ast.Call)
            and ast.unparse(node.func) == 'flash_attn_with_kvcache')
        adapted_call.keywords = [keyword for keyword in adapted_call.keywords if keyword.arg != 'return_softmax_lse']
        self.assertEqual(ast.dump(original_call), ast.dump(adapted_call))

    def test_entire_auxiliary_body_diff_is_only_return_adaptation(self):
        selected = (REPO/'reviews/attention-v2/fusion-lookahead-probe/candidate.py').read_text()
        original = next(node for node in ast.parse(selected).body
            if isinstance(node, ast.FunctionDef) and node.name == '_fa3_full_verify_fused')
        _, adapted_source = w.make_aux_adapter(selected, {})
        adapted = ast.parse(adapted_source).body[0]
        adapted.name = original.name
        restored = []
        for node in adapted.body:
            if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == 'out' and ast.unparse(node.value) == '_diagnostic_aux_return[0]':
                continue
            if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == '_diagnostic_aux_return':
                node.targets = [ast.Name(id='out', ctx=ast.Store())]
                node.value.keywords = [keyword for keyword in node.value.keywords if keyword.arg != 'return_softmax_lse']
            if isinstance(node, ast.Return):
                self.assertIsInstance(node.value, ast.Tuple)
                self.assertEqual(ast.unparse(node.value.elts[1]), '_diagnostic_aux_return')
                node.value = node.value.elts[0]
            restored.append(node)
        adapted.body = restored
        self.assertEqual(ast.dump(original), ast.dump(adapted))

    def test_missing_profiler_executable_not_called(self):
        with patch.object(w.shutil, 'which', return_value=None), patch.object(w.subprocess, 'run') as run:
            self.assertEqual(w.command_version('ncu'), {'installed_on_path': False})
            run.assert_not_called()

    def test_no_service_or_candidate_edits_in_worker(self):
        tree = ast.parse((HERE/'worker.py').read_text())
        imports = [node.module or '' for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
        self.assertNotIn('modal', imports)
        self.assertNotIn('requests', imports)
        self.assertNotIn('urllib', imports)
        self.assertLessEqual(w.MAX_TRACE_GZ*4//3, 8*1024*1024)
        self.assertEqual(w.MAX_TRACE, 32*1024*1024)


if __name__ == '__main__':
    unittest.main()
