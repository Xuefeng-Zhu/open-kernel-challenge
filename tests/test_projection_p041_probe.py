"""Local mock/static gates only. No CUDA candidate or profiler execution."""
import ast
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import projection_p041_probe as probe


class P041ProbeTests(unittest.TestCase):
    def test_frozen_selected_source(self):
        self.assertEqual(probe.verify_sources()[probe.SOURCE], probe.SOURCE_SHA256)

    def test_source_mismatch_refuses_before_execution(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(probe.hashlib, 'sha256') as sha, patch.object(Path, 'read_bytes', return_value=b'x'):
                sha.return_value.hexdigest.return_value = 'wrong'
                with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                    probe.verify_sources(root)

    def test_device_gate(self):
        torch = Mock()
        torch.cuda.is_available.return_value = False
        with self.assertRaisesRegex(RuntimeError, 'CUDA unavailable'):
            probe.require_h200(torch)
        torch.cuda.is_available.return_value = True
        torch.cuda.get_device_name.return_value = 'NVIDIA H100'
        with self.assertRaisesRegex(RuntimeError, 'H200 SM90'):
            probe.require_h200(torch)
        torch.cuda.get_device_name.return_value = 'NVIDIA H200'
        torch.cuda.get_device_capability.return_value = (9, 0)
        self.assertEqual(probe.require_h200(torch), 'NVIDIA H200')

    def test_timeout_and_mode_declarations(self):
        good = dict(mode='timing', deadline_seconds=150, outer_timeout_seconds=170,
                    include_warm_cache_diagnostic=True)
        probe.validate_options(SimpleNamespace(**good))
        for change in ({'outer_timeout_seconds':150}, {'deadline_seconds':0},
                       {'deadline_seconds':301}, {'mode':'activity'}):
            with self.assertRaises(ValueError):
                probe.validate_options(SimpleNamespace(**{**good, **change}))

    def test_balanced_order_and_raw_pair_summary(self):
        samples = [dict(round=i, order=list(probe.runner_order(i)),
                        us=dict(candidate=10+i%2, baseline=20+2*(i%2))) for i in range(24)]
        summary = probe.paired_summary(samples)
        self.assertEqual(summary['pair_ratios'], [2.0]*12)
        self.assertEqual(summary['median_paired_gross_ratio'], 2)
        self.assertEqual(summary['median_gross_us']['candidate'], 10.5)
        self.assertEqual(probe.condition_order(0, True), ('cold','warm'))
        self.assertEqual(probe.condition_order(2, True), ('warm','cold'))
        self.assertEqual(probe.condition_order(3, False), ('cold',))
        with self.assertRaises(ValueError):
            probe.paired_summary(samples[:-1])
        samples[0]['us']['candidate'] = 0
        with self.assertRaises(ValueError):
            probe.paired_summary(samples)

    def test_warm_replay_immediately_precedes_timer(self):
        events = []
        graph = Mock()
        graph.replay.side_effect = lambda: events.append('warm_replay')
        eviction = object()
        def timer(g, e):
            self.assertIs(g, graph)
            self.assertIsInstance(e, probe.NoEviction)
            events.append('timed_replay')
            return 10
        self.assertEqual(probe.timed_replay(graph, eviction, 'warm', timer), 10)
        self.assertEqual(events, ['warm_replay','timed_replay'])
        timer = Mock(return_value=12)
        graph.reset_mock()
        self.assertEqual(probe.timed_replay(graph, eviction, 'cold', timer), 12)
        graph.replay.assert_not_called()
        timer.assert_called_once_with(graph, eviction)

    def test_immutable_checks_detect_input_and_weight_mutations(self):
        torch = Mock()
        x = Mock()
        x.data_ptr.return_value = 7
        torch.equal.side_effect = [True, True]
        probe.assert_immutable(torch, {'x':x}, 'expected', {'w':'live'}, {'w':'frozen'}, 7)
        torch.equal.side_effect = [True, False]
        with self.assertRaisesRegex(RuntimeError, 'weight tensor mutated'):
            probe.assert_immutable(torch, {'x':x}, 'expected', {'w':'live'}, {'w':'frozen'}, 7)
        with self.assertRaisesRegex(RuntimeError, 'address/content'):
            probe.assert_immutable(torch, {'x':x}, 'expected', {}, {}, 8)

    def test_prepare_duration_includes_completion_and_excludes_prior_work(self):
        events = []
        torch, module = Mock(), Mock()
        weights, state = object(), object()
        torch.cuda.synchronize.side_effect = lambda: events.append('synchronize')
        module.prepare.side_effect = lambda value: (events.append('prepare'), state)[1]
        def clock():
            events.append('clock')
            return 10.0 if events.count('clock') == 1 else 12.5
        with patch.object(probe.time, 'perf_counter', side_effect=clock):
            result, elapsed = probe.prepare_with_timing(torch, module, weights)
        self.assertIs(result, state)
        self.assertEqual(elapsed, 2.5)
        module.prepare.assert_called_once_with(weights)
        self.assertEqual(events, ['synchronize', 'clock', 'prepare', 'synchronize', 'clock'])

    def test_trace_preserves_raw_and_extracts_only_emitted_kernel_resources(self):
        raw = dict(traceEvents=[
            dict(ph='X', cat='cpu_op', name='aten::mm', dur=100, args={'grid':[99]}),
            dict(ph='X', cat='kernel', name='library_kernel', dur=4.5, ts=123,
                 args={'grid':[8,1,1], 'block':[128,1,1], 'registers per thread':64,
                       'shared memory':4096, 'blocks per SM':0.5, 'stream':7}),
            dict(ph='X', cat='cuda_kernel', name='unknown_resources', dur=1.0, args={}),
            dict(ph='X', cat='gpu_memcpy', name='Memcpy', dur=1, args={}),
        ])
        with TemporaryDirectory() as folder:
            path = Path(folder)/'activity.case00.pooler.m8.candidate.trace.json'
            encoded = json.dumps(raw).encode()
            prof = Mock()
            prof.export_chrome_trace.side_effect = lambda target: Path(target).write_bytes(encoded)
            result = probe.export_activity_trace(prof, path)
            self.assertEqual(path.read_bytes(), encoded)
            self.assertEqual(result['file'], path.name)
            self.assertEqual(result['bytes'], len(encoded))
            self.assertEqual(result['sha256'], probe.hashlib.sha256(encoded).hexdigest())
            self.assertEqual(len(result['chrome_kernels']), 2)
            kernel = result['chrome_kernels'][0]
            self.assertEqual(kernel['duration_us'], 4.5)
            self.assertEqual(kernel['trace_args'], raw['traceEvents'][1]['args'])
            self.assertEqual(kernel['resources'], dict(grid=[8,1,1], block=[128,1,1],
                registers_per_thread=64, shared_memory_bytes=4096, blocks_per_sm=0.5))
            self.assertEqual(result['chrome_kernels'][1]['resources'], {})

    def test_malformed_trace_fails_without_inventing_metadata(self):
        for raw in (dict(traceEvents={}), dict(traceEvents=[dict(ph='X', cat='kernel', args=[])])):
            with self.subTest(raw=raw), TemporaryDirectory() as folder:
                path = Path(folder)/'trace.json'
                prof = Mock()
                prof.export_chrome_trace.side_effect = lambda target: Path(target).write_text(json.dumps(raw))
                with self.assertRaises(ValueError):
                    probe.export_activity_trace(prof, path)
                self.assertEqual(json.loads(path.read_text()), raw)

    def test_cli_source_failure_never_calls_worker(self):
        with TemporaryDirectory() as folder:
            path = Path(folder)/'result.json'
            with patch.object(probe, 'verify_sources', side_effect=ValueError('frozen mismatch')), \
                 patch.object(probe, 'execute') as execute, patch('builtins.print'):
                rc = probe.main(['--mode','correctness','--output',str(path),
                                 '--outer-timeout-seconds','180'])
            self.assertEqual(rc, 1)
            execute.assert_not_called()
            self.assertFalse(json.loads(path.read_text())['passed'])

    def test_source_has_no_provisioning_or_subprocess(self):
        tree = ast.parse(Path(probe.__file__).read_text())
        imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        self.assertFalse(set(imports) & {'modal','subprocess','requests','urllib','socket'})
        self.assertEqual(probe.EVICTION_BYTES, 256*1024**2)
        self.assertEqual(probe.WARMUP_REPLAYS, 6)
        self.assertEqual(probe.MEASURED_ROUNDS, 24)
        self.assertEqual(probe.REPRESENTATIVES,
                         (('pooler',8),('pooler',128),('pooler',256),('qkv_gate',128),('unpooler',16)))
        profiler_options = {kw.arg: ast.literal_eval(kw.value) for node in ast.walk(tree)
                            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                            and node.func.attr == 'profile' for kw in node.keywords
                            if kw.arg in {'record_shapes', 'with_stack', 'profile_memory'}}
        self.assertEqual(profiler_options, dict(record_shapes=False, with_stack=False,
                                               profile_memory=False))


if __name__ == '__main__':
    unittest.main()
