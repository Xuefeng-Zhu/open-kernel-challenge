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
        self.assertEqual(probe.REPRESENTATIVES,
                         (('pooler',8),('pooler',128),('pooler',256),('qkv_gate',128),('unpooler',16)))


if __name__ == '__main__':
    unittest.main()
