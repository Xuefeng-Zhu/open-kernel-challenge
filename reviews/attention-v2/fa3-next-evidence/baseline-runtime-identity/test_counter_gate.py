"""CPU-only counter protocol tests; never imports Torch or invokes NCU."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('counter_gate_under_test', HERE/'counter_gate.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class CounterProtocolTests(unittest.TestCase):
    def test_exact_metric_query(self):
        self.assertEqual(gate.query_argv('/ncu'), ['/ncu', '--query-metrics', '--query-metrics-mode',
            'suffix', '--metrics', 'sm__cycles_elapsed'])
        self.assertTrue(gate.metric_supported('sm__cycles_elapsed.avg\n'))
        self.assertFalse(gate.metric_supported('sm__cycles_elapsed.avg.per_second'))

    def test_no_graph_nvtx_or_application_replay(self):
        for mode in ('tiny', 'selected'):
            argv = gate.profile_argv('/ncu', '/root', mode)
            self.assertEqual(argv[argv.index('--replay-mode')+1], 'kernel')
            self.assertEqual(argv[argv.index('--launch-count')+1], '1')
            self.assertEqual(argv[argv.index('--target-processes')+1], 'application-only')
            self.assertEqual(argv[argv.index('--metrics')+1], gate.METRIC)
            self.assertEqual(argv[argv.index('--clock-control')+1], 'none')
            self.assertFalse(any('nvtx' in value.lower() or 'graph' in value.lower()
                                 or value == 'application' for value in argv))

    def test_long_csv_requires_numeric_positive_metric(self):
        text = ('"ID","Metric Name","Metric Value"\n'
            '"0","sm__cycles_elapsed.avg","1,234.5"\n')
        self.assertEqual(gate.metric_values(text), [1234.5])
        for value in ('nan', '-4', '0', 'N/A', 'inf'):
            self.assertEqual(gate.metric_values(text.replace('1,234.5', value)), [])

    def test_wide_csv_skips_units_and_command_echo(self):
        text = ('echo sm__cycles_elapsed.avg\n'
            '"ID","Kernel Name","sm__cycles_elapsed.avg"\n'
            '"","","cycle"\n"0","probe","1234.5"\n')
        self.assertEqual(gate.metric_values(text), [1234.5])
        self.assertEqual(gate.metric_values('command --metrics sm__cycles_elapsed.avg'), [])

    def test_single_child_success_marker(self):
        text = 'COUNTER_GATE_CHILD_JSON={"mode":"tiny","passed":true}\n'
        self.assertTrue(gate._child_passed(text, 'tiny'))
        self.assertFalse(gate._child_passed(text, 'selected'))
        self.assertFalse(gate._child_passed(text+text, 'tiny'))

    def test_insufficient_time_has_no_subprocess_or_path_access(self):
        with patch.object(gate, '_command', side_effect=AssertionError('no call')):
            report = gate.run_gate('/missing', '/missing', 49.9)
        self.assertEqual(report['state'], 'not_attempted')
        self.assertEqual(report['hardware_counter_sessions'], 0)

    def test_missing_ncu_has_no_probe(self):
        with patch.object(gate.shutil, 'which', return_value=None), \
             patch.object(gate, '_command', side_effect=AssertionError('no call')):
            report = gate.run_gate('/missing', '/missing', 50)
        self.assertEqual(report['state'], 'not_attempted')

    def test_tiny_timeout_prevents_selected_attempt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in gate.SOURCE_SHA:
                (root/name).write_text('fixture')
            query = {'state':'complete','stdout':gate.METRIC,'stderr':''}
            tiny = {'state':'timeout','returncode':-9,'stdout':'Connected to process',
                    'stderr':''}
            with patch.object(gate.shutil, 'which', return_value='/ncu'), \
                 patch.object(gate.hashlib, 'sha256') as digest, \
                 patch.object(gate, '_command', side_effect=[query, tiny]) as commands:
                digest.return_value.hexdigest.side_effect = list(gate.SOURCE_SHA.values())
                report = gate.run_gate(root, root, 60)
            self.assertEqual(commands.call_count, 2)
            self.assertEqual(report['state'], 'tiny_failed')
            self.assertEqual(report['hardware_counter_sessions'], 1)
            self.assertFalse(report['permission_denial_observed'])
            self.assertNotIn('selected', report)

    def test_explicit_permission_denial_distinct_from_timeout(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in gate.SOURCE_SHA:
                (root/name).write_text('fixture')
            query = {'state':'complete','stdout':gate.METRIC,'stderr':''}
            tiny = {'state':'nonzero_exit','returncode':1,'stdout':'',
                    'stderr':'ERR_NVGPUCTRPERM'}
            with patch.object(gate.shutil, 'which', return_value='/ncu'), \
                 patch.object(gate.hashlib, 'sha256') as digest, \
                 patch.object(gate, '_command', side_effect=[query, tiny]):
                digest.return_value.hexdigest.side_effect = list(gate.SOURCE_SHA.values())
                report = gate.run_gate(root, root, 60)
            self.assertTrue(report['permission_denial_observed'])
            self.assertTrue(report['tiny']['explicit_permission_denial'])


if __name__ == '__main__':
    unittest.main()
