import importlib.util
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('counter_repair_help_worker', HERE/'worker.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


def profile():
    return w.profile_argv('/ncu', Path('/payload'), 'sm__cycles_elapsed.avg')


def help_fixture(argv=None):
    argv = argv or profile()
    flags = [a for a in argv[1:argv.index(w.sys.executable)] if a.startswith('--')]
    return '\n'.join(flags + ['--query-metrics', '--query-metrics-mode',
        'application-only', 'demangled', 'kernel', 'suffix'])


class HelpContractTests(unittest.TestCase):
    def test_remote_upload_and_execution_paths_match(self):
        tree = ast.parse((HERE/'controller.py').read_text())
        prefixes = [node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            and node.value.startswith('/opt/')]
        self.assertEqual(len(prefixes), 2)
        self.assertEqual({p.rstrip('/') for p in prefixes},
            {'/opt/attention-eager-counter-repair'})

    def test_payload_scope_and_closure_exclude_credentials_and_attention(self):
        cspec = importlib.util.spec_from_file_location('counter_repair_contract_controller',
            HERE/'controller.py')
        controller = importlib.util.module_from_spec(cspec)
        cspec.loader.exec_module(controller)
        self.assertEqual(set(controller.payload()),
            {'worker.py', 'counter_helper.py', 'LICENSE', 'THIRD_PARTY_NOTICES.md'})
        self.assertEqual(controller.make_remote_entry().__code__.co_freevars, ())

    def test_exact_command_has_no_unsupported_units_flag(self):
        args = profile()
        self.assertNotIn('--units', args)
        self.assertNotIn('--print-units', args)
        self.assertIn('--csv', args)
        self.assertEqual(args[args.index('--replay-mode')+1], 'kernel')
        self.assertEqual(args[args.index('--launch-skip')+1], '1')
        self.assertEqual(args[args.index('--launch-count')+1], '1')

    def test_child_flags_are_not_ncu_flags(self):
        result = w.validate_installed_help(help_fixture(), profile())
        self.assertTrue(result['all_flags_advertised'])
        self.assertNotIn('--tiny-child', result['used_ncu_flags'])
        self.assertNotIn('--tiny-child', help_fixture())

    def test_missing_installed_flag_stops_gate(self):
        with self.assertRaisesRegex(ValueError, 'clock-control'):
            w.validate_installed_help(help_fixture().replace('--clock-control', ''), profile())

    def test_longer_flag_does_not_satisfy_exact_option(self):
        with self.assertRaisesRegex(ValueError, '--csv'):
            w.validate_installed_help(help_fixture().replace('--csv', '--csv-more'), profile())

    def test_known_bad_units_rejected_even_if_fixture_advertises_it(self):
        args = profile()
        args[1:1] = ['--units', 'base']
        with self.assertRaisesRegex(ValueError, 'Known unsupported'):
            w.validate_installed_help(help_fixture(args), args)

    def test_missing_required_choice_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'demangled'):
            w.validate_installed_help(help_fixture().replace('demangled', ''), profile())

    def test_query_flags_are_also_checked(self):
        query = ['/ncu', '--query-metrics', '--query-metrics-mode', 'suffix',
                 '--metrics', 'sm__cycles_elapsed']
        with self.assertRaisesRegex(ValueError, '--query-metrics-mode'):
            w.validate_installed_help(help_fixture().replace('--query-metrics-mode', ''),
                profile(), query)

    def test_failed_help_contract_never_reaches_query_or_profile(self):
        calls = []

        class Helper:
            METRIC = 'sm__cycles_elapsed.avg'

            @staticmethod
            def query_argv(ncu):
                return [ncu, '--query-metrics', '--query-metrics-mode', 'suffix',
                        '--metrics', 'sm__cycles_elapsed']

            @staticmethod
            def _command(argv, dest, stem, seconds):
                calls.append((argv, stem, seconds))
                text = 'NCU version fixture' if stem == 'version' else help_fixture().replace('--csv', '')
                return {'state': 'complete', 'returncode': 0, 'stdout': text, 'stderr': ''}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root/'result'
            with patch.object(w, 'FIXED', {}), patch.object(w, 'load_helper', return_value=Helper), \
                    patch.object(w.shutil, 'which', return_value='/ncu'), \
                    patch.object(w.signal, 'signal'), patch.object(w.signal, 'alarm'):
                code = w.run(root, output, 65)
            report = json.loads((output/'summary.json').read_text())
        self.assertEqual(code, 1)
        self.assertEqual([c[1] for c in calls], ['version', 'help'])
        self.assertEqual(report['hardware_counter_sessions'], 0)
        self.assertFalse(report['passed'])
        self.assertIn('--csv', report['error'])


if __name__ == '__main__':
    unittest.main()
