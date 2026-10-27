"""CPU fixtures only: no NCU process, service or GPU is started."""
import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('offline_export_tested', HERE/'offline_report_export.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)
REPORT = HERE/'artifacts/baseline.ncu-rep'
NATIVE = json.loads((HERE/'result.json').read_text())
HELP = next(row['stdout'] for row in NATIVE['commands'] if '--help' in row['argv'])
VERSION = next(row['stdout'] for row in NATIVE['commands'] if '--version' in row['argv'])
CSV = (HERE/'artifacts/profile-imported.csv').read_text()


def entry_at(path):
    """Remap only the container path to the retained local CPU test fixture."""
    code = m.import_existing_report.__code__
    values = tuple(str(path) if value == '/opt/attention-retained-report/baseline.ncu-rep'
        else value for value in code.co_consts)
    return types.FunctionType(code.replace(co_consts=values), m.__dict__)


class OfflineExportTests(unittest.TestCase):
    def flow(self, *, report=REPORT, help_text=HELP, import_failure=None):
        calls = []
        def run(argv, **kwargs):
            calls.append(list(argv))
            self.assertTrue(kwargs['capture_output'])
            self.assertTrue(kwargs['text'])
            self.assertLessEqual(kwargs['timeout'], 10)
            if '--version' in argv:
                out = VERSION
            elif '--help' in argv:
                out = help_text
            else:
                self.assertIn('--import', argv)
                self.assertEqual(argv[argv.index('--import')+1], str(report))
                if isinstance(import_failure, Exception):
                    raise import_failure
                if import_failure is not None:
                    return subprocess.CompletedProcess(argv, 1, import_failure, 'stderr preserved')
                out = CSV
            return subprocess.CompletedProcess(argv, 0, out, '')
        with patch('shutil.which', return_value='/installed/ncu'), patch('subprocess.run', side_effect=run):
            result = entry_at(report)()
        return result, calls

    def test_entry_is_closure_free_cpu_only_and_uploads_only_retained_report(self):
        self.assertIsNone(m.import_existing_report.__closure__)
        tree = ast.parse(inspect.getsource(m.launch))
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and node.func.attr == 'function']
        self.assertEqual(len(functions), 1)
        gpu = next(kw.value for kw in functions[0].keywords if kw.arg == 'gpu')
        self.assertIsInstance(gpu, ast.Constant)
        self.assertIsNone(gpu.value)
        uploads = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_local_file']
        self.assertEqual(len(uploads), 1)
        self.assertEqual(ast.unparse(uploads[0].args[0]), 'report')
        self.assertEqual(uploads[0].args[1].value, '/opt/attention-retained-report/baseline.ncu-rep')

    def test_actual_help_and_retained_fixtures_drive_exact_two_import_commands(self):
        before = hashlib.sha256(REPORT.read_bytes()).hexdigest()
        self.assertEqual(before, m.REPORT_SHA)
        result, calls = self.flow()
        self.assertTrue(result['passed'])
        self.assertEqual(result['status'], 'complete')
        self.assertFalse(result['gpu_requested'])
        self.assertFalse(result['profiling_target_launched'])
        self.assertEqual(len(calls), 4)
        self.assertEqual(calls[2], calls[3])
        self.assertIn('--print-fp', calls[2])
        self.assertEqual(calls[2][calls[2].index('--print-units')+1], 'base')
        self.assertNotIn('--units', calls[2])
        for flag in ('--export', '--replay-mode', '--kernel-name', '--metrics', '--launch-count'):
            self.assertNotIn(flag, calls[2])
        self.assertTrue(result['canonical_stdout_identical'])
        self.assertEqual(result['report_before_sha256'], before)
        self.assertEqual(result['report_after_sha256'], before)
        self.assertEqual(hashlib.sha256(REPORT.read_bytes()).hexdigest(), before)

    def test_nonzero_import_retains_exact_partial_command_output(self):
        result, calls = self.flow(import_failure='offline failure stdout')
        self.assertFalse(result['passed'])
        self.assertEqual(result['phase'], 'first_import')
        self.assertEqual(len(calls), 3)
        self.assertEqual(result['commands'][-1]['returncode'], 1)
        self.assertEqual(result['commands'][-1]['stdout'], 'offline failure stdout')
        self.assertEqual(result['commands'][-1]['stderr'], 'stderr preserved')

    def test_timeout_retains_partial_streams_without_second_import(self):
        timeout = subprocess.TimeoutExpired(['ncu'], 10, output=b'partial stdout', stderr=b'partial stderr')
        result, calls = self.flow(import_failure=timeout)
        self.assertFalse(result['passed'])
        self.assertEqual(result['phase'], 'first_import')
        self.assertEqual(len(calls), 3)
        self.assertEqual(result['commands'][-1]['state'], 'timeout')
        self.assertEqual(result['commands'][-1]['stdout'], 'partial stdout')
        self.assertEqual(result['commands'][-1]['stderr'], 'partial stderr')

    def test_unsupported_help_or_changed_report_stops_before_import(self):
        result, calls = self.flow(help_text=HELP.replace('--print-fp', '--print-fpp'))
        self.assertFalse(result['passed'])
        self.assertEqual(len(calls), 2)
        self.assertIn('--print-fp', result['error'])
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'baseline.ncu-rep'
            raw = REPORT.read_bytes()
            path.write_bytes(bytes([raw[0] ^ 1])+raw[1:])
            result, calls = self.flow(report=path)
            self.assertFalse(result['passed'])
            self.assertEqual(result['phase'], 'report_validation')
            self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
