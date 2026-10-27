"""CPU-only end-to-end tests at the collector's command execution boundary."""
import copy
import ast
from contextlib import redirect_stdout
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


w = load(HERE/'worker.py', 'capture_executor_test_worker')
reader = load(HERE/'csv_helpers.py', 'capture_executor_test_csv')
EXPECTED = json.loads((HERE/'expected-native.json').read_text())
VALUES = (11.47, 36.77, 43.40, 12.49)


def metric_csv(*, signature=None, grid=None, block=None, values=VALUES,
               units=('%', '%', '%', '%'), data_row=True):
    stream = io.StringIO()
    output = csv.writer(stream)
    output.writerow(['ID', 'Kernel Name', 'Grid Size', 'Block Size', *w.METRICS])
    output.writerow(['', '', '', '', *units])
    if data_row:
        dimensions = lambda value: '('+', '.join(map(str, value))+')'
        output.writerow(['0', signature or EXPECTED['kernel_name'],
            dimensions(grid or EXPECTED['grid']), dimensions(block or EXPECTED['block']),
            *map(str, values)])
    return stream.getvalue()


class FakeCommandHelper:
    """No subprocess: preserve the same files/records as the real boundary."""
    def __init__(self, *, capture_text=None, import_text=None, write_report=True,
                 report_bytes=b'fake durable NCU report', import_code=0, mutate_report=False):
        self.capture_text = metric_csv() if capture_text is None else capture_text
        self.import_text = metric_csv() if import_text is None else import_text
        self.write_report = write_report
        self.report_bytes = report_bytes
        self.import_code = import_code
        self.mutate_report = mutate_report
        self.calls = []

    def _command(self, argv, output, name, timeout):
        argv, output = list(argv), Path(output)
        self.calls.append(dict(argv=argv, name=name, timeout=timeout))
        importing = '--import' in argv
        report_path = Path(argv[argv.index('--import' if importing else '--export')+1])
        if importing:
            if self.mutate_report:
                with report_path.open('ab') as stream:
                    stream.write(b'changed')
            text, code = self.import_text, self.import_code
        else:
            if self.write_report:
                report_path.write_bytes(self.report_bytes)
            text, code = self.capture_text, 0
        (output/(name+'.stdout.txt')).write_text(text)
        (output/(name+'.stderr.txt')).write_text('')
        raw = text.encode()
        return dict(argv=argv, state='complete' if code == 0 else 'nonzero_exit',
            returncode=code, stdout=text, stderr='', wall_seconds=0.01,
            stdout_bytes=len(raw), stdout_sha256=hashlib.sha256(raw).hexdigest(),
            stderr_bytes=0, stderr_sha256=hashlib.sha256(b'').hexdigest())


class Tests(unittest.TestCase):
    def collect(self, output, helper, *, bindings=None, report=None, post=None, remaining=None):
        return w.collect_counter('/fake/ncu', HERE, Path(output), helper, reader,
            copy.deepcopy(EXPECTED), copy.deepcopy(EXPECTED['bindings']) if bindings is None else bindings,
            report=report, post_profile=post, remaining=remaining, save=lambda: None)

    def test_valid_real_collection_import_flow_and_exact_commands(self):
        with tempfile.TemporaryDirectory() as temporary:
            helper, partial, post = FakeCommandHelper(), {}, []
            result = self.collect(temporary, helper, report=partial, post=lambda: post.append(True))
            self.assertIs(result, partial)
            self.assertEqual(post, [True])
            self.assertTrue(result['capture_prerequisites']['passed'])
            self.assertTrue(result['report_roundtrip']['passed'])
            self.assertTrue(result['native_capture_identity']['accepted'])
            self.assertEqual(len(helper.calls), 2)
            capture, imported = [row['argv'] for row in helper.calls]
            self.assertEqual(capture[capture.index('--kernel-name')+1], 'regex:.*FlashAttnFwdSm90.*')
            for argv in (capture, imported):
                self.assertEqual(argv[argv.index('--rename-kernels')+1], '0')
                self.assertEqual(argv[argv.index('--config-file')+1], '0')
                self.assertNotIn('--units', argv)
            target = str(Path(temporary).resolve()/'baseline.ncu-rep')
            self.assertEqual(capture[capture.index('--export')+1], target)
            self.assertEqual(imported[imported.index('--import')+1], target)
            self.assertEqual(imported[imported.index('--print-units')+1], 'base')
            self.assertIn('--print-fp', imported)
            self.assertNotIn('--replay-mode', imported)
            for name in ('baseline.ncu-rep', 'profile-original.csv', 'profile-imported.csv'):
                self.assertIn(name, result['artifact_manifest'])
                self.assertTrue((Path(temporary)/name).is_file())

    def test_actual_exit_zero_empty_capture_fails_before_import(self):
        text = (HERE.parent/'fa3-b32-identity-recapture/6-stdout.txt').read_text()
        self.assertIn('No kernels were profiled', text)
        with tempfile.TemporaryDirectory() as temporary:
            helper, partial, post = FakeCommandHelper(capture_text=text), {}, []
            with self.assertRaises(ValueError):
                self.collect(temporary, helper, report=partial, post=lambda: post.append(True))
            self.assertEqual(len(helper.calls), 1)
            self.assertEqual(helper.calls[0]['name'], 'ncu-baseline')
            self.assertEqual(post, [True])
            self.assertFalse(partial['capture_prerequisites']['passed'])
            self.assertFalse(partial['capture_prerequisites']['no_empty_capture_warning'])
            self.assertEqual((Path(temporary)/'profile-original.csv').read_text(), text)
            self.assertNotIn('imported_counter_report', partial)

    def test_header_only_capture_fails_before_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            helper, partial = FakeCommandHelper(capture_text=metric_csv(data_row=False)), {}
            with self.assertRaises(ValueError):
                self.collect(temporary, helper, report=partial)
            self.assertEqual(len(helper.calls), 1)
            self.assertNotIn('counter_report', partial)

    def test_missing_durable_report_fails_before_import_but_keeps_partial(self):
        with tempfile.TemporaryDirectory() as temporary:
            helper, partial, post = FakeCommandHelper(write_report=False), {}, []
            with self.assertRaises(ValueError):
                self.collect(temporary, helper, report=partial, post=lambda: post.append(True))
            self.assertEqual(len(helper.calls), 1)
            self.assertIn('ncu_report_error', partial)
            self.assertIn('profile-original.csv', partial['artifact_manifest'])
            self.assertEqual(post, [True])

    def test_empty_report_fails_before_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            helper = FakeCommandHelper(report_bytes=b'')
            with self.assertRaises(ValueError):
                self.collect(temporary, helper)
            self.assertEqual(len(helper.calls), 1)

    def test_wrong_native_signature_is_not_accepted_by_family_membership(self):
        wrong = EXPECTED['kernel_name'].replace('(int)2,', '(int)3,', 1)
        self.assertNotEqual(wrong, EXPECTED['kernel_name'])
        with tempfile.TemporaryDirectory() as temporary:
            partial = {}
            helper = FakeCommandHelper(capture_text=metric_csv(signature=wrong),
                import_text=metric_csv(signature=wrong))
            with self.assertRaises(ValueError):
                self.collect(temporary, helper, report=partial)
            self.assertFalse(partial['native_capture_identity']['expected_native_signature_and_launch'])

    def test_wrong_native_geometry_is_rejected(self):
        for geometry in ({'grid': [131, 1, 1]}, {'block': [128, 1, 1]}):
            with self.subTest(geometry=geometry), tempfile.TemporaryDirectory() as temporary:
                partial = {}
                text = metric_csv(**geometry)
                with self.assertRaises(ValueError):
                    self.collect(temporary, FakeCommandHelper(capture_text=text, import_text=text), report=partial)
                self.assertFalse(partial['native_capture_identity']['accepted'])

    def test_wrong_fixture_binding_is_rejected(self):
        bindings = copy.deepcopy(EXPECTED['bindings'])
        bindings['fixture.qkv'] = '0'*64
        with tempfile.TemporaryDirectory() as temporary:
            partial = {}
            with self.assertRaises(ValueError):
                self.collect(temporary, FakeCommandHelper(), report=partial, bindings=bindings)
            self.assertFalse(partial['native_capture_identity']['expected_source_binary_fixture_bindings'])

    def test_import_failure_preserves_original_and_failed_roundtrip(self):
        with tempfile.TemporaryDirectory() as temporary:
            partial, helper = {}, FakeCommandHelper(import_code=1, import_text='Import failed\n')
            with self.assertRaises(ValueError):
                self.collect(temporary, helper, report=partial)
            self.assertEqual(len(helper.calls), 2)
            self.assertFalse(partial['report_roundtrip']['passed'])
            self.assertEqual(partial['report_roundtrip']['import_exit_code'], 1)
            self.assertIn('counter_report', partial)
            self.assertIn('profile-imported.csv', partial['artifact_manifest'])

    def test_report_mutation_and_imported_value_mismatch_are_rejected(self):
        cases = [FakeCommandHelper(mutate_report=True),
            FakeCommandHelper(import_text=metric_csv(values=(99., *VALUES[1:])))]
        for helper in cases:
            with self.subTest(case=helper.mutate_report), tempfile.TemporaryDirectory() as temporary:
                partial = {}
                with self.assertRaises(ValueError):
                    self.collect(temporary, helper, report=partial)
                self.assertFalse(partial['report_roundtrip']['passed'])

    def test_existing_report_is_not_overwritten_or_profiled(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary)/'baseline.ncu-rep'
            report.write_bytes(b'existing')
            helper = FakeCommandHelper()
            with self.assertRaises(ValueError):
                self.collect(temporary, helper)
            self.assertEqual(helper.calls, [])
            self.assertEqual(report.read_bytes(), b'existing')

    def test_insufficient_budget_stops_before_command(self):
        with tempfile.TemporaryDirectory() as temporary:
            helper = FakeCommandHelper()
            with self.assertRaises(TimeoutError):
                self.collect(temporary, helper, remaining=lambda: 1)
            self.assertEqual(helper.calls, [])

    def test_no_vendor_imports_at_module_top_level(self):
        tree = ast.parse((HERE/'worker.py').read_text())
        names = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                names += [name.name.split('.')[0] for name in node.names]
            elif isinstance(node, ast.ImportFrom):
                names.append((node.module or '').split('.')[0])
        self.assertFalse(set(names) & {'torch', 'triton', 'sgl_kernel', 'sglang', 'modal'})

    def test_child_exactly_three_selected_calls_with_pre_post_oracles(self):
        calls, oracles = [], []
        raw = SimpleNamespace()
        raw.clone = lambda: raw
        helpers = SimpleNamespace(clone_inputs=lambda fresh: {},
            restore=lambda work, fresh: calls.append('restore'))
        reference = SimpleNamespace(generate_input=lambda **kwargs: {})
        selected = SimpleNamespace(kernel_fn=lambda work: calls.append('selected'))
        torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: calls.append('sync')))
        def check(reference, helpers, work, fresh, out, expected, oracle=False):
            oracles.append(oracle)
            return dict(passed=True), raw
        output = io.StringIO()
        with patch.object(w, 'bind_sources'), \
                patch.object(w, 'runtime_modules', return_value=(helpers, reference, selected, {})), \
                patch.object(w, 'fixture_fingerprint', return_value={}), \
                patch.object(w, 'check_output', side_effect=check), \
                patch.dict(sys.modules, {'torch': torch}), redirect_stdout(output):
            self.assertEqual(w.counter_child(HERE), 0)
        self.assertEqual(calls, ['restore', 'selected', 'sync']*3)
        self.assertEqual(oracles, [True, False, True])
        markers = w.progress_rows(output.getvalue())
        self.assertEqual([row['phase'] for row in markers if row.get('passed') is True],
            ['pre', 'measured', 'post'])

    def test_actual_canonical_imports_are_byte_identical_and_exact_to_capture(self):
        exported = json.loads((HERE/'OFFLINE_EXPORT.json').read_text())
        original = json.loads((HERE/'result.json').read_text())
        first, second = exported['commands'][2:4]
        self.assertEqual([first['returncode'], second['returncode']], [0, 0])
        self.assertEqual(first['stdout'].encode(), second['stdout'].encode())
        self.assertTrue(exported['canonical_stdout_identical'])
        self.assertEqual(exported['report_before_sha256'], exported['report_after_sha256'])
        parsed = [reader.parse_ncu_csv(command['stdout'], w.METRICS) for command in (first, second)]
        capture = original['counter_report']
        before = dict(bytes=exported['report_bytes'], sha256=exported['report_before_sha256'])
        after = dict(bytes=exported['report_bytes'], sha256=exported['report_after_sha256'])
        for item in parsed:
            self.assertEqual(item['row_count'], 1)
            self.assertEqual(item['kernel_name'], capture['kernel_name'])
            self.assertTrue(w.compare_roundtrip(capture, item, before, after)['passed'])
            self.assertEqual({name: (item['metrics'][name]['value'], item['metrics'][name]['unit'])
                for name in w.METRICS}, {name: (capture['metrics'][name]['value'], capture['metrics'][name]['unit'])
                for name in w.METRICS})
        self.assertEqual(parsed[0], parsed[1])

    def test_legacy_default_import_comparator_still_fails_without_rounding(self):
        original = json.loads((HERE/'result.json').read_text())
        self.assertFalse(original['passed'])
        self.assertFalse(original['report_roundtrip']['passed'])
        roundtrip = original['report_roundtrip']
        result = w.compare_roundtrip(original['counter_report'], original['imported_counter_report'],
            roundtrip['report_before_import'], roundtrip['report_after_import'])
        self.assertFalse(result['passed'])
        self.assertFalse(result['metric_values_units_equal'])
        self.assertTrue(result['report_unchanged'])

    def test_real_import_builder_matches_both_empirically_validated_commands(self):
        exported = json.loads((HERE/'OFFLINE_EXPORT.json').read_text())
        for command in exported['commands'][2:4]:
            argv = command['argv']
            path = Path(argv[argv.index('--import')+1])
            self.assertEqual(w.import_argv(argv[0], path), argv)
            self.assertEqual(argv[argv.index('--print-units')+1], 'base')
            self.assertIn('--print-fp', argv)
            for forbidden in ('--units', '--export', '--replay-mode', '--kernel-name'):
                self.assertNotIn(forbidden, argv)

    def test_actual_installed_help_advertises_canonical_import_flags(self):
        exported = json.loads((HERE/'OFFLINE_EXPORT.json').read_text())
        help_text = exported['commands'][1]['stdout']
        self.assertIn('--print-units', help_text)
        self.assertIn('--print-fp', help_text)
        argv = exported['commands'][2]['argv']
        contract = w.validate_import_help(help_text, argv)
        self.assertTrue(contract['all_flags_advertised'])
        self.assertTrue(contract['offline_only'])
        with self.assertRaises(ValueError):
            w.validate_import_help(help_text.replace('--print-fp', '--not-print-fp'), argv)


if __name__ == '__main__':
    unittest.main()
