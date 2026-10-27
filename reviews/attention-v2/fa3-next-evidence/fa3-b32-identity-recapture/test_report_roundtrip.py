"""CPU-only contract tests for durable NCU reports and offline re-import."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('recapture_roundtrip_worker', HERE/'worker.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


def parsed():
    return dict(kernel_name='void flash::Kernel<64, 128, 64>(Params)',
        grid=[132, 1, 1], block=[256, 1, 1], row_count=1,
        metrics={name: dict(value=12.5+i, unit='%', raw=str(12.5+i))
            for i, name in enumerate(w.METRICS)})


class Tests(unittest.TestCase):
    def test_report_metadata_hashes_exact_nonempty_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary)/'capture.ncu-rep'
            raw = b'durable-report-fixture\x00\xff'
            report.write_bytes(raw)
            self.assertEqual(w.report_metadata(report),
                dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))

    def test_missing_empty_nonregular_relative_or_wrong_extension_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            empty = root/'empty.ncu-rep'
            empty.touch()
            directory = root/'directory.ncu-rep'
            directory.mkdir()
            wrong = root/'wrong.txt'
            wrong.write_bytes(b'x')
            for report in (root/'missing.ncu-rep', empty, directory, Path('relative.ncu-rep'), wrong):
                with self.subTest(report=report), self.assertRaises((OSError, ValueError)):
                    w.report_metadata(report)

    def test_symlink_is_not_a_durable_regular_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root/'target.ncu-rep'
            target.write_bytes(b'content')
            link = root/'link.ncu-rep'
            link.symlink_to(target)
            with self.assertRaises(ValueError):
                w.report_metadata(link)

    def test_oversize_report_rejected_before_content_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary)/'oversize.ncu-rep'
            with report.open('wb') as stream:
                stream.truncate(w.MAX_NCU_REPORT+1)
            with self.assertRaises(ValueError):
                w.report_metadata(report)

    def test_report_mutation_during_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary)/'changing.ncu-rep'
            report.write_bytes(b'original')
            digest = hashlib.sha256()
            class MutatingDigest:
                changed = False
                def update(self, raw):
                    digest.update(raw)
                    if not self.changed:
                        self.changed = True
                        with report.open('ab') as stream:
                            stream.write(b'changed')
                def hexdigest(self):
                    return digest.hexdigest()
            with patch.object(w.hashlib, 'sha256', return_value=MutatingDigest()):
                with self.assertRaises(ValueError):
                    w.report_metadata(report)

    def test_profile_command_has_exact_durable_export_and_supported_flags(self):
        report = Path('/tmp/frozen.ncu-rep')
        symbol = 'void Fake<T>(Params)'
        argv = w.profile_argv('/ncu', Path('/payload'), report, symbol)
        for flag, value in (('--export', str(report)), ('--rename-kernels', '0'),
                ('--print-kernel-base', 'demangled'), ('--import-sass', 'off'),
                ('--replay-mode', 'kernel'), ('--launch-count', '1'),
                ('--kernel-name', symbol), ('--config-file', '0')):
            self.assertEqual(argv[argv.index(flag)+1], value)
        self.assertNotIn('regex:', argv[argv.index('--kernel-name')+1])
        self.assertEqual(argv[-1], '--counter-child')
        for unsupported in ('--units', '--nvtx', '--graph-profiling'):
            self.assertNotIn(unsupported, argv)
        with self.assertRaises(ValueError):
            w.profile_argv('/ncu', Path('/payload'), Path('relative.ncu-rep'), symbol)

    def test_import_command_only_reopens_same_report_without_target(self):
        report = Path('/tmp/frozen.ncu-rep')
        argv = w.import_argv('/ncu', report)
        self.assertEqual(argv[argv.index('--import')+1], str(report))
        self.assertEqual(argv[argv.index('--rename-kernels')+1], '0')
        self.assertEqual(argv[argv.index('--print-kernel-base')+1], 'demangled')
        self.assertEqual(argv[argv.index('--config-file')+1], '0')
        for absent in ('--export', '--metrics', '--replay-mode', '--counter-child', '--units', sys.executable):
            self.assertNotIn(absent, argv)
        with self.assertRaises(ValueError):
            w.import_argv('/ncu', Path('/tmp/not-a-report.txt'))
        query = w.query_argv('/ncu', w.METRIC_BASES[0])
        self.assertEqual(query[query.index('--config-file')+1], '0')

    def test_roundtrip_ignores_raw_formatting_but_matches_values_and_units(self):
        original = parsed()
        imported = copy.deepcopy(original)
        for value in imported['metrics'].values():
            value['raw'] = 'different numeric text formatting'
        metadata = dict(bytes=100, sha256='a'*64)
        result = w.compare_roundtrip(original, imported, metadata, dict(metadata))
        self.assertTrue(result['passed'])
        self.assertTrue(result['report_unchanged'])

    def test_roundtrip_rejects_report_name_geometry_or_row_count_changes(self):
        original = parsed()
        metadata = dict(bytes=100, sha256='a'*64)
        for field, value in (('kernel_name', original['kernel_name']+' '),
                ('grid', [131, 1, 1]), ('block', [128, 1, 1]), ('row_count', 2)):
            imported = copy.deepcopy(original)
            imported[field] = value
            with self.subTest(field=field):
                self.assertFalse(w.compare_roundtrip(original, imported, metadata, metadata)['passed'])
        for changed in (dict(metadata, bytes=101), dict(metadata, sha256='b'*64)):
            with self.subTest(metadata=changed):
                self.assertFalse(w.compare_roundtrip(original, original, metadata, changed)['passed'])

    def test_roundtrip_rejects_missing_extra_metric_changed_value_or_unit(self):
        original = parsed()
        metadata = dict(bytes=100, sha256='a'*64)
        metric = w.METRICS[0]
        for variation in ('missing', 'extra', 'value', 'unit'):
            imported = copy.deepcopy(original)
            if variation == 'missing':
                del imported['metrics'][metric]
            elif variation == 'extra':
                imported['metrics']['unrequested'] = dict(value=1, unit='%')
            else:
                imported['metrics'][metric][variation] = 999 if variation == 'value' else 'cycle'
            with self.subTest(variation=variation):
                self.assertFalse(w.compare_roundtrip(original, imported, metadata, metadata)['passed'])


if __name__ == '__main__':
    unittest.main()
