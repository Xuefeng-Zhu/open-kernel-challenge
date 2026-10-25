"""Identity inspection guards using only fake functions, files and metadata."""
import hashlib
import importlib.util
import inspect
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('unsplit_identity_tests', HERE/'identity.py')
identity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(identity)


def explicit(q, *, num_splits=0):
    raise AssertionError('Inspection must never call the function')


def kwargs_only(q, **kwargs):
    raise AssertionError('Inspection must never call the function')


def positional_only(q, num_splits, /):
    raise AssertionError('Inspection must never call the function')


class IdentityGuards(unittest.TestCase):
    def test_explicit_keyword_api_admitted_without_calling_or_importing_gpu(self):
        with patch.object(identity.importlib.metadata, 'version', return_value='test-version'):
            result = identity.inspect_fa3(explicit)
        self.assertTrue(result['admitted'])
        self.assertEqual(result['num_splits_kind'], 'KEYWORD_ONLY')
        self.assertFalse(result['follow_wrapped'])
        self.assertEqual(result['packages']['torch']['version'], 'test-version')
        self.assertEqual(result['source']['function_source_sha256'],
                         hashlib.sha256(inspect.getsource(explicit).encode()).hexdigest())
        self.assertFalse(result['exact_observed_extensions_verified'])

    def test_unsupported_api_fails_before_provenance_reads(self):
        with patch.object(identity, '_source_record', side_effect=AssertionError('must not read')):
            missing = identity.inspect_fa3(kwargs_only)
            positional = identity.inspect_fa3(positional_only)
        self.assertFalse(missing['admitted'])
        self.assertEqual(missing['reason'], 'explicit_num_splits_parameter_missing')
        self.assertFalse(positional['admitted'])
        self.assertEqual(positional['reason'], 'num_splits_not_keyword_capable')

    def test_signature_unavailable_stops(self):
        with patch.object(identity.inspect, 'signature', side_effect=ValueError('unavailable')):
            result = identity.inspect_fa3(explicit)
        self.assertFalse(result['admitted'])
        self.assertEqual(result['reason'], 'signature_unavailable')

    def test_file_hash_budgets_and_nonfiles(self):
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp)/'binary.so'
            file.write_bytes(b'12345678')
            budget = {'remaining': 8, 'hashed_bytes': 0}
            result = identity._file_record(file, budget)
            self.assertEqual(result['sha256'], hashlib.sha256(b'12345678').hexdigest())
            self.assertEqual(budget, {'remaining': 0, 'hashed_bytes': 8})
            blocked = identity._file_record(file, {'remaining': 7, 'hashed_bytes': 0})
            self.assertEqual(blocked['hash_omitted_reason'], 'total_hash_byte_limit')
            with patch.object(identity, 'MAX_HASH_FILE_BYTES', 4):
                blocked = identity._file_record(file, {'remaining': 8, 'hashed_bytes': 0})
            self.assertEqual(blocked['hash_omitted_reason'], 'per_file_hash_byte_limit')
            self.assertEqual(identity._file_record(temp, budget)['hash_omitted_reason'], 'not_regular_file')

    def test_loaded_extensions_limited_to_relevant_already_loaded_files(self):
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp)/'already_loaded.so'
            file.write_bytes(b'extension bytes')
            modules = {'sgl_kernel.fake_extension': SimpleNamespace(__file__=str(file)),
                       'other.fake_extension': SimpleNamespace(__file__=str(file)),
                       'flash_attn.alias': SimpleNamespace(__file__=str(file))}
            with patch.dict(sys.modules, modules), patch.object(identity.importlib.metadata, 'version', return_value='test'):
                report = identity.inspect_fa3(explicit)
            rows = [row for row in report['loaded_binaries'] if row['path'] == str(file.resolve())]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['sha256'], hashlib.sha256(b'extension bytes').hexdigest())
            self.assertTrue(rows[0]['module'].startswith(('sgl_kernel', 'flash_attn')))
            self.assertTrue(report['exact_observed_extensions_verified'])

    def test_torch_loaded_libraries_discovers_relevant_paths_without_import(self):
        with tempfile.TemporaryDirectory() as temp:
            relevant = Path(temp)/'libsgl_flash_attn.so'
            unrelated = Path(temp)/'libunrelated.so'
            relevant.write_bytes(b'loaded fa3 extension')
            unrelated.write_bytes(b'not part of this identity inventory')
            fake_torch = SimpleNamespace(ops=SimpleNamespace(loaded_libraries={
                str(relevant), str(unrelated)}))
            with patch.dict(sys.modules, {'torch': fake_torch}), patch.object(
                    identity.importlib.metadata, 'version', return_value='test'):
                report = identity.inspect_fa3(explicit)
            self.assertTrue(report['exact_observed_extensions_verified'])
            row = next(row for row in report['loaded_binaries'] if row['path'] == str(relevant.resolve()))
            self.assertIn('torch.ops.loaded_libraries', row['discovered_by'])
            self.assertEqual(row['sha256'], hashlib.sha256(relevant.read_bytes()).hexdigest())
            self.assertNotIn(str(unrelated), [row['path'] for row in report['loaded_binaries']])

    def test_canonical_module_and_loaded_library_aliases_hash_once(self):
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp)/'libflash_attn.so'
            alias = Path(temp)/'alias_sgl_flash_attn.so'
            file.write_bytes(b'one canonical binary')
            alias.symlink_to(file)
            fake_torch = SimpleNamespace(ops=SimpleNamespace(loaded_libraries={str(alias)}))
            modules = {'torch': fake_torch, 'sgl_kernel.extension': SimpleNamespace(__file__=str(file))}
            with patch.dict(sys.modules, modules), patch.object(identity.importlib.metadata, 'version', return_value='test'):
                report = identity.inspect_fa3(explicit)
            relevant = [row for row in report['loaded_binaries'] if row['path'] == str(file.resolve())]
            self.assertEqual(len(relevant), 1)
            self.assertEqual(set(relevant[0]['observed_paths']), {str(file), str(alias)})
            self.assertEqual(set(relevant[0]['discovered_by']), {
                'module:sgl_kernel.extension', 'torch.ops.loaded_libraries'})
            self.assertTrue(report['exact_observed_extensions_verified'])

    def test_missing_or_omitted_extension_hash_fails_strict_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp)/'libflash_attn.so'
            file.write_bytes(b'extension')
            modules = {'sgl_flash_attn.fake': SimpleNamespace(__file__=str(file))}
            with patch.dict(sys.modules, modules), patch.object(identity, 'MAX_HASH_FILE_BYTES', 4), patch.object(
                    identity.importlib.metadata, 'version', return_value='test'):
                report = identity.inspect_fa3(explicit)
            self.assertFalse(report['exact_observed_extensions_verified'])
            self.assertEqual(report['extension_identity_reason'], 'observed_extension_hash_unavailable')
            with patch.dict(sys.modules, modules), patch.object(identity, 'MAX_LOADED_BINARIES', 0), patch.object(
                    identity.importlib.metadata, 'version', return_value='test'):
                report = identity.inspect_fa3(explicit)
            self.assertFalse(report['exact_observed_extensions_verified'])
            self.assertEqual(report['loaded_binary_paths_omitted'], 1)

    def test_read_permission_failure_recorded_without_exposing_error_text(self):
        with patch.object(identity.Path, 'stat', side_effect=PermissionError('sensitive path detail')):
            row = identity._file_record('/not/read', {'remaining': 8, 'hashed_bytes': 0})
        self.assertIsNone(row['sha256'])
        self.assertEqual(row['read_error_type'], 'PermissionError')
        self.assertNotIn('sensitive path detail', str(row))

    def test_worker_timeout_is_never_converted_to_provenance_warning(self):
        with patch.object(identity.Path, 'stat', side_effect=TimeoutError('worker deadline')):
            with self.assertRaises(TimeoutError):
                identity._file_record('/not/read', {'remaining': 8, 'hashed_bytes': 0})
        with patch.object(identity.inspect, 'getsourcefile', side_effect=TimeoutError('worker deadline')):
            with self.assertRaises(TimeoutError):
                identity._source_record(explicit, {'remaining': 8, 'hashed_bytes': 0})
        with patch.object(identity.importlib.metadata, 'version', side_effect=TimeoutError('worker deadline')):
            with self.assertRaises(TimeoutError):
                identity.inspect_fa3(explicit)


if __name__ == '__main__':
    unittest.main(verbosity=2)
