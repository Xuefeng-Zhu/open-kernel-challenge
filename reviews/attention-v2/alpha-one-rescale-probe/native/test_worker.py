"""Offline compiler-protocol and bounded artifact transport tests."""
import argparse
import ast
import base64
import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('alpha_compile_worker_protocol', HERE/'worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class CompileProtocol(unittest.TestCase):
    def test_exact_source_and_license_hashes_reject_mutation(self):
        self.assertEqual(worker.source_gate(), {**worker.SOURCE_HASHES, **worker.LICENSE_HASHES})
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            for name in worker.SOURCE_HASHES:
                (source/name).write_bytes((worker.PROBE/name).read_bytes())
            (source/'candidate.py').write_bytes(b'changed\n')
            with self.assertRaisesRegex(ValueError, 'Frozen source/license mismatch: candidate.py'):
                worker.source_gate(source, worker.REPO)
            (source/'candidate.py').unlink()
            (source/'candidate.py').symlink_to(worker.PROBE/'candidate.py')
            with self.assertRaisesRegex(ValueError, 'Symlink source/license refused'):
                worker.source_gate(source, worker.REPO)

    def test_only_four_fixed_main_units_and_strict_precision_options(self):
        options = worker.producer_options()
        self.assertEqual({key: options[key] for key in ['L', 'G', 'HQ', 'HKV', 'D', 'BLOCK_M', 'BLOCK_N', 'BLOCK_L', 'SPLITS']},
                         dict(L=1, G=8, HQ=16, HKV=2, D=64, BLOCK_M=16, BLOCK_N=64, BLOCK_L=16, SPLITS=11))
        self.assertTrue(options['PAGE_LOOKAHEAD'])
        self.assertTrue(options['SEPARATE_MERGE'])
        self.assertEqual(options['CHUNK_SIZE'], 8192)
        self.assertFalse(options['ROPE'] or options['WINDOWED'])
        self.assertEqual((options['num_warps'], options['num_stages']), (4, 3))
        self.assertEqual((worker.merge_options()['SPLITS'], worker.merge_options()['BS']), (11, 16))
        self.assertEqual(len(worker.ALLOWED), 24)
        self.assertEqual(worker.FORMATS, ['ttir', 'ttgir', 'llir', 'ptx', 'cubin', 'sass'])
        self.assertEqual((86001+8191)//8192, 11)
        self.assertEqual((2*16+2*2)*64, 2304)

    def test_no_subscript_kernel_calls_entrypoint_or_timing(self):
        source = (HERE/'worker.py').read_text()
        tree = ast.parse(source)
        self.assertFalse(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript) for node in ast.walk(tree)))
        for forbidden in ['.kernel_fn(', '.replay(', '.profiler', '.timed(', 'import modal', 'subprocess.Popen']:
            self.assertNotIn(forbidden, source)
        for required in ['module._fused.warmup(', 'module._merge.warmup(', 'kernel._init_handles()',
                         'DOT=tl.bfloat16, QK_DOT=tl.float8e4nv', 'grid=(32, 2, 11)', 'grid=(32, 16)',
                         'torch.empty(1,', 'timeout=10']:
            self.assertIn(required, source)
        compile(tree, str(HERE/'worker.py'), 'exec')

    def test_ptx_inventory_handles_predicates_ftz_comments_and_no_operand_ops(self):
        text = '''
        .reg .f32 %f<8>;
        // mul.f32 %f0, %f1, %f2;
        $L__BB0_1:
        @!%p1 bra.uni $L__BB0_1;
        mul.f32 %f0, %f1, %f2;
        mul.ftz.f32 %f1, %f2, %f3; // inline comment
        fma.rn.ftz.f32 %f0, %f1, %f2, %f3;
        mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%f0}, {%r0}, {%r1}, {%f0};
        wgmma.commit_group.sync.aligned;
        ld.local.u32 %r0, [%rd0];
        st.local.u32 [%rd0], %r0;
        ret;
        '''
        result = worker.ptx_instruction_counts(text)
        self.assertEqual(result['families']['fp32_mul'], 2)
        self.assertEqual(result['families']['fp32_fma'], 1)
        self.assertEqual(result['families']['branch'], 1)
        self.assertEqual(result['families']['mma'], 1)
        self.assertEqual(result['families']['wgmma'], 1)
        self.assertEqual(result['families']['local_load'], 1)
        self.assertEqual(result['families']['local_store'], 1)
        self.assertEqual(result['opcodes']['ret'], 1)

    def test_deterministic_full_artifact_bundle_round_trip_and_individual_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for index, name in enumerate(sorted(worker.ALLOWED)):
                (folder/name).write_bytes(bytes([index, 0, 255])+b'compiler evidence\n')
            left = worker.bundle_artifacts(folder)
            right = worker.bundle_artifacts(folder)
            self.assertEqual(left, right)
            packed = base64.b64decode(left['data'], validate=True)
            self.assertEqual(worker.digest(packed), left['compressed_sha256'])
            self.assertEqual(len(packed), left['compressed_size'])
            raw = gzip.decompress(packed)
            self.assertEqual(worker.digest(raw), left['uncompressed_sha256'])
            self.assertEqual(len(raw), left['uncompressed_size'])
            decoded = json.loads(raw)['files']
            self.assertEqual(set(decoded), worker.ALLOWED)
            for name, value in decoded.items():
                data = base64.b64decode(value, validate=True)
                self.assertEqual(data, (folder/name).read_bytes())
                self.assertEqual(left['files'][name], {'sha256':worker.digest(data), 'size':len(data)})
            self.assertLess(len(json.dumps(left).encode()), worker.MAX_REPORT)

    def test_artifact_bundle_refuses_unknown_symlink_and_directory_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder/'unexpected.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'outside allowlist'):
                worker.bundle_artifacts(folder)
            (folder/'unexpected.json').unlink()
            target = folder/'source'
            target.write_text('data')
            (folder/'baseline.producer.ptx').symlink_to(target)
            target.unlink()
            with self.assertRaises(ValueError):
                worker.bundle_artifacts(folder)
            (folder/'baseline.producer.ptx').unlink()
            (folder/'baseline.producer.ptx').mkdir()
            with self.assertRaises(ValueError):
                worker.bundle_artifacts(folder)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            target = folder/'real'
            target.mkdir()
            link = folder/'link'
            link.symlink_to(target)
            with self.assertRaisesRegex(ValueError, 'regular private directory'):
                worker.bundle_artifacts(link)

    def test_per_file_uncompressed_and_encoded_size_guards(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder/'baseline.producer.ptx').write_bytes(b'abcd')
            with patch.object(worker, 'MAX_FILE', 3):
                with self.assertRaisesRegex(ValueError, 'per-file bound'):
                    worker.bundle_artifacts(folder)
            with patch.object(worker, 'MAX_UNCOMPRESSED', 8):
                with self.assertRaisesRegex(ValueError, 'uncompressed bound'):
                    worker.bundle_artifacts(folder)
            with patch.object(worker, 'MAX_REPORT', 65540):
                with self.assertRaisesRegex(ValueError, 'encoded transport bound'):
                    worker.bundle_artifacts(folder)

    def test_source_failure_preserves_bounded_partial_without_gpu_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'result.json'
            args = argparse.Namespace(output=output, deadline=1, artifact_dir=None)
            with patch.object(worker, 'source_gate', side_effect=ValueError('source mismatch')):
                code = worker.main(args)
            self.assertEqual(code, 1)
            result = json.loads(output.read_text())
            self.assertFalse(result['passed'] or result['compile_complete'])
            self.assertEqual(result['kernel_launches'], 0)
            self.assertTrue(result['protocol']['compile_only'])
            self.assertFalse(result['correctness_tested'] or result['performance_tested'])
            self.assertEqual(result['kernels'], [])
            self.assertIsNone(result['artifact_bundle'])
            self.assertFalse(output.with_suffix('.tmp').exists())
            self.assertLess(output.stat().st_size, worker.MAX_REPORT)

    def test_deadline_cannot_expand_beyond_reviewed_worker_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            for deadline in [0, -1, 91, float('inf')]:
                with self.assertRaises(ValueError):
                    worker.main(argparse.Namespace(output=Path(temporary)/'result.json', deadline=deadline, artifact_dir=None))

    def test_metadata_serialization_is_bounded_and_json_safe(self):
        value = {'a':(1, 2.5, True, None), 'b':object()}
        converted = worker.metadata_value(value)
        json.dumps(converted)
        self.assertEqual(converted['a'], [1,2.5,True,None])
        self.assertIsInstance(converted['b'], str)
        self.assertLessEqual(len(converted['b']), 1024)


if __name__ == '__main__':
    unittest.main(verbosity=2)
