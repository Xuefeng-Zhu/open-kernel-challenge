"""CPU checks for the bounded worker's source, command, parsing and pairing gates."""
import ast
import csv
import io
from pathlib import Path
import tempfile
import unittest

import worker

ROOT = Path(__file__).resolve().parent
ATTENTION = ROOT.parent / 'attention-fusion-lookahead/reviews/attention-v2'


def csv_fixture(name='_full2h', count=1, bad=None):
    text = io.StringIO(); writer = csv.writer(text)
    columns = ['ID', 'Kernel Name', 'Grid Size', 'Block Size', *worker.METRICS]
    writer.writerow(columns)
    writer.writerow(['', '', '', '', 'ns', 'byte', 'byte', 'sector', 'request', '%', '%', '%', 'register/thread'])
    for i in range(count):
        metrics = ['1234', '1,234', '1,000', '100', '25', '11.5', '12.5', '43.3', '128']
        if bad is not None: metrics[0] = bad
        writer.writerow([str(i), name, '(132, 1, 1)', '(256, 1, 1)', *metrics])
    return text.getvalue()


class SourceAndProtocolTests(unittest.TestCase):
    def test_all_twelve_shapes_are_contract_cases(self):
        cases = [worker.case(b, length, layer) for length in (1, 8)
                 for b in (1, 16, 32) for layer in ('sliding', 'full')]
        self.assertEqual(len(cases), 12)
        self.assertEqual(len({(x['batch'], x['q_len'], x['layer']) for x in cases}), 12)
        self.assertTrue(all(x['page'] == 1 and x['ctx'] == 80000 for x in cases))

    def test_reverse_order_within_each_pair(self):
        for i in range(0, 12, 2):
            self.assertEqual(worker.order(i), worker.order(i+1)[::-1])
        self.assertEqual(worker.order(0), worker.order(3))

    def test_pair_ratio_cancels_order_bias(self):
        samples = [dict(champion=12., v4=5.) if i % 2 == 0 else dict(champion=10., v4=6.)
                   for i in range(12)]
        summary = worker.paired_summary(samples)
        self.assertAlmostEqual(summary['median_pair_speedup'], 2.)
        self.assertEqual(summary['favorable_pairs'], 6)

    def test_pair_input_incomplete_or_invalid_rejected(self):
        for samples in ([], [dict(champion=1., v4=1.)]*11,
                        [dict(champion=1., v4=0.)]*12,
                        [dict(champion=float('nan'), v4=1.)]*12):
            with self.assertRaises(ValueError): worker.paired_summary(samples)

    def test_actual_control_routes_are_distinct(self):
        self.assertEqual(worker.family('champion', 16), 'FlashAttnFwdSm90')
        self.assertEqual(worker.family('champion', 32), '_fused')
        self.assertEqual(worker.family('v4', 32), '_full2h')
        with self.assertRaises(ValueError): worker.family('fa3', 32)

    def test_exact_saved_source_bindings(self):
        sources = {'selected.py': ATTENTION/'fusion-lookahead-probe/candidate.py',
                   'v4.py': ATTENTION/'user-v4-probe/submission_v4.py',
                   'reference.py': ATTENTION/'official/reference.py',
                   'checks.py': ATTENTION/'official/checks.py'}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, source in sources.items(): (root/name).write_bytes(source.read_bytes())
            self.assertEqual(worker.bind_sources(root), worker.SOURCES)
            (root/'v4.py').write_text('tampered')
            with self.assertRaisesRegex(ValueError, 'v4.py'): worker.bind_sources(root)

    def test_torch_imports_deferred_until_native_run(self):
        tree = ast.parse((ROOT/'worker.py').read_text())
        top_imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any('torch' in ast.unparse(n) or 'triton' in ast.unparse(n) for n in top_imports))


class CaptureTests(unittest.TestCase):
    def test_command_against_real_saved_2025_3_help(self):
        help_text = (ATTENTION/'fa3-next-evidence/fa3-b32-capture-executor/1-stdout.txt').read_text()
        capture = worker.capture_argv('/usr/bin/ncu', Path('/opt/probe'), Path('/tmp/probe/test.ncu-rep'), 'champion', 32)
        worker.validate_help(help_text, capture)
        worker.validate_help(help_text, worker.import_argv('/usr/bin/ncu', Path('/tmp/test.ncu-rep')))
        self.assertNotIn('--units', capture)
        self.assertEqual(capture[capture.index('--replay-mode')+1], 'kernel')
        self.assertEqual(capture[capture.index('--profile-from-start')+1], 'off')
        self.assertEqual(capture[capture.index('--launch-count')+1], '1')
        self.assertEqual(capture[capture.index('--metrics')+1].split(','), list(worker.HARDWARE_METRICS))
        self.assertEqual(len(worker.HARDWARE_METRICS), 7)
        self.assertEqual(len(worker.METRICS), 9)
        self.assertIn('--export', capture)
        self.assertNotIn('sudo', capture)

    def test_unsupported_flag_rejected_before_native(self):
        with self.assertRaises(ValueError): worker.validate_help('--csv', ['--made-up-flag'])
        with self.assertRaises(ValueError): worker.validate_help('--units', ['--units'])

    def test_relative_report_path_rejected(self):
        with self.assertRaises(ValueError):
            worker.capture_argv('ncu', Path('/opt/probe'), Path('test.ncu-rep'), 'v4', 16)

    def test_full_identity_units_and_values_retained(self):
        parsed = worker.parse_csv(csv_fixture('void namespace::_full2h<X>'), '_full2h')
        self.assertEqual(parsed['identity']['kernel_name'], 'void namespace::_full2h<X>')
        self.assertEqual(parsed['identity']['grid'], [132, 1, 1])
        self.assertEqual(parsed['metrics']['dram__bytes_read.sum'], dict(value=1234., unit='byte'))

    def test_empty_wrong_family_multi_capture_rejected(self):
        for text in ('==PROF== No kernels were profiled', csv_fixture(count=0),
                     csv_fixture(name='_merge'), csv_fixture(count=2)):
            with self.assertRaises(ValueError): worker.parse_csv(text, '_full2h')

    def test_missing_metric_does_not_fail_open(self):
        text = csv_fixture().replace(worker.METRICS[0], 'not_a_requested_metric')
        with self.assertRaises(ValueError): worker.parse_csv(text, '_full2h')

    def test_nonfinite_metrics_rejected(self):
        for number in ('nan', 'inf', '-1'):
            with self.assertRaises(ValueError): worker.parse_csv(csv_fixture(bad=number), '_full2h')

    def test_missing_units_and_launch_metadata_rejected(self):
        text = csv_fixture()
        rows = text.splitlines()
        for invalid in ('\n'.join([rows[0], rows[2]]), text.replace('(132, 1, 1)', 'abc132')):
            with self.assertRaises(ValueError): worker.parse_csv(invalid, '_full2h')

    def test_saved_four_metric_report_is_not_nine_metric_evidence(self):
        old = ATTENTION/'fa3-next-evidence/fa3-b32-capture-executor/artifacts/profile-imported.csv'
        with self.assertRaises(ValueError): worker.parse_csv(old.read_text(), 'FlashAttnFwdSm90')


class RealTorchByteTests(unittest.TestCase):
    """Real CPU tensors, never stubs or CUDA. Imports stay inside test execution."""

    def setUp(self):
        import torch
        self.torch = torch

    def test_zero_dim_float(self):
        t = self.torch
        a = t.tensor(1.25, dtype=t.float32, device='cpu')
        self.assertTrue(worker.bytes_equal(t, a, a.clone()))
        self.assertFalse(worker.bytes_equal(t, a, t.tensor(1.5, dtype=t.float32, device='cpu')))

    def test_other_scalar_types(self):
        t = self.torch
        for dtype, value in ((t.float64, 2.), (t.int64, 5), (t.int32, 3),
                             (t.bfloat16, 1.5), (t.bool, True), (t.uint8, 42)):
            with self.subTest(dtype=dtype):
                a = t.tensor(value, dtype=dtype, device='cpu')
                self.assertTrue(worker.bytes_equal(t, a, a.clone()))

    def test_noncontiguous_and_offset_views(self):
        t = self.torch
        base = t.arange(24, dtype=t.float32, device='cpu').reshape(4, 6)
        for a in (base.t(), base[1::2, ::2], base.reshape(-1)[3:15]):
            with self.subTest(shape=tuple(a.shape), stride=tuple(a.stride())):
                b = a.contiguous().clone()
                self.assertTrue(worker.bytes_equal(t, a, b))
                b.reshape(-1)[0] += 1
                self.assertFalse(worker.bytes_equal(t, a, b))

    def test_bfloat16_and_encoded_fp8(self):
        t = self.torch
        bf16 = t.tensor([[1., -2.], [3., 4.]], dtype=t.bfloat16, device='cpu').t()
        fp8 = t.tensor([0x00, 0x38, 0x80, 0x7F], dtype=t.uint8, device='cpu').view(t.float8_e4m3fn).reshape(2, 2).t()
        for a in (bf16, fp8, fp8[0, 0]):
            self.assertTrue(worker.bytes_equal(t, a, a.clone()))
        altered = fp8.contiguous().clone()
        altered.view(t.uint8)[0, 0] ^= 1
        self.assertFalse(worker.bytes_equal(t, fp8, altered))

    def test_signed_zero_is_not_byte_equal(self):
        t = self.torch
        positive = t.tensor(0., dtype=t.float32, device='cpu')
        negative = t.tensor(-0., dtype=t.float32, device='cpu')
        self.assertTrue(t.equal(positive, negative))
        self.assertFalse(worker.bytes_equal(t, positive, negative))

    def test_nan_payloads_use_byte_consistency(self):
        t = self.torch
        first = t.tensor([0x7FC00001], dtype=t.int32, device='cpu').view(t.float32).squeeze(0)
        second = t.tensor([0x7FC00002], dtype=t.int32, device='cpu').view(t.float32).squeeze(0)
        self.assertTrue(bool(t.isnan(first)))
        self.assertFalse(t.equal(first, first.clone()))
        self.assertTrue(worker.bytes_equal(t, first, first.clone()))
        self.assertFalse(worker.bytes_equal(t, first, second))

    def test_empty_and_layout_mismatches(self):
        t = self.torch
        empty = t.empty((0, 3), dtype=t.float32, device='cpu')
        self.assertTrue(worker.bytes_equal(t, empty, empty.clone()))
        self.assertFalse(worker.bytes_equal(t, empty, empty.reshape(0, 1)))
        a = t.tensor([1.], dtype=t.float32, device='cpu')
        self.assertFalse(worker.bytes_equal(t, a, a.to(t.float64)))

    def test_ownership_accepts_scalar_inputs_and_only_new_fp8_rows(self):
        t = self.torch
        raw = (t.arange(3*2*64, dtype=t.int64, device='cpu') % 256).to(t.uint8)
        fresh = {'config': {'num_kv_heads': 2, 'head_dim': 64},
                 'cache_loc': t.tensor([1], dtype=t.int64, device='cpu'),
                 'k_scale': t.tensor(1., dtype=t.float32, device='cpu'),
                 'v_scale': t.tensor(1., dtype=t.float32, device='cpu'),
                 'qkv': t.arange(16, dtype=t.bfloat16, device='cpu'),
                 'k_cache': raw.clone().view(t.float8_e4m3fn).reshape(3, 2, 64),
                 'v_cache': raw.clone().view(t.float8_e4m3fn).reshape(3, 2, 64)}
        work = worker.clone(t, fresh)
        work['k_cache'].view(t.uint8)[1].fill_(0x38)
        work['v_cache'].view(t.uint8)[1].fill_(0x40)
        worker.check_ownership(t, work, fresh)
        self.assertTrue(bool((work['k_cache'].view(t.uint8)[1] == 0x38).all()))
        work['k_scale'].add_(0.25)
        with self.assertRaisesRegex(ValueError, 'Immutable input changed: k_scale'):
            worker.check_ownership(t, work, fresh)
        work['k_scale'].copy_(fresh['k_scale'])
        work['v_cache'].view(t.uint8)[0, 0, 0] ^= 1
        with self.assertRaisesRegex(ValueError, 'Historical cache rows changed: v_cache'):
            worker.check_ownership(t, work, fresh)


if __name__ == '__main__':
    unittest.main()
