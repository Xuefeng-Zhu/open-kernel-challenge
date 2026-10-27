import csv
import importlib.util
import io
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('test_fa3_csv_helpers', HERE/'csv_helpers.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
METRICS = ('sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed',
    'dram__throughput.avg.pct_of_peak_sustained_elapsed',
    'smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct',
    'sm__warps_active.avg.pct_of_peak_sustained_active')


def fixture(values=('17.2', '45.6', '0', '12.5'), name='void kernel<64, 128, 64>(int)',
            units=('%', '%', '%', '%'), extra=False):
    stream = io.StringIO()
    stream.write('==PROF== Connected to process 12\n')
    out = csv.writer(stream)
    header = ['ID', 'Kernel Name', 'Grid Size', 'Block Size', *METRICS]
    out.writerow(header)
    out.writerow(['', '', '', '', *units])
    row = ['0', name, '(132, 1, 1)', '(256, 1, 1)', *values]
    out.writerow(row)
    if extra:
        out.writerow(['1', *row[1:]])
    stream.write('FA3_PROGRESS_JSON={"phase":"post","passed":true}\n')
    return stream.getvalue()


class Tests(unittest.TestCase):
    def test_full_name_and_geometry(self):
        name = 'void cutlass::kernel<'+'some::Type<64, 128>, '*80+'float>(Params)'
        result = c.parse_ncu_csv(fixture(name=name), METRICS)
        self.assertEqual(result['kernel_name'], name)
        self.assertEqual(result['grid'], [132, 1, 1])
        self.assertEqual(result['block'], [256, 1, 1])
        self.assertEqual(result['metrics'][METRICS[2]]['value'], 0)

    def test_actual_saved_tiny_wide_csv(self):
        text = (HERE.parent/'eager-counter-repair/3-stdout.txt').read_text()
        result = c.parse_ncu_csv(text, ['sm__cycles_elapsed.avg'])
        self.assertEqual(result['kernel_name'], 'okc_counter_gate')
        self.assertEqual(result['metrics']['sm__cycles_elapsed.avg'],
            dict(value=5541.35, unit='cycle', raw='5,541.35'))

    def test_multiple_kernel_rows_rejected(self):
        with self.assertRaises(ValueError):
            c.parse_ncu_csv(fixture(extra=True), METRICS)

    def test_missing_metric_rejected(self):
        with self.assertRaises(ValueError):
            c.parse_ncu_csv(fixture().replace(METRICS[0], 'other_metric'), METRICS)

    def test_duplicate_metric_column_rejected(self):
        with self.assertRaises(ValueError):
            c.parse_ncu_csv(fixture().replace(METRICS[1], METRICS[0]), METRICS)

    def test_nonfinite_and_bad_number_rejected(self):
        for value in ('NaN', 'inf', '1e999', '1,23', '', '-0.1'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                c.parse_ncu_csv(fixture(values=(value, '1', '1', '1')), METRICS)

    def test_percentage_over_peak_preserved_with_warning(self):
        result = c.parse_ncu_csv(fixture(values=('100.1', '1', '1', '1')), METRICS)
        self.assertEqual(result['metrics'][METRICS[0]]['value'], 100.1)
        self.assertEqual(result['range_warnings'][0]['metric'], METRICS[0])
        with self.assertRaises(ValueError):
            c.parse_ncu_csv(fixture(units=('cycle', '%', '%', '%')), METRICS)
        self.assertEqual(c.parse_ncu_csv(fixture(values=('100', '0', '0', '0')), METRICS)['range_warnings'], [])

    def test_grid_block_bad_or_zero_rejected(self):
        for replacement in ('(0, 1, 1)', '(132.0, 1, 1)', '(132, 1)', '132'):
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                c.parse_ncu_csv(fixture().replace('(132, 1, 1)', replacement), METRICS)

    def test_logs_and_echo_cannot_count_as_kernel(self):
        with self.assertRaises(ValueError):
            c.parse_ncu_csv('ncu --metrics '+','.join(METRICS)+'\n', METRICS)

    def test_malformed_actual_csv_rejected(self):
        with self.assertRaises(ValueError):
            c.parse_ncu_csv(fixture().replace('ID,Kernel Name', '"ID"broken,Kernel Name'), METRICS)

    def test_identity_only_normalizes_whitespace(self):
        row = c.parse_ncu_csv(fixture(name='void  kernel<64, 128, 64>(int)'), METRICS)
        self.assertTrue(c.compare_kernel_identity(row, 'void kernel<64, 128, 64>(int)', [132, 1, 1], [256, 1, 1])['matches'])
        self.assertFalse(c.compare_kernel_identity(row, 'void kernel<64,128,64>(int)', [132, 1, 1], [256, 1, 1])['matches'])
        self.assertFalse(c.compare_kernel_identity(row, row['kernel_name'], [131, 1, 1], [256, 1, 1])['matches'])

    def test_size_and_requested_metric_contract(self):
        with self.assertRaises(ValueError):
            c.parse_ncu_csv('x'*(c.MAX_CSV_BYTES+1), METRICS)
        with self.assertRaises(ValueError):
            c.parse_ncu_csv(fixture(), [METRICS[0], METRICS[0]])


if __name__ == '__main__':
    unittest.main()
