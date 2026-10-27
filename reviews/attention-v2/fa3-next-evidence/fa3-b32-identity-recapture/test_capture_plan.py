import copy
import csv
import importlib.util
import io
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('capture_plan_tested', HERE/'capture_plan.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.real = p.saved_regression_fixtures(HERE/'result.json')

    def header(self):
        stream = io.StringIO()
        csv.writer(stream).writerow(['ID', 'Kernel Name', 'Grid Size', 'Block Size', *p.METRICS])
        return stream.getvalue()

    def state(self):
        return dict(returncode=0, stdout=self.header(), report_path='/tmp/baseline.ncu-rep',
            bindings=copy.deepcopy(self.real['expected_native_identity']['bindings']),
            report={'bytes':123, 'sha256':'a'*64}, row=dict(row_count=1,
                kernel_name=self.real['ncu_available_name'], grid=[132,1,1], block=[256,1,1],
                metrics={name:{'value':1.25,'unit':'%'} for name in p.METRICS}))

    def test_family_matches_both_actual_demanglers(self):
        self.assertEqual((self.real['graph_characters'], self.real['ncu_available_characters']), (1380,870))
        self.assertTrue(p.matches_family(self.real['graph_name']))
        self.assertTrue(p.matches_family(self.real['ncu_available_name']))
        self.assertEqual(sum(p.matches_family(name) for name in self.real['available_kernel_names']), 1)
        self.assertFalse(p.matches_family('void flash::FlashAttnFwdCombine<Sm90>(Params)'))

    def test_graph_cross_demangler_equality_stays_unknown(self):
        result = p.graph_name_gate(self.real['graph_name'], self.real['ncu_available_name'])
        self.assertFalse(result['raw_string_equal'])
        self.assertEqual(result['cross_demangler_equivalence'], 'UNRESOLVED')
        self.assertIn('(int)64', self.real['ncu_available_name'])
        self.assertIn('T1::Params', self.real['ncu_available_name'])
        self.assertNotEqual(self.real['graph_name'], self.real['ncu_available_name'])

    def test_builder_uses_family_not_exact_torch_name(self):
        args = p.capture_command('/ncu', '/python', '/payload/worker.py', '/tmp/baseline.ncu-rep')
        self.assertEqual(args[args.index('--kernel-name')+1], 'regex:.*FlashAttnFwdSm90.*')
        self.assertNotIn(self.real['graph_name'], args)
        for flag,value in [('--kernel-name-base','demangled'),('--rename-kernels','0'),
                ('--print-kernel-base','demangled'),('--config-file','0'),
                ('--target-processes','application-only'),('--replay-mode','kernel'),
                ('--launch-skip','1'),('--launch-count','1')]:
            self.assertEqual(args[args.index(flag)+1], value)
        self.assertNotIn('--units', args)
        self.assertNotIn('--force-overwrite', args)
        self.assertNotIn('--nvtx', args)

    def test_import_same_report_and_optional_mangled_evidence(self):
        args = p.import_command('/ncu', '/tmp/baseline.ncu-rep')
        self.assertEqual(args[args.index('--import')+1], '/tmp/baseline.ncu-rep')
        self.assertEqual(args[args.index('--print-kernel-base')+1], 'demangled')
        self.assertNotIn('--replay-mode', args)
        mangled = p.mangled_import_command('/ncu', '/tmp/baseline.ncu-rep')
        self.assertEqual(mangled[mangled.index('--print-kernel-base')+1], 'mangled')
        self.assertEqual(mangled[mangled.index('--import')+1], '/tmp/baseline.ncu-rep')
        with self.assertRaises(ValueError):
            p.import_command('/ncu', 'relative.ncu-rep')

    def test_actual_empty_capture_is_rejected_despite_exit_zero_and_correctness(self):
        self.assertEqual(self.real['failed_returncode'], 0)
        self.assertTrue(all(row['passed'] for row in self.real['child_correctness']))
        result = p.capture_prerequisites(self.real['failed_returncode'], self.real['failed_stdout'], self.real['report_meta'])
        self.assertFalse(result['passed'])
        self.assertFalse(result['no_empty_capture_warning'])
        self.assertFalse(result['csv_header_present'])
        self.assertFalse(result['durable_report_present'])
        self.assertFalse(result['permission_denial_observed'])

    def test_empty_warning_beats_fabricated_header_and_report(self):
        result = p.capture_prerequisites(0, 'No kernels were profiled.\n'+self.header(), self.state()['report'])
        self.assertFalse(result['passed'])

    def test_valid_ncu_roundtrip_does_not_prove_graph_equivalence(self):
        state = self.state()
        result = p.capture_import_gate(state, copy.deepcopy(state), self.real['expected_native_identity'])
        self.assertTrue(result['accepted'])
        self.assertTrue(result['exact_ncu_signature_equal'])
        self.assertTrue(result['graph_abi_equivalence'].startswith('UNRESOLVED'))
        self.assertEqual(result['execution_state'], 'plan_only_not_executed')

    def test_roundtrip_rejects_each_material_mismatch(self):
        mutations = [lambda s:s.update(returncode=1), lambda s:s.update(report_path='/tmp/other.ncu-rep'),
            lambda s:s['report'].update(sha256='b'*64), lambda s:s['row'].update(kernel_name='wrong'),
            lambda s:s['row'].update(row_count=0), lambda s:s['row'].update(grid=[1,1,1]),
            lambda s:s['row']['metrics'][p.METRICS[0]].update(value=2),
            lambda s:s['row']['metrics'][p.METRICS[0]].update(unit='cycle'),
            lambda s:s.update(stdout='No kernels were profiled.'), lambda s:s.update(report=None)]
        for mutate in mutations:
            state = self.state(); other = copy.deepcopy(state); mutate(other)
            self.assertFalse(p.capture_import_gate(state, other, self.real['expected_native_identity'])['accepted'])

    def test_same_wrong_native_identity_on_both_sides_still_fails(self):
        expected = self.real['expected_native_identity']
        self.assertEqual(expected['grid'], [132,1,1])
        self.assertEqual(expected['block'], [256,1,1])
        changes = [lambda s:s['row'].update(grid=[1,1,1]),
            lambda s:s['row'].update(block=[128,1,1]),
            lambda s:s['row'].update(kernel_name=s['row']['kernel_name']+' other'),
            lambda s:s['bindings'].update(binary_sha256='b'*64)]
        for change in changes:
            state = self.state(); change(state)
            result = p.capture_import_gate(state, copy.deepcopy(state), expected)
            self.assertFalse(result['accepted'])
        self.assertFalse(p.capture_import_gate(self.state(), self.state(), None)['accepted'])

    def test_expected_binding_set_requires_all_15_saved_keys(self):
        expected = self.real['expected_native_identity']
        self.assertEqual(len(p.REQUIRED_BINDING_KEYS), 15)
        self.assertEqual(set(expected['bindings']), p.REQUIRED_BINDING_KEYS)
        incomplete = []
        for missing in p.REQUIRED_BINDING_KEYS:
            binding = dict(expected['bindings']); del binding[missing]
            incomplete.append(binding)
        incomplete.extend([{'foo': 'a'*64}, dict(expected['bindings'], foo='a'*64)])
        for bindings in incomplete:
            identity = dict(expected, bindings=bindings)
            state = self.state(); state['bindings'] = bindings
            self.assertFalse(p.capture_import_gate(state, copy.deepcopy(state), identity)['accepted'])

    def test_no_filter_single_launch_is_not_safe_for_current_child(self):
        self.assertEqual(len(self.real['graph_kernel_names']), 5)
        self.assertGreater(len(self.real['available_kernel_names']), 5)
        result = p.no_filter_assessment(self.real)
        self.assertFalse(result['safe_for_current_child'])
        self.assertIn('original-oracle', result['reason'])


if __name__ == '__main__':
    unittest.main()
