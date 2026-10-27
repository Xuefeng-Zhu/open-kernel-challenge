import ast
import importlib.util
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
ATTENTION = HERE.parents[1]
spec = importlib.util.spec_from_file_location('fa3_b32_worker_test', HERE/'worker.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


class Tests(unittest.TestCase):
    def test_exact_case_is_verify_not_decode(self):
        self.assertEqual((w.CASE['batch'], w.CASE['q_len'], w.CASE['layer'],
            w.CASE['page'], w.CASE['heads'], w.CASE['ctx']), (32, 8, 'full', 1, 'tp8', 80000))
        self.assertEqual(w.CASE['seed'], 135741)

    def test_current_selected_route_is_fused_fa3(self):
        text = (ATTENTION/'fusion-lookahead-probe/candidate.py').read_text()
        self.assertEqual(w.selected_route(text)['helper'], '_fa3_full_verify_fused')
        with self.assertRaises(ValueError):
            w.selected_route(text.replace('(32, 8)', '(32, 1)'))

    def test_profile_command_keeps_working_minimal_contract(self):
        a = w.profile_argv('/ncu', Path('/payload'), Path('/tmp/baseline.ncu-rep'), 'void Test(Params)')
        self.assertNotIn('--units', a)
        self.assertNotIn('--nvtx', a)
        self.assertEqual(a[a.index('--replay-mode')+1], 'kernel')
        self.assertEqual(a[a.index('--launch-skip')+1], '1')
        self.assertEqual(a[a.index('--launch-count')+1], '1')
        self.assertEqual(a[a.index('--metrics')+1].split(','), list(w.METRICS))
        self.assertEqual(a[-1], '--counter-child')

    def test_four_declared_metrics_only(self):
        self.assertEqual(len(w.METRICS), 4)
        self.assertIn('long_scoreboard_per_warp_active', w.METRIC_BASES[2])
        for metric in w.METRICS:
            self.assertTrue(w.metric_supported(' '+metric+' ', metric))
            self.assertFalse(w.metric_supported(metric+'.extra', metric))

    def test_current_installed_help_accepts_constructed_flags(self):
        p = ATTENTION/'fa3-next-evidence/eager-counter-repair'
        s = importlib.util.spec_from_file_location('actual_help_contract', p/'worker.py')
        module = importlib.util.module_from_spec(s)
        s.loader.exec_module(module)
        result = module.validate_installed_help((p/'1-stdout.txt').read_text(),
            w.profile_argv('/ncu', Path('/payload'), Path('/tmp/baseline.ncu-rep'), 'void Test(Params)'), w.query_argv('/ncu', w.METRIC_BASES[0]))
        self.assertTrue(result['all_flags_advertised'])

    def test_actual_help_accepts_offline_import_without_application(self):
        text = (ATTENTION/'fa3-next-evidence/eager-counter-repair/1-stdout.txt').read_text()
        args = w.import_argv('/ncu', Path('/tmp/baseline.ncu-rep'))
        self.assertNotIn(sys.executable, args)
        self.assertTrue(w.validate_import_help(text, args)['offline_only'])
        with self.assertRaises(ValueError):
            w.validate_import_help(text, args+['--replay-mode', 'kernel'])
        with self.assertRaises(ValueError):
            w.validate_import_help(text.replace('--print-kernel-base', '--unrelated-flag'), args)

    def test_csv_interpretation_follows_durable_import(self):
        tree = ast.parse((HERE/'worker.py').read_text())
        run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
        imports = [n for n in ast.walk(run) if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'import_command' for t in n.targets)]
        parses = [n for n in ast.walk(run) if isinstance(n, ast.Call)
            and ast.unparse(n.func) == 'csv_reader.parse_ncu_csv']
        self.assertEqual(len(imports), 1)
        self.assertEqual(len(parses), 2)
        self.assertTrue(all(n.lineno > imports[0].lineno for n in parses))

    def test_only_three_selected_eager_calls_no_capture(self):
        tree = ast.parse((HERE/'worker.py').read_text())
        child = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'counter_child')
        loops = [n for n in ast.walk(child) if isinstance(n, ast.For)]
        self.assertEqual([ast.literal_eval(n.iter) for n in loops], [('pre', 'measured', 'post')])
        calls = [ast.unparse(n.func) for n in ast.walk(child) if isinstance(n, ast.Call)]
        self.assertEqual(calls.count('selected.kernel_fn'), 1)
        self.assertNotIn('helpers.capture', calls)
        self.assertNotIn('torch.cuda.graph', calls)

    def test_progress_parser_requires_explicit_prefix(self):
        text = '==PROF== connected\nFA3_PROGRESS_JSON={"phase":"pre","passed":true}\n'
        self.assertEqual(w.progress_rows(text), [{'phase': 'pre', 'passed': True}])
        self.assertEqual(w.progress_rows('pre passed=true'), [])

    def test_top_level_is_cpu_only(self):
        tree = ast.parse((HERE/'worker.py').read_text())
        imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        names = {n.module for n in imports if isinstance(n, ast.ImportFrom)}
        names.update(alias.name for n in imports if isinstance(n, ast.Import) for alias in n.names)
        self.assertFalse({'torch', 'triton', 'modal', 'sgl_kernel'} & names)

    def test_fixed_source_hashes(self):
        paths = {'selected.py': ATTENTION/'fusion-lookahead-probe/candidate.py',
            'reference.py': ATTENTION/'official/reference.py', 'checks.py': ATTENTION/'official/checks.py',
            'modal_worker.py': ATTENTION/'modal_worker.py'}
        for name, path in paths.items():
            self.assertEqual(w.sha(path), w.EXPECTED[name])


if __name__ == '__main__':
    unittest.main()
