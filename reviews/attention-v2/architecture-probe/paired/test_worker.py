"""Local semantic/protocol safety checks; no candidate imports, CUDA or network."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('paired_worker', HERE/'worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class PairedProtocolTests(unittest.TestCase):
    def test_helper_is_original_with_only_operand_selection(self):
        original_tree = ast.parse((HERE.parent/'probe/candidate.py').read_text())
        original = copy.deepcopy(next(n for n in original_tree.body if isinstance(n, ast.FunctionDef) and n.name == '_e2_kernel_fn'))
        helper_tree = ast.parse((HERE/'candidatecustomhelper.py').read_text())
        helper = copy.deepcopy(next(n for n in helper_tree.body if isinstance(n, ast.FunctionDef)))
        helper.name = '_e2_kernel_fn'
        class Normalize(ast.NodeTransformer):
            def visit_Attribute(self, node):
                if isinstance(node.value, ast.Name) and node.value.id == 'candidate':
                    return ast.copy_location(ast.Name(id=node.attr, ctx=node.ctx), node)
                return self.generic_visit(node)
            def visit_keyword(self, node):
                if node.arg == 'QK_DOT':
                    self.assert_operand(node.value)
                    node.value = ast.Name(id='_QK_DOT', ctx=ast.Load())
                    return node
                return self.generic_visit(node)
            @staticmethod
            def assert_operand(node):
                assert ast.unparse(node) == 'tl.bfloat16'
        helper = Normalize().visit(helper)
        self.assertEqual(ast.dump(helper, include_attributes=False), ast.dump(original, include_attributes=False))
        self.assertFalse(any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) for t in n.targets)
                             for n in ast.walk(helper_tree)))

    def test_boundary_lengths_and_l8_metadata(self):
        a, b = worker.boundary_lengths(0), worker.boundary_lengths(1)
        self.assertEqual(len(a), 16)
        self.assertEqual(set(a), {0, 1, 63, 64, 65, 8191, 8192, 8193})
        self.assertEqual(set(a), set(b))
        self.assertNotEqual(a, b)
        positions = [[seq+j for j in range(8)] for seq in a]
        self.assertEqual(sum(map(len, positions)), 128)
        self.assertEqual((max(a)+8+63)//64, 129)
        text = (HERE/'worker.py').read_text()
        self.assertIn("pos = lengths.to(t.int64)[:, None]+t.arange(8, device='cuda')", text)
        self.assertIn("data['cache_seqlens'].copy_(lengths+8)", text)
        self.assertIn('valid = (seq+8+63)//64', text)

    def test_scored_families_and_strict_support(self):
        self.assertEqual([x['batch'] for x in worker.TARGETS], [16, 32])
        for case in worker.TARGETS:
            self.assertEqual((case['q_len'], case['ctx'], case['page'], case['layer'], case['dist']), (8, 80000, 1, 'full', 'coherent'))
        self.assertEqual(worker.ADVERSARIAL['kv_scale'], 1/64)
        root = HERE.parent.parent
        for name, sha in worker.SUPPORT.items():
            self.assertEqual(hashlib.sha256((root/name).read_bytes()).hexdigest(), sha)

    def test_aba_oracle_gate_fails_closed(self):
        calls, rows = [], []
        class Fake:
            def __init__(self, name): self.name = name
            def replay_and_check(self, label):
                calls.append((self.name, label))
                return dict(passed=not(self.name == 'bf16_custom' and label == 'B'))
        with self.assertRaises(AssertionError):
            worker.execute_aba({n: Fake(n) for n in worker.NAMES}, lambda: None, rows.append)
        self.assertFalse(any(label == 'A2' for _, label in calls))
        self.assertFalse(rows[-1]['passed'])
        tree = ast.parse((HERE/'worker.py').read_text())
        replay = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'replay_and_check')
        self.assertIn('self.checked(version, True)', ast.unparse(replay))

    def test_reverse_order_pairs_hold_input_fixed(self):
        for i in range(0, worker.WARMUP_PAIRS+worker.MEASURED_PAIRS, 2):
            a, av = worker.order_and_version(i)
            b, bv = worker.order_and_version(i+1)
            self.assertEqual(a, list(reversed(b)))
            self.assertEqual(av, bv)
        samples = [dict(raw_us=dict(fa3=float(i+1), bf16_custom=float(i+1)/2)) for i in range(12)]
        result = worker.summarize_timing(samples)
        self.assertEqual(result['median_reverse_order_pair_ratio'], 2)
        self.assertEqual(len(result['reverse_order_pair_geomeans']), 6)
        with self.assertRaises(ValueError): worker.summarize_timing(samples[:-1])
        samples[0]['raw_us']['bf16_custom'] = float('nan')
        with self.assertRaises(ValueError): worker.summarize_timing(samples)

    def test_deadline_precedes_replay(self):
        class Never:
            def replay_and_check(self, label): raise AssertionError('Must not execute')
        def expired(): raise TimeoutError('Expired')
        with self.assertRaises(TimeoutError):
            worker.execute_aba({n: Never() for n in worker.NAMES}, expired, lambda _: None)

    def test_optional_profile_names_have_total_byte_bound(self):
        result = worker.bounded_kernel_names(['x'*17000, 'small']+[f'{i:03d}'+'é'*180 for i in range(100)])
        self.assertLessEqual(result['kernel_names_utf8_bytes'], 16384)
        self.assertLessEqual(len(result['kernel_names']), 64)
        self.assertGreater(result['omitted_kernel_name_count'], 0)
        self.assertFalse(result['kernel_names_truncated'])
        self.assertEqual(result['kernel_names_utf8_bytes'], sum(len(n.encode()) for n in result['kernel_names']))


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PairedProtocolTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    (HERE/'worker-local-verification.json').write_text(json.dumps(dict(
        status='passed' if result.wasSuccessful() else 'failed', tests_run=result.testsRun,
        gpu_launched=False, network_called=False,
        worker_sha256=hashlib.sha256((HERE/'worker.py').read_bytes()).hexdigest(),
        helper_sha256=hashlib.sha256((HERE/'candidatecustomhelper.py').read_bytes()).hexdigest()), indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
