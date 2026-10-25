"""Local source/protocol guards. No candidate import, GPU or network activity."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('probe_worker', HERE/'worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class ProbeSourceTests(unittest.TestCase):
    def test_immutable_source_bytes(self):
        raw = (HERE/'candidate.py').read_bytes()
        original = (HERE.parent.parent/'mixed_fa3_fp8_qk.py').read_bytes()
        self.assertEqual(raw, original)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), worker.SOURCE_SHA256)

    def test_only_operand_selector_changes(self):
        tree = ast.parse((HERE/'worker.py').read_text())
        warmups = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Attribute) and n.func.attr == 'warmup']
        self.assertEqual(len(warmups), 1)
        self.assertEqual(ast.unparse(warmups[0].func), 'module._fused.warmup')
        keywords = {k.arg: ast.unparse(k.value) for k in warmups[0].keywords if k.arg}
        self.assertEqual(keywords, {'DOT': 'tl.bfloat16', 'QK_DOT': 'qk_dot', 'grid': '(batch, 2, 11)'})
        self.assertIn("[('fp8', tl.float8e4nv), ('bf16', tl.bfloat16)]", ast.unparse(tree))

    def test_full_verify_geometry_and_counter_merge(self):
        self.assertEqual(worker.specialization(), dict(L=8, G=8, HQ=16, HKV=2, D=64,
            BLOCK_M=64, BLOCK_N=64, BLOCK_L=16, SPLITS=11, ROPE=False,
            WINDOWED=False, CHUNK_SIZE=8192, SEPARATE_MERGE=False, num_warps=4, num_stages=3))
        self.assertEqual((86000+8+8191)//8192, 11)

    def test_no_candidate_kernel_execution(self):
        tree = ast.parse((HERE/'worker.py').read_text())
        self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Subscript)
                             for n in ast.walk(tree)))
        self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                             and n.func.attr in {'kernel_fn', '_e2_kernel_fn', 'remote', 'remote_gen'}
                             for n in ast.walk(tree)))

    def test_precision_and_dispatch_preserved(self):
        tree = ast.parse((HERE/'candidate.py').read_text())
        fused = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_fused')
        text = ast.unparse(fused)
        self.assertIn('q.to(tl.bfloat16).to(tl.float8e4nv).to(QK_DOT)', text)
        self.assertIn('kq.to(QK_DOT)', text)
        self.assertIn('other=0.0).to(QK_DOT)', text)
        qk_dots = [n for n in ast.walk(fused) if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Attribute) and n.func.attr == 'dot'
                   and any(k.arg == 'max_num_imprecise_acc' for k in n.keywords)]
        self.assertEqual(len(qk_dots), 2)
        self.assertTrue(all(any(k.arg == 'max_num_imprecise_acc' and isinstance(k.value, ast.Constant)
                               and k.value.value == 0 for k in n.keywords) for n in qk_dots))
        self.assertIn('sem=\'acq_rel\'', text)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ProbeSourceTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    receipt = dict(status='passed' if result.wasSuccessful() else 'failed',
        tests_run=result.testsRun, gpu_compiled=False, gpu_launched=False,
        correctness_tested=False, performance_tested=False,
        source_sha256=worker.SOURCE_SHA256,
        worker_sha256=hashlib.sha256((HERE/'worker.py').read_bytes()).hexdigest())
    (HERE/'local-verification.json').write_text(json.dumps(receipt, indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
