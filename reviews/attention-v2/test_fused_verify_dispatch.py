"""A2-06 complete-file dispatch/AST proofs; CPU stubs, not Triton execution."""
import ast
import copy
from pathlib import Path
import unittest

import test_fused_prepare_component as fused

HERE = Path(__file__).resolve().parent


class FusedVerifyDispatch(unittest.TestCase):
    def test_only_batch_restriction_changes(self):
        old = ast.parse((HERE / 'mixed_fa3_fused_prepare.py').read_text())
        new = ast.parse((HERE / 'mixed_fa3_fused_verify.py').read_text())
        before = {n.name: n for n in old.body if isinstance(n, ast.FunctionDef)}
        after = {n.name: n for n in new.body if isinstance(n, ast.FunctionDef)}
        self.assertEqual(before.keys(), after.keys())
        for name in before:
            expected = copy.deepcopy(before[name])
            if name == 'kernel_fn':
                del expected.body[1].test.values[0]  # only batch == 1 removed
            self.assertEqual(ast.dump(expected), ast.dump(after[name]), name)
        compile((HERE / 'mixed_fa3_fused_verify.py').read_bytes(), 'candidate', 'exec')

    def test_complete_dispatch_preserves_all_other_paths(self):
        tree = ast.parse((HERE / 'mixed_fa3_fused_verify.py').read_text())
        entry = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'kernel_fn')
        ns = dict(torch=fused.torch, Inputs=dict, _fa3_sliding_verify_fused=lambda _: 'fused',
                  _fa3_sliding_verify=lambda _: 'sliding', _fa3_full_verify=lambda _: 'full',
                  _e2_kernel_fn=lambda _: 'e2')
        exec(compile(ast.Module(body=[entry], type_ignores=[]), 'dispatch', 'exec'), ns)
        for b in [1, 2, 3, 4, 5, 8, 16, 32, 33]:
            for length in [1, 7, 8, 9]:
                for window, rope in [(0, 0), (2048, 500000), (2047, 500000), (2048, 0)]:
                    expected = 'e2'
                    if window == rope == 0:
                        if length == 8 and b in (1, 16, 32): expected = 'full'
                        if length == 1 and b in (1, 16): expected = 'full'
                    if length == 8 and window == 2048 and rope == 500000: expected = 'fused'
                    self.assertEqual(ns['kernel_fn']({'config': dict(batch=b, q_len=length,
                        window=window, rope_theta=rope)}), expected)

    def test_complete_entrypoint_uses_generic_component_CPU_model(self):
        tree = ast.parse((HERE / 'mixed_fa3_fused_verify.py').read_text())
        entry = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'kernel_fn')
        for b, page, scale in [(3, 1, 1/64), (4, 64, 1.), (16, 1, 1.), (32, 1, 1.)]:
            fresh = fused.sliding.old.reference.generate_input(batch=b, ctx=32, heads='tp8',
                layer='sliding', q_len=8, spread='mixed', page=page, kv_scale=scale,
                seed=4800+b, device='cpu')
            work = fused.sliding.old.buffers.clone_inputs(fresh)
            trace = []
            ns, _ = fused.adapter(work, trace)
            exec(compile(ast.Module(body=[entry], type_ignores=[]), 'entrypoint', 'exec'), ns)
            out = ns['kernel_fn'](work)
            oracle = fused.sliding.old.reference.ref_kernel(work)
            self.assertTrue(all(fused.torch.isfinite(x.float()).all() for x in oracle))
            ok, msg = fused.sliding.old.reference.check_implementation(work,
                fused.sliding.old.reference.collect(work, out))
            self.assertTrue(ok, msg)
            self.assertTrue(fused.sliding.old.buffers.unchanged(work, fresh)[0])
            self.assertEqual(trace, ['prepare_fused_model', 'attention', 'gate'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
