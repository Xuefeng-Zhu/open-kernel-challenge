"""A2-01 local full-verify1 adapter tests; CPU stand-ins, not native CUDA."""
import ast
import copy
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_mixed_fa3 as old
import torch

SOURCE = HERE/'mixed_fa3_verify1.py'


def entrypoint():
    return next(n for n in ast.parse(SOURCE.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'kernel_fn')


def adapter(data, trace):
    # Parent full-helper body is AST-proven unchanged. Exercise the actual new
    # entrypoint with the existing independent full-attention CPU stand-ins.
    ns = old.cpu_adapter(data, trace)
    exec(compile(ast.Module(body=[entrypoint()], type_ignores=[]), 'A2_01_entry', 'exec'), ns)
    return ns


class FullVerify1(unittest.TestCase):
    def test_only_one_condition_changes_and_other_functions_unchanged(self):
        a = ast.parse((HERE/'mixed_fa3_sliding.py').read_text())
        b = ast.parse(SOURCE.read_text())
        oldf = {n.name: n for n in a.body if isinstance(n, ast.FunctionDef)}
        newf = {n.name: n for n in b.body if isinstance(n, ast.FunctionDef)}
        self.assertEqual(oldf.keys(), newf.keys())
        for name, node in oldf.items():
            if name != 'kernel_fn':
                self.assertEqual(ast.dump(node), ast.dump(newf[name]), name)
        actual = copy.deepcopy(newf['kernel_fn'])
        # Remove only added literal1 from full-attention batch membership.
        full = actual.body[2]
        membership = full.test.values[-1]
        self.assertEqual([e.value for e in membership.comparators[0].elts], [1, 16, 32])
        del membership.comparators[0].elts[0]
        self.assertEqual(ast.dump(actual), ast.dump(oldf['kernel_fn']))
        compile(SOURCE.read_bytes(), str(SOURCE), 'exec')

    def test_dispatch_controls_and_nearby_shapes(self):
        ns = dict(torch=torch, Inputs=dict, _fa3_sliding_verify=lambda _: 'sliding',
                  _fa3_full_verify=lambda _: 'full', _e2_kernel_fn=lambda _: 'e2')
        exec(compile(ast.Module(body=[entrypoint()], type_ignores=[]), 'A2_01_dispatch', 'exec'), ns)
        for batch in [1, 2, 3, 8, 16, 32, 33]:
            for length in [1, 7, 8, 9]:
                for window, rope in [(0, 0), (2048, 500000), (0, 500000), (2047, 500000)]:
                    cfg = dict(batch=batch, q_len=length, window=window, rope_theta=rope)
                    expected = 'e2'
                    if batch == 1 and length == 8 and window == 2048 and rope == 500000:
                        expected = 'sliding'
                    if batch in (1, 16, 32) and length == 8 and window == rope == 0:
                        expected = 'full'
                    self.assertEqual(ns['kernel_fn']({'config': cfg}), expected)

    def test_full_boundaries_page_scale_cache_zero_and_coherent(self):
        for ctx, page, scale, dist in [(0, 1, 1., 'normal'), (1, 1, 1., 'normal'),
                                      (64, 64, 1/64, 'soft'), (2048, 1, 1., 'coherent')]:
            with self.subTest(ctx=ctx, page=page, scale=scale, dist=dist):
                fresh = old.reference.generate_input(batch=1, ctx=ctx, heads='tp8', layer='full',
                    q_len=8, page=page, kv_scale=scale, dist=dist, seed=1700+ctx, device='cpu')
                if ctx % page == 0:
                    old.zero_new_page(fresh)
                work = old.buffers.clone_inputs(fresh)
                trace = []
                out = adapter(work, trace)['kernel_fn'](work)
                ok, msg = old.reference.check_implementation(work, old.reference.collect(work, out))
                self.assertTrue(ok, msg)
                self.assertTrue(old.buffers.unchanged(work, fresh)[0])
                self.assertEqual(trace, ['norm', 'store', 'attention', 'gate'])

    def test_full_same_address_A_B_A_changed_lengths(self):
        for page, scale in [(1, 1.), (64, 1/64)]:
            args = dict(batch=1, ctx=64, heads='tp8', layer='full', q_len=8,
                        spread='mixed', page=page, kv_scale=scale, device='cpu')
            a = old.reference.generate_input(**args, seed=803)
            b = old.reference.generate_input(**args, seed=804)
            self.assertFalse(torch.equal(a['seq_lens'], b['seq_lens']))
            work = old.buffers.clone_inputs(a)
            pointers = {k: v.data_ptr() for k, v in work.items() if isinstance(v, torch.Tensor)}
            ns = adapter(work, [])
            first = None
            for i, fresh in enumerate([a, b, a]):
                old.buffers.restore(work, fresh)
                out = ns['kernel_fn'](work)
                raw = old.reference.collect(work, out)
                ok, msg = old.reference.check_implementation(work, raw)
                self.assertTrue(ok, msg)
                self.assertTrue(old.buffers.unchanged(work, fresh)[0])
                self.assertTrue(all(work[k].data_ptr() == ptr for k, ptr in pointers.items()))
                if i == 0:
                    first = raw.clone()
                if i == 2:
                    self.assertTrue(torch.equal(first, raw))


if __name__ == '__main__':
    unittest.main(verbosity=2)
