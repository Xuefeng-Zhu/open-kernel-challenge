"""A2-02 larger verify sliding CPU stand-ins; no native execution."""
import ast
import copy
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_mixed_fa3_sliding as sliding
import torch

SOURCE = HERE/'mixed_fa3_all_verify.py'


def entrypoint():
    return next(n for n in ast.parse(SOURCE.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'kernel_fn')


def adapter(data, trace):
    # Actual adapter helpers are AST-identical to E6; replace only entrypoint.
    ns = sliding.adapter(data, trace)
    exec(compile(ast.Module(body=[entrypoint()], type_ignores=[]), 'A2_02_entry', 'exec'), ns)
    return ns


class AllVerify(unittest.TestCase):
    def test_only_sliding_batch_dispatch_changes(self):
        old = {n.name: n for n in ast.parse((HERE/'mixed_fa3_verify1.py').read_text()).body
               if isinstance(n, ast.FunctionDef)}
        new = {n.name: n for n in ast.parse(SOURCE.read_text()).body if isinstance(n, ast.FunctionDef)}
        self.assertEqual(old.keys(), new.keys())
        for name in old:
            if name != 'kernel_fn':
                self.assertEqual(ast.dump(old[name]), ast.dump(new[name]), name)
        patched = copy.deepcopy(new['kernel_fn'])
        compare = patched.body[1].test.values[0]
        self.assertEqual([n.value for n in compare.comparators[0].elts], [1, 16, 32])
        patched.body[1].test.values[0] = copy.deepcopy(old['kernel_fn'].body[1].test.values[0])
        self.assertEqual(ast.dump(patched), ast.dump(old['kernel_fn']))
        compile(SOURCE.read_bytes(), str(SOURCE), 'exec')

    def test_controls_and_nearby_dispatch(self):
        ns = dict(torch=torch, Inputs=dict, _fa3_sliding_verify=lambda _: 'sliding',
                  _fa3_full_verify=lambda _: 'full', _e2_kernel_fn=lambda _: 'e2')
        exec(compile(ast.Module(body=[entrypoint()], type_ignores=[]), 'A2_02_dispatch', 'exec'), ns)
        for b in [1, 2, 8, 15, 16, 17, 31, 32, 33]:
            for l in [1, 7, 8, 9]:
                for window, rope in [(0, 0), (2048, 500000), (0, 500000), (2047, 500000)]:
                    expected = 'e2'
                    if b in (1, 16, 32) and l == 8:
                        if window == 2048 and rope == 500000:
                            expected = 'sliding'
                        if window == rope == 0:
                            expected = 'full'
                    cfg = dict(batch=b, q_len=l, window=window, rope_theta=rope)
                    self.assertEqual(ns['kernel_fn']({'config': cfg}), expected)

    def test_larger_batches_page_scale_zero_and_untouched_cache(self):
        for batch, ctx, page, scale in [(16, 17, 1, 1.), (32, 64, 64, 1/64)]:
            fresh = sliding.old.reference.generate_input(batch=batch, ctx=ctx, heads='tp8',
                layer='sliding', q_len=8, page=page, kv_scale=scale, seed=1900+batch, device='cpu')
            sliding.old.zero_new_page(fresh)
            work = sliding.old.buffers.clone_inputs(fresh)
            trace = []
            out = adapter(work, trace)['kernel_fn'](work)
            ok, msg = sliding.old.reference.check_implementation(work, sliding.old.reference.collect(work, out))
            self.assertTrue(ok, msg)
            self.assertTrue(sliding.old.buffers.unchanged(work, fresh)[0])
            self.assertEqual(trace, ['norm_rope', 'store', 'attention', 'gate'])

    def test_same_address_A_B_A_changed_lengths(self):
        for batch, page, scale in [(16, 1, 1.), (32, 64, 1/64)]:
            args = dict(batch=batch, ctx=32, heads='tp8', layer='sliding', q_len=8,
                        spread='mixed', page=page, kv_scale=scale, device='cpu')
            a = sliding.old.reference.generate_input(**args, seed=803)
            b = sliding.old.reference.generate_input(**args, seed=804)
            self.assertFalse(torch.equal(a['seq_lens'], b['seq_lens']))
            work = sliding.old.buffers.clone_inputs(a)
            pointers = {k: v.data_ptr() for k, v in work.items() if isinstance(v, torch.Tensor)}
            ns = adapter(work, [])
            first = None
            for i, fresh in enumerate([a, b, a]):
                sliding.old.buffers.restore(work, fresh)
                out = ns['kernel_fn'](work)
                raw = sliding.old.reference.collect(work, out)
                ok, msg = sliding.old.reference.check_implementation(work, raw)
                self.assertTrue(ok, msg)
                self.assertTrue(sliding.old.buffers.unchanged(work, fresh)[0])
                self.assertTrue(all(work[k].data_ptr() == ptr for k, ptr in pointers.items()))
                if i == 0:
                    first = raw.clone()
                if i == 2:
                    self.assertTrue(torch.equal(first, raw))


if __name__ == '__main__':
    unittest.main(verbosity=2)
