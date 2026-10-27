"""A2-04 single-request full decode local tests; native execution remains separate."""
import ast
import copy
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_mixed_fa3 as old
import torch

SOURCE = HERE/'mixed_fa3_decode_small.py'


def entrypoint():
    return next(n for n in ast.parse(SOURCE.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'kernel_fn')


def adapter(data, trace):
    ns = old.cpu_adapter(data, trace)
    exec(compile(ast.Module(body=[entrypoint()], type_ignores=[]), 'A2_04_entry', 'exec'), ns)
    return ns


class FullDecode1(unittest.TestCase):
    def test_only_decode_batch_membership_changes(self):
        oldf = {n.name: n for n in ast.parse((HERE/'mixed_fa3_decode16.py').read_text()).body
                if isinstance(n, ast.FunctionDef)}
        newf = {n.name: n for n in ast.parse(SOURCE.read_text()).body if isinstance(n, ast.FunctionDef)}
        self.assertEqual(oldf.keys(), newf.keys())
        for name in oldf:
            if name != 'kernel_fn':
                self.assertEqual(ast.dump(oldf[name]), ast.dump(newf[name]), name)
        actual = copy.deepcopy(newf['kernel_fn'])
        compare = actual.body[1].test.values[0]
        self.assertEqual([n.value for n in compare.comparators[0].elts], [1, 16])
        actual.body[1].test.values[0] = copy.deepcopy(oldf['kernel_fn'].body[1].test.values[0])
        self.assertEqual(ast.dump(actual), ast.dump(oldf['kernel_fn']))
        compile(SOURCE.read_bytes(), str(SOURCE), 'exec')

    def test_all_controls_and_nearby_dispatch(self):
        ns = dict(torch=torch, Inputs=dict, _fa3_sliding_verify=lambda _: 'sliding',
                  _fa3_full_verify=lambda _: 'full', _e2_kernel_fn=lambda _: 'e2')
        exec(compile(ast.Module(body=[entrypoint()], type_ignores=[]), 'A2_04_dispatch', 'exec'), ns)
        for b in [1, 2, 8, 15, 16, 17, 31, 32, 33]:
            for l in [1, 2, 7, 8, 9]:
                for window, rope in [(0, 0), (2048, 500000), (0, 500000), (2047, 500000)]:
                    expected = 'e2'
                    if b in (1, 16, 32) and l == 8:
                        if window == 2048 and rope == 500000:
                            expected = 'sliding'
                        if window == rope == 0:
                            expected = 'full'
                    if b in (1, 16) and l == 1 and window == rope == 0:
                        expected = 'full'
                    self.assertEqual(ns['kernel_fn']({'config': dict(batch=b, q_len=l,
                        window=window, rope_theta=rope)}), expected)

    def test_new_path_oracle_cache_scales_heads_and_scheduler(self):
        for ctx, page, scale, heads, dist in [(0, 1, 1., 'tp8', 'normal'),
                (64, 64, 1/64, 'dp', 'soft'), (3000, 1, 1., 'tp8', 'coherent')]:
            fresh = old.reference.generate_input(batch=1, ctx=ctx, heads=heads, layer='full',
                q_len=1, page=page, kv_scale=scale, dist=dist, seed=2300+ctx, device='cpu')
            fresh['scheduler_metadata'] = torch.arange(64, dtype=torch.int32)
            old.zero_new_page(fresh)
            work = old.buffers.clone_inputs(fresh)
            trace = []
            out = adapter(work, trace)['kernel_fn'](work)
            ok, msg = old.reference.check_implementation(work, old.reference.collect(work, out))
            self.assertTrue(ok, msg)
            self.assertTrue(old.buffers.unchanged(work, fresh)[0])
            self.assertEqual(trace, ['norm', 'store', 'attention', 'gate'])

    def test_changed_content_lengths_schedule_same_address_A_B_A(self):
        for page, scale in [(1, 1.), (64, 1/64)]:
            args = dict(batch=1, ctx=32, heads='tp8', layer='full', q_len=1,
                        spread='mixed', page=page, kv_scale=scale, device='cpu')
            a = old.reference.generate_input(**args, seed=803)
            b = old.reference.generate_input(**args, seed=804)
            a['scheduler_metadata'] = torch.arange(64, dtype=torch.int32)
            b['scheduler_metadata'] = torch.arange(64, dtype=torch.int32)+1
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
                self.assertTrue(all(work[k].data_ptr() == p for k, p in pointers.items()))
                if i == 0:
                    first = raw.clone()
                if i == 2:
                    self.assertTrue(torch.equal(first, raw))


if __name__ == '__main__':
    unittest.main(verbosity=2)
