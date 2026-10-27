"""A2-08 single-token extension; CPU stand-ins and source invariants only."""
import ast
import copy
from pathlib import Path
import unittest

import test_full_fused_prepare_component as full

HERE = Path(__file__).resolve().parent
SOURCE = HERE/'mixed_fa3_fused_full_small.py'


def entry():
    return next(n for n in ast.parse(SOURCE.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'kernel_fn')


class FullSmallDispatch(unittest.TestCase):
    def test_only_length_guard_changes(self):
        old = {n.name: n for n in ast.parse((HERE/'mixed_fa3_fused_full_verify1.py').read_text()).body
               if isinstance(n, ast.FunctionDef)}
        new = {n.name: n for n in ast.parse(SOURCE.read_text()).body if isinstance(n, ast.FunctionDef)}
        self.assertEqual(old.keys(), new.keys())
        for name in old:
            expected = copy.deepcopy(old[name])
            if name == 'kernel_fn':
                expected.body[1].test.values[1] = ast.parse('cfg["q_len"] in (1, 8)', mode='eval').body
            self.assertEqual(ast.dump(expected), ast.dump(new[name]), name)
        compile(SOURCE.read_bytes(), str(SOURCE), 'exec')

    def test_all_dispatch_paths(self):
        ns = dict(Inputs=dict, torch=full.torch, _fa3_full_verify_fused=lambda _: 'new',
                  _fa3_sliding_verify_fused=lambda _: 'sliding-fused',
                  _fa3_sliding_verify=lambda _: 'sliding', _fa3_full_verify=lambda _: 'full',
                  _e2_kernel_fn=lambda _: 'e2')
        exec(compile(ast.Module(body=[entry()], type_ignores=[]), 'entry', 'exec'), ns)
        for b in [1, 2, 3, 4, 5, 8, 16, 32, 33]:
            for length in [1, 7, 8, 9]:
                for window, rope in [(0, 0), (2048, 500000), (2047, 500000), (2048, 0)]:
                    expected = 'e2'
                    if window == rope == 0:
                        if length == 8 and b in (1, 16, 32): expected = 'full'
                        if length == 1 and b in (1, 16): expected = 'full'
                    if length == 8 and window == 2048 and rope == 500000: expected = 'sliding-fused'
                    if b == 1 and length in (1, 8) and window == rope == 0: expected = 'new'
                    self.assertEqual(ns['kernel_fn']({'config': dict(batch=b, q_len=length,
                        window=window, rope_theta=rope)}), expected)

    def test_actual_complete_entrypoint_L1_CPU(self):
        for page, scale in [(1, 1.), (64, 1/64)]:
            fresh = full.old.reference.generate_input(batch=1, ctx=64, heads='tp8', layer='full',
                q_len=1, page=page, kv_scale=scale, seed=5301+page, device='cpu')
            # Host-forwarding sentinel only, NOT an actual FA3 scheduler encoding.
            fresh['scheduler_metadata'] = full.torch.zeros(8, dtype=full.torch.int32)
            work = full.old.buffers.clone_inputs(fresh)
            trace = []
            ns, state = full.adapter(work, trace)
            exec(compile(ast.Module(body=[entry()], type_ignores=[]), 'entry', 'exec'), ns)
            out = ns['kernel_fn'](work)
            self.assertTrue(all(full.torch.isfinite(v.float()).all() for v in full.old.reference.ref_kernel(work)))
            ok, message = full.old.reference.check_implementation(work, full.old.reference.collect(work, out))
            self.assertTrue(ok, message)
            self.assertTrue(full.old.buffers.unchanged(work, fresh)[0])
            self.assertEqual(trace, ['prepare_full_model', 'attention', 'gate'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
