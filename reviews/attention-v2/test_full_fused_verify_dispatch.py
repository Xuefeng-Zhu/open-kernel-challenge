"""A2-07 complete entrypoint tests use CPU stand-ins, not native Triton/FA3."""
import ast
from pathlib import Path
import unittest

from build_full_fused_verify import build
import test_full_fused_prepare_component as full

HERE = Path(__file__).resolve().parent
SOURCE = HERE/'mixed_fa3_fused_full_verify1.py'


def entry():
    return next(n for n in ast.parse(SOURCE.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'kernel_fn')


class FullVerifyDispatch(unittest.TestCase):
    def test_exact_assembly_and_all_parent_helpers_unchanged(self):
        self.assertEqual(SOURCE.read_text(), build())

    def test_only_one_scored_path_changes(self):
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
                    if b == 1 and length == 8 and window == rope == 0: expected = 'new'
                    self.assertEqual(ns['kernel_fn']({'config': dict(batch=b, q_len=length,
                        window=window, rope_theta=rope)}), expected)

    def test_entrypoint_with_complete_full_host_adapter_CPU_oracle(self):
        for page, scale in [(1, 1.), (64, 1/64)]:
            fresh = full.old.reference.generate_input(batch=1, ctx=64, heads='tp8', layer='full',
                q_len=8, page=page, kv_scale=scale, seed=5201+page, device='cpu')
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
