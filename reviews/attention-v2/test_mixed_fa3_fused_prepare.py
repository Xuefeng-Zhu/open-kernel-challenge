"""Complete fused candidate wiring/parent-preservation tests with CPU stand-ins."""
import ast
import copy
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_fused_prepare_component as fused
import torch

SOURCE = HERE/'mixed_fa3_fused_prepare.py'


def entrypoint():
    return next(n for n in ast.parse(SOURCE.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'kernel_fn')


class CompleteFused(unittest.TestCase):
    def test_all_parent_functions_and_component_bodies_preserved(self):
        old = {n.name: n for n in ast.parse((HERE/'mixed_fa3_decode_small.py').read_text()).body
               if isinstance(n, ast.FunctionDef)}
        new = {n.name: n for n in ast.parse(SOURCE.read_text()).body if isinstance(n, ast.FunctionDef)}
        component = {n.name: n for n in ast.parse(fused.COMPONENT.read_text()).body
                     if isinstance(n, ast.FunctionDef)}
        self.assertEqual(set(new)-set(old), set(component))
        for name in component:
            self.assertEqual(ast.dump(new[name]), ast.dump(component[name]))
        for name in old:
            if name != 'kernel_fn':
                self.assertEqual(ast.dump(old[name]), ast.dump(new[name]), name)
        entry = copy.deepcopy(new['kernel_fn'])
        del entry.body[1]
        self.assertEqual(ast.dump(entry), ast.dump(old['kernel_fn']))
        compile(SOURCE.read_bytes(), str(SOURCE), 'exec')

    def test_exactly_one_scored_path_changes_and_nearby_fallback(self):
        ns = dict(torch=torch, Inputs=dict, _fa3_sliding_verify_fused=lambda _: 'fused',
                  _fa3_sliding_verify=lambda _: 'sliding', _fa3_full_verify=lambda _: 'full',
                  _e2_kernel_fn=lambda _: 'e2')
        exec(compile(ast.Module(body=[entrypoint()], type_ignores=[]), 'fused_dispatch', 'exec'), ns)
        for b in [1, 2, 8, 16, 32, 33]:
            for length in [1, 7, 8, 9]:
                for window, rope in [(0, 0), (2048, 500000), (0, 500000), (2047, 500000)]:
                    expected = 'e2'
                    if b in (1, 16, 32) and length == 8:
                        if window == 2048 and rope == 500000:
                            expected = 'sliding'
                        if window == rope == 0:
                            expected = 'full'
                    if b in (1, 16) and length == 1 and window == rope == 0:
                        expected = 'full'
                    if b == 1 and length == 8 and window == 2048 and rope == 500000:
                        expected = 'fused'
                    self.assertEqual(ns['kernel_fn']({'config': dict(batch=b, q_len=length,
                        window=window, rope_theta=rope)}), expected)

    def test_actual_entrypoint_calls_fused_host_adapter_CPU_oracle(self):
        fresh = fused.sliding.old.reference.generate_input(batch=1, ctx=64, heads='tp8', layer='sliding',
            q_len=8, page=64, kv_scale=1/64, seed=2681, device='cpu')
        fused.sliding.old.zero_new_page(fresh)
        work = fused.sliding.old.buffers.clone_inputs(fresh)
        trace = []
        ns, _ = fused.adapter(work, trace)
        exec(compile(ast.Module(body=[entrypoint()], type_ignores=[]), 'fused_entrypoint', 'exec'), ns)
        out = ns['kernel_fn'](work)
        ok, msg = fused.sliding.old.reference.check_implementation(work,
                    fused.sliding.old.reference.collect(work, out))
        self.assertTrue(ok, msg)
        self.assertTrue(fused.sliding.old.buffers.unchanged(work, fresh)[0])
        self.assertEqual(trace, ['prepare_fused_model', 'attention', 'gate'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
