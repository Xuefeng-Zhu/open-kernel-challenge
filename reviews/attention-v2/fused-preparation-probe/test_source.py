"""Frozen-source and CPU host-model gates; no Triton/FA3/CUDA claim."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
BASE_SHA = 'e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e'
CANDIDATE_SHA = '5c3d1c5b00f0e759f1c499b8f567a0f47c77ce0fa38566f3028a571fccc614a0'
SELECTED_SHA = 'c9a0847502a0b2fe2be55f546a2b3ea7eefa931c851e328e680302ebd828baf0'
TARGETS = {(16, 1), (16, 8), (32, 8)}


def nodes(path):
    return ast.parse(path.read_bytes())


def functions(path):
    return {n.name: n for n in nodes(path).body if isinstance(n, ast.FunctionDef)}


def dispatch(path):
    namespace = {'torch': type('Torch', (), {'Tensor': object}), 'Inputs': dict}
    for name in ('_fa3_full_verify_fused', '_fa3_sliding_verify_fused',
                 '_fa3_full_verify', '_fa3_sliding_verify', '_e2_kernel_fn'):
        namespace[name] = lambda _, label=name: label
    entry = functions(path)['kernel_fn']
    exec(compile(ast.Module(body=[entry], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['kernel_fn']


class SourceGates(unittest.TestCase):
    def test_hashes_and_only_first_guard_changed(self):
        for path, sha in [(HERE/'baseline.py', BASE_SHA), (HERE/'candidate.py', CANDIDATE_SHA),
                          (PARENT/'mixed_fa3_fp8_qk.py', BASE_SHA),
                          (PARENT/'mixed_fa3_decode_parallel_merge.py', SELECTED_SHA)]:
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), sha, path.name)
        before, after = nodes(HERE/'baseline.py'), nodes(HERE/'candidate.py')
        a = next(n for n in before.body if isinstance(n, ast.FunctionDef) and n.name == 'kernel_fn')
        b = next(n for n in after.body if isinstance(n, ast.FunctionDef) and n.name == 'kernel_fn')
        self.assertNotEqual(ast.dump(a.body[1].test), ast.dump(b.body[1].test))
        b.body[1].test = a.body[1].test
        self.assertEqual(ast.dump(before), ast.dump(after))
        compile((HERE/'candidate.py').read_bytes(), 'candidate.py', 'exec')

    def test_exact_target_route_difference_and_unchanged_nearby_routes(self):
        old, new = dispatch(HERE/'baseline.py'), dispatch(HERE/'candidate.py')
        differences = set()
        for batch in (1, 2, 8, 15, 16, 17, 31, 32, 33):
            for length in (1, 2, 7, 8, 9):
                for window in (0, 2047, 2048):
                    for rope in (0, 500000):
                        cfg = dict(batch=batch, q_len=length, window=window, rope_theta=rope)
                        data = {'config': cfg}
                        if old(data) != new(data):
                            differences.add((batch, length, window, rope))
                            self.assertEqual(old(data), '_fa3_full_verify')
                            self.assertEqual(new(data), '_fa3_full_verify_fused')
        self.assertEqual(differences, {(b, l, 0, 0) for b, l in TARGETS})
        self.assertEqual(new({'config': dict(batch=32, q_len=1, window=0, rope_theta=0)}), '_e2_kernel_fn')

    def test_fa3_schedule_precision_and_postprocessing_unchanged(self):
        f = functions(HERE/'candidate.py')
        def call(node, name):
            return next(n for n in ast.walk(node) if isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Name) and n.func.id == name)
        a, b = f['_fa3_full_verify'], f['_fa3_full_verify_fused']
        ordinary = call(a, 'flash_attn_with_kvcache')
        fused = call(b, 'flash_attn_with_kvcache')
        self.assertEqual({k.arg: ast.dump(k.value) for k in ordinary.keywords if k.arg != 'q'},
                         {k.arg: ast.dump(k.value) for k in fused.keywords if k.arg != 'q'})
        self.assertEqual([ast.dump(n) for n in a.body[-4:]], [ast.dump(n) for n in b.body[-4:]])
        self.assertEqual(ast.dump(f['_fa3_prepare_fused']),
                         ast.dump(functions(PARENT/'mixed_fa3_decode_parallel_merge.py')['_fa3_prepare_fused']))


class CPUHostModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name, expected in json.loads((HERE/'cpu-support-hashes.json').read_text()).items():
            observed = hashlib.sha256((PARENT/name).read_bytes()).hexdigest()
            if observed != expected:
                raise ValueError('Frozen CPU-model dependency mismatch: '+name)
        sys.path.insert(0, str(PARENT))
        import test_full_fused_prepare_component as full
        cls.full = full
        cls.old = full.old
        cls.torch = full.torch
        # The existing adapter models this exact helper; this is not native execution.
        assert ast.dump(full.host_node()) == ast.dump(functions(HERE/'candidate.py')['_fa3_full_verify_fused'])

    def execute(self, work, fresh, ns=None, state=None):
        if ns is None:
            ns, state = self.full.adapter(work, [])
            exec(compile(ast.Module(body=[functions(HERE/'candidate.py')['kernel_fn']],
                                    type_ignores=[]), 'candidate_dispatch_cpu', 'exec'), ns)
        out = ns['kernel_fn'](work)
        raw = self.old.reference.collect(work, out)
        okay, why = self.old.reference.check_implementation(work, raw)
        self.assertTrue(okay, why)
        self.assertTrue(self.old.buffers.unchanged(work, fresh)[0])
        ordinary = self.old.buffers.clone_inputs(fresh)
        ordinary_out = self.old.cpu_adapter(ordinary, [])['_fa3_full_verify'](ordinary)
        self.assertTrue(self.torch.equal(raw, self.old.reference.collect(ordinary, ordinary_out)))
        for cache, key in [('k_cache', 'k'), ('v_cache', 'v')]:
            written = work[cache].view(-1, work['config']['num_kv_heads'], 64).view(self.torch.uint8)[work['cache_loc'].long()]
            self.assertTrue(self.torch.equal(written, state[key].contiguous().view(self.torch.uint8)))
        return raw, ns, state

    def test_all_three_targets_page_scale_live_zero_and_A_B_A(self):
        for batch, length in sorted(TARGETS):
            for page, scale in [(1, 1.), (64, 1/64)]:
                with self.subTest(batch=batch, length=length, page=page, scale=scale):
                    args = dict(batch=batch, q_len=length, ctx=64, heads='tp8', layer='full',
                                page=page, kv_scale=scale, spread='mixed', device='cpu')
                    a = self.old.reference.generate_input(**args, seed=803)
                    b = self.old.reference.generate_input(**args, seed=804)
                    # Valid synthetic CPU identity sentinel only, not a native FA3 schedule.
                    if length == 1:
                        a['scheduler_metadata'] = self.torch.arange(64, dtype=self.torch.int32)
                        b['scheduler_metadata'] = self.torch.arange(64, dtype=self.torch.int32)+100
                    # Force a new page at position zero for a dedicated first call.
                    zero = self.old.reference.generate_input(**{**args, 'ctx': 0, 'spread': 'fixed'}, seed=805)
                    self.old.zero_new_page(zero)
                    self.assertEqual(int(zero['cache_loc'][0]), 0)
                    self.execute(self.old.buffers.clone_inputs(zero), zero)
                    self.assertFalse(self.torch.equal(a['seq_lens'], b['seq_lens']))
                    work = self.old.buffers.clone_inputs(a)
                    addresses = {k: v.data_ptr() for k, v in work.items() if isinstance(v, self.torch.Tensor)}
                    first = ns = state = None
                    for i, fresh in enumerate((a, b, a)):
                        self.old.buffers.restore(work, fresh)
                        raw, ns, state = self.execute(work, fresh, ns, state)
                        self.assertTrue(all(work[k].data_ptr() == value for k, value in addresses.items()))
                        if i == 0:
                            first = raw.clone()
                        if i == 2:
                            self.assertTrue(self.torch.equal(first, raw))


if __name__ == '__main__':
    unittest.main(verbosity=2)
