"""Source and CPU host-model gates only; installed FA3 behavior remains unknown."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
BASE = '5c3d1c5b00f0e759f1c499b8f567a0f47c77ce0fa38566f3028a571fccc614a0'
CANDIDATE = '586074a4bbff57ff28c53070dfcdfcf881631b9aeffeb016032a7a07b32309cf'
SELECTED = 'c9a0847502a0b2fe2be55f546a2b3ea7eefa931c851e328e680302ebd828baf0'


def tree(path):
    return ast.parse(path.read_bytes())


def function(path, name):
    return next(x for x in tree(path).body if isinstance(x, ast.FunctionDef) and x.name == name)


def attention_call(node):
    return next(x for x in ast.walk(node) if isinstance(x, ast.Call)
                and isinstance(x.func, ast.Name) and x.func.id == 'flash_attn_with_kvcache')


class SourceGates(unittest.TestCase):
    def test_frozen_parent_selected_and_single_keyword_only(self):
        for file, sha in [(HERE/'baseline.py', BASE), (HERE/'candidate.py', CANDIDATE),
                          (V2/'fused-preparation-probe/candidate.py', BASE),
                          (V2/'mixed_fa3_decode_parallel_merge.py', SELECTED)]:
            self.assertEqual(hashlib.sha256(file.read_bytes()).hexdigest(), sha)
        a, b = tree(HERE/'baseline.py'), tree(HERE/'candidate.py')
        helper = next(x for x in b.body if isinstance(x, ast.FunctionDef)
                      and x.name == '_fa3_sliding_verify_fused')
        call = attention_call(helper)
        override = call.keywords.pop(-2)
        self.assertIsNone(override.arg)
        self.assertIsInstance(override.value, ast.IfExp)
        self.assertEqual(ast.dump(a), ast.dump(b))
        compile((HERE/'candidate.py').read_bytes(), 'conditional_candidate.py', 'exec')

    def test_only_two_target_call_contracts_receive_override(self):
        helper = function(HERE/'candidate.py', '_fa3_sliding_verify_fused')
        keyword = attention_call(helper).keywords[-2]
        expression = compile(ast.Expression(keyword.value), 'shape_only_keyword', 'eval')
        for batch in range(1, 35):
            for length in [1, 2, 7, 8, 9]:
                actual = eval(expression, {'batch': batch, 'length': length})
                self.assertEqual(actual, {'num_splits': 1} if batch in (16, 32) and length == 8 else {})
        self.assertEqual(ast.dump(function(HERE/'baseline.py', 'kernel_fn')),
                         ast.dump(function(HERE/'candidate.py', 'kernel_fn')))


class CPUHostContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name, expected in json.loads((HERE/'cpu-support-hashes.json').read_text()).items():
            if hashlib.sha256((V2/name).read_bytes()).hexdigest() != expected:
                raise ValueError('Frozen CPU support mismatch: '+name)
        sys.path.insert(0, str(V2))
        import test_fused_prepare_component as fused
        cls.fused, cls.old, cls.torch = fused, fused.sliding.old, fused.torch
        cls.torch.set_num_threads(1)
        assert ast.dump(function(V2/'fa3_prologue_component.py', '_fa3_sliding_verify_fused')) == ast.dump(
            function(HERE/'baseline.py', '_fa3_sliding_verify_fused'))

    def namespace(self, data, candidate):
        ns, state = self.fused.adapter(data, [])
        # CPU stand-in extent only: mixed histories can reach 4*ctx+L.
        # The candidate's unchanged config table still has 2**21 positions.
        ns['_FA3_ROPE_POSITIONS'] = 16384
        ordinary_attention = ns['flash_attn_with_kvcache']
        cfg = data['config']
        def modeled_attention(**kwargs):
            requested = kwargs.pop('num_splits', None)
            expected = 1 if candidate and cfg['batch'] in (16, 32) and cfg['q_len'] == 8 else None
            self.assertEqual(requested, expected)
            # This independent CPU operator computes the mathematical attention;
            # it does not model GPU splitting, reduction order or scheduling.
            return ordinary_attention(**kwargs)
        ns['flash_attn_with_kvcache'] = modeled_attention
        path = HERE/('candidate.py' if candidate else 'baseline.py')
        selected = [function(path, '_fa3_sliding_verify_fused'), function(path, 'kernel_fn')]
        exec(compile(ast.Module(body=selected, type_ignores=[]), 'CPU_host_contract', 'exec'), ns)
        return ns, state

    def test_boundaries_scales_live_zero_and_fixed_address_replay(self):
        for batch, ctx, page, scale, spread, dist in [
                (16, 2048, 1, 1., 'fixed', 'edges'),
                (32, 2049, 64, 1/64, 'mixed', 'soft'),
                (1, 0, 64, 1/64, 'fixed', 'soft')]:
            with self.subTest(batch=batch, page=page, scale=scale):
                args = dict(batch=batch, ctx=ctx, heads='tp8', layer='sliding', q_len=8,
                            page=page, kv_scale=scale, spread=spread, dist=dist, device='cpu')
                a = self.old.reference.generate_input(**args, seed=803)
                b = self.old.reference.generate_input(**args, seed=804)
                if ctx in (0, 2048):
                    self.old.zero_new_page(a)
                    self.old.zero_new_page(b)
                    self.assertEqual(int(a['cache_loc'][0]), 0)
                work = self.old.buffers.clone_inputs(a)
                ns, state = self.namespace(work, True)
                addresses = {k: v.data_ptr() for k, v in work.items() if isinstance(v, self.torch.Tensor)}
                first = None
                for index, fresh in enumerate((a, b, a)):
                    self.old.buffers.restore(work, fresh)
                    out = ns['kernel_fn'](work)
                    raw = self.old.reference.collect(work, out)
                    okay, why = self.old.reference.check_implementation(work, raw)
                    self.assertTrue(okay, why)
                    self.assertTrue(self.old.buffers.unchanged(work, fresh)[0])
                    self.assertTrue(all(work[k].data_ptr() == pointer for k, pointer in addresses.items()))
                    ordinary = self.old.buffers.clone_inputs(fresh)
                    old_ns, _ = self.namespace(ordinary, False)
                    original = self.old.reference.collect(ordinary, old_ns['kernel_fn'](ordinary))
                    self.assertTrue(self.torch.equal(raw, original))
                    for cache, key in [('k_cache', 'k'), ('v_cache', 'v')]:
                        new = work[cache].view(-1, fresh['config']['num_kv_heads'], 64).view(self.torch.uint8)[work['cache_loc'].long()]
                        self.assertTrue(self.torch.equal(new, state[key].contiguous().view(self.torch.uint8)))
                    if index == 0:
                        first = raw.clone()
                    if index == 2:
                        self.assertTrue(self.torch.equal(first, raw))

if __name__ == '__main__':
    unittest.main(verbosity=2)
