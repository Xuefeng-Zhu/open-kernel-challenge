"""E6 CPU-only adapter tests. Native SGLang/FA3/Triton/CUDA do not execute."""
import ast
import copy
import json
from pathlib import Path
import sys
import unittest
import warnings

warnings.filterwarnings('ignore', message='Failed to initialize NumPy')
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_mixed_fa3 as old

SOURCE = HERE / 'mixed_fa3_sliding.py'


def functions():
    names = {'_fa3_norm', '_fa3_cos_sin', '_fa3_full_verify', '_fa3_sliding_verify', 'kernel_fn'}
    return [n for n in ast.parse(SOURCE.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name in names]


def adapter(data, trace):
    cfg = data['config']
    batch = cfg['batch']
    hq, hk, d = [cfg[k] for k in ['num_q_heads', 'num_kv_heads', 'head_dim']]

    class Norm:
        def __init__(self, **kw):
            assert kw == dict(head_dim=d, num_q_heads=hq, num_kv_heads=hk, eps=1e-5)

        def __call__(self, x, positions, table):
            assert positions is data['positions']
            q = x[:, :hq*d].reshape(-1, hq, d).float()
            k = x[:, hq*d:(hq+hk)*d].reshape(-1, hk, d).float()
            # Pinned FusedQKNorm rounds RMSNorm to BF16 BEFORE interleaved RoPE,
            # and rounds RoPE outputs to BF16. This differs from ideal FP32 ref.
            def norm_rope(y):
                y = (y * torch.rsqrt(y.square().mean(-1, keepdim=True)+1e-5)).to(torch.bfloat16).float()
                c, s = table[positions].float().chunk(2, dim=-1)
                even, odd = y[..., 0::2], y[..., 1::2]
                return torch.stack((even*c[:, None]-odd*s[:, None],
                                    even*s[:, None]+odd*c[:, None]), -1).flatten(-2).to(torch.bfloat16)
            trace.append('norm_rope')
            return norm_rope(q), norm_rope(k)

    def store(k, v, kc, vc, loc, *, reserved_skip_index):
        assert reserved_skip_index == -1
        assert k.dtype == v.dtype == kc.dtype == vc.dtype == torch.uint8
        kc[loc.long()] = k
        vc[loc.long()] = v
        trace.append('store')

    def attention(**kw):
        assert kw['q'].dtype == torch.float8_e4m3fn
        assert kw['causal'] and kw['window_size'] == (2047, 0)
        assert kw['max_seqlen_q'] == 8
        assert kw['softmax_scale'] == cfg['sm_scale']*cfg['qk_scale_factor']
        for key in ['k_cache', 'v_cache', 'page_table', 'cache_seqlens', 'cu_seqlens_q', 'scheduler_metadata']:
            assert kw[key] is data[key]
        if cfg['kv_scale'] != 1:
            assert torch.equal(kw['k_descale'], data['k_scale'].expand(batch, hk))
            assert torch.equal(kw['v_descale'], data['v_scale'].expand(batch, hk))
        else:
            assert 'k_descale' not in kw and 'v_descale' not in kw
        q = kw['q'].float().reshape(batch, 8, hk, hq//hk, d)
        kc = kw['k_cache'].view(-1, hk, d).float()*data['k_scale']
        vc = kw['v_cache'].view(-1, hk, d).float()*data['v_scale']
        out = torch.empty((batch, 8, hq, d), dtype=torch.bfloat16)
        for bi in range(batch):
            for j in range(8):
                end = int(data['cache_seqlens'][bi])-8+j+1
                # Window applies to each query separately, including the new tokens.
                pos = torch.arange(max(0, end-cfg['window']), end)
                page = cfg['page_size']
                rows = data['page_table'][bi].long()[pos//page]*page+pos%page
                score = torch.einsum('hgd,thd->hgt', q[bi, j], kc[rows])*kw['softmax_scale']
                out[bi, j] = torch.einsum('hgt,thd->hgd', score.softmax(-1), vc[rows]).reshape(hq, d).to(torch.bfloat16)
        trace.append('attention')
        return (out.reshape(batch*8, hq, d), None)

    def finish(a, gate, weight, **kw):
        assert a.dtype == torch.bfloat16 and weight is None
        assert kw == dict(head_dim=d, eps=1e-5, has_norm=True)
        y = a.reshape(-1, hq, d).float()
        y *= torch.rsqrt(y.square().mean(-1, keepdim=True)+1e-5)
        trace.append('gate')
        return (y*torch.sigmoid(gate.float().reshape_as(y))).reshape_as(a).to(torch.bfloat16)

    # Only CPU table extent is reduced; the candidate constant remains 2^21.
    ns = dict(torch=torch, Inputs=dict, RMS_EPS=1e-5, _FA3_NORMS={}, _FA3_ROPE={},
              _FA3_ROPE_POSITIONS=8192, FusedQKNorm=Norm, store_cache=store,
              flash_attn_with_kvcache=attention, rms_norm_gate=finish)
    exec(compile(ast.Module(body=functions(), type_ignores=[]), 'E6_cpu_adapter', 'exec'), ns)
    return ns


class SlidingFA3(unittest.TestCase):
    def test_parent_functions_unchanged_except_dispatch(self):
        parent = {n.name: n for n in ast.parse((HERE/'mixed_fa3.py').read_text()).body
                  if isinstance(n, ast.FunctionDef)}
        current = {n.name: n for n in ast.parse(SOURCE.read_text()).body if isinstance(n, ast.FunctionDef)}
        for name, node in parent.items():
            if name != 'kernel_fn':
                self.assertEqual(ast.dump(node), ast.dump(current[name]), name)
        actual = copy.deepcopy(current['kernel_fn'])
        del actual.body[1]  # exactly one added dispatch, after cfg assignment
        self.assertEqual(ast.dump(parent['kernel_fn']), ast.dump(actual))
        self.assertEqual(set(current)-set(parent), {'_fa3_cos_sin', '_fa3_sliding_verify'})
        compile(SOURCE.read_bytes(), str(SOURCE), 'exec')

    def test_only_one_scored_shape_changes_and_nearby_dispatch(self):
        entry = next(n for n in functions() if n.name == 'kernel_fn')
        ns = dict(torch=torch, Inputs=dict, _fa3_sliding_verify=lambda _: 'sliding',
                  _fa3_full_verify=lambda _: 'full', _e2_kernel_fn=lambda _: 'e2')
        exec(compile(ast.Module(body=[entry], type_ignores=[]), 'E6_dispatch', 'exec'), ns)
        changed = 0
        for b in [1, 2, 4, 8, 16, 32, 33]:
            for l in [1, 7, 8, 9, 16]:
                for window in [0, 1, 2047, 2048, 2049]:
                    for rope in [0, 499999, 500000, 500001]:
                        cfg = dict(batch=b, q_len=l, window=window, rope_theta=rope)
                        expected = 'full' if b in (16, 32) and l == 8 and window == rope == 0 else 'e2'
                        if b == 1 and l == 8 and window == 2048 and rope == 500000:
                            expected = 'sliding'
                            changed += 1
                        self.assertEqual(ns['kernel_fn']({'config': cfg}), expected)
        self.assertEqual(changed, 1)

    def test_rope_table_layout_cache_and_constant_bound(self):
        ns = adapter({'config': dict(batch=1, num_q_heads=16, num_kv_heads=2, head_dim=64)}, [])
        cache = ns['_fa3_cos_sin'](500000., 64, torch.device('cpu'))
        self.assertIs(cache, ns['_fa3_cos_sin'](500000., 64, torch.device('cpu')))
        self.assertEqual(cache.dtype, torch.float32)
        self.assertEqual(cache.shape, (8192, 64))
        pos = torch.tensor([0, 1, 2047, 2048, 8191])
        freq = pos.float()[:, None] / 500000.**(torch.arange(0, 64, 2).float()/64)
        self.assertTrue(torch.allclose(cache[pos], torch.cat((freq.cos(), freq.sin()), -1), atol=5e-4))
        tree = ast.parse(SOURCE.read_text())
        value = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == '_FA3_ROPE_POSITIONS' for t in n.targets))
        self.assertEqual(eval(compile(ast.Expression(value), '<constant>', 'eval')), 1 << 21)

    def check_case(self, fresh, zero=False):
        if zero:
            old.zero_new_page(fresh)
        work = old.buffers.clone_inputs(fresh)
        trace = []
        ns = adapter(work, trace)
        out = ns['kernel_fn'](work)
        ok, msg = old.reference.check_implementation(work, old.reference.collect(work, out))
        self.assertTrue(ok, msg)
        self.assertTrue(old.buffers.unchanged(work, fresh)[0])
        self.assertEqual(trace, ['norm_rope', 'store', 'attention', 'gate'])
        self.assertEqual(out.shape, (1, 8, fresh['config']['num_q_heads']*64))
        return msg

    def test_window_boundary_cache_zero_scales_and_heads(self):
        cases = [(0, 1, 1., 'normal', 'tp8'), (1, 1, 1., 'normal', 'tp8'),
                 (2047, 1, 1., 'edges', 'tp8'), (2048, 64, 1/64, 'edges', 'tp8'),
                 (2049, 1, 1., 'soft', 'tp8'), (64, 64, 1/64, 'normal', 'dp')]
        for ctx, page, scale, dist, heads in cases:
            with self.subTest(ctx=ctx, page=page, scale=scale, dist=dist, heads=heads):
                fresh = old.reference.generate_input(batch=1, ctx=ctx, heads=heads, layer='sliding',
                            q_len=8, page=page, kv_scale=scale, dist=dist, seed=912+ctx, device='cpu')
                self.check_case(fresh, zero=ctx % page == 0)

    def test_cpu_same_address_A_B_A_changed_lengths_and_content(self):
        for page, scale in [(1, 1.), (64, 1/64)]:
            args = dict(batch=1, ctx=64, heads='tp8', layer='sliding', q_len=8,
                        page=page, kv_scale=scale, spread='mixed', device='cpu')
            a = old.reference.generate_input(**args, seed=803)
            b = old.reference.generate_input(**args, seed=804)
            self.assertFalse(torch.equal(a['seq_lens'], b['seq_lens']))
            work = old.buffers.clone_inputs(a)
            pointers = {k: v.data_ptr() for k, v in work.items() if isinstance(v, torch.Tensor)}
            trace = []
            ns = adapter(work, trace)
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
