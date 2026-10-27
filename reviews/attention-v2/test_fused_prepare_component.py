"""CPU models/host-argument/index proofs for proposed prologue, not Triton execution."""
import ast
from pathlib import Path
import sys
import symtable
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_mixed_fa3_sliding as sliding
import torch

COMPONENT = HERE/'fa3_prologue_component.py'


def prepare_model(qkv, positions, table, scale_k, scale_v, cfg):
    """Independent tensor model with explicit baseline BF16 rounding boundaries."""
    hq, hk, d = [cfg[k] for k in ['num_q_heads', 'num_kv_heads', 'head_dim']]
    q = qkv[:, :hq*d].reshape(-1, hq, d).float()
    k = qkv[:, hq*d:(hq+hk)*d].reshape(-1, hk, d).float()
    v = qkv[:, (hq+hk)*d:(hq+2*hk)*d].reshape(-1, hk, d)
    def rms_rope(y):
        y = (y*torch.rsqrt(y.square().mean(-1, keepdim=True)+1e-5)).to(torch.bfloat16).float()
        c, s = table[positions].float().chunk(2, dim=-1)
        even, odd = y[..., 0::2], y[..., 1::2]
        return torch.stack((even*c[:, None]-odd*s[:, None],
                            even*s[:, None]+odd*c[:, None]), -1).flatten(-2).to(torch.bfloat16)
    q, k = rms_rope(q), rms_rope(k)
    if cfg['kv_scale'] != 1:
        k = (k.float()/scale_k).to(torch.bfloat16)
        v = (v.float()/scale_v).to(torch.bfloat16)
    return (q.to(torch.float8_e4m3fn), k.to(torch.float8_e4m3fn),
            v.to(torch.float8_e4m3fn))


def adapter(data, trace):
    ns = sliding.adapter(data, trace)
    ns['triton'] = SimpleNamespace(cdiv=lambda a, b: (a+b-1)//b,
                                   next_power_of_2=lambda a: 1 << (a-1).bit_length())
    cfg = data['config']
    state = {}
    class Launch:
        def __getitem__(self, grid):
            def run(qkv, positions, table, loc, kc, vc, ks, vs, qout, stride, capacity, eps, **meta):
                b, l, hq, hk, d = [cfg[k] for k in ['batch', 'q_len', 'num_q_heads', 'num_kv_heads', 'head_dim']]
                self_grid = ((b*l*(hq+2*hk)+3)//4,)
                assert grid == self_grid
                assert meta == dict(T=b*l, HQ=hq, HKV=hk, D=d, BD=64, BM=4,
                                    ROPE=True, SCALED=cfg['kv_scale'] != 1., num_warps=4)
                assert qkv.data_ptr() == data['qkv'].data_ptr() and stride == qkv.stride(0)
                assert positions is data['positions'] and loc is data['cache_loc']
                assert kc is data['k_cache'] and vc is data['v_cache']
                assert ks is data['k_scale'] and vs is data['v_scale']
                assert capacity == kc.numel()//(hk*d) and eps == 1e-5
                assert table is ns['_FA3_ROPE'][(cfg['rope_theta'], d, qkv.device)]
                assert qout.shape == (b*l, hq, d) and qout.dtype == torch.float8_e4m3fn
                q, k, v = prepare_model(qkv, positions, table, ks, vs, cfg)
                qout.view(torch.uint8).copy_(q.contiguous().view(torch.uint8))
                kc.view(-1, hk*d).view(torch.uint8)[loc.long()] = k.reshape(b*l, hk*d).view(torch.uint8)
                vc.view(-1, hk*d).view(torch.uint8)[loc.long()] = v.reshape(b*l, hk*d).view(torch.uint8)
                state.update(q=q.clone(), k=k.clone(), v=v.clone())
                trace.append('prepare_fused_model')
            return run
    ns['_fa3_prepare_fused'] = Launch()
    host = next(n for n in ast.parse(COMPONENT.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == '_fa3_sliding_verify_fused')
    exec(compile(ast.Module(body=[host], type_ignores=[]), 'fused_prologue_host', 'exec'), ns)
    return ns, state


class FusedPrepare(unittest.TestCase):
    def test_scalar_dtype_promotion_and_direct_quantization(self):
        self.assertEqual((torch.ones(1, dtype=torch.bfloat16)/torch.tensor(1/64)).dtype, torch.bfloat16)
        x = torch.tensor([0., 1., 2., 448.])
        expected = sliding.old.reference._quant(x, torch.tensor(1.)).view(torch.uint8)
        self.assertTrue(torch.equal(x.to(torch.float8_e4m3fn).view(torch.uint8), expected))
        # CPU overflow conversion is environment-specific; no CUDA conclusion from it.
        overflow = torch.tensor([512., -1000.])
        self.assertTrue(torch.equal(overflow.to(torch.float8_e4m3fn).view(torch.uint8),
                                   sliding.old.reference._quant(overflow, 1.).view(torch.uint8)))

    def test_host_arguments_oracle_cache_zero_boundaries_and_overflow(self):
        for ctx, page, scale, overflow in [(2048, 1, 1., False), (64, 64, 1/64, False), (0, 1, 1/64, True)]:
            with self.subTest(ctx=ctx, page=page, scale=scale, overflow=overflow):
                fresh = sliding.old.reference.generate_input(batch=1, ctx=ctx, heads='tp8', layer='sliding',
                    q_len=8, page=page, kv_scale=scale, seed=2500+ctx, device='cpu')
                if overflow:
                    cfg = fresh['config'];hq, hk, d = cfg['num_q_heads'], cfg['num_kv_heads'], cfg['head_dim']
                    flat = fresh['qkv'].view(8, -1)
                    flat[:, hq*d:(hq+hk)*d] = 0
                    flat[:, hq*d] = 1000
                    flat[:, (hq+hk)*d] = 1000
                sliding.old.zero_new_page(fresh)
                work = sliding.old.buffers.clone_inputs(fresh)
                trace = []
                ns, state = adapter(work, trace)
                out = ns['_fa3_sliding_verify_fused'](work)
                oracle = sliding.old.reference.ref_kernel(work)
                self.assertTrue(all(torch.isfinite(v.float()).all() for v in oracle), "Nonfinite oracle cannot establish correctness")
                ok, msg = sliding.old.reference.check_implementation(work, sliding.old.reference.collect(work, out))
                self.assertTrue(ok, msg)
                self.assertTrue(sliding.old.buffers.unchanged(work, fresh)[0])
                self.assertEqual(trace, ['prepare_fused_model', 'attention', 'gate'])
                self.assertTrue(torch.isfinite(state['q'].float()).all())
                self.assertTrue(torch.isfinite(state['k'].float()).all())
                self.assertTrue(torch.isfinite(state['v'].float()).all())
                if overflow:
                    self.assertEqual(float(state['k'][0, 0, 0]), 448.)
                    self.assertEqual(float(state['v'][0, 0, 0]), 448.)

    def test_CPU_same_address_A_B_A_and_changed_lengths(self):
        for page, scale in [(1, 1.), (64, 1/64)]:
            args = dict(batch=1, ctx=32, heads='tp8', layer='sliding', q_len=8,
                        spread='mixed', page=page, kv_scale=scale, device='cpu')
            a = sliding.old.reference.generate_input(**args, seed=803)
            b = sliding.old.reference.generate_input(**args, seed=804)
            self.assertFalse(torch.equal(a['seq_lens'], b['seq_lens']))
            work = sliding.old.buffers.clone_inputs(a)
            pointers = {k: v.data_ptr() for k, v in work.items() if isinstance(v, torch.Tensor)}
            ns, _ = adapter(work, [])
            first = None
            for i, fresh in enumerate([a, b, a]):
                sliding.old.buffers.restore(work, fresh)
                out = ns['_fa3_sliding_verify_fused'](work)
                raw = sliding.old.reference.collect(work, out)
                ok, msg = sliding.old.reference.check_implementation(work, raw)
                self.assertTrue(ok, msg)
                self.assertTrue(sliding.old.buffers.unchanged(work, fresh)[0])
                self.assertTrue(all(work[k].data_ptr() == p for k, p in pointers.items()))
                if i == 0:
                    first = raw.clone()
                if i == 2:
                    self.assertTrue(torch.equal(first, raw))

    def test_integer_ownership_indices_cover_exact_outputs_and_new_cache(self):
        # Contract-derived expected sets, independent of any native launch.
        for tokens, hq, hk in [(1, 16, 2), (8, 16, 2), (8, 128, 16), (32, 16, 2)]:
            d = 64;heads = hq+2*hk;locs = [0]+list(range(3, tokens+2))
            query, keys, values = [], [], []
            for row in range(((tokens*heads+3)//4)*4):
                if row >= tokens*heads:
                    continue
                t, h = divmod(row, heads)
                for col in range(d):
                    if h < hq:
                        query.append((t*hq+h)*d+col)
                    elif h < hq+hk:
                        keys.append((locs[t]*hk+h-hq)*d+col)
                    else:
                        values.append((locs[t]*hk+h-hq-hk)*d+col)
            expected = {(loc*hk+h)*d+c for loc in locs for h in range(hk) for c in range(d)}
            self.assertEqual(set(query), set(range(tokens*hq*d)))
            self.assertEqual(len(query), len(set(query)))
            self.assertEqual(set(keys), expected)
            self.assertEqual(set(values), expected)
            self.assertEqual(len(keys), len(expected))
            self.assertEqual(len(values), len(expected))

    def test_syntax_and_JIT_globals_without_native_import(self):
        text = COMPONENT.read_text()
        compile(text, str(COMPONENT), 'exec')
        scope = symtable.symtable(text, str(COMPONENT), 'exec')
        kernel = next(s for s in scope.get_children() if s.get_name() == '_fa3_prepare_fused')
        globals_used = {s.get_name() for s in kernel.get_symbols() if s.is_global() and s.is_referenced()}
        # No plain numeric global can trip Triton's constexpr check. Only tl is external.
        self.assertEqual(globals_used, {'tl'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
