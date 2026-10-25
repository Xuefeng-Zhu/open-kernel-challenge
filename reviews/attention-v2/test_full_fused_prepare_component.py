"""CPU-only full-prologue component model and host-contract tests.

No complete-candidate dispatch is changed or tested here. Triton/FA3 are replaced
by CPU models; no native compile, CUDA graph, timing or GPU pass is claimed.
"""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_mixed_fa3 as old
import torch

SOURCE = HERE / 'fa3_full_prologue_component.py'
PARENT = HERE / 'mixed_fa3_fused_prepare.py'


def host_node():
    return next(n for n in ast.parse(SOURCE.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == '_fa3_full_verify_fused')


def prepare_model(qkv, scale_k, scale_v, cfg):
    """Baseline's full-layer BF16 normalization and direct FP8 conversion order."""
    hq, hk, d = [cfg[k] for k in ['num_q_heads', 'num_kv_heads', 'head_dim']]
    q = qkv[:, :hq*d].reshape(-1, hq, d).float()
    k = qkv[:, hq*d:(hq+hk)*d].reshape(-1, hk, d).float()
    v = qkv[:, (hq+hk)*d:(hq+2*hk)*d].reshape(-1, hk, d)
    def norm(y):
        return (y * torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-5)).to(torch.bfloat16)
    q, k = norm(q), norm(k)
    if cfg['kv_scale'] != 1.:
        k = (k.float() / scale_k).to(torch.bfloat16)
        v = (v.float() / scale_v).to(torch.bfloat16)
    return tuple(value.to(torch.float8_e4m3fn) for value in (q, k, v))


def adapter(data, trace):
    ns = old.cpu_adapter(data, trace)
    cfg = data['config']
    state = {}
    ns['triton'] = SimpleNamespace(cdiv=lambda a, b: (a+b-1)//b,
                                   next_power_of_2=lambda a: 1 << (a-1).bit_length())
    def forbidden_rope(*args, **kwargs):
        raise AssertionError('Full adapter must not allocate or consult a RoPE table')
    ns['_fa3_cos_sin'] = forbidden_rope
    class Launch:
        def __getitem__(self, grid):
            def run(qkv, positions, unused_table, loc, kc, vc, ks, vs, qout,
                    stride, capacity, eps, **meta):
                b, length, hq, hk, d = [cfg[k] for k in
                    ['batch', 'q_len', 'num_q_heads', 'num_kv_heads', 'head_dim']]
                assert cfg['rope_theta'] == cfg['window'] == 0
                assert grid == ((b*length*(hq+2*hk)+3)//4,)
                assert meta == dict(T=b*length, HQ=hq, HKV=hk, D=d, BD=64, BM=4,
                                    ROPE=False, SCALED=cfg['kv_scale'] != 1., num_warps=4)
                assert qkv.data_ptr() == data['qkv'].data_ptr()
                assert unused_table is qkv and stride == qkv.stride(0)
                assert positions is data['positions'] and loc is data['cache_loc']
                assert kc is data['k_cache'] and vc is data['v_cache']
                assert ks is data['k_scale'] and vs is data['v_scale']
                assert capacity == kc.numel()//(hk*d) and eps == 1e-5
                assert qout.shape == (b*length, hq, d) and qout.dtype == torch.float8_e4m3fn
                q, k, v = prepare_model(qkv, ks, vs, cfg)
                qout.view(torch.uint8).copy_(q.contiguous().view(torch.uint8))
                kc.view(-1, hk*d).view(torch.uint8)[loc.long()] = k.reshape(b*length, hk*d).view(torch.uint8)
                vc.view(-1, hk*d).view(torch.uint8)[loc.long()] = v.reshape(b*length, hk*d).view(torch.uint8)
                state.update(q=q.clone(), k=k.clone(), v=v.clone())
                trace.append('prepare_full_model')
            return run
    ns['_fa3_prepare_fused'] = Launch()
    exec(compile(ast.Module(body=[host_node()], type_ignores=[]), 'full_fused_host', 'exec'), ns)
    return ns, state


class FullFusedPrepareComponent(unittest.TestCase):
    def check_case(self, work, fresh, namespace=None, trace=None, state=None):
        if namespace is None:
            trace = []
            namespace, state = adapter(work, trace)
        before = len(trace)
        out = namespace['_fa3_full_verify_fused'](work)
        cfg = work['config']
        self.assertEqual(out.shape, (cfg['batch'], cfg['q_len'], cfg['num_q_heads']*64))
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(trace[before:], ['prepare_full_model', 'attention', 'gate'])
        self.assertTrue(torch.isfinite(out.float()).all())
        self.assertTrue(all(torch.isfinite(x.float()).all() for x in state.values()))
        reference_values = old.reference.ref_kernel(work)
        self.assertTrue(all(torch.isfinite(x.float()).all() for x in reference_values))
        raw = old.reference.collect(work, out)
        ok, message = old.reference.check_implementation(work, raw)
        self.assertTrue(ok, message)
        intact, message = old.buffers.unchanged(work, fresh)
        self.assertTrue(intact, message)
        for cache, value in [('k_cache', 'k'), ('v_cache', 'v')]:
            written = work[cache].view(-1, cfg['num_kv_heads'], 64).view(torch.uint8)[work['cache_loc'].long()]
            self.assertTrue(torch.equal(written, state[value].contiguous().view(torch.uint8)))

        # Independent parent CPU adapter must have exactly the same modeled bytes.
        parent_work = old.buffers.clone_inputs(fresh)
        parent_ns = old.cpu_adapter(parent_work, [])
        parent_out = parent_ns['_fa3_full_verify'](parent_work)
        parent_raw = old.reference.collect(parent_work, parent_out)
        self.assertTrue(torch.equal(raw, parent_raw))
        return raw.clone()

    def test_full_attention_call_and_gate_arguments_preserved_by_AST(self):
        original = next(n for n in ast.parse(PARENT.read_text()).body
                        if isinstance(n, ast.FunctionDef) and n.name == '_fa3_full_verify')
        def calls(node, name):
            return [n for n in ast.walk(node) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name) and n.func.id == name]
        a, b = calls(original, 'flash_attn_with_kvcache')[0], calls(host_node(), 'flash_attn_with_kvcache')[0]
        before = {k.arg: ast.dump(k.value) for k in a.keywords if k.arg != 'q'}
        after = {k.arg: ast.dump(k.value) for k in b.keywords if k.arg != 'q'}
        self.assertEqual(before, after)
        self.assertEqual(ast.dump(next(k.value for k in b.keywords if k.arg == 'q')),
                         ast.dump(ast.Name(id='q', ctx=ast.Load())))
        self.assertEqual(ast.dump(calls(original, 'rms_norm_gate')[0]),
                         ast.dump(calls(host_node(), 'rms_norm_gate')[0]))
        self.assertEqual([ast.dump(n) for n in original.body[-4:]],
                         [ast.dump(n) for n in host_node().body[-4:]])
        self.assertEqual(calls(host_node(), '_fa3_cos_sin'), [])
        compile(SOURCE.read_bytes(), str(SOURCE), 'exec')

    def test_page_scales_live_zero_DP_and_reference_cache_output(self):
        for ctx, page, scale, heads in [(0, 1, 1., 'tp8'), (64, 64, 1/64, 'tp8'),
                                         (32, 1, 1/64, 'tp8'), (64, 64, 1., 'dp')]:
            with self.subTest(ctx=ctx, page=page, scale=scale, heads=heads):
                fresh = old.reference.generate_input(batch=1, ctx=ctx, heads=heads, layer='full',
                    q_len=8, page=page, kv_scale=scale, seed=5101+ctx, device='cpu')
                old.zero_new_page(fresh)
                self.assertEqual(int(fresh['cache_loc'][0]), 0)
                self.check_case(old.buffers.clone_inputs(fresh), fresh)

    def test_CPU_same_address_A_B_A_changed_actual_lengths(self):
        for page, scale in [(1, 1.), (64, 1/64)]:
            args = dict(batch=1, ctx=64, heads='tp8', layer='full', q_len=8,
                        page=page, kv_scale=scale, spread='mixed', device='cpu')
            a = old.reference.generate_input(**args, seed=803)
            b = old.reference.generate_input(**args, seed=804)
            self.assertFalse(torch.equal(a['seq_lens'], b['seq_lens']))
            self.assertFalse(torch.equal(a['positions'], b['positions']))
            self.assertFalse(torch.equal(a['qkv'], b['qkv']))
            work = old.buffers.clone_inputs(a)
            pointers = {k: v.data_ptr() for k, v in work.items() if isinstance(v, torch.Tensor)}
            trace = []
            namespace, state = adapter(work, trace)
            first = None
            for index, fresh in enumerate((a, b, a)):
                old.buffers.restore(work, fresh)
                raw = self.check_case(work, fresh, namespace, trace, state)
                self.assertTrue(all(work[k].data_ptr() == p for k, p in pointers.items()))
                if index == 0:
                    first = raw
                elif index == 2:
                    self.assertTrue(torch.equal(raw, first))

    def test_scheduler_metadata_forwarding_only_with_synthetic_sentinel(self):
        # Real full-verify fixtures use None. This tensor is only a host identity
        # sentinel; its contents are NOT asserted to be a valid native FA3 schedule.
        fresh = old.reference.generate_input(batch=1, ctx=16, heads='tp8', layer='full',
                                             q_len=8, page=1, seed=5120, device='cpu')
        fresh['scheduler_metadata'] = torch.zeros(8, dtype=torch.int32)
        self.check_case(old.buffers.clone_inputs(fresh), fresh)


if __name__ == '__main__':
    unittest.main(verbosity=2)
