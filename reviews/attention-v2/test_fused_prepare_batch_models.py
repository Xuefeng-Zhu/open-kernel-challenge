"""Component-only CPU models for possible batched fused-prologue dispatch.

The current complete A2-05 candidate routes only B1/L8/sliding to this component.
These tests call its generic host adapter directly, replacing Triton and FA3 with
CPU models. They do not extend candidate dispatch, execute native kernels, capture
CUDA graphs, or establish GPU correctness/performance. Contexts stay <= 2,050 tokens.
"""
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_fused_prepare_component as fused
import torch


reference = fused.sliding.old.reference
buffers = fused.sliding.old.buffers


class FusedPrepareBatchModels(unittest.TestCase):
    def _execute_and_check(self, work, fresh, namespace=None, trace=None, state=None):
        if namespace is None:
            trace = []
            namespace, state = fused.adapter(work, trace)
        before = len(trace)
        out = namespace['_fa3_sliding_verify_fused'](work)
        cfg = work['config']
        self.assertEqual(out.shape, (cfg['batch'], 8, cfg['num_q_heads'] * 64))
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(out.float()).all())
        self.assertEqual(trace[before:], ['prepare_fused_model', 'attention', 'gate'])

        # Reject a nonfinite oracle before comparisons: a NaN error is not a pass.
        oracle = reference.ref_kernel(work)
        self.assertTrue(all(torch.isfinite(v.float()).all() for v in oracle))
        raw = reference.collect(work, out)
        ok, message = reference.check_implementation(work, raw)
        self.assertTrue(ok, message)
        intact, message = buffers.unchanged(work, fresh)
        self.assertTrue(intact, message)

        # Verify all requested stores, including byte-level FP8 values. This checks
        # the CPU adapter's writeback, not actual Triton address generation.
        slots = work['cache_loc'].long()
        hkv = cfg['num_kv_heads']
        for cache, produced in [('k_cache', 'k'), ('v_cache', 'v')]:
            written = work[cache].view(-1, hkv, 64).view(torch.uint8)[slots]
            expected = state[produced].contiguous().view(torch.uint8)
            self.assertTrue(torch.equal(written, expected))
            self.assertTrue(torch.isfinite(state[produced].float()).all())
        self.assertTrue(torch.isfinite(state['q'].float()).all())
        return raw.clone()

    def test_prerequisite_like_batches_paging_scales_and_live_zero(self):
        cases = [
            dict(batch=3, ctx=64, page=64, kv_scale=1/64,
                 spread='fixed', dist='normal', seed=4601, live_zero=True),
            # This length plants both the oldest visible and first excluded key.
            dict(batch=4, ctx=2050, page=1, kv_scale=1.,
                 spread='fixed', dist='edges', seed=4602, live_zero=False),
            dict(batch=3, ctx=67, page=1, kv_scale=1/64,
                 spread='mixed', dist='soft', seed=4603, live_zero=False),
        ]
        for case in cases:
            with self.subTest(case=case):
                args = dict(case)
                live_zero = args.pop('live_zero')
                fresh = reference.generate_input(**args, heads='tp8', layer='sliding',
                                                 q_len=8, device='cpu')
                if live_zero:
                    fused.sliding.old.zero_new_page(fresh)
                    self.assertEqual(int(fresh['cache_loc'][0]), 0)
                self._execute_and_check(buffers.clone_inputs(fresh), fresh)

    def test_batch16_batch32_and_dp_head_layout_small_contexts(self):
        cases = [
            dict(batch=16, ctx=32, heads='tp8', page=1, kv_scale=1., seed=4611),
            dict(batch=32, ctx=16, heads='tp8', page=64, kv_scale=1/64, seed=4612),
            dict(batch=1, ctx=16, heads='dp', page=64, kv_scale=1., seed=4613),
        ]
        for case in cases:
            with self.subTest(case=case):
                fresh = reference.generate_input(**case, layer='sliding', q_len=8,
                                                 spread='mixed', dist='soft', device='cpu')
                self._execute_and_check(buffers.clone_inputs(fresh), fresh)

    def test_cpu_same_address_batch3_A_B_A_with_changed_actual_lengths(self):
        args = dict(batch=3, ctx=64, heads='tp8', layer='sliding', q_len=8,
                    page=64, kv_scale=1/64, spread='mixed', dist='soft', device='cpu')
        a = reference.generate_input(**args, seed=803)
        b = reference.generate_input(**args, seed=804)
        self.assertFalse(torch.equal(a['seq_lens'], b['seq_lens']))
        self.assertFalse(torch.equal(a['positions'], b['positions']))
        self.assertFalse(torch.equal(a['qkv'], b['qkv']))
        work = buffers.clone_inputs(a)
        addresses = {k: v.data_ptr() for k, v in work.items() if isinstance(v, torch.Tensor)}
        trace = []
        namespace, state = fused.adapter(work, trace)
        first = None
        for index, fresh in enumerate((a, b, a)):
            buffers.restore(work, fresh)
            raw = self._execute_and_check(work, fresh, namespace, trace, state)
            self.assertTrue(all(work[k].data_ptr() == address
                                for k, address in addresses.items()))
            if index == 0:
                first = raw
            elif index == 2:
                self.assertTrue(torch.equal(first, raw))
        self.assertEqual(trace.count('prepare_fused_model'), 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
