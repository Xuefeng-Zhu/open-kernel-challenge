"""CPU-only L1 checks for the unchanged generic full fused host component.

No candidate dispatch is changed. The Triton producer and FA3 remain CPU stand-ins;
same-address tests below are ordinary calls, not captured CUDA graph execution.
Real FA3 decode schedules are unavailable here; sentinel tests prove forwarding
and in-place metadata updates only, not validity of any native schedule encoding.
"""
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_full_fused_prepare_component as full
import torch


class FullFusedDecodeModel(unittest.TestCase):
    def check_case(self, work, fresh, namespace=None, trace=None, state=None):
        self.assertEqual(work['config']['batch'], 1)
        self.assertEqual(work['config']['q_len'], 1)
        # Reuse the existing rigorous finite-oracle/cache/immutability checks
        # and exact parent CPU-adapter byte comparison without editing them.
        return full.FullFusedPrepareComponent.check_case(
            self, work, fresh, namespace, trace, state)

    def test_L1_TP8_DP_page1_page64_unit_nonunit_live_cache_zero(self):
        for heads in ['tp8', 'dp']:
            for page in [1, 64]:
                for scale in [1., 1/64]:
                    with self.subTest(heads=heads, page=page, scale=scale):
                        fresh = full.old.reference.generate_input(
                            batch=1, ctx=64, heads=heads, layer='full', q_len=1,
                            page=page, kv_scale=scale, seed=6100+page, device='cpu')
                        full.old.zero_new_page(fresh)
                        self.assertEqual(int(fresh['cache_loc'][0]), 0)
                        self.check_case(full.old.buffers.clone_inputs(fresh), fresh)

    def test_empty_history_one_new_token_nonunit_scale(self):
        fresh = full.old.reference.generate_input(
            batch=1, ctx=0, heads='tp8', layer='full', q_len=1,
            page=64, kv_scale=1/64, seed=6120, device='cpu')
        full.old.zero_new_page(fresh)
        self.assertEqual(int(fresh['cache_loc'][0]), 0)
        self.check_case(full.old.buffers.clone_inputs(fresh), fresh)

    def test_CPU_A_B_A_same_addresses_changed_lengths_and_metadata(self):
        for page, scale in [(1, 1.), (64, 1/64)]:
            with self.subTest(page=page, scale=scale):
                args = dict(batch=1, ctx=64, heads='tp8', layer='full', q_len=1,
                            page=page, kv_scale=scale, spread='mixed', device='cpu')
                a = full.old.reference.generate_input(**args, seed=803)
                b = full.old.reference.generate_input(**args, seed=804)
                self.assertFalse(torch.equal(a['seq_lens'], b['seq_lens']))
                self.assertFalse(torch.equal(a['qkv'], b['qkv']))
                # Synthetic, fixed-size CPU metadata; not a valid FA3 encoding.
                for fresh, marker in [(a, 101), (b, 202)]:
                    fresh['scheduler_metadata'] = torch.tensor(
                        [marker, int(fresh['cache_seqlens'][0]), 0, 0], dtype=torch.int32)
                self.assertFalse(torch.equal(a['scheduler_metadata'], b['scheduler_metadata']))
                work = full.old.buffers.clone_inputs(a)
                addresses = {k: v.data_ptr() for k, v in work.items() if isinstance(v, torch.Tensor)}
                trace = []
                namespace, state = full.adapter(work, trace)
                first = None
                for index, fresh in enumerate((a, b, a)):
                    full.old.buffers.restore(work, fresh)
                    self.assertTrue(torch.equal(work['scheduler_metadata'], fresh['scheduler_metadata']))
                    raw = self.check_case(work, fresh, namespace, trace, state)
                    self.assertTrue(all(work[k].data_ptr() == p for k, p in addresses.items()))
                    if index == 0:
                        first = raw
                    elif index == 2:
                        self.assertTrue(torch.equal(raw, first))
                self.assertEqual(trace.count('prepare_full_model'), 3)

    def test_supplied_scheduler_tensor_identity_for_L1(self):
        fresh = full.old.reference.generate_input(
            batch=1, ctx=32, heads='tp8', layer='full', q_len=1,
            page=1, kv_scale=1., seed=6130, device='cpu')
        fresh['scheduler_metadata'] = torch.tensor([7, 11, 13, 17], dtype=torch.int32)
        work = full.old.buffers.clone_inputs(fresh)
        trace = []
        namespace, state = full.adapter(work, trace)
        attention = namespace['flash_attn_with_kvcache']
        def checked_attention(**kwargs):
            self.assertIs(kwargs['scheduler_metadata'], work['scheduler_metadata'])
            self.assertIs(kwargs['cache_seqlens'], work['cache_seqlens'])
            self.assertIs(kwargs['cu_seqlens_q'], work['cu_seqlens_q'])
            self.assertEqual(kwargs['max_seqlen_q'], 1)
            self.assertEqual(kwargs['window_size'], (-1, -1))
            self.assertIs(kwargs['causal'], True)
            return attention(**kwargs)
        namespace['flash_attn_with_kvcache'] = checked_attention
        self.check_case(work, fresh, namespace, trace, state)


if __name__ == '__main__':
    unittest.main(verbosity=2)
