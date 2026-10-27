"""Independent CPU checks for physical page permutation, never a GPU timing model."""
import dataclasses
import random
import unittest
from collections import Counter

from fixture import build_plan, permute_rows_bytes

SEEDS = (135831, 135832)


def original(batch, width, lengths, q_len, seed, extra=7):
    capacity = batch * width + extra
    permutation = list(range(capacity))
    random.Random(seed).shuffle(permutation)
    table = [[permutation[b * width + j] if j < lengths[b] else -1
              for j in range(width)] for b in range(batch)]
    loc = [table[b][lengths[b] - q_len + j]
           for b in range(batch) for j in range(q_len)]
    return table, list(lengths), loc, capacity


def cache_bytes(capacity, row_bytes=128, salt=0):
    rows = []
    for row in range(capacity):
        payload = bytes(((row * row_bytes + col + salt) & 255)
                        for col in range(row_bytes))
        rows.append(row.to_bytes(8, 'little') + payload[8:])
    return b''.join(rows)


def row(blob, index, row_bytes=128):
    return blob[index * row_bytes:(index + 1) * row_bytes]


class LocalityFixture(unittest.TestCase):
    def test_full_bijection_canonical_prefix_suffix_and_inverse(self):
        for seed in SEEDS:
            a, lens, loc, cap = original(4, 19, [8, 10, 15, 19], 8, seed)
            plan = build_plan(a, lens, loc, 8, cap)
            self.assertEqual((plan.batch, plan.width, plan.capacity_pages, plan.q_len), (4,19,cap,8))
            self.assertEqual(plan.valid_pages, sum(lens))
            self.assertEqual(sorted(plan.new_to_old), list(range(cap)))
            self.assertEqual(sorted(plan.old_to_new), list(range(cap)))
            for new, old in enumerate(plan.new_to_old):
                self.assertEqual(plan.old_to_new[old], new)
            for b in range(4):
                self.assertEqual(plan.page_table_b[b][:lens[b]], tuple(b*19+j for j in range(lens[b])))
                self.assertEqual(plan.page_table_b[b][lens[b]:], tuple([-1]*(19-lens[b])))
                for j in range(lens[b]):
                    self.assertEqual(plan.new_to_old[plan.page_table_b[b][j]], a[b][j])
            self.assertEqual(len(plan.page_table_b), len(a))
            self.assertTrue(all(len(r) == 19 for r in plan.page_table_b))

    def test_complete_kv_byte_permutation_preserves_unused_pages(self):
        for seed in SEEDS:
            a, lens, loc, cap = original(3, 23, [8, 13, 21], 8, seed)
            plan = build_plan(a,lens,loc,8,cap)
            for salt in (0, 91):
                before = cache_bytes(cap, salt=salt)
                saved = before[:]
                after = permute_rows_bytes(before,128,plan)
                self.assertEqual(before,saved)
                self.assertEqual(len(after),len(before))
                for n in range(cap): self.assertEqual(row(after,n), row(before,plan.new_to_old[n]))
                self.assertEqual(Counter(row(after,n) for n in range(cap)),
                                 Counter(row(before,n) for n in range(cap)))
                self.assertEqual(permute_rows_bytes(after,128,plan,inverse=True),before)
                valid_old = {p for r,l in zip(a,lens) for p in r[:l]}
                valid_new = {p for r,l in zip(plan.page_table_b,lens) for p in r[:l]}
                self.assertEqual({plan.new_to_old[n] for n in range(cap) if n not in valid_new},
                                 set(range(cap))-valid_old)

    def test_logical_kv_exact_for_both_scale_labels_and_all_encoded_values(self):
        a,lens,loc,cap = original(3,17,[8,13,17],8,SEEDS[0])
        plan=build_plan(a,lens,loc,8,cap)
        for scale in (1.0,1.0/64):
            # Byte permutation is independent of scales; no quantization is performed here.
            k=bytes(range(256))*((cap*128+255)//256);k=k[:cap*128]
            v=bytes(x^0x5a for x in k)
            self.assertEqual(set(k),set(range(256)))
            kb=permute_rows_bytes(k,128,plan);vb=permute_rows_bytes(v,128,plan)
            for b in range(3):
                for j in range(lens[b]):
                    old=a[b][j];new=plan.page_table_b[b][j]
                    self.assertEqual((row(kb,new),row(vb,new)),(row(k,old),row(v,old)),(scale,b,j))

    def test_live_new_kv_write_equivalence_and_unused_bytes_unchanged(self):
        for q_len,batch,width,hist in [(1,1,12,[0]),(8,16,49,[i*3%42 for i in range(16)]),
                                       (1,32,19,[i%19 for i in range(32)])]:
            lens=[h+q_len for h in hist]
            a,lens,loc,cap=original(batch,width,lens,q_len,SEEDS[1])
            plan=build_plan(a,lens,loc,q_len,cap)
            for salt in (0,71):
                ka=bytearray(cache_bytes(cap,salt=salt));initial=bytes(ka)
                kb=bytearray(permute_rows_bytes(ka,128,plan))
                for t,(pa,pb) in enumerate(zip(loc,plan.cache_loc_b)):
                    replacement=bytes(((t+salt+53*i)&255) for i in range(128))
                    ka[pa*128:(pa+1)*128]=replacement
                    kb[pb*128:(pb+1)*128]=replacement
                    b=t//q_len;j=t%q_len
                    self.assertEqual(pb,b*width+lens[b]-q_len+j)
                self.assertEqual(permute_rows_bytes(kb,128,plan,inverse=True),bytes(ka))
                for p in set(range(cap))-set(loc): self.assertEqual(row(ka,p),row(initial,p))

    def test_slot_zero_is_live_and_never_filtered(self):
        a=[[7,4,0,-1,-1],[1,6,-1,-1,-1]]
        lens=[3,2];loc=[0,6]
        plan=build_plan(a,lens,loc,1,12)
        self.assertEqual(plan.new_to_old[2],0)
        self.assertEqual(plan.cache_loc_b,(2,6))
        b=permute_rows_bytes(cache_bytes(12),128,plan)
        self.assertEqual(row(b,2),row(cache_bytes(12),0))

    def test_all_query_full_and_sliding_key_sets_including_80k_address_only(self):
        histories=[0,1,7,31,32,33,2046,2047,2048,2049,80000]
        for q_len in (1,8):
            for hist in histories:
                lens=[hist+q_len];width=lens[0]+3
                a,lens,loc,cap=original(1,width,lens,q_len,SEEDS[0],extra=3)
                plan=build_plan(a,lens,loc,q_len,cap)
                for j in range(q_len):
                    endpoint=hist+j
                    for window in (0,2048):
                        first=0 if window==0 else max(0,endpoint-window+1)
                        expected=list(range(first,endpoint+1))
                        old_physical=[a[0][k] for k in expected]
                        new_physical=[plan.new_to_old[plan.page_table_b[0][k]] for k in expected]
                        self.assertEqual(new_physical,old_physical,(q_len,hist,j,window))
                        self.assertTrue(all(k<=endpoint for k in expected))
                        if window:self.assertLessEqual(len(expected),2048)
                        future=set(loc[j+1:])
                        self.assertFalse(future.intersection(old_physical))
                # No large encoded KV allocation in this address-only case.
                self.assertEqual(plan.width,width)

    def test_no_alias_or_mutation_of_input_metadata_and_cache_buffer(self):
        a,lens,loc,cap=original(2,17,[8,13],8,SEEDS[0])
        expected_a=tuple(tuple(r) for r in a);expected_lens=tuple(lens);expected_loc=tuple(loc)
        plan=build_plan(a,lens,loc,8,cap)
        self.assertEqual(tuple(tuple(r) for r in a),expected_a)
        self.assertIsNot(plan.new_to_old,plan.old_to_new)
        a[0][0]=999;lens[0]=999;loc[0]=999
        self.assertEqual(plan.page_table_a,expected_a)
        self.assertEqual(plan.cache_seqlens,expected_lens)
        self.assertEqual(plan.cache_loc_a,expected_loc)
        blob=bytearray(cache_bytes(cap));before=bytes(blob)
        result=permute_rows_bytes(blob,128,plan)
        self.assertEqual(bytes(blob),before);self.assertIsNot(result,blob)
        if dataclasses.is_dataclass(plan):
            with self.assertRaises((dataclasses.FrozenInstanceError,AttributeError)):
                plan.width=123

    def test_fixed_address_style_a_b_a_restoration_and_fresh_plan(self):
        a,lens,loc,cap=original(2,21,[8,19],8,SEEDS[0])
        p=build_plan(a,lens,loc,8,cap);buffer=bytearray(cache_bytes(cap));addr=id(buffer);saved=bytes(buffer)
        buffer[:]=permute_rows_bytes(saved,128,p)
        self.assertEqual(id(buffer),addr)
        for b in range(2):
            for j in range(lens[b]):self.assertEqual(row(buffer,p.page_table_b[b][j]),row(saved,a[b][j]))
        buffer[:]=permute_rows_bytes(buffer,128,p,inverse=True)
        self.assertEqual((id(buffer),bytes(buffer)),(addr,saved))
        a2,l2,c2,_=original(2,21,[13,8],8,SEEDS[1])
        p2=build_plan(a2,l2,c2,8,cap)
        self.assertNotEqual(tuple(p.new_to_old),tuple(p2.new_to_old))
        self.assertEqual(p2.page_table_b[1][8:],tuple([-1]*13))

    def test_refuses_duplicate_live_physical_pages_within_and_across_requests(self):
        for a,lens,loc in [([[0,0]],[2],[0]),([[0,1],[0,-1]],[2,1],[1,0])]:
            with self.assertRaises(ValueError):build_plan(a,lens,loc,1,8)

    def test_refuses_malformed_tables_lengths_locations_and_capacity(self):
        cases=[([[0,-1]],[2],[0],1,4), # missing live page
               ([[0,4]],[2],[4],1,4), # physical out-of-bounds
               ([[0,1]],[1],[0],1,4), # non--1 unused suffix
               ([[0,-1],[1]],[1,1],[0,1],1,4), # ragged table
               ([[0,-1]],[0],[0],1,4), # fewer cached tokens than query block
               ([[0,1]],[2],[0],1,4), # cache_loc not matching new token
               ([[0,-1],[1,-1]],[1,1],[0,1],1,2), # canonical second-batch target OOB
               ([[0,1]],[3],[1],1,4), # length beyond width
               ([[0,1]],[-1],[1],1,4),
               ([[0,1]],[2],[],1,4),
               ([[0,1]],[2],[1],0,4)]
        for args in cases:
            with self.subTest(args=args):
                with self.assertRaises((ValueError,TypeError)):build_plan(*args)

    def test_byte_helper_refuses_invalid_width_or_length(self):
        a,lens,loc,cap=original(1,4,[2],1,SEEDS[0])
        p=build_plan(a,lens,loc,1,cap)
        for blob,row_bytes in [(b'',128),(b'x'*cap,0),(b'x'*(cap*128-1),128)]:
            with self.assertRaises((ValueError,TypeError)):permute_rows_bytes(blob,row_bytes,p)

if __name__=='__main__':unittest.main()
