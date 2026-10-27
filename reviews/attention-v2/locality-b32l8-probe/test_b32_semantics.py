"""CPU opaque-byte proof for the exact B32/L8/page1 diagnostic remap."""
import importlib.util
from pathlib import Path
import random
import sys
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('b32_locality_fixture', HERE/'fixture.py')
f = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = f
spec.loader.exec_module(f)


class B32Semantics(unittest.TestCase):
    def test_both_predeclared_seeds_preserve_all_causal_inputs_and_writes(self):
        for seed in (135831, 135832):
            with self.subTest(seed=seed):
                batch, length, width, row_bytes = 32, 8, 13, 128
                capacity = batch*width
                lengths = [length+b%6 for b in range(batch)]
                permutation = list(range(capacity))
                random.Random(seed).shuffle(permutation)
                table, offset = [], 0
                for n in lengths:
                    table.append(permutation[offset:offset+n]+[-1]*(width-n))
                    offset += n
                locations = [p for b,n in enumerate(lengths) for p in table[b][n-length:n]]
                plan = f.build_plan(table, lengths, locations, length, capacity)
                self.assertEqual(len(set(plan.new_to_old)), capacity)
                self.assertEqual(len(set(plan.old_to_new)), capacity)
                for salt in (17, 83):  # independent opaque K/V encoded bytes
                    original = b''.join(row.to_bytes(8,'little')+bytes([salt])+
                        bytes((row*19+i+salt)%256 for i in range(row_bytes-9))
                        for row in range(capacity))
                    canonical = f.permute_rows_bytes(original, row_bytes, plan)
                    self.assertEqual(f.permute_rows_bytes(canonical,row_bytes,plan,inverse=True),original)
                    for b,n in enumerate(lengths):
                        self.assertEqual(plan.page_table_b[b][n:], tuple([-1]*(width-n)))
                        for q in range(length):
                            rows = f.causal_window_rows(n,length,q,0)
                            self.assertEqual(list(rows),list(range(n-length+q+1)))
                            for logical in rows:
                                old,new = table[b][logical],plan.page_table_b[b][logical]
                                self.assertEqual(original[old*row_bytes:(old+1)*row_bytes],
                                    canonical[new*row_bytes:(new+1)*row_bytes])
                    a,b = bytearray(original),bytearray(canonical)
                    for token,(old,new) in enumerate(zip(plan.cache_loc_a,plan.cache_loc_b)):
                        self.assertEqual(new,plan.old_to_new[old])
                        value=bytes((token*7+i+salt)%256 for i in range(row_bytes))
                        a[old*row_bytes:(old+1)*row_bytes]=value
                        b[new*row_bytes:(new+1)*row_bytes]=value
                    self.assertEqual(f.permute_rows_bytes(b,row_bytes,plan,inverse=True),bytes(a))
                    allowed=set(locations)
                    for row in range(capacity):
                        if row not in allowed:
                            self.assertEqual(a[row*row_bytes:(row+1)*row_bytes],
                                original[row*row_bytes:(row+1)*row_bytes])


if __name__ == '__main__':
    unittest.main()
