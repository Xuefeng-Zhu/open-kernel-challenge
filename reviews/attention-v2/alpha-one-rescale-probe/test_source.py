"""Offline source/predicate checks; no Torch, Triton, GPU or native claims."""
import ast
import copy
import hashlib
import math
from pathlib import Path
import random
import struct
import unittest

HERE = Path(__file__).resolve().parent
BASE_SHA = '3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8'
OLD = '        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16).to(DOT), v)\n'


def nodes(raw):
    return ast.parse(raw)


def dump(node):
    return ast.dump(node, include_attributes=False)


def functions(tree):
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def all_valid_alpha_one(alpha, valid):
    if len(alpha) != len(valid):
        raise ValueError('Matching alpha/query-lane shapes required')
    return sum(bool(ok and value != 1.0) for value, ok in zip(alpha, valid)) == 0


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


class SourceChecks(unittest.TestCase):
    def setUp(self):
        self.baseline = (HERE/'baseline.py').read_text()
        self.candidate = (HERE/'candidate.py').read_text()
        self.before = nodes(self.baseline)
        self.after = nodes(self.candidate)

    def test_exact_control_and_one_historical_accumulator_patch(self):
        self.assertEqual(hashlib.sha256(self.baseline.encode()).hexdigest(), BASE_SHA)
        self.assertEqual((HERE.parent/'fusion-lookahead-probe/candidate.py').read_text(), self.baseline)
        before, after = functions(self.before), functions(self.after)
        self.assertEqual(before.keys(), after.keys())
        for name in before.keys() - {'_fused'}:
            self.assertEqual(dump(before[name]), dump(after[name]), name)
        old_lines = self.baseline.splitlines(keepends=True)
        new_lines = self.candidate.splitlines(keepends=True)
        first = old_lines.index(OLD)
        inserted = ''.join(new_lines[first:first+9])
        self.assertIn('all_alpha_one = tl.sum((row_ok & (alpha != 1.0)).to(tl.int32), 0) == 0', inserted)
        self.assertEqual(self.candidate.replace(inserted, OLD, 1), self.baseline)

    def test_both_slow_paths_are_the_exact_old_expression(self):
        fused = functions(self.after)['_fused']
        branch = next(node for node in ast.walk(fused) if isinstance(node, ast.If)
            and any(isinstance(x, ast.Name) and x.id == 'all_alpha_one' for x in ast.walk(node))
            and isinstance(node.test, ast.Name) and node.test.id == 'PAGE_LOOKAHEAD')
        inner = next(node for node in branch.body if isinstance(node, ast.If))
        old = nodes(OLD.strip()).body[0]
        self.assertEqual(dump(inner.orelse[0]), dump(old))
        self.assertEqual(dump(branch.orelse[0]), dump(old))
        self.assertEqual(dump(inner.body[0]), dump(nodes('acc = acc + tl.dot(p.to(tl.bfloat16).to(DOT), v)').body[0]))

    def test_denom_and_new_token_tail_remain_byte_exact(self):
        self.assertEqual(self.baseline.count('l_i = l_i * alpha + tl.sum(p, 1)'), 2)
        self.assertEqual(self.candidate.count('l_i = l_i * alpha + tl.sum(p, 1)'), 2)
        tail = '    if split == SPLITS - 1:\n'
        self.assertEqual(self.baseline.split(tail, 1)[1], self.candidate.split(tail, 1)[1])
        self.assertEqual(self.baseline.count('max_num_imprecise_acc=0'), self.candidate.count('max_num_imprecise_acc=0'))

    def test_only_existing_exact_dispatch_guard_enables_patch(self):
        launch = functions(self.after)['_e2_kernel_fn']
        expected = nodes('page_lookahead = batch == 32 and length == 1 and window == 0 and not rope').body[0]
        self.assertIn(dump(expected), [dump(node) for node in launch.body])
        self.assertEqual(dump(functions(self.before)['_e2_kernel_fn']), dump(launch))
        # There is no new timing/cache/profiler or alternate precision mechanism.
        for token in ['ALPHA_THRESHOLD', 'rescale_threshold', 'clock64', 'atomic_add(counter']:
            self.assertNotIn(token, self.candidate)

    def test_predicate_exactness_and_padding(self):
        valid = [True]*8 + [False]*8
        self.assertTrue(all_valid_alpha_one([1.0]*8+[math.nan, 0.0, -0.0, -1.0, 2.0, math.inf, -math.inf, .5], valid))
        for special in [0.0, -0.0, .5, math.nan, math.inf, -math.inf,
                        math.nextafter(1.0, 0.0), math.nextafter(1.0, 2.0)]:
            alpha = [1.0]*16
            alpha[3] = special
            self.assertFalse(all_valid_alpha_one(alpha, valid), repr(special))
        self.assertFalse(all_valid_alpha_one([0.0]*16, valid))  # First nonempty tile.
        with self.assertRaises(ValueError):
            all_valid_alpha_one([1.0], [])

    def test_ieee_scalar_identity_is_a_model_not_native_equivalence(self):
        examples = [0.0, -0.0, 2.0**-149, -(2.0**-149), 2.0**-126,
                    -(2.0**-126), 1.0, -1.0, 2.0**100]
        rng = random.Random(135731)
        for _ in range(256):
            bits = rng.getrandbits(32)
            value = struct.unpack('<f', struct.pack('<I', bits))[0]
            if math.isfinite(value):
                examples.append(value)
        for old in examples:
            for update in [0.0, -0.0, 2.0**-149, 1.0, -1.0]:
                # Host IEEE round-to-nearest scalar model only. Native FTZ,
                # tensor-core accumulation order and compiler folding are gates.
                slow = f32(f32(old*1.0)+update)
                fast = f32(old+update)
                self.assertEqual(struct.pack('<f', slow), struct.pack('<f', fast))


if __name__ == '__main__':
    unittest.main(verbosity=2)
