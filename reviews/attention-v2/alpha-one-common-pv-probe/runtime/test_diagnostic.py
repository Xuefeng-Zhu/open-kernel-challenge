"""Offline diagnostic-source proof and independent historical-tile count model."""
import ast
import hashlib
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
SOURCE_SHA = '14a8fa2e3d7b64fa299fe69c8da6569501c069425a62e07cc5f11b1e9b15f3e5'
SOURCE = HERE.parent/'candidate.py'
DIAGNOSTIC = HERE/'diagnostic.py'

PATCHES = [
    ('           o_ptr, m_ptr, l_ptr, ticket_ptr, out_ptr,\n',
     '           o_ptr, m_ptr, l_ptr, ticket_ptr, out_ptr, stats_ptr,\n'),
    ('    offs_n = tl.arange(0, BLOCK_N)\n',
     '    stats_total = tl.full((), 0, tl.int32)\n'
     '    stats_fast = tl.full((), 0, tl.int32)\n'
     '    stats_slow = tl.full((), 0, tl.int32)\n'
     '    offs_n = tl.arange(0, BLOCK_N)\n'),
    ('            all_alpha_one = tl.sum((row_ok & (alpha != 1.0)).to(tl.int32), 0) == 0\n',
     '            all_alpha_one = tl.sum((row_ok & (alpha != 1.0)).to(tl.int32), 0) == 0\n'
     '            stats_total += 1\n'
     '            stats_fast += all_alpha_one.to(tl.int32)\n'
     '            stats_slow += 1 - all_alpha_one.to(tl.int32)\n'),
    ('    if split == SPLITS - 1:\n',
     '    stats_row = ((b * HKV + kvh) * SPLITS + split) * 3\n'
     '    tl.store(stats_ptr + stats_row, stats_total)\n'
     '    tl.store(stats_ptr + stats_row + 1, stats_fast)\n'
     '    tl.store(stats_ptr + stats_row + 2, stats_slow)\n\n'
     '    if split == SPLITS - 1:\n'),
    ('        data["k_scale"], data["v_scale"], part_o, part_m, part_l, tickets, out,\n',
     '        data["k_scale"], data["v_scale"], part_o, part_m, part_l, tickets, out, data["alpha_stats"],\n'),
]


def dump(node):
    return ast.dump(node, include_attributes=False)


def function(tree, name):
    return next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name)


def historical_tiles(seq, split, chunk=8192, block=64):
    if seq < 0 or split < 0 or chunk <= 0 or block <= 0:
        raise ValueError('Nonnegative lengths/splits and positive geometry required')
    span = max(0, min(seq, (split+1)*chunk)-split*chunk)
    return (span+block-1)//block


class DiagnosticSourceChecks(unittest.TestCase):
    def setUp(self):
        self.original = SOURCE.read_text()
        self.instrumented = DIAGNOSTIC.read_text()
        self.tree = ast.parse(self.instrumented)
        self.fused = function(self.tree, '_fused')

    def test_exact_source_and_full_reverse_patch(self):
        self.assertEqual(hashlib.sha256(self.original.encode()).hexdigest(), SOURCE_SHA)
        restored = self.instrumented
        for before, after in reversed(PATCHES):
            self.assertEqual(restored.count(after), 1)
            restored = restored.replace(after, before)
        self.assertEqual(restored, self.original)
        # Stronger than selected expression checks: every original byte returns.
        self.assertEqual(dump(ast.parse(restored)), dump(ast.parse(self.original)))

    def test_only_fused_and_its_launch_function_change(self):
        original = ast.parse(self.original)
        for node in original.body:
            if isinstance(node, ast.FunctionDef) and node.name not in {'_fused', '_e2_kernel_fn'}:
                self.assertEqual(dump(node), dump(function(self.tree, node.name)), node.name)
        old_args = [x.arg for x in function(original, '_fused').args.args]
        new_args = [x.arg for x in self.fused.args.args]
        self.assertEqual([x for x in new_args if x != 'stats_ptr'], old_args)
        self.assertEqual(new_args.index('stats_ptr'), new_args.index('out_ptr')+1)
        self.assertEqual(self.instrumented.count('data["alpha_stats"]'), 1)

    def test_three_int32_loop_counters_reuse_the_original_predicate(self):
        loop = next(x for x in self.fused.body if isinstance(x, ast.For))
        inits = [x for x in self.fused.body if isinstance(x, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id in {'stats_total', 'stats_fast', 'stats_slow'} for t in x.targets)]
        self.assertEqual(len(inits), 3)
        for init in inits:
            self.assertEqual(dump(init.value), dump(ast.parse('tl.full((), 0, tl.int32)', mode='eval').body))
        branch = next(x for x in loop.body if isinstance(x, ast.If)
                      and isinstance(x.test, ast.Name) and x.test.id == 'PAGE_LOOKAHEAD'
                      and any(isinstance(y, ast.Name) and y.id == 'all_alpha_one' for y in ast.walk(x)))
        expected = ast.parse('''all_alpha_one = tl.sum((row_ok & (alpha != 1.0)).to(tl.int32), 0) == 0
stats_total += 1
stats_fast += all_alpha_one.to(tl.int32)
stats_slow += 1 - all_alpha_one.to(tl.int32)
''').body
        self.assertEqual([dump(x) for x in branch.body[:4]], [dump(x) for x in expected])
        self.assertEqual(sum(isinstance(x, ast.AugAssign) and isinstance(x.target, ast.Name)
                             and x.target.id.startswith('stats_') for x in ast.walk(self.fused)), 3)
        # No atomic/counter or stats pointer is allowed to feed original float math.
        for node in ast.walk(branch):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {'acc', 'l_i', 'p', 'm_i'} for t in node.targets):
                self.assertFalse(any(isinstance(x, ast.Name) and x.id.startswith('stats_') for x in ast.walk(node.value)))
        self.assertEqual(self.instrumented.count('tl.atomic_'), self.original.count('tl.atomic_'))

    def test_all_ctas_store_three_owned_slots_before_unchanged_tail(self):
        tail_index = next(i for i, node in enumerate(self.fused.body)
                          if isinstance(node, ast.If) and dump(node.test) == dump(ast.parse('split == SPLITS - 1', mode='eval').body))
        stores = self.fused.body[tail_index-3:tail_index]
        expected = ast.parse('''tl.store(stats_ptr + stats_row, stats_total)
tl.store(stats_ptr + stats_row + 1, stats_fast)
tl.store(stats_ptr + stats_row + 2, stats_slow)
''').body
        self.assertEqual([dump(x) for x in stores], [dump(x) for x in expected])
        self.assertEqual(dump(self.fused.body[tail_index-4]),
                         dump(ast.parse('stats_row = ((b * HKV + kvh) * SPLITS + split) * 3').body[0]))
        self.assertEqual(self.instrumented.split('    if split == SPLITS - 1:\n', 1)[1].split('def _e2_kernel_fn', 1)[0],
                         self.original.split('    if split == SPLITS - 1:\n', 1)[1].split('def _e2_kernel_fn', 1)[0])
        for hkv, splits in [(2, 11), (16, 2), (16, 3)]:
            slots = [((b*hkv+k)*splits+s)*3+col
                     for b in range(32) for k in range(hkv) for s in range(splits) for col in range(3)]
            self.assertEqual(sorted(slots), list(range(32*hkv*splits*3)))

    def test_cpu_tile_model_matches_independent_range_enumeration(self):
        lengths = [0, 1, 63, 64, 65, 127, 128, 129, 8191, 8192, 8193,
                   16383, 16384, 16385, 80000, 86000]
        for seq in lengths:
            totals = []
            for split in range(11):
                expected = len(range(split*8192, min((split+1)*8192, seq), 64))
                self.assertEqual(historical_tiles(seq, split), expected)
                self.assertLessEqual(expected, 128)
                totals.append(expected)
            self.assertEqual(sum(totals), (seq+63)//64)
        self.assertEqual(historical_tiles(0, 10), 0)
        self.assertEqual(historical_tiles(8192, 1), 0)
        self.assertEqual(historical_tiles(8193, 1), 1)
        for seq, split in [(-1, 0), (1, -1)]:
            with self.assertRaises(ValueError):
                historical_tiles(seq, split)


if __name__ == '__main__':
    unittest.main(verbosity=2)
