"""A2-11 source/CPU checks; no candidate imports or Triton/GPU execution.

The FP32 models round individual operations and use an illustrative tree sum.
They permit reordered sums and do not model native exp2, FMA, or GPU layouts.
Passing these tests establishes neither native equivalence nor judge accuracy.
"""
import ast
import copy
import math
from pathlib import Path
import random
import struct
import unittest

HERE = Path(__file__).resolve().parent
SPLITS = (1, 2, 5, 9, 11, 16, 66)


def fn(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def expr(text):
    return ast.dump(ast.parse(text, mode='eval').body)


def f32(x):
    return struct.unpack('f', struct.pack('f', x))[0]


def tree_sum(xs, cast):
    xs = list(xs)
    while len(xs) > 1:
        xs = [cast(xs[i] + xs[i + 1]) if i + 1 < len(xs) else xs[i]
              for i in range(0, len(xs), 2)]
    return xs[0] if xs else 0.0


def partials(chunks, width=4):
    result = []
    for chunk in chunks:
        m = max((x for x, _ in chunk), default=-math.inf)
        weights = [2.0 ** (x - m) for x, _ in chunk]
        result.append((m, sum(weights), [sum(w * row[d] for w, (_, row) in zip(weights, chunk))
                                        for d in range(width)]))
    return result


def serial_merge(parts, cast=float, vs=1.0):
    m, den, acc = -math.inf, 0.0, [0.0] * len(parts[0][2])
    for sm, sl, sa in parts:
        sm, sl, sa = cast(sm), cast(sl), list(map(cast, sa))
        nm = max(m, sm)
        safe = 0.0 if nm == -math.inf else nm
        wt, ws = cast(2.0 ** cast(m - safe)), cast(2.0 ** cast(sm - safe))
        den = cast(cast(den * wt) + cast(sl * ws))
        acc = [cast(cast(a * wt) + cast(b * ws)) for a, b in zip(acc, sa)]
        m = nm
    return [cast(cast(a * cast(vs)) / den) for a in acc]


def parallel_merge(parts, cast=float, padded=True, vs=1.0):
    rows = [(cast(m), cast(l), list(map(cast, a))) for m, l, a in parts]
    width = len(rows[0][2])
    if padded:
        bs = 1 << (len(rows) - 1).bit_length()
        rows += [(-math.inf, 0.0, [0.0] * width)] * (bs - len(rows))
    m = max(x[0] for x in rows)
    safe = 0.0 if m == -math.inf else m
    weights = [cast(2.0 ** cast(x[0] - safe)) for x in rows]
    den = tree_sum((cast(l * w) for (_, l, _), w in zip(rows, weights)), cast)
    acc = [tree_sum((cast(a[d] * w) for (_, _, a), w in zip(rows, weights)), cast)
           for d in range(width)]
    return [cast(cast(a * cast(vs)) / den) for a in acc]


def direct(chunks):
    tokens = [token for chunk in chunks for token in chunk]
    m = max(x for x, _ in tokens)
    weights = [2.0 ** (x - m) for x, _ in tokens]
    den = sum(weights)
    width = len(tokens[0][1])
    out = [sum(w * row[d] for w, (_, row) in zip(weights, tokens)) / den for d in range(width)]
    moments = [sum(w * abs(row[d]) for w, (_, row) in zip(weights, tokens)) / den
               for d in range(width)]
    return out, moments


def norm_gate(values, gates, cast=float):
    square_sum = tree_sum((cast(x * x) for x in values), cast)
    inv = cast(1.0 / math.sqrt(cast(cast(square_sum / len(values)) + cast(1e-5))))
    return [cast(cast(x * inv) * cast(1.0 / (1.0 + math.exp(-g))))
            for x, g in zip(values, gates)]


def cases():
    rng = random.Random(20261025)
    for count in SPLITS:
        # All historical chunks empty, with only the newest token in the last.
        yield count, 'last-token-only', [[] for _ in range(count - 1)] + [[(-3.5, [1., -2., 0., .125])]]
        for kind in ('mixed', 'sparse', 'cancellation', 'wide-logits'):
            n = max(1, count // 2) if kind == 'sparse' else count * 3 + 7
            tokens = []
            for i in range(n):
                logit = rng.uniform(-80., 80.) if kind == 'wide-logits' else rng.uniform(-12., 12.)
                values = [rng.uniform(-3., 3.) for _ in range(4)]
                if kind == 'cancellation':
                    logit = 0.
                    sign = -1. if i % 2 else 1.
                    values = [sign * 10000. + .03125, -sign * 1024. + .0009765625,
                              sign * .0001, 0.]
                tokens.append((logit, values))
            # The newest token is always assigned to the final split, as in _fused.
            chunks = [[] for _ in range(count)]
            for i, token in enumerate(tokens[:-1]):
                chunks[min(count - 1, i * count // max(1, n - 1))].append(token)
            chunks[-1].append(tokens[-1])
            yield count, kind, chunks


class SourceContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = ast.parse((HERE / 'mixed_fa3_decode_merge.py').read_text())
        cls.source = (HERE / 'mixed_fa3_decode_parallel_merge.py').read_text()
        cls.new = ast.parse(cls.source)

    def test_only_merge_and_host_ast_change(self):
        def retained(tree):
            return [ast.dump(n) for n in tree.body if not
                    (isinstance(n, ast.FunctionDef) and n.name in ('_merge', '_e2_kernel_fn'))]
        self.assertEqual(retained(self.old), retained(self.new))
        compile(self.source, 'A2-11 candidate syntax only', 'exec')

    def test_exact_host_edits_and_l1_counter_bypass(self):
        old, new = copy.deepcopy(fn(self.old, '_e2_kernel_fn')), copy.deepcopy(fn(self.new, '_e2_kernel_fn'))
        branch = new.body[-2]
        self.assertEqual(ast.dump(branch.test), expr('separate_merge and splits > 1'))
        launch = branch.body[0].value
        self.assertEqual(ast.dump(launch.func), expr('_merge[batch, hq]'))
        bs = [k for k in launch.keywords if k.arg == 'BS']
        self.assertEqual(len(bs), 1)
        self.assertEqual(ast.dump(bs[0].value), expr('triton.next_power_of_2(splits)'))
        launch.keywords.remove(bs[0])
        launch.func.slice.elts[1].id = 'hkv'
        self.assertEqual(ast.dump(old), ast.dump(new))
        assigns = {n.targets[0].id: n.value for n in new.body if isinstance(n, ast.Assign)
                   and isinstance(n.targets[0], ast.Name)}
        self.assertEqual(ast.dump(assigns['separate_merge']), expr('length == 1'))
        self.assertEqual(ast.dump(assigns['tickets']), expr('out if separate_merge else _tickets(dev, batch * hkv)'))

    def test_merge_signature_masking_geometry_and_no_ticket_ops(self):
        old, new = fn(self.old, '_merge'), fn(self.new, '_merge')
        self.assertEqual([ast.dump(x) for x in new.args.args[:-1]], [ast.dump(x) for x in old.args.args])
        self.assertEqual(ast.dump(new.args.args[-1]), ast.dump(ast.arg(arg='BS', annotation=ast.parse('tl.constexpr', mode='eval').body)))
        self.assertEqual([ast.dump(x) for x in new.decorator_list], [expr('triton.jit')])
        self.assertEqual(ast.dump(new.body[0].value), expr('tl.static_assert(L == 1)'))
        attrs = {n.attr for n in ast.walk(new) if isinstance(n, ast.Attribute)}
        self.assertFalse({'atomic_add', 'atomic_xchg', 'debug_barrier'} & attrs)
        self.assertFalse(any(isinstance(n, (ast.For, ast.While)) for n in ast.walk(new)))
        assignments = {n.targets[0].id: n.value for n in new.body if isinstance(n, ast.Assign)}
        for name, value in {'head': 'tl.program_id(1)', 'kvh': 'head // G', 'g': 'head % G',
                            'offs_s': 'tl.arange(0, BS)', 'split_ok': 'offs_s < SPLITS',
                            'qb': '((b * HKV + kvh) * SPLITS + offs_s) * BLOCK_M + g'}.items():
            self.assertEqual(ast.dump(assignments[name]), expr(value), name)
        for name, mask, other in [('m_s', 'split_ok', 'float("-inf")'),
                                  ('l_s', 'split_ok', '0.0'), ('o_s', 'split_ok[:, None]', '0.0')]:
            kws = {k.arg: k.value for k in assignments[name].keywords}
            self.assertEqual(ast.dump(kws['mask']), expr(mask))
            self.assertEqual(ast.dump(kws['other']), expr(other))
        for name, value in {'m_t': 'tl.max(m_s, 0)', 'm_z': 'tl.where(m_t == float("-inf"), 0.0, m_t)',
                            'w_s': 'tl.exp2(m_s - m_z)', 'l_t': 'tl.sum(l_s * w_s, 0)'}.items():
            self.assertEqual(ast.dump(assignments[name]), expr(value), name)
        a_values = [ast.dump(n.value) for n in new.body if isinstance(n, ast.Assign) and n.targets[0].id == 'a']
        self.assertEqual(a_values, [expr('tl.sum(o_s * w_s[:, None], 0)'), expr('a * vs / l_t'),
                                    expr('a * tl.rsqrt(tl.sum(a * a, 0) / D + eps)')])

    def test_valid_head_scratch_addresses_are_unique_and_in_bounds(self):
        for batch, hkv, g in [(1, 2, 8), (3, 2, 8), (2, 16, 8)]:
            for count in SPLITS:
                with self.subTest(batch=batch, hkv=hkv, splits=count):
                    addresses = {((b * hkv + head // g) * count + s) * 16 + head % g
                                 for b in range(batch) for head in range(hkv * g) for s in range(count)}
                    self.assertEqual(len(addresses), batch * hkv * g * count)
                    self.assertGreaterEqual(min(addresses), 0)
                    self.assertLess(max(addresses), batch * hkv * count * 16)
                    self.assertTrue(all(index % 16 < g for index in addresses))


class ArithmeticModels(unittest.TestCase):
    def test_scaled_64d_rms_gate_has_forward_error_bound(self):
        rng = random.Random(111025)
        gates = [rng.uniform(-6., 6.) for _ in range(64)]
        for count in SPLITS:
            for vs in (0.015625, 1., 3.0):
                for cancellation in (False, True):
                    with self.subTest(splits=count, scale=vs, cancellation=cancellation):
                        chunks = [[] for _ in range(count)]
                        for i in range(2 * count + 1):
                            row = [((-1. if i % 2 else 1.) * 100. + (d - 32) * .0001
                                    if cancellation else rng.uniform(-3., 3.)) for d in range(64)]
                            chunks[min(i // 2, count - 1)].append((0. if cancellation else rng.uniform(-12., 12.), row))
                        parts = partials(chunks, width=64)
                        avg, moments = direct(chunks)
                        reference_x = [x * vs for x in avg]
                        reference_y = norm_gate(reference_x, gates)
                        # Coordinate merge error is bounded using the absolute
                        # first moment, not the cancellation-sensitive output.
                        errors = [8e-6 * vs * max(1., m) for m in moments]
                        rms_error = math.sqrt(sum(e * e for e in errors) / 64)
                        reference_rms = math.sqrt(sum(x * x for x in reference_x) / 64 + 1e-5)
                        min_rms = max(math.sqrt(1e-5), reference_rms - rms_error)
                        for merge in (serial_merge, parallel_merge):
                            values = merge(parts, f32, vs=vs)
                            output = norm_gate(values, gates, f32)
                            for x, y, ref_x, ref_y, error in zip(values, output, reference_x, reference_y, errors):
                                self.assertTrue(math.isfinite(y))
                                self.assertLessEqual(abs(x - ref_x), error)
                                # Reverse triangle inequality bounds the RMS
                                # change; sigmoid magnitude is at most one.
                                bound = error / min_rms + abs(ref_x) * rms_error / (min_rms * reference_rms)
                                self.assertLessEqual(abs(y - ref_y), bound + 2e-5)

    def test_float64_serial_parallel_and_direct_agree(self):
        for count, kind, chunks in cases():
            with self.subTest(splits=count, kind=kind):
                parts = partials(chunks)
                expected, moments = direct(chunks)
                for got in (serial_merge(parts), parallel_merge(parts)):
                    for value, target, moment in zip(got, expected, moments):
                        self.assertTrue(math.isfinite(value))
                        self.assertLessEqual(abs(value - target), 2e-12 * max(1., moment))

    def test_float32_reordered_sums_have_condition_aware_error_bound(self):
        for count, kind, chunks in cases():
            with self.subTest(splits=count, kind=kind):
                parts = partials(chunks)
                expected, moments = direct(chunks)
                # Absolute first moments account for signed cancellation; relative
                # error against a near-zero output would be misleading.
                for got in (serial_merge(parts, f32), parallel_merge(parts, f32)):
                    for value, target, moment in zip(got, expected, moments):
                        self.assertTrue(math.isfinite(value))
                        self.assertLessEqual(abs(value - target), 8e-6 * max(1., moment))

    def test_empty_padding_is_neutral_in_both_cpu_precisions(self):
        for count, kind, chunks in cases():
            with self.subTest(splits=count, kind=kind):
                for cast in (float, f32):
                    self.assertEqual(parallel_merge(partials(chunks), cast, padded=True),
                                     parallel_merge(partials(chunks), cast, padded=False))


if __name__ == '__main__':
    unittest.main(verbosity=2)
