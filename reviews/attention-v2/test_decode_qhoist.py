"""A2-12 AST and CPU component checks; never imports the candidate or GPU code.

The stdlib FP32/BF16 model illustrates rounding boundaries and fresh inputs.
It is not a model of native Triton exp2/trig, reduction layout, or FMA behavior.
"""
import ast
import copy
import math
from pathlib import Path
import random
import struct
import unittest

HERE = Path(__file__).resolve().parent


def fn(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def expr(source):
    return ast.dump(ast.parse(source, mode='eval').body)


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def bf16(x):
    bits = struct.unpack('<I', struct.pack('<f', x))[0]
    rounded = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
    return struct.unpack('<f', struct.pack('<I', rounded))[0]


def sum32(values):
    values = list(values)
    while len(values) > 1:
        values = [f32(values[i] + values[i + 1]) for i in range(0, len(values), 2)]
    return values[0]


def q_component(row, position, theta=500000., factor=5.473, early_round=False):
    """Illustrative FP32 operations, with the existing final BF16 boundary."""
    row = list(map(bf16, row))
    d = len(row)
    inv = f32(1.0 / math.sqrt(f32(f32(sum32(f32(x*x) for x in row) / d) + f32(1e-5))))
    norm = [f32(x * inv) for x in row]
    if early_round:
        norm = list(map(bf16, norm))
    output = []
    for i, x in enumerate(norm):
        frequency = f32(2.0 ** f32(f32(-(2.0 * (i // 2) / d)) * f32(math.log2(theta))))
        angle = f32(f32(position) * frequency)
        cosine, sine = f32(math.cos(angle)), f32(math.sin(angle))
        sign = -1.0 if i % 2 == 0 else 1.0
        rotated = f32(f32(x * cosine) + f32(sign * f32(norm[i ^ 1] * sine)))
        output.append(bf16(f32(rotated * f32(factor))))
    return output


class QHoistSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = ast.parse((HERE / 'mixed_fa3_decode_parallel_merge.py').read_text())
        cls.source = (HERE / 'mixed_fa3_decode_qhoist.py').read_text()
        cls.new = ast.parse(cls.source)

    def test_other_top_level_nodes_including_routing_unchanged(self):
        def kept(tree):
            return [ast.dump(n) for n in tree.body if not
                    (isinstance(n, ast.FunctionDef) and n.name in ('_prepare_q', '_fused', '_e2_kernel_fn'))]
        self.assertEqual(kept(self.old), kept(self.new))
        compile(self.source, 'A2-12 syntax check only', 'exec')

    def test_fused_reconstructs_parent_exactly(self):
        old, new = fn(self.old, '_fused'), copy.deepcopy(fn(self.new, '_fused'))
        self.assertEqual(new.args.args.pop(1).arg, 'prepared_q_ptr')
        last = new.args.args.pop()
        self.assertEqual(last.arg, 'PREPARED_Q')
        self.assertEqual(ast.dump(last.annotation), expr('tl.constexpr'))
        index = next(i for i, n in enumerate(new.body) if isinstance(n, ast.If)
                     and isinstance(n.test, ast.Name) and n.test.id == 'PREPARED_Q')
        branch = new.body[index]
        expected = ast.parse('''q = tl.load(prepared_q_ptr + tok[:, None] * (HQ * D)
            + ((kvh * G + g) * D)[:, None] + offs_d[None, :],
            mask=row_ok[:, None], other=0.0).to(DOT)''').body
        self.assertEqual([ast.dump(n) for n in branch.body], [ast.dump(n) for n in expected])
        new.body[index:index + 1] = branch.orelse
        self.assertEqual(ast.dump(new), ast.dump(old))

    def test_host_reconstructs_parent_and_preparation_is_local_and_ordered(self):
        old, new = fn(self.old, '_e2_kernel_fn'), copy.deepcopy(fn(self.new, '_e2_kernel_fn'))
        i = next(i for i, n in enumerate(new.body) if isinstance(n, ast.Assign)
                 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'prepare_q')
        self.assertEqual(ast.dump(new.body[i].value), expr('batch == 1 and length == 1 and window == 2048 and cfg["rope_theta"] == 500000'))
        self.assertEqual(ast.dump(new.body[i + 1].value), expr('qkv'))
        branch = new.body[i + 2]
        self.assertEqual(ast.dump(branch.test), expr('prepare_q'))
        expected = ast.parse('''if prepare_q:
    prepared_q = torch.empty((hq, d), dtype=torch.bfloat16, device=dev)
    _prepare_q[(triton.cdiv(hq, block_m),)](qkv, data["seq_lens"], prepared_q,
        theta, RMS_EPS, float(cfg["qk_scale_factor"]), HQ=hq, D=d,
        BLOCK_M=block_m, num_warps=4, num_stages=3)
''').body[0]
        self.assertEqual(ast.dump(branch), ast.dump(expected))
        launch = new.body[i + 3].value
        self.assertEqual(launch.func.value.id, '_fused')
        self.assertEqual(ast.dump(launch.args.pop(1)), expr('prepared_q'))
        prepared = [k for k in launch.keywords if k.arg == 'PREPARED_Q']
        self.assertEqual(len(prepared), 1)
        self.assertEqual(ast.dump(prepared[0].value), expr('prepare_q'))
        launch.keywords.remove(prepared[0])
        del new.body[i:i + 3]
        self.assertEqual(ast.dump(new), ast.dump(old))

    def test_preparation_has_exact_position_norm_and_rounding_boundaries(self):
        helper = fn(self.new, '_prepare_q')
        self.assertEqual([arg.arg for arg in helper.args.args],
                         ['qkv_ptr', 'seq_ptr', 'prepared_q_ptr', 'theta', 'eps', 'qk_factor', 'HQ', 'D', 'BLOCK_M'])
        self.assertEqual([ast.dump(n) for n in helper.decorator_list], [expr('triton.jit')])
        expected = ast.parse('''heads = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
row_ok = heads < HQ
offs_d = tl.arange(0, D)
qpos = tl.full((BLOCK_M,), 0, tl.int64) + tl.load(seq_ptr).to(tl.int64)
q_rows = qkv_ptr + heads[:, None] * D
q = _norm_rope(q_rows, offs_d, qpos[:, None], theta, eps, True, D, row_ok[:, None])
q = (q * qk_factor).to(tl.bfloat16)
tl.store(prepared_q_ptr + heads[:, None] * D + offs_d[None, :], q, mask=row_ok[:, None])
''').body
        self.assertEqual([ast.dump(n) for n in helper.body], [ast.dump(n) for n in expected])

    def test_all_scored_dispatches_and_nearby_shape_guards(self):
        host = fn(self.new, '_e2_kernel_fn')
        condition = next(n.value for n in host.body if isinstance(n, ast.Assign)
                         and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'prepare_q')
        predicate = compile(ast.Expression(condition), 'shape-only predicate', 'eval')
        # Evaluate only the inspected shape/config predicate; no candidate function runs.
        def enabled(batch, length, window, theta, **extra_cfg):
            return eval(predicate, {'__builtins__': {}}, dict(batch=batch, length=length,
                        window=window, cfg=dict(extra_cfg, rope_theta=theta)))
        def route(tree, cfg):
            for node in fn(tree, 'kernel_fn').body[1:]:
                if isinstance(node, ast.If):
                    yes = eval(compile(ast.Expression(node.test), 'dispatch predicate', 'eval'),
                               {'__builtins__': {}}, {'cfg': cfg})
                    if yes:
                        return node.body[0].value.func.id
                elif isinstance(node, ast.Return):
                    return node.value.func.id
        checked = selected = 0
        for batch in (1, 16, 32):
            for length in (1, 8):
                for window, theta in ((0, 0), (2048, 500000)):
                    cfg = dict(batch=batch, q_len=length, window=window, rope_theta=theta)
                    self.assertEqual(route(self.old, cfg), route(self.new, cfg))
                    self.assertEqual(enabled(batch, length, window, theta),
                                     batch == 1 and length == 1 and window == 2048)
                    selected += enabled(batch, length, window, theta)
                    checked += 1
        self.assertEqual(checked, 12)
        self.assertEqual(selected, 1)
        for batch, length, window, theta in [(2,1,2048,500000), (15,1,2048,500000),
                (17,1,2048,500000), (33,1,2048,500000), (1,2,2048,500000),
                (1,7,2048,500000), (1,9,2048,500000), (1,1,2047,500000),
                (1,1,2049,500000), (1,1,2048,499999), (1,1,2048,500001)]:
            self.assertFalse(enabled(batch, length, window, theta))
        for context in (10, 2047, 2048, 79999, 80000, 80001, 81920):
            for page, hq, hkv in ((1, 16, 2), (64, 16, 2), (64, 128, 16)):
                self.assertTrue(enabled(1, 1, 2048, 500000, ctx=context,
                                        page_size=page, num_q_heads=hq, num_kv_heads=hkv))
        names = {n.id for n in ast.walk(condition) if isinstance(n, ast.Name)}
        self.assertEqual(names, {'batch', 'length', 'window', 'cfg'})


class QHoistComponents(unittest.TestCase):
    def test_bf16_scratch_roundtrip_and_round_to_even_boundaries(self):
        for value in (0., -0., 1., -1., 1. + 2.**-8, 1. + 3.*2.**-8,
                      -1. - 2.**-8, 123.456, .000123):
            rounded = bf16(value)
            self.assertEqual(bf16(f32(rounded)), rounded)
            self.assertTrue(math.isfinite(rounded))
        self.assertEqual(bf16(1. + 2.**-8), 1.)
        self.assertEqual(bf16(1. + 3.*2.**-8), 1. + 2.**-6)

    def test_component_detects_fresh_data_positions_and_premature_rounding(self):
        rng = random.Random(121025)
        row = [bf16(rng.uniform(-3., 3.)) for _ in range(64)]
        changed = row.copy()
        changed[0] += 1.
        original = q_component(row, 80000)
        self.assertNotEqual(original, q_component(changed, 80000))
        self.assertNotEqual(original, q_component(row, 80001))
        self.assertNotEqual(original, q_component(row, 80000, early_round=True))
        self.assertNotEqual(original, q_component(row, 80000, factor=1.))
        for position in (0, 10, 2047, 2048, 79999, 80000, 80001):
            result = q_component(row, position)
            self.assertEqual(len(result), 64)
            self.assertTrue(all(math.isfinite(x) and bf16(f32(x)) == x for x in result))


if __name__ == '__main__':
    unittest.main(verbosity=2)
