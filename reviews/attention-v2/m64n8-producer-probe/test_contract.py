"""Independent CPU contract/models, never a GPU precision/oracle result.

The models use exact dyadic fixtures so FP32/BF16-representable intermediate
state can be compared algebraically. They do not execute a candidate or vendor
package and cannot establish instruction lowering, FMA/reduction behavior,
buffer lifetimes, native accuracy, performance or graph replay safety.
"""
import ast
import copy
import hashlib
import importlib.util
import math
from pathlib import Path
import random
import struct
import textwrap
import unittest

HERE = Path(__file__).resolve().parent
SELECTED = HERE.parent / 'fusion-lookahead-probe/candidate.py'
SELECTED_SHA = '3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8'
G, D, BN, BL, ABI_M = 8, 64, 64, 16, 16


def ceildiv(a, b):
    return (a + b - 1) // b


def split_intervals(seq, window, splits, chunk=0):
    """Selected L1 partition contract; ranges may be empty/reversed."""
    lo = max(seq - window + 1, 0) if window else 0
    if chunk:
        return [(s * chunk, min((s + 1) * chunk, seq)) for s in range(splits)]
    width = ceildiv(ceildiv(seq - lo, splits), BN) * BN
    return [(lo + s * width, min(lo + (s + 1) * width, seq)) for s in range(splits)]


def live_blocks(interval):
    begin, end = interval
    return [list(range(at, min(at + BN, end))) for at in range(begin, end, BN)]


def qk_rows(q, keys):
    return [[sum(a * b for a, b in zip(head, key)) for key in keys] for head in q]


def qk_key_major(q, keys):
    return [[sum(a * b for a, b in zip(key, head)) for head in q] for key in keys]


def exp2_dyadic(value):
    if value == -math.inf:
        return 0.0
    if value != int(value):
        raise ValueError('Fixtures require integer logits/exact power-of-two probabilities')
    return math.ldexp(1.0, int(value))


def update_rows(state, scores, values):
    maxima, mass, acc = state
    for g, row in enumerate(scores):
        new_max = max(maxima[g], max(row, default=-math.inf))
        safe = 0.0 if new_max == -math.inf else new_max
        alpha = exp2_dyadic(maxima[g] - safe)
        prob = [exp2_dyadic(s - safe) for s in row]
        mass[g] = mass[g] * alpha + sum(prob)
        acc[g] = [acc[g][d] * alpha + sum(p * values[t][d] for t, p in enumerate(prob))
                  for d in range(D)]
        maxima[g] = new_max
    return maxima, mass, acc


def update_key_major(state, scores_t, values):
    """Independent transposed reductions: max/sum over key axis; V.T @ P."""
    maxima, mass, acc = state
    next_max = [max(maxima[g], max((row[g] for row in scores_t), default=-math.inf))
                for g in range(G)]
    safe = [0.0 if m == -math.inf else m for m in next_max]
    alpha = [exp2_dyadic(m - z) for m, z in zip(maxima, safe)]
    p_t = [[exp2_dyadic(row[g] - safe[g]) for g in range(G)] for row in scores_t]
    # Key-major probability publication is BF16-exact on these dyadic fixtures.
    new_mass = [mass[g] * alpha[g] + sum(row[g] for row in p_t) for g in range(G)]
    pv_t = [[sum(values[t][d] * p_t[t][g] for t in range(len(values)))
             for g in range(G)] for d in range(D)]
    new_acc = [[acc[g][d] * alpha[g] + pv_t[d][g] for d in range(D)] for g in range(G)]
    return next_max, new_mass, new_acc


def empty_state():
    return ([-math.inf] * G, [0.0] * G, [[0.0] * D for _ in range(G)])


def dyadic_fixture(count, maximum_at=None):
    q = [[float(d == g) for d in range(D)] for g in range(G)]
    # Each Q head selects a distinct feature; each score stays an integer.
    keys = [[float(((t + d) % 5) - 2) if d < G else 0.0 for d in range(D)]
            for t in range(count)]
    if maximum_at is not None:
        keys[maximum_at] = [3.0 if d < G else 0.0 for d in range(D)]
    values = [[float(((3 * t + d) % 7) - 3) for d in range(D)] for t in range(count)]
    return q, keys, values


def producer_state(q, keys, values, blocks, reversed_mapping=False):
    state = empty_state()
    for indices in blocks:
        kk = [keys[i] for i in indices]
        vv = [values[i] for i in indices]
        if reversed_mapping:
            state = update_key_major(state, qk_key_major(q, kk), vv)
        else:
            state = update_rows(state, qk_rows(q, kk), vv)
    return state


def merge_states(states):
    result = empty_state()
    for g in range(G):
        maximum = max(state[0][g] for state in states)
        safe = maximum if maximum != -math.inf else 0.0
        weights = [exp2_dyadic(state[0][g] - safe) for state in states]
        result[0][g] = maximum
        result[1][g] = sum(state[1][g] * w for state, w in zip(states, weights))
        result[2][g] = [sum(state[2][g][d] * w for state, w in zip(states, weights))
                        for d in range(D)]
    return result


def bf16(value):
    word = struct.unpack('<I', struct.pack('<f', value))[0]
    word = (word + 0x7fff + ((word >> 16) & 1)) & 0xffff0000
    return struct.unpack('<f', struct.pack('<I', word))[0]


def e4m3(value):
    """Finite RN-even mathematical encoding model, not a CUDA cast emulator."""
    if not math.isfinite(value) or abs(value) > 448:
        raise ValueError('Finite in-range fixtures only')
    sign = -1.0 if value < 0 else 1.0
    points = [(0.0, 0)]
    for code in range(1, 127):
        exponent, mantissa = code >> 3, code & 7
        magnitude = math.ldexp(float(mantissa), -9) if exponent == 0 else math.ldexp(1 + mantissa / 8, exponent - 7)
        points.append((magnitude, code))
    magnitude, _ = min(points, key=lambda pair: (abs(pair[0] - abs(value)), pair[1] & 1))
    return sign * magnitude


class PartitionContract(unittest.TestCase):
    def test_exact_four_route_geometry(self):
        selected_tree = ast.parse(SELECTED.read_text())
        node = next(n for n in selected_tree.body if isinstance(n, ast.FunctionDef) and n.name == '_num_splits')
        env = {'NUM_SMS': 132}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), 'selected-pure-split-helper', 'exec'), env)
        self.assertEqual([env['_num_splits'](b, 2, 2048, BN) for b in (1, 16, 32)], [16, 9, 5])
        self.assertEqual(ceildiv(86001, 8192), 11)
        self.assertEqual([b * 2 * s for b, s in ((1, 16), (16, 9), (32, 5), (32, 11))], [32, 288, 320, 704])

    def test_window_and_chunk_boundaries_cover_every_key_once(self):
        for seq in (0, 1, 63, 64, 65, 127, 128, 2047, 2048, 2049, 8191, 8192, 8193, 16384):
            for window, splits, chunk in [(2048, s, 0) for s in (5, 9, 16)] + [(0, 11, 8192)]:
                pieces = split_intervals(seq, window, splits, chunk)
                visited = [key for p in pieces for block in live_blocks(p) for key in block]
                first = max(0, seq - 2047) if window else 0
                self.assertEqual(visited, list(range(first, seq)))
                self.assertEqual(len(visited), len(set(visited)))
                self.assertTrue(all(len(block) <= BN for p in pieces for block in live_blocks(p)))

    def test_empty_last_split_still_owns_exact_n16_tail(self):
        for seq in (0, 1, 63, 64, 65, 8191, 8192, 8193):
            for splits, window, chunk in ((16, 2048, 0), (9, 2048, 0), (5, 2048, 0), (11, 0, 8192)):
                intervals = split_intervals(seq, window, splits, chunk)
                writers = [(s, token) for s in range(splits) for token in range(BL)
                           if s == splits - 1 and token < 1]
                self.assertEqual(writers, [(splits - 1, 0)])
                tail_keys = [seq + token for _, token in writers]
                self.assertEqual(tail_keys, [seq])
                if seq < (splits - 1) * 8192 and chunk:
                    self.assertEqual(live_blocks(intervals[-1]), [])
                self.assertEqual(BL, 16)

    def test_page1_live_mutation_and_lookahead_never_read_new_token(self):
        for seq in (0, 1, 63, 64, 65, 8193):
            width = seq + 3
            a = list(range(width)); b = list(range(width))
            random.Random(seq + 171).shuffle(a); random.Random(seq + 172).shuffle(b)
            table = a[:]; pointer = id(table)
            pieces = split_intervals(seq, 0, 11, 8192)
            logical = [i for piece in pieces for block in live_blocks(piece) for i in block]
            old = [table[i] for i in logical]
            table[:] = b
            self.assertEqual(id(table), pointer)
            self.assertEqual([table[i] for i in logical], [b[i] for i in range(seq)])
            if seq > 1: self.assertNotEqual(old, [table[i] for i in logical])
            # A page-lookahead implementation reads only future historical tiles.
            accesses = [i for piece in pieces for block in live_blocks(piece) for i in block]
            self.assertTrue(all(0 <= i < seq for i in accesses))
            self.assertNotIn(seq, accesses)
            # Live physical slot0 remains a valid source/destination row.
            table[seq] = 0
            self.assertEqual(table[seq], 0)

    def test_two_slot_payload_schedule_waits_before_read_or_reuse(self):
        # Abstract worst-case committed-group ordering, not hardware execution.
        for tiles in range(8):
            pending, ready, issued, consumed = [], {}, [], []
            def issue(tile, slot):
                self.assertFalse(any(s == slot for _, s in pending))
                if slot in ready: self.assertIn(ready[slot], consumed)
                pending.append((tile, slot)); issued.append(tile)
            for tile in range(min(2, tiles)): issue(tile, tile)
            slot = 0
            for tile in range(tiles):
                retain = 1 if tile + 1 < tiles else 0
                while len(pending) > retain:
                    done, target = pending.pop(0); ready[target] = done
                self.assertEqual(ready[slot], tile)
                # Source's CTA barrier follows raw reads into separate BF16
                # K/V buffers before this raw slot is reused for tile+2.
                consumed.append(tile)
                if tile + 2 < tiles: issue(tile + 2, slot)
                slot ^= 1
            self.assertEqual(issued, list(range(tiles)))
            self.assertEqual(consumed, list(range(tiles)))
            self.assertFalse(pending)


class OnlineTransposeContract(unittest.TestCase):
    def test_row_major_and_key_major_dots_and_online_states(self):
        for seq in (0, 1, 63, 64, 65, 129):
            for maximum_at in (0, seq // 2, seq):
                q, keys, values = dyadic_fixture(seq + 1, maximum_at)
                scores = qk_rows(q, keys); scores_t = qk_key_major(q, keys)
                self.assertEqual(scores, [list(row) for row in zip(*scores_t)])
                for splits in (2, 3, 11):
                    states_a, states_b = [], []
                    for s, interval in enumerate(split_intervals(seq, 0, splits, 8192)):
                        blocks = live_blocks(interval)
                        if s == splits - 1: blocks += [[seq]]
                        states_a.append(producer_state(q, keys, values, blocks))
                        states_b.append(producer_state(q, keys, values, blocks, True))
                    self.assertEqual(states_a, states_b)
                    combined = merge_states(states_a)
                    direct = producer_state(q, keys, values, [list(range(seq + 1))])
                    self.assertEqual(combined, direct)
                    for head in range(G):
                        self.assertGreater(combined[1][head], 0)
                        self.assertEqual([v / combined[1][head] for v in combined[2][head]],
                                         [v / direct[1][head] for v in direct[2][head]])

    def test_fully_masked_history_preserves_neutral_valid_state_until_tail(self):
        q, keys, values = dyadic_fixture(1)
        empty = producer_state(q, keys, values, [])
        self.assertEqual(empty, empty_state())
        own = producer_state(q, keys, values, [[0]], True)
        self.assertEqual(own[1], [1.0] * G)
        self.assertEqual(merge_states([empty, empty, own]), own)

    def test_scale_placement_and_rounding_boundaries_are_not_interchangeable(self):
        self.assertEqual(e4m3(bf16(1.0626)), 1.0)
        self.assertEqual(e4m3(1.0626), 1.125)
        self.assertNotEqual(e4m3(bf16(1.0626 * 5.4730064863875)),
                            e4m3(bf16(1.0626)) * 5.4730064863875)
        q = [[0.0] * D for _ in range(G)]
        keys = [[1.0] * D for _ in range(3)]
        values = [[float(i + 1)] * D for i in range(3)]
        # The encoded-key scale is in QK once; V scale is applied after merging.
        for ks, vs in ((1.0, 1.0), (1.0 / 64, 1.0 / 64)):
            scale = (.125 * 5.4730064863875) * ks * 1.4426950408889634
            self.assertEqual(scale, (.125 * 5.4730064863875 * 1.4426950408889634) * ks)
            states = [producer_state(q, keys, values, [[0, 1]]), producer_state(q, keys, values, [[2]], True)]
            merged = merge_states(states)
            output = [[v * vs / merged[1][g] for v in merged[2][g]] for g in range(G)]
            self.assertEqual(merged[1], [3.0] * G)
            self.assertEqual(output, [[2.0 * vs] * D for _ in range(G)])

    def test_m16_partial_abi_never_reads_ignored_query_or_bs_padding(self):
        class Poison:
            def __float__(self): raise AssertionError('Ignored partial row read')
        batch, hkv, splits, bs = 2, 2, 3, 4
        storage = [Poison()] * (batch * hkv * splits * ABI_M)
        for b in range(batch):
            for kv in range(hkv):
                for s in range(splits):
                    for g in range(G):
                        at = ((b * hkv + kv) * splits + s) * ABI_M + g
                        storage[at] = float(1000*b + 100*kv + 10*s + g)
        for b in range(batch):
            for head in range(hkv * G):
                kv, g = divmod(head, G)
                loaded = [float(storage[((b*hkv+kv)*splits+s)*ABI_M+g])
                          if s < splits else 0.0 for s in range(bs)]
                self.assertEqual(loaded, [float(1000*b+100*kv+10*s+g) for s in range(splits)] + [0.0])
        self.assertEqual(16 * (64 + 2) * 4, 4224)
        self.assertEqual(32 * 2 * 11 * 4224, 2973696)

    def test_original_padded_query_rows_need_not_be_neutral(self):
        # Selected masked Q is zero, but visibility does not mask invalid M rows.
        # Historical score0 yields nonzero mass/acc; parallel merge never reads it.
        invalid_q = [[0.0] * D for _ in range(G)]
        _, keys, _ = dyadic_fixture(3)
        values = [[1.0] * D for _ in range(3)]
        padded = producer_state(invalid_q, keys, values, [[0, 1, 2]])
        self.assertEqual(padded[0], [0.0] * G)
        self.assertEqual(padded[1], [3.0] * G)
        self.assertEqual(padded[2], [[3.0] * D for _ in range(G)])
        self.assertNotEqual(padded, empty_state())


class BuilderContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('owned_candidate_builder_contract', HERE/'build_candidate.py')
        cls.builder = importlib.util.module_from_spec(spec); spec.loader.exec_module(cls.builder)
        cls.selected = SELECTED.read_bytes()

    def test_exact_selected_roundtrip_and_original_merge_abi(self):
        self.assertEqual(hashlib.sha256(self.selected).hexdigest(), SELECTED_SHA)
        kernel = 'def _m64n8_fused(*args, **kwargs):\n    pass\n'
        result = self.builder.assemble(self.selected, kernel)
        recovered = result.split(self.builder.MARKER)[0].replace(self.builder.NEW_LAUNCH, self.builder.OLD_LAUNCH)
        self.assertEqual(recovered.encode(), self.selected)
        old = ast.parse(self.selected.decode()); new = ast.parse(result)
        merge = lambda tree: next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_merge')
        self.assertEqual(ast.dump(merge(old)), ast.dump(merge(new)))

    def test_builder_refuses_changed_base_and_all_global_shadowing(self):
        kernel = 'def _m64n8_fused(*args, **kwargs):\n    pass\n'
        with self.assertRaises(ValueError): self.builder.assemble(self.selected + b'\n', kernel)
        for shadow in ('RMS_EPS = 0.1\n', 'import math as tl\n', 'class Inputs: pass\n', 'def _merge(): pass\n'):
            with self.subTest(shadow=shadow):
                with self.assertRaises(ValueError): self.builder.assemble(self.selected, shadow + kernel)

    def test_dispatch_exact_scope_and_single_split_fallback(self):
        prefix = textwrap.dedent(self.builder.NEW_LAUNCH.split('    producer[')[0])
        assignment = next(n for n in ast.parse(prefix).body if isinstance(n, ast.Assign)
                          and n.targets[0].id == 'use_m64n8')
        def guard(**updates):
            env = dict(length=1, g=8, hq=16, hkv=2, d=64, page=1, batch=32,
                       window=0, rope=False, splits=11, cfg={'rope_theta': 0})
            env.update(updates)
            exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(assignment)], type_ignores=[])), 'pure-owned-guard', 'exec'), env)
            return env['use_m64n8']
        self.assertTrue(guard())
        for b in (1,16,32): self.assertTrue(guard(batch=b, window=2048, rope=True, cfg={'rope_theta':500000}, splits=5))
        for update in ({'splits':1},{'length':8},{'page':64},{'batch':16},{'hq':128,'hkv':16},{'g':4},{'batch':4,'window':2048,'rope':True,'cfg':{'rope_theta':500000}}):
            self.assertFalse(guard(**update))
        for capacity in (1, 8191, 8192): self.assertFalse(guard(splits=ceildiv(capacity,8192)))
        self.assertTrue(guard(splits=ceildiv(8193,8192)))


class IntegratedSourceContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = (HERE/'kernel.py').read_text()
        cls.tree = ast.parse(cls.text)
        cls.baseline = ast.parse(SELECTED.read_text())

    def test_norm_rope_expressions_match_selected_before_rounding(self):
        original = copy.deepcopy(next(n for n in self.baseline.body
            if isinstance(n, ast.FunctionDef) and n.name == '_norm_rope'))
        owned = copy.deepcopy(next(n for n in self.tree.body
            if isinstance(n, ast.FunctionDef) and n.name == '_m64n8_norm_rope'))
        class LanguageName(ast.NodeTransformer):
            def visit_Name(self, node):
                return ast.copy_location(ast.Name(id='tl', ctx=node.ctx), node) if node.id == '_m64n8_gl' else node
        owned = LanguageName().visit(owned)
        for node in (original, owned):
            node.name = 'norm'; node.decorator_list = []
            if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body = node.body[1:]
        self.assertEqual(ast.dump(owned, include_attributes=False), ast.dump(original, include_attributes=False))

    def test_signature_n16_tail_and_dtype_boundaries_are_explicit(self):
        original = next(n for n in self.baseline.body if isinstance(n, ast.FunctionDef) and n.name == '_fused')
        owned = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == '_m64n8_fused')
        self.assertEqual([a.arg for a in owned.args.args], [a.arg for a in original.args.args])
        code = ast.unparse(owned)
        self.assertIn('SPLITS > 1 and SEPARATE_MERGE', code)
        self.assertIn('BLOCK_L == 16', code)
        self.assertIn('q.to(_m64n8_gl.bfloat16).to(_m64n8_gl.float8e4nv).to(_m64n8_gl.bfloat16)', code)
        self.assertIn('(kn / ks).to(kc_ptr.dtype.element_ty)', code)
        self.assertIn('(vn / vs).to(vc_ptr.dtype.element_ty)', code)
        tail = next(n for n in owned.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'split == SPLITS - 1')
        tail_code = ast.unparse(tail)
        self.assertIn('nl = _m64n8_gl.arange(0, 16,', tail_code)
        self.assertIn('tail_n = _m64n8_gl.arange(0, 16,', tail_code)
        self.assertIn('(tail_n < L)[None, :]', tail_code)
        self.assertIn('vq.to(_m64n8_gl.float32).to(_m64n8_gl.bfloat16)', tail_code)
        self.assertEqual(sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                             and n.func.id == '_m64n8_mma_v2' for n in ast.walk(tail)), 2)
        wgmma = [n for n in ast.walk(owned) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Name) and n.func.id == '_m64n8_warpgroup_mma']
        self.assertEqual(len(wgmma), 2)
        for call in wgmma:
            self.assertEqual(next(k.value.value for k in call.keywords if k.arg == 'max_num_imprecise_acc'), 0)

    def test_actual_assembly_preserves_complete_selected_roundtrip(self):
        spec = importlib.util.spec_from_file_location('actual_owned_builder_contract', HERE/'build_candidate.py')
        builder = importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)
        selected = SELECTED.read_bytes()
        result = builder.assemble(selected, self.text)
        recovered = result.split(builder.MARKER)[0].replace(builder.NEW_LAUNCH, builder.OLD_LAUNCH)
        self.assertEqual(recovered.encode(), selected)
        if (HERE/'candidate.py').is_file():
            self.assertEqual((HERE/'candidate.py').read_text(), result)


if __name__ == '__main__':
    unittest.main()
