"""CPU contract models; these tests do not compile or execute the GPU kernels."""

from collections import Counter
import ast
import math
from pathlib import Path
import random
import struct
from types import SimpleNamespace
import unittest


D, G, N, EPS = 64, 8, 64, 1e-5


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def chunks(seq, splits):
    per = math.ceil(math.ceil(seq / splits) / N) * N
    return [range(s * per, min((s + 1) * per, seq)) for s in range(splits)]


def partial_row(b, kvh, split, qrow, hkv, splits, block_m):
    return ((b * hkv + kvh) * splits + split) * block_m + qrow


def partial(scores, values, cast=float):
    """Online base-2 state, retaining neutral state for an empty interval."""
    top, mass, acc = -math.inf, cast(0), [cast(0)] * D
    for start in range(0, len(scores), N):
        tile = scores[start:start + N]
        new_top = max(top, max(tile))
        alpha = cast(2 ** (top - new_top))
        weights = [cast(2 ** (x - new_top)) for x in tile]
        mass = cast(cast(mass * alpha) + cast(sum(weights)))
        acc = [cast(cast(old * alpha) + cast(sum(
            cast(p * v[d]) for p, v in zip(weights, values[start:start + N])
        ))) for d, old in enumerate(acc)]
        top = new_top
    return top, mass, acc


def merge(states, v_scale, gate, cast=float):
    top = max(s[0] for s in states)
    weights = [cast(2 ** (s[0] - top)) for s in states]
    mass = cast(sum(cast(w * s[1]) for w, s in zip(weights, states)))
    numerator = [cast(sum(cast(w * s[2][d]) for w, s in zip(weights, states)))
                 for d in range(D)]
    a = [cast(cast(x * v_scale) / mass) for x in numerator]
    inv = cast(1 / math.sqrt(cast(sum(cast(x * x) for x in a) / D) + EPS))
    return [cast(cast(x * inv) / (1 + math.exp(-g))) for x, g in zip(a, gate)]


def dense_reference(scores, values, v_scale, gate):
    """Independent dense normalization, with no tile or split state."""
    top = max(scores)
    weights = [2 ** (x - top) for x in scores]
    norm = math.fsum(weights)
    a = [math.fsum(w * v[d] * v_scale for w, v in zip(weights, values)) / norm
         for d in range(D)]
    inv = 1 / math.sqrt(math.fsum(x * x for x in a) / D + EPS)
    return [x * inv / (1 + math.exp(-g)) for x, g in zip(a, gate)]


class DualHeadContractTests(unittest.TestCase):
    def test_pair_query_mapping_is_bijective(self):
        for hkv in (2, 16):
            for length in (1, 8):
                mapped = [(j, (2 * pair + h) * G + g)
                          for pair in range(hkv // 2) for h in range(2)
                          for j in range(length) for g in range(G)]
                expected = {(j, head) for j in range(length) for head in range(hkv * G)}
                self.assertEqual(set(mapped), expected)
                self.assertEqual(len(mapped), len(expected))

    def test_balanced_history_covers_every_key_once(self):
        for seq in (0, 1, 63, 64, 65, 129):
            for splits in (1, 2, 5, 9, 17, 132):
                visited = Counter(t for chunk in chunks(seq, splits) for t in chunk)
                self.assertEqual(visited, Counter(range(seq)), (seq, splits))
                for chunk in chunks(seq, splits):
                    self.assertEqual(chunk.start % N, 0)

    def test_physical_page_addressing_matches_nested_storage(self):
        rng = random.Random(3701)
        for hkv in (2, 16):
            for page in (1, 64):
                table = list(range(6))
                rng.shuffle(table)
                for token in range(3 * page + 1):
                    physical_page, in_page = table[token // page], token % page
                    for pair in range(hkv // 2):
                        for h in range(2):
                            kvh = 2 * pair + h
                            for d in (0, 1, 31, 32, 63):
                                # Oracle treats storage as [page, offset, head, D].
                                nested = (((physical_page * page + in_page) * hkv + kvh) * D + d)
                                row = table[token // page] * page + token % page
                                candidate = row * (hkv * D) + (pair * 2 + h) * D + d
                                self.assertEqual(candidate, nested)
                                self.assertEqual(divmod(candidate // D, hkv), (row, kvh))

    def test_last_split_owns_new_writes_even_without_history(self):
        for hkv in (2, 16):
            for length in (1, 8):
                for seq in (0, 1, 63, 64, 65, 129):
                    for splits in (1, 5, 17, 132):
                        # Slot zero is intentionally a valid live destination.
                        cache_loc = list(reversed(range(length)))
                        writes = Counter()
                        for split, _ in enumerate(chunks(seq, splits)):
                            if split == splits - 1:
                                for pair in range(hkv // 2):
                                    for h in range(2):
                                        for j, row in enumerate(cache_loc):
                                            writes[(j, row, pair * 2 + h)] += 1
                        expected = Counter((j, cache_loc[j], head)
                                           for j in range(length) for head in range(hkv))
                        self.assertEqual(writes, expected)

    def test_causal_key_set_and_two_heads_do_not_mix(self):
        for length in (1, 8):
            for seq in (0, 1, 63, 64, 65, 129):
                for splits in (1, 9, 132):
                    for h in range(2):
                        for j in range(length):
                            keys = [(h, t) for c in chunks(seq, splits) for t in c]
                            keys += [(h, seq + k) for k in range(length) if k <= j]
                            self.assertEqual(Counter(keys), Counter((h, t) for t in range(seq + j + 1)))
                            self.assertFalse(any(kvh != h for kvh, _ in keys))

    def test_partial_abi_merge_selects_correct_token_head(self):
        for hkv in (2, 16):
            for length, block_m in ((1, 16), (8, 64)):
                splits = 9
                storage = {}
                for b in range(2):
                    for pair in range(hkv // 2):
                        for h in range(2):
                            for s in range(splits):
                                for qrow in range(block_m):
                                    index = partial_row(b, 2 * pair + h, s, qrow, hkv, splits, block_m)
                                    self.assertNotIn(index, storage)
                                    storage[index] = (b, 2 * pair + h, s, qrow)
                for tok in range(2 * length):
                    for head in range(hkv * G):
                        b, j, kvh, g = tok // length, tok % length, head // G, head % G
                        for s in range(splits):
                            index = partial_row(b, kvh, s, j * G + g, hkv, splits, block_m)
                            self.assertEqual(storage[index], (b, kvh, s, j * G + g))

    def test_online_split_merge_matches_dense_with_independent_scales(self):
        rng = random.Random(3702)
        # Exact representable raw values stand in for already-encoded cache bytes.
        raw_values = [[rng.randrange(-8, 9) / 8 for _ in range(D)] for _ in range(137)]
        gate = [rng.uniform(-2, 2) for _ in range(D)]
        for seq, length in ((0, 1), (1, 8), (65, 8), (129, 8)):
            for ks, vs in ((1.0, 1.0), (1 / 64, 1 / 32), (1 / 32, 1 / 64)):
                raw_scores = [rng.uniform(-9, 9) for _ in range(seq + length)]
                for max_at in (0, (seq + length) // 2, seq + length - 1):
                    logits = [x * ks for x in raw_scores]
                    logits[max_at] += 12 * ks
                    for j in range(length):
                        expected = dense_reference(logits[:seq + j + 1], raw_values[:seq + j + 1], vs, gate)
                        for splits in (1, 5, 132):
                            for cast, tolerance in ((float, 1e-12), (f32, 2e-6)):
                                states = []
                                for s, c in enumerate(chunks(seq, splits)):
                                    ids = list(c)
                                    if s == splits - 1:
                                        ids += list(range(seq, seq + j + 1))
                                    states.append(partial([logits[t] for t in ids], [raw_values[t] for t in ids], cast))
                                got = merge(states, vs, gate, cast)
                                self.assertLessEqual(max(abs(a - b) for a, b in zip(got, expected)), tolerance)

    def test_rms_epsilon_cannot_drop_mass(self):
        # Large mass/near cancellation makes normalization epsilon material.
        scores = [0.0, 0.0]
        values = [[1.0] * D, [-1.0 + 1e-4] * D]
        gate = [0.0] * D
        expected = dense_reference(scores, values, 1 / 64, gate)
        got = merge([partial(scores, values)], 1 / 64, gate)
        self.assertAlmostEqual(got[0], expected[0], places=14)
        numerator = [sum(v[d] for v in values) / 64 for d in range(D)]
        incorrectly_drop_mass = numerator[0] / math.sqrt(sum(x * x for x in numerator) / D + EPS) / 2
        self.assertGreater(abs(incorrectly_drop_mass - expected[0]), 1e-5)


class FakeTensor:
    def __init__(self, shape, *, dtype="input", device="cuda", owner=None, strides=None):
        self.shape = tuple(shape)
        self.dtype, self.device = dtype, device
        self.owner = owner if owner is not None else object()
        self._strides = strides

    def view(self, *shape):
        elements = math.prod(self.shape)
        shape = list(shape)
        if -1 in shape:
            shape[shape.index(-1)] = elements // math.prod(x for x in shape if x != -1)
        if math.prod(shape) != elements:
            raise AssertionError("Invalid host view")
        return FakeTensor(shape, dtype=self.dtype, device=self.device, owner=self.owner)

    def stride(self, dim):
        return self._strides[dim] if self._strides else math.prod(self.shape[dim + 1:])


class FakeJIT:
    def __init__(self, parameters):
        self.parameters, self.calls = parameters, []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            if len(args) > len(self.parameters):
                raise AssertionError("Host kernel positional ABI mismatch")
            bound = dict(zip(self.parameters, args))
            bound.update(kwargs)
            self.calls.append(dict(grid=grid, bound=bound))
        return launch


def load_host_adapter():
    """Execute only the two actual host AST nodes; never import vendor code."""
    source = Path(__file__).with_name("candidate.py").read_text()
    tree = ast.parse(source)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    allocations, fallbacks = [], []

    def empty(shape, *, device, dtype):
        result = FakeTensor(shape, device=device, dtype=dtype)
        allocations.append(result)
        return result

    def champion(data):
        fallbacks.append(data)
        return "champion fallback"

    def no_fa3(*args, **kwargs):
        raise AssertionError("Full adapter called FA3")

    torch = SimpleNamespace(empty=empty, float32="float32", bfloat16="bfloat16", Tensor=FakeTensor)
    env = dict(torch=torch, Inputs=dict, NUM_SMS=132, RMS_EPS=EPS,
               _QK_DOT="FP8 QK", _DOT="BF16 PV", _champion_kernel_fn=champion,
               flash_attn_with_kvcache=no_fa3,
               triton=SimpleNamespace(cdiv=lambda x, y: (x + y - 1) // y,
                                      next_power_of_2=lambda n: 1 << (n - 1).bit_length()))
    for name in ("_dual_full", "_dual_merge"):
        env[name] = FakeJIT([a.arg for a in functions[name].args.args])
    nodes = [functions[name] for name in ("_dual_full_call", "kernel_fn")]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "candidate-host-only", "exec"), env)
    return env, allocations, fallbacks


def fake_data(batch=32, length=8, hkv=2, page=1, width=86008):
    hq = hkv * G
    qkv_width = (2 * hq + 2 * hkv) * D
    cfg = dict(batch=batch, q_len=length, num_q_heads=hq, num_kv_heads=hkv,
               head_dim=D, page_size=page, window=0, rope_theta=0,
               qk_scale_factor=5.4730064863875, sm_scale=0.125, kv_scale=1 / 64)
    data = dict(config=cfg, qkv=FakeTensor((batch, length, qkv_width)),
                page_table=FakeTensor((batch, width), strides=(width + 13, 1)))
    for name in ("seq_lens", "cache_loc", "k_cache", "v_cache", "k_scale", "v_scale"):
        data[name] = FakeTensor((1,), dtype=name)
    return data


class ActualHostAdapterTests(unittest.TestCase):
    def test_tp_grids_split_policy_and_live_abi(self):
        for batch, expected_s in ((1, 132), (16, 9), (32, 5)):
            for length in (1, 8):
                env, allocations, fallback = load_host_adapter()
                data = fake_data(batch=batch, length=length)
                result = env["kernel_fn"](data)
                producer, merge_call = env["_dual_full"].calls[0], env["_dual_merge"].calls[0]
                self.assertEqual(producer["grid"], (batch, 1, expected_s))
                self.assertEqual(merge_call["grid"], (batch * length, 16))
                p = producer["bound"]
                self.assertEqual((p["L"], p["G"], p["HQ"], p["HKV"], p["D"]), (length, 8, 16, 2, 64))
                self.assertEqual(p["BLOCK_M"], 64 if length == 8 else 16)
                self.assertEqual(p["num_warps"], 8 if length == 8 else 4)
                self.assertEqual(p["stride_pt"], 86021)
                for formal, key in (("seq_ptr", "seq_lens"), ("pt_ptr", "page_table"),
                                    ("loc_ptr", "cache_loc"), ("kc_ptr", "k_cache"),
                                    ("vc_ptr", "v_cache"), ("ks_ptr", "k_scale"), ("vs_ptr", "v_scale")):
                    self.assertIs(p[formal], data[key])
                self.assertEqual(result.shape, (batch, length, 16 * 64))
                self.assertEqual(len(allocations), 4)
                self.assertFalse(fallback)

    def test_dp_passes_actual_head_count_and_scaled_page_layout(self):
        for batch, expected_s in ((2, 9), (8, 3)):
            for length in (1, 8):
                env, _, fallback = load_host_adapter()
                data = fake_data(batch=batch, length=length, hkv=16, page=64, width=150)
                result = env["kernel_fn"](data)
                producer = env["_dual_full"].calls[0]
                self.assertEqual(producer["grid"], (batch, 8, expected_s))
                p = producer["bound"]
                self.assertEqual((p["HKV"], p["HQ"], p["page"]), (16, 128, 64))
                self.assertEqual(p["stride_tok"], (2 * 128 + 2 * 16) * 64)
                self.assertEqual(p["stride_pt"], 163)
                self.assertIsNot(p["ks_ptr"], p["vs_ptr"])
                self.assertEqual(result.shape, (batch, length, 128 * 64))
                self.assertFalse(fallback)

    def test_scratch_is_private_per_invocation_and_shared_only_with_own_merge(self):
        env, allocations, _ = load_host_adapter()
        data = fake_data()
        first, second = env["kernel_fn"](data), env["kernel_fn"](data)
        self.assertEqual(len(allocations), 8)
        self.assertEqual(len({id(t.owner) for t in allocations}), 8)
        self.assertIsNot(first.owner, second.owner)
        for producer, merge_call in zip(env["_dual_full"].calls, env["_dual_merge"].calls):
            for name in ("o_ptr", "m_ptr", "l_ptr"):
                self.assertIs(producer["bound"][name], merge_call["bound"][name])
        self.assertIs(first.owner, allocations[3].owner)
        self.assertIs(second.owner, allocations[7].owner)

    def test_small_table_caps_splits_without_zero_and_nearby_batch_is_supported(self):
        for width, expected_s in ((65, 1), (128, 1), (256, 2), (86008, 44)):
            env, _, fallback = load_host_adapter()
            env["kernel_fn"](fake_data(batch=3, width=width))
            self.assertEqual(env["_dual_full"].calls[0]["grid"], (3, 1, expected_s))
            self.assertFalse(fallback)

    def test_only_explicit_unsupported_configs_use_champion_fallback(self):
        variants = [dict(window=2048, rope_theta=500000), dict(rope_theta=1),
                    dict(q_len=4), dict(head_dim=32), dict(num_kv_heads=3),
                    dict(num_q_heads=8)]
        for changes in variants:
            env, allocations, fallback = load_host_adapter()
            data = fake_data()
            data["config"].update(changes)
            self.assertEqual(env["kernel_fn"](data), "champion fallback")
            self.assertEqual(fallback, [data])
            self.assertFalse(allocations)
            self.assertFalse(env["_dual_full"].calls)
            self.assertFalse(env["_dual_merge"].calls)


if __name__ == "__main__":
    unittest.main()
