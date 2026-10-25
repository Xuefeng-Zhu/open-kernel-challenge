"""CPU analytical checks for the proposed FP8 QK encoding; no candidate import.

Torch runs only on CPU, with no NumPy use. Decoded FP8 operands are multiplied
in CPU FP32 and checked against CPU FP64 arithmetic. These tests do not validate
Triton compilation, native FP8 MMA, max_num_imprecise_acc, GPU rounding, accuracy
against the judge, or performance. The norm/RoPE model illustrates the source's
operation boundaries; CPU transcendental/reduction behavior is not GPU behavior.
"""
import math
import unittest

import torch


CPU = torch.device("cpu")
E4M3 = torch.float8_e4m3fn
FACTOR = 5.4730064863875
SM_SCALE = 1.0 / 8.0
LOG2E = math.log2(math.e)
EPS = 1e-5
D = 64


def tensor(values, dtype=torch.float32):
    return torch.tensor(values, dtype=dtype, device=CPU)


def encode_query(normalized_rotated_query):
    """Chosen boundary: unscaled normalized/RoPE Q -> BF16 -> E4M3."""
    return normalized_rotated_query.to(torch.bfloat16).to(E4M3)


def norm_rope_cpu(raw_bf16, positions, theta=500000.0):
    """Stateless FP32 component model, independent of all candidate code."""
    assert raw_bf16.device.type == "cpu" and raw_bf16.dtype == torch.bfloat16
    x = raw_bf16.float()
    inv = torch.rsqrt(torch.sum(x * x, dim=-1, keepdim=True) / D + EPS)
    indices = torch.arange(D, device=CPU)
    frequency = torch.exp2(-(2.0 * (indices // 2).float() / D) * torch.log2(tensor(theta)))
    angle = positions.to(device=CPU, dtype=torch.float32)[:, None] * frequency[None, :]
    sign = torch.where(indices % 2 == 0, -1.0, 1.0)
    partner = x[:, indices ^ 1] * inv
    return x * inv * torch.cos(angle) + sign * partner * torch.sin(angle)


def qk_logits_cpu(q8, k8, ks, factor=FACTOR):
    """Log2-domain CPU FP32 dot; factor and cache scale each appear once."""
    assert q8.device.type == k8.device.type == "cpu"
    assert q8.dtype == k8.dtype == E4M3
    scale = tensor(SM_SCALE) * tensor(factor) * tensor(ks) * tensor(LOG2E)
    return (q8.float() @ k8.float().T) * scale


def make_components():
    generator = torch.Generator(device=CPU).manual_seed(20261025)
    raw = torch.randn((3, D), generator=generator, device=CPU).to(torch.bfloat16)
    query = norm_rope_cpu(raw, tensor([0, 2048, 80000], torch.int64))
    keys = torch.randn((17, D), generator=generator, device=CPU)
    keys = keys * torch.rsqrt(torch.mean(keys * keys, dim=-1, keepdim=True) + EPS)
    return raw, query, keys


class FP8Encoding(unittest.TestCase):
    def test_round_to_even_ties_and_subnormals(self):
        values = tensor([1.0625, 1.1875, -1.0625, -1.1875,
                         2.0**-10, 3.0 * 2.0**-10])
        expected = tensor([1.0, 1.25, -1.0, -1.25, 0.0, 2.0**-8])
        self.assertTrue(torch.equal(values.to(E4M3).float(), expected))
        midpoint = tensor(1.0625)
        below = torch.nextafter(midpoint, tensor(-math.inf))
        above = torch.nextafter(midpoint, tensor(math.inf))
        self.assertEqual(below.to(E4M3).float().item(), 1.0)
        self.assertEqual(above.to(E4M3).float().item(), 1.125)

    def test_all_finite_e4m3_codes_roundtrip_including_signed_zero(self):
        codes = torch.arange(256, device=CPU, dtype=torch.int32).to(torch.uint8)
        decoded = codes.view(E4M3).float()
        finite = torch.isfinite(decoded)
        self.assertEqual(int(finite.sum()), 254)
        # BF16 exactly represents finite E4M3 values; this covers K/V upcasts too.
        for values in (decoded[finite], decoded[finite].to(torch.bfloat16).float()):
            self.assertTrue(torch.equal(values.to(E4M3).view(torch.uint8), codes[finite]))
        self.assertFalse(bool(torch.signbit(decoded[0])))
        self.assertTrue(bool(torch.signbit(decoded[128])))

    def test_bf16_double_rounding_is_explicitly_not_direct_fp8_rounding(self):
        witness = tensor(1.0625 + 2.0**-12)
        self.assertEqual(witness.to(torch.bfloat16).float().item(), 1.0625)
        self.assertEqual(witness.to(E4M3).float().item(), 1.125)
        self.assertEqual(encode_query(witness).float().item(), 1.0)

    def test_quantizing_scaled_query_changes_rounding(self):
        query = tensor([[1.125]])  # Exactly representable in BF16 and E4M3.
        key = tensor([[2.0]]).to(E4M3)
        correct = qk_logits_cpu(encode_query(query), key, ks=1.0 / 64.0)
        scaled_then_quantized = (query * tensor(FACTOR)).to(torch.bfloat16).to(E4M3)
        wrong = qk_logits_cpu(scaled_then_quantized, key, ks=1.0 / 64.0, factor=1.0)
        self.assertEqual(scaled_then_quantized.float().item(), 6.0)
        self.assertGreater(abs((correct - wrong).item()), 1e-4)


class QKScalePlacement(unittest.TestCase):
    def test_factor_and_cache_scale_once_match_dequantized_reference(self):
        _, query, keys = make_components()
        q8 = encode_query(query)
        for ks in (1.0 / 64.0, 0.25, 1.0, 2.0):
            with self.subTest(k_scale=ks):
                k8 = (keys / ks).to(E4M3)
                self.assertTrue(bool(torch.isfinite(k8.float()).all()))
                got = qk_logits_cpu(q8, k8, ks)
                # Independent placement: dequantize operands before the FP64 dot.
                q_ref = q8.double() * FACTOR
                k_ref = k8.double() * ks
                reference = (q_ref @ k_ref.T) * SM_SCALE * LOG2E
                moment = (q_ref.abs() @ k_ref.abs().T) * SM_SCALE * LOG2E
                # Conservative absolute FP32 forward-error envelope, including
                # the 64-term dot and scalar products; robust to cancellation.
                bound = (D + 8) * torch.finfo(torch.float32).eps * moment + 1e-6
                self.assertEqual(got.dtype, torch.float32)
                self.assertEqual(got.device.type, "cpu")
                self.assertTrue(bool((torch.abs(got.double() - reference) <= bound).all()))

    def test_nonunit_scale_witness_rejects_double_factor_and_wrong_k_scale(self):
        q8, k8 = tensor([[1.125]]).to(E4M3), tensor([[2.0]]).to(E4M3)
        ks = 1.0 / 64.0
        good = qk_logits_cpu(q8, k8, ks)
        expected = 1.125 * FACTOR * (2.0 * ks) * SM_SCALE * LOG2E
        self.assertAlmostEqual(good.item(), expected, delta=1e-8)
        wrong = {
            "factor_twice": qk_logits_cpu(q8, k8, ks, factor=FACTOR * FACTOR),
            "factor_missing": qk_logits_cpu(q8, k8, ks, factor=1.0),
            "k_scale_missing": qk_logits_cpu(q8, k8, 1.0),
            "k_scale_twice": qk_logits_cpu(q8, k8, ks * ks),
        }
        for name, value in wrong.items():
            with self.subTest(error=name):
                self.assertGreater(abs((value - good).item()), abs(good.item()) * 0.5)


class NormRopeComponents(unittest.TestCase):
    def test_zero_position_and_interleaved_rotation_preserve_pair_energy(self):
        raw, _, _ = make_components()
        zero = norm_rope_cpu(raw, torch.zeros(3, device=CPU, dtype=torch.int64))
        x = raw.float()
        expected = x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + EPS)
        torch.testing.assert_close(zero, expected, rtol=1e-6, atol=1e-7)
        rotated = norm_rope_cpu(raw, tensor([1, 2048, 80000], torch.int64))
        zero_energy = zero.square().reshape(3, D // 2, 2).sum(dim=-1)
        rotated_energy = rotated.square().reshape(3, D // 2, 2).sum(dim=-1)
        torch.testing.assert_close(rotated_energy, zero_energy, rtol=2e-6, atol=1e-6)

    def test_fresh_inputs_and_positions_change_encoded_cpu_queries(self):
        raw, _, _ = make_components()
        positions = tensor([10, 2048, 80000], torch.int64)
        original = encode_query(norm_rope_cpu(raw, positions)).float()
        changed = raw.clone()
        changed[0, 0] += 2.0
        changed_input = encode_query(norm_rope_cpu(changed, positions)).float()
        changed_position = encode_query(norm_rope_cpu(raw, positions + 1)).float()
        self.assertFalse(torch.equal(original[0], changed_input[0]))
        self.assertTrue(torch.equal(original[1:], changed_input[1:]))
        self.assertFalse(torch.equal(original, changed_position))
        self.assertTrue(torch.equal(original, encode_query(norm_rope_cpu(raw, positions)).float()))
        self.assertTrue(bool(torch.isfinite(original).all()))
        # This tests CPU component dependencies, not native graph/cache ownership.


if __name__ == "__main__":
    unittest.main(verbosity=2)
