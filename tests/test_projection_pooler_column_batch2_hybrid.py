"""P046 local CPU/dispatch tests; no CUDA execution or graph validation."""
import ast
import hashlib
import unittest
from unittest.mock import patch

import torch

from harness import ROOT, load_module

BASE = ROOT / "candidates/projection_router_m1_fp32_gemv.py"
PARENT = ROOT / "candidates/projection_pooler_column_batch2.py"
SOURCE = ROOT / "candidates/projection_pooler_column_batch2_hybrid.py"
BASE_SHA = "0c0c41119ef925b03a33ccab230bc64176cc22c242e3fea77ce01e317376defc"
PARENT_SHA = "c751d3ec5bff008e9f893f3cc6113184e3a78ae255578e6b398c9d22122d3f58"


class PoolerColumnBatch2Hybrid(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.candidate = load_module(SOURCE, "p046_candidate")
        cls.parent = load_module(PARENT, "p046_p045")
        cls.base = load_module(BASE, "p046_p039")
        cls.oracle = load_module(ROOT / "official/reference.py", "p046_oracle")
        gen = torch.Generator().manual_seed(46001)
        cls.weights = {"router": torch.randn((7, 131), generator=gen),
                       "pooler": (torch.randn((4096, 8192), generator=gen) / 8192**.5).bfloat16()}
        cls.state = cls.candidate.prepare(cls.weights)

    def check_output(self, data, output, weights=None):
        ok, message = self.oracle.check_implementation(data, output, weights or self.weights)
        self.assertTrue(ok, message)
        self.assertEqual(output.shape, (data["x"].shape[0], data["config"]["n"]))
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(output.is_contiguous())

    def test_frozen_sources_and_exact_single_predicate_change(self):
        self.assertEqual(hashlib.sha256(BASE.read_bytes()).hexdigest(), BASE_SHA)
        self.assertEqual(hashlib.sha256(PARENT.read_bytes()).hexdigest(), PARENT_SHA)
        def normalize(path):
            tree = ast.parse(path.read_text())
            tree.body.pop(0)  # The documentation string is the only non-executable difference.
            return ast.dump(tree, include_attributes=False)
        changed = SOURCE.read_text().replace("x.shape[0] in (8, 16, 32)", "2 <= x.shape[0] <= 128")
        tree = ast.parse(changed)
        tree.body.pop(0)
        self.assertEqual(ast.dump(tree, include_attributes=False), normalize(PARENT))
        self.assertEqual(SOURCE.read_text().count("x.shape[0] in (8, 16, 32)"), 1)
        marker = "# Applicable project/parent Apache License 2.0 text for standalone upload:"
        self.assertEqual(PARENT.read_text().split(marker)[1].split("import torch")[0],
                         SOURCE.read_text().split(marker)[1].split("import torch")[0])

    def test_exact_three_scored_routes_and_nearby_fallbacks(self):
        for m in (1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 127, 128, 129, 256):
            x = torch.empty((m, 8192), dtype=torch.bfloat16)
            self.assertEqual(self.candidate._pooler_column_batch2_compatible(x, self.state, 4096), m in (8, 16, 32))
        x = torch.empty((8, 8192), dtype=torch.bfloat16)
        for n in (0, 1, 2048, 4095, 4097):
            self.assertFalse(self.candidate._pooler_column_batch2_compatible(x, self.state, n))
        for bad_x in (x.float(), torch.empty((8, 16384), dtype=x.dtype)[:, ::2],
                      torch.empty((8, 8193), dtype=x.dtype), torch.empty((8, 8191), dtype=x.dtype)):
            self.assertFalse(self.candidate._pooler_column_batch2_compatible(bad_x, self.state, 4096))
        self.assertFalse(self.candidate._pooler_column_batch2_compatible(x, dict(self.state, pooler_column_batch2=None), 4096))

    def test_full_width_dispatch_spies_and_fallback_to_p039(self):
        for m in (7, 8, 9, 16, 31, 32, 33, 128):
            x = torch.empty((m, 8192), dtype=torch.bfloat16)
            data = {"x": x, "config": {"m": m, "n": 4096, "k": 8192, "layer": "pooler"}}
            sentinel = torch.empty((m, 4096), dtype=x.dtype)
            if m in (8, 16, 32):
                with patch.object(self.candidate, "_pooler_column_batch2", return_value=sentinel) as batch:
                    self.assertIs(self.candidate.kernel_fn(data, self.state), sentinel)
                    self.assertEqual(batch.call_count, 1)
            else:
                with patch.object(self.candidate, "_pooler_column_batch2", side_effect=AssertionError("Unexpected batch path")), \
                     patch.object(torch.Tensor, "__matmul__", return_value=sentinel) as mm:
                    self.assertIs(self.candidate.kernel_fn(data, self.state), sentinel)
                    self.assertIs(self.base.kernel_fn(data, self.state), sentinel)
                    self.assertEqual(mm.call_count, 2)

    def test_full_k_m8_normal_outlier_sparse_zero(self):
        for dist in ("normal", "outliers", "sparse", "zero"):
            data = self.oracle.generate_input(8, "pooler", 46002, dist="outliers" if dist == "outliers" else "normal", device="cpu")
            if dist in ("sparse", "zero"):
                data["x"].zero_()
            if dist == "sparse":
                for row, col in enumerate((0, 63, 2047, 2048, 4095, 4096, 8190, 8191)):
                    data["x"][row, col] = -128 if row % 2 else 8
            before = data["x"].clone()
            output = self.candidate.kernel_fn(data, self.state)
            self.check_output(data, output)
            self.assertTrue(torch.equal(data["x"], before))
            if dist == "sparse":
                self.assertTrue(torch.equal(output, self.oracle.ref_kernel(data, self.weights)))
            if dist == "zero":
                self.assertEqual(torch.count_nonzero(output), 0)

    def test_full_k_m8_same_address_aba_and_immutable_state(self):
        a = self.oracle.generate_input(8, "pooler", 46011, device="cpu")
        b = self.oracle.generate_input(8, "pooler", 46012, dist="outliers", device="cpu")
        slot = torch.empty_like(a["x"])
        address = slot.data_ptr()
        before = {key: value.clone() for key, value in self.state.items()}
        outputs = []
        for data in (a, b, a):
            slot.copy_(data["x"])
            current = {"x": slot, "config": data["config"]}
            output = self.candidate.kernel_fn(current, self.state)
            self.check_output(current, output)
            self.assertEqual(slot.data_ptr(), address)
            self.assertTrue(torch.equal(slot, data["x"]))
            outputs.append(output.clone())
        self.assertTrue(torch.equal(outputs[0], outputs[2]))
        self.assertFalse(torch.equal(outputs[0], outputs[1]))
        for key, value in before.items():
            self.assertTrue(torch.equal(self.state[key], value), key)

    def test_other_target_m_reduced_k_helper_and_exact_output_addresses(self):
        # K131 helper models are not production dispatch or full-K execution.
        for seed in (46021, 46022):
            gen = torch.Generator().manual_seed(seed)
            w = (torch.randn((4096, 131), generator=gen) / 131**.5).bfloat16()
            packed = w.reshape(2, 2048, 131).transpose(1, 2).contiguous()
            for m in (16, 32):
                x = torch.randn((m, 131), generator=gen).bfloat16()
                data = {"x": x, "config": {"m": m, "n": 4096, "k": 131, "layer": "pooler"}}
                self.check_output(data, self.candidate._pooler_column_batch2(x, packed), {"pooler": w})
                offsets = (torch.arange(2)[:, None, None] * 2048 + torch.arange(m)[None, :, None] * 4096
                           + torch.arange(2048)[None, None, :])
                self.assertTrue(torch.equal(offsets.flatten().sort().values, torch.arange(m * 4096)))

    def test_nearby_and_other_layers_numerically_match_p039(self):
        gen = torch.Generator().manual_seed(46031)
        weights = {"router": torch.randn((31, 131), generator=gen), **{
            layer: torch.randn((37, 131), generator=gen).bfloat16()
            for layer in ("pooler", "qkv_gate", "unpooler", "o")}}
        state, base = self.candidate.prepare(weights), self.base.prepare(weights)
        for layer in weights:
            for m in (1, 7, 8, 9, 16, 31, 32, 33, 128, 256):
                x = torch.randn((m, 131), generator=gen).bfloat16()
                data = {"x": x, "config": {"m": m, "n": 29, "k": 131, "layer": layer}}
                torch.testing.assert_close(self.candidate.kernel_fn(data, state), self.base.kernel_fn(data, base), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
