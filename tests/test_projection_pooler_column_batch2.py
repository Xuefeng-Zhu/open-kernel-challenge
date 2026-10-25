"""CPU/structural checks only: these do not execute CUDA bmm or CUDA graphs."""
import ast
import hashlib
import unittest
from unittest.mock import patch

import torch

from harness import ROOT, load_module


PARENT = ROOT / "candidates/projection_router_m1_fp32_gemv.py"
SOURCE = ROOT / "candidates/projection_pooler_column_batch2.py"
PARENT_SHA = "0c0c41119ef925b03a33ccab230bc64176cc22c242e3fea77ce01e317376defc"


def dump(node):
    return ast.dump(node, include_attributes=False)


class PoolerColumnBatch2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.candidate = load_module(SOURCE, "p045_candidate")
        cls.parent = load_module(PARENT, "p045_parent")
        cls.oracle = load_module(ROOT / "official/reference.py", "p045_oracle")
        gen = torch.Generator().manual_seed(45001)
        cls.weights = {
            "router": torch.randn((7, 131), generator=gen),
            "pooler": (torch.randn((4096, 8192), generator=gen) / 8192**0.5).bfloat16(),
        }
        cls.state = cls.candidate.prepare(cls.weights)

    def assert_oracle(self, data, output, weights=None):
        ok, message = self.oracle.check_implementation(data, output, weights or self.weights)
        self.assertTrue(ok, message)
        self.assertEqual(output.shape, (data["x"].shape[0], data["config"]["n"]))
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(output.is_contiguous())

    def test_frozen_parent_and_unchanged_kernels(self):
        self.assertEqual(hashlib.sha256(PARENT.read_bytes()).hexdigest(), PARENT_SHA)
        def functions(path):
            return {n.name: n for n in ast.walk(ast.parse(path.read_text()))
                    if isinstance(n, ast.FunctionDef)}
        old, new = functions(PARENT), functions(SOURCE)
        self.assertEqual(new.keys() - old.keys(),
                         {"_pooler_column_batch2_compatible", "_pooler_column_batch2"})
        for name in old.keys() - {"prepare", "kernel_fn"}:
            self.assertEqual(dump(old[name]), dump(new[name]), name)
        pooler_if = new["prepare"].body[2]
        self.assertEqual(ast.unparse(pooler_if.test), "'pooler' in weights")
        added = pooler_if.body.pop()
        self.assertIsInstance(added, ast.If)
        self.assertIn("pooler_column_batch2", ast.unparse(added))
        self.assertEqual(dump(old["prepare"]), dump(new["prepare"]))
        added = new["kernel_fn"].body.pop(3)
        self.assertIn("_pooler_column_batch2_compatible", ast.unparse(added))
        self.assertEqual(dump(old["kernel_fn"]), dump(new["kernel_fn"]))

    def test_embedded_license_and_no_timed_state_write(self):
        old, new = PARENT.read_text(), SOURCE.read_text()
        marker = "# Applicable project/parent Apache License 2.0 text for standalone upload:"
        self.assertEqual(old.split(marker)[1].split("import torch")[0],
                         new.split(marker)[1].split("import torch")[0])
        tree = ast.parse(new)
        functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        for name in ("kernel_fn", "_pooler_column_batch2", "_pooler_column_batch2_compatible"):
            for node in ast.walk(functions[name]):
                if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
                    self.assertNotEqual(ast.unparse(node.value), "state")
        helper = ast.unparse(functions["_pooler_column_batch2"])
        self.assertEqual(helper.count("torch.bmm("), 1)
        for forbidden in (".float(", ".clone(", ".contiguous(", "torch.cat(", "torch.cuda", ".to("):
            self.assertNotIn(forbidden, helper)

    def test_all_bf16_payloads_pack_bit_exactly(self):
        # Includes signed zero, subnormals, infinities, and every NaN payload.
        bits = torch.arange(65536, dtype=torch.int32).to(torch.int16)
        weight = bits.repeat(512).reshape(4096, 8192).view(torch.bfloat16)
        before = weight.view(torch.int16).clone()
        state = self.candidate.prepare({"pooler": weight, "router": self.weights["router"]})
        packed = state["pooler_column_batch2"]
        self.assertEqual(packed.shape, (2, 8192, 2048))
        self.assertEqual(packed.stride(), (16777216, 2048, 1))
        restored = packed.transpose(1, 2).reshape(4096, 8192).view(torch.int16)
        self.assertTrue(torch.equal(restored, before))
        self.assertTrue(torch.equal(weight.view(torch.int16), before))
        self.assertEqual(packed.numel() * packed.element_size(), 64 * 1024**2)
        self.assertNotEqual(packed.data_ptr(), weight.data_ptr())

    def test_output_view_addresses_are_unique_and_cover_output(self):
        for m in (2, 7, 8, 16, 32, 128):
            out = torch.empty((m, 4096), dtype=torch.bfloat16)
            view = out.as_strided((2, m, 2048), (2048, 4096, 1))
            self.assertEqual(torch._debug_has_internal_overlap(view), 0)
            offsets = (torch.arange(2)[:, None, None] * 2048
                       + torch.arange(m)[None, :, None] * 4096
                       + torch.arange(2048)[None, None, :])
            self.assertTrue(torch.equal(offsets.flatten().sort().values, torch.arange(m * 4096)))
            view[0].fill_(3)
            view[1].fill_(-5)
            self.assertTrue(torch.all(out[:, :2048] == 3))
            self.assertTrue(torch.all(out[:, 2048:] == -5))
            x = torch.empty((m, 8192), dtype=torch.bfloat16)
            expanded = x.unsqueeze(0).expand(2, -1, -1)
            self.assertEqual(expanded.stride(), (0, 8192, 1))
            self.assertEqual(expanded.data_ptr(), x.data_ptr())

    def test_exact_guard_and_nearby_dimensions(self):
        guard = self.candidate._pooler_column_batch2_compatible
        for m in (2, 7, 8, 16, 32, 127, 128):
            self.assertTrue(guard(torch.empty((m, 8192), dtype=torch.bfloat16), self.state, 4096))
        for m in (0, 1, 129, 256):
            self.assertFalse(guard(torch.empty((m, 8192), dtype=torch.bfloat16), self.state, 4096))
        x = torch.empty((2, 8192), dtype=torch.bfloat16)
        for n in (0, 1, 2048, 4095, 4097):
            self.assertFalse(guard(x, self.state, n))
        for k in (8191, 8193):
            self.assertFalse(guard(torch.empty((2, k), dtype=torch.bfloat16), self.state, 4096))
        self.assertFalse(guard(torch.empty((8192,), dtype=torch.bfloat16), self.state, 4096))

    def test_strides_dtypes_devices_and_bad_state_reject_new_path(self):
        guard = self.candidate._pooler_column_batch2_compatible
        x = torch.empty((2, 8192), dtype=torch.bfloat16)
        for bad_x in (x.float(), torch.empty((2, 16384), dtype=x.dtype)[:, ::2],
                      torch.empty((2, 8193), dtype=x.dtype)[:, :8192],
                      torch.empty((8192, 2), dtype=x.dtype).T,
                      torch.empty((2, 8192), dtype=x.dtype, device="meta")):
            self.assertFalse(guard(bad_x, self.state, 4096))
        for replacement in (None, torch.empty((2, 8192, 2047), dtype=x.dtype),
                            torch.empty((2, 2048, 8192), dtype=x.dtype).transpose(1, 2),
                            torch.empty((2, 8192, 2048), dtype=torch.int8),
                            torch.empty((2, 8192, 2048), dtype=x.dtype, device="meta")):
            state = dict(self.state, pooler_column_batch2=replacement)
            self.assertFalse(guard(x, state, 4096))
        self.assertFalse(guard(x, dict(self.state, pooler=None), 4096))
        self.assertFalse(guard(x, dict(self.state, pooler=self.weights["pooler"][:4095]), 4096))

    def test_prepare_ignores_noncanonical_weights(self):
        for shape, dtype in (((33, 131), torch.bfloat16), ((4095, 8192), torch.bfloat16),
                             ((4096, 8191), torch.bfloat16), ((4096, 8192), torch.float32)):
            weights = {"router": self.weights["router"], "pooler": torch.zeros(shape, dtype=dtype)}
            state = self.candidate.prepare(weights)
            self.assertNotIn("pooler_column_batch2", state)

    def test_full_width_small_m_normal_and_outliers_against_official_fp64_oracle(self):
        # Full K=8192 candidate execution is deliberately bounded to small M.
        # Larger M uses real output addressing and reduced-K helper checks below.
        for m in (2, 7):
            for dist in ("normal", "outliers"):
                with self.subTest(m=m, dist=dist):
                    data = self.oracle.generate_input(m, "pooler", 45002 + m, dist=dist, device="cpu")
                    snapshot = data["x"].clone()
                    output = self.candidate.kernel_fn(data, self.state)
                    self.assert_oracle(data, output)
                    self.assertTrue(torch.equal(data["x"], snapshot))

    def test_full_width_sparse_and_zero(self):
        x = torch.zeros((2, 8192), dtype=torch.bfloat16)
        for row, col in enumerate((2047, 8191)):
            x[row, col] = -128 if row % 2 else 8
        data = {"x": x, "config": {"m": 2, "n": 4096, "k": 8192, "layer": "pooler"}}
        output = self.candidate.kernel_fn(data, self.state)
        self.assert_oracle(data, output)
        self.assertTrue(torch.equal(output, self.oracle.ref_kernel(data, self.weights)))
        x.zero_()
        output = self.candidate.kernel_fn(data, self.state)
        self.assert_oracle(data, output)
        self.assertEqual(torch.count_nonzero(output), 0)

    def test_same_address_aba_and_state_immutability_cpu_only(self):
        a = self.oracle.generate_input(2, "pooler", 45011, device="cpu")
        b = self.oracle.generate_input(2, "pooler", 45012, dist="outliers", device="cpu")
        slot = torch.empty_like(a["x"])
        address = slot.data_ptr()
        state_before = {key: value.clone() for key, value in self.state.items()}
        weights_before = {key: value.clone() for key, value in self.weights.items()}
        outputs = []
        for data in (a, b, a):
            slot.copy_(data["x"])
            current = {"x": slot, "config": data["config"]}
            output = self.candidate.kernel_fn(current, self.state)
            self.assert_oracle(current, output)
            self.assertEqual(slot.data_ptr(), address)
            self.assertTrue(torch.equal(slot, data["x"]))
            outputs.append(output.clone())
        self.assertTrue(torch.equal(outputs[0], outputs[2]))
        self.assertFalse(torch.equal(outputs[0], outputs[1]))
        for key, before in state_before.items():
            self.assertTrue(torch.equal(self.state[key], before), key)
        for key, before in weights_before.items():
            self.assertTrue(torch.equal(self.weights[key], before), key)

    def test_larger_m_reduced_k_helper_math_with_real_output_layout(self):
        # K=131 is a CPU algebra/layout model, NOT a routed candidate shape.
        # Repeated fresh weights check that the helper uses the current packing.
        for seed in (45041, 45042):
            gen = torch.Generator().manual_seed(seed)
            k = 131
            weight = (torch.randn((4096, k), generator=gen) / k**0.5).bfloat16()
            packed = weight.reshape(2, 2048, k).transpose(1, 2).contiguous()
            before = packed.clone()
            for m in (8, 16, 32, 127, 128):
                x = torch.randn((m, k), generator=gen).bfloat16()
                x[0].zero_()
                x[0, k - 1] = -128
                snapshot = x.clone()
                data = {"x": x, "config": {"m": m, "n": 4096, "k": k, "layer": "pooler"}}
                output = self.candidate._pooler_column_batch2(x, packed)
                self.assert_oracle(data, output, {"pooler": weight})
                self.assertFalse(self.candidate._pooler_column_batch2_compatible(x, self.state, 4096))
                self.assertTrue(torch.equal(x, snapshot))
            self.assertTrue(torch.equal(packed, before))

    def test_full_width_fallback_dispatch_matches_parent(self):
        # Full fallback expressions are AST-identical and numerically checked at
        # nearby small dimensions below. Avoid repeating large unchanged GEMMs.
        gen = torch.Generator().manual_seed(45021)
        with patch.object(self.candidate, "_pooler_column_batch2", side_effect=AssertionError("Wrong dispatch")):
            for m, n in ((1, 4096), (129, 4096), (256, 4096), (2, 4095), (7, 2048)):
                x = torch.randn((m, 8192), generator=gen).bfloat16()
                data = {"x": x, "config": {"m": m, "n": n, "k": 8192, "layer": "pooler"}}
                sentinel = torch.zeros((m, n), dtype=torch.bfloat16)
                with patch.object(torch.Tensor, "__matmul__", return_value=sentinel) as matmul:
                    self.assertIs(self.candidate.kernel_fn(data, self.state), sentinel)
                    self.assertIs(self.parent.kernel_fn(data, self.state), sentinel)
                    self.assertEqual(matmul.call_count, 2)
            x = torch.randn((2, 16384), generator=gen).bfloat16()[:, ::2]
            data = {"x": x, "config": {"m": 2, "n": 4096, "k": 8192, "layer": "pooler"}}
            sentinel = torch.zeros((2, 4096), dtype=torch.bfloat16)
            with patch.object(torch.Tensor, "__matmul__", return_value=sentinel) as matmul:
                self.assertIs(self.candidate.kernel_fn(data, self.state), sentinel)
                self.assertIs(self.parent.kernel_fn(data, self.state), sentinel)
                self.assertEqual(matmul.call_count, 2)

    def test_other_layers_and_nearby_pooler_match_parent(self):
        gen = torch.Generator().manual_seed(45031)
        weights = {"router": torch.randn((31, 131), generator=gen), **{
            layer: torch.randn((37, 131), generator=gen).bfloat16()
            for layer in ("pooler", "qkv_gate", "unpooler", "o")}}
        state = self.candidate.prepare(weights)
        parent = self.parent.prepare(weights)
        with patch.object(self.candidate, "_pooler_column_batch2", side_effect=AssertionError("Wrong dispatch")):
            for layer in weights:
                for m in (1, 2, 7, 8, 9, 16, 32, 128, 129, 256):
                    for n in (1, 29):
                        x = torch.randn((m, 131), generator=gen).bfloat16()
                        data = {"x": x, "config": {"m": m, "n": n, "k": 131, "layer": layer}}
                        torch.testing.assert_close(self.candidate.kernel_fn(data, state),
                                                   self.parent.kernel_fn(data, parent), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
