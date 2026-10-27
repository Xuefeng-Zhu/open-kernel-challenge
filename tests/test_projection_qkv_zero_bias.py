"""P050 CPU numerical/AST tests; CUDA execution and performance remain unverified."""
import ast
import hashlib
import unittest
from unittest.mock import patch

import torch

from harness import ROOT, correctness_cases, load_module

PARENT = ROOT / "candidates/projection_pooler_column_batch2_hybrid.py"
PREVIOUS = ROOT / "candidates/projection_qkv_unpooler_zero_bias.py"
SOURCE = ROOT / "candidates/projection_qkv_zero_bias.py"
PARENT_SHA = "001ea12e8764564d507de51e433769f5bd9a9916bfc4c014f0f87522c278545e"
PREVIOUS_SHA = "4c3b0ed4181cbdc70f8557fd7445b6937c1ae0b236724c87178faf9d02116ad7"


def functions(path):
    return {node.name: node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}


class QKVZeroBias(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(4)
        cls.candidate = load_module(SOURCE, "p050_candidate")
        cls.parent = load_module(PARENT, "p050_parent")
        cls.oracle = load_module(ROOT / "official/reference.py", "p050_oracle")
        gen = torch.Generator().manual_seed(50001)
        cls.weights = {
            "router": torch.randn((7, 131), generator=gen),
            "qkv_gate": (torch.randn((2304, 8192), generator=gen) / 8192**.5).bfloat16(),
            "unpooler": (torch.randn((8192, 4096), generator=gen) / 4096**.5).bfloat16(),
        }
        cls.state = cls.candidate.prepare(cls.weights)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def check_output(self, data, out, weights=None):
        ok, message = self.oracle.check_implementation(data, out, weights or self.weights)
        self.assertTrue(ok, message)
        self.assertEqual(out.shape, (data["x"].shape[0], data["config"]["n"]))
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertTrue(out.is_contiguous())
        self.assertEqual(out.stride(), (out.shape[1], 1))

    def test_frozen_parents_exact_ast_and_license_preservation(self):
        self.assertEqual(hashlib.sha256(PARENT.read_bytes()).hexdigest(), PARENT_SHA)
        self.assertEqual(hashlib.sha256(PREVIOUS.read_bytes()).hexdigest(), PREVIOUS_SHA)
        parent, candidate = ast.parse(PARENT.read_text()), ast.parse(SOURCE.read_text())
        parent.body.pop(0)
        candidate.body.pop(0)
        candidate.body = [node for node in candidate.body if not (
            isinstance(node, ast.FunctionDef) and node.name in {"_zero_bias_addmm", "_zero_bias_addmm_compatible"})]
        for node in candidate.body:
            if isinstance(node, ast.FunctionDef) and node.name == "prepare":
                added = node.body[3].body[0].body.pop()
                self.assertIsInstance(added, ast.If)
                self.assertEqual(ast.unparse(added.test), "layer == 'qkv_gate'")
                self.assertEqual(len(added.body), 1)
                self.assertIn("_zero_bias", ast.unparse(added.body[0]))
            if isinstance(node, ast.FunctionDef) and node.name == "kernel_fn":
                added = node.body.pop(3)
                self.assertIsInstance(added, ast.If)
                self.assertIn("_zero_bias_addmm_compatible", ast.unparse(added.test))
        self.assertEqual(ast.dump(candidate, include_attributes=False), ast.dump(parent, include_attributes=False))
        old, new = functions(PREVIOUS), functions(SOURCE)
        for name in old.keys() - {"prepare", "_zero_bias_addmm_compatible"}:
            self.assertEqual(ast.dump(new[name], include_attributes=False), ast.dump(old[name], include_attributes=False), name)
        # Restoring the one nested guard recovers P049's exact preparation function.
        prepare = new["prepare"]
        body = prepare.body[3].body[0].body
        body[-1:] = body[-1].body
        self.assertEqual(ast.dump(prepare, include_attributes=False), ast.dump(old["prepare"], include_attributes=False))
        marker = "# Applicable project/parent Apache License 2.0 text for standalone upload:"
        self.assertEqual(SOURCE.read_text().split(marker)[1].split("import torch")[0],
                         PARENT.read_text().split(marker)[1].split("import torch")[0])

    def test_prepare_adds_only_immutable_qkv_zeros_and_preserves_weight_bits(self):
        old = self.parent.prepare(self.weights)
        self.assertEqual(set(self.state) - set(old), {"qkv_gate_zero_bias"})
        self.assertNotIn("unpooler_zero_bias", self.state)
        bias = self.state["qkv_gate_zero_bias"]
        self.assertEqual(bias.shape, (2304,))
        self.assertEqual(bias.dtype, torch.bfloat16)
        self.assertEqual(bias.device, self.weights["qkv_gate"].device)
        self.assertTrue(bias.is_contiguous())
        self.assertEqual(bias.numel() * bias.element_size(), 4608)
        self.assertEqual(torch.count_nonzero(bias), 0)
        for key in old:
            self.assertTrue(torch.equal(old[key].view(torch.uint8), self.state[key].view(torch.uint8)), key)

    def test_prepare_without_qkv_creates_no_bias_state(self):
        weights = {"router": torch.zeros((7, 31)), "unpooler": torch.zeros((37, 31), dtype=torch.bfloat16)}
        actual, expected = self.candidate.prepare(weights), self.parent.prepare(weights)
        self.assertEqual(actual.keys(), expected.keys())
        for name in actual:
            self.assertTrue(torch.equal(actual[name], expected[name]))

    def test_all_scored_and_prerequisite_metadata_exactly_qkv_mgreater1(self):
        targets = []
        for m in (1, 16, 32, 8, 128, 256):
            for layer, (_, k) in self.oracle.LAYERS.items():
                x = torch.empty((m, k), dtype=torch.bfloat16)
                actual = self.candidate._zero_bias_addmm_compatible(x, self.state, layer, self.oracle.out_features(layer, m))
                self.assertEqual(actual, layer == "qkv_gate" and m > 1, (layer, m))
                if actual:
                    targets.append((layer, m))
        self.assertEqual(targets, [("qkv_gate", m) for m in (16, 32, 8, 128, 256)])
        prerequisite_targets = []
        for case in correctness_cases():
            layer, m = case["layer"], case["m"]
            x = torch.empty((m, self.oracle.LAYERS[layer][1]), dtype=torch.bfloat16)
            actual = self.candidate._zero_bias_addmm_compatible(x, self.state, layer, self.oracle.out_features(layer, m))
            self.assertEqual(actual, layer == "qkv_gate" and m > 1)
            if actual:
                prerequisite_targets.append((layer, m))
        self.assertEqual(prerequisite_targets, [("qkv_gate", 512)])

    def test_nearby_qkv_rows_and_unpooler_always_restored(self):
        for m in (0, 1, 2, 5, 7, 8, 9, 15, 17, 31, 33, 127, 129, 130, 255, 257, 512, 1000):
            q = torch.empty((m, 8192), dtype=torch.bfloat16)
            self.assertEqual(self.candidate._zero_bias_addmm_compatible(q, self.state, "qkv_gate", 2304), m > 1)
            u = torch.empty((m, 4096), dtype=torch.bfloat16)
            for n in (1024, 8192):
                self.assertFalse(self.candidate._zero_bias_addmm_compatible(u, self.state, "unpooler", n))

    def test_precision_stride_dimensions_and_device_guards(self):
        x = torch.empty((8, 8192), dtype=torch.bfloat16)
        self.assertTrue(self.candidate._zero_bias_addmm_compatible(x, self.state, "qkv_gate", 2304))
        for bad in (x.float(), torch.empty((8, 16384), dtype=x.dtype)[:, ::2],
                    torch.empty((8192, 8), dtype=x.dtype).T, torch.empty((8, 8191), dtype=x.dtype),
                    torch.empty(8192, dtype=x.dtype), torch.empty((8, 8192), device="meta", dtype=x.dtype)):
            self.assertFalse(self.candidate._zero_bias_addmm_compatible(bad, self.state, "qkv_gate", 2304))
        for n in (0, 1, 2303, 2305):
            self.assertFalse(self.candidate._zero_bias_addmm_compatible(x, self.state, "qkv_gate", n))
        for weight in (None, torch.empty((2304, 8192), dtype=torch.float32),
                       torch.empty((8192, 2304), dtype=x.dtype).T,
                       torch.empty((2303, 8192), dtype=x.dtype),
                       torch.empty((2304, 8192), dtype=x.dtype, device="meta")):
            state = dict(self.state, qkv_gate=weight)
            self.assertFalse(self.candidate._zero_bias_addmm_compatible(x, state, "qkv_gate", 2304))
        for bias in (None, torch.empty(2304, dtype=torch.float32), torch.empty(4608, dtype=x.dtype)[::2],
                     torch.empty(2303, dtype=x.dtype), torch.empty(2304, dtype=x.dtype, device="meta")):
            state = dict(self.state, qkv_gate_zero_bias=bias)
            self.assertFalse(self.candidate._zero_bias_addmm_compatible(x, state, "qkv_gate", 2304))

    def test_canonical_dispatch_calls_helper(self):
        for m in (2, 8, 16, 32, 127, 128, 129, 256, 512):
            x = torch.empty((m, 8192), dtype=torch.bfloat16)
            data = {"x": x, "config": dict(layer="qkv_gate", m=m, n=2304, k=8192)}
            sentinel = object()
            with patch.object(self.candidate, "_zero_bias_addmm", return_value=sentinel) as call:
                self.assertIs(self.candidate.kernel_fn(data, self.state), sentinel)
            self.assertEqual(call.call_count, 1)
            self.assertIs(call.call_args.args[0], x)
            self.assertIs(call.call_args.args[1], self.state["qkv_gate"])

    def test_full_k_m2_m8_normal_outlier_sparse_zero_original_oracle(self):
        for m in (2, 8):
            for dist in ("normal", "outliers", "sparse", "zero"):
                data = self.oracle.generate_input(m, "qkv_gate", 50002 + m,
                    dist="outliers" if dist == "outliers" else "normal", device="cpu")
                if dist in ("sparse", "zero"):
                    data["x"].zero_()
                if dist == "sparse":
                    for row in range(m):
                        data["x"][row, (row * 1171) % 8192] = -128 if row % 2 else 8
                initial = data["x"].clone()
                output = self.candidate.kernel_fn(data, self.state)
                self.check_output(data, output)
                self.assertTrue(torch.equal(data["x"], initial))
                if dist == "sparse":
                    self.assertTrue(torch.equal(output, self.oracle.ref_kernel(data, self.weights)))
                if dist == "zero":
                    self.assertEqual(torch.count_nonzero(output), 0)

    def test_full_k_m8_same_address_aba_fresh_outputs_and_immutable_state(self):
        before = {key: value.clone() for key, value in self.state.items()}
        a = self.oracle.generate_input(8, "qkv_gate", 50011, device="cpu")
        b = self.oracle.generate_input(8, "qkv_gate", 50012, dist="outliers", device="cpu")
        slot = torch.empty_like(a["x"])
        address = slot.data_ptr()
        outputs = []
        for fresh in (a, b, a):
            slot.copy_(fresh["x"])
            data = {"x": slot, "config": dict(fresh["config"])}
            output = self.candidate.kernel_fn(data, self.state)
            self.check_output(data, output)
            self.assertEqual(slot.data_ptr(), address)
            self.assertTrue(torch.equal(slot, fresh["x"]))
            self.assertEqual(data["config"], fresh["config"])
            outputs.append(output)
        self.assertEqual(len({o.data_ptr() for o in outputs}), 3)
        self.assertTrue(torch.equal(outputs[0], outputs[2]))
        self.assertFalse(torch.equal(outputs[0], outputs[1]))
        for key, old in before.items():
            self.assertTrue(torch.equal(old.view(torch.uint8), self.state[key].view(torch.uint8)), key)

    def test_larger_and_nearby_m_reduced_k_helper_models_two_fresh_weights(self):
        # K131 helper models: these do not claim full-size or CUDA coverage.
        for seed in (50021, 50022):
            gen = torch.Generator().manual_seed(seed)
            weight = (torch.randn((2304, 131), generator=gen) / 131**.5).bfloat16()
            bias = torch.zeros(2304, dtype=torch.bfloat16)
            for m in (7, 16, 32, 127, 128, 129, 256, 512):
                x = torch.randn((m, 131), generator=gen).bfloat16()
                x[0].zero_()
                x[0, 130] = -64
                data = {"x": x, "config": dict(m=m, n=2304, k=131, layer="qkv_gate")}
                self.check_output(data, self.candidate._zero_bias_addmm(x, weight, bias, 2304), {"qkv_gate": weight})
            self.assertEqual(torch.count_nonzero(bias), 0)

    def test_restored_qkv_m1_and_unpooler_canonical_outputs_match_parent(self):
        for layer, m in (("qkv_gate", 1), ("unpooler", 1), ("unpooler", 8), ("unpooler", 16)):
            data = self.oracle.generate_input(m, layer, 50031 + m, device="cpu")
            with patch.object(self.candidate, "_zero_bias_addmm", side_effect=AssertionError("restored path changed")):
                actual = self.candidate.kernel_fn(data, self.state)
            expected = self.parent.kernel_fn(data, self.state)
            self.assertTrue(torch.equal(actual, expected), (layer, m))

    def test_noncanonical_all_family_and_strided_fallbacks_match_parent(self):
        gen = torch.Generator().manual_seed(50041)
        weights = {"router": torch.randn((31, 131), generator=gen), **{
            layer: torch.randn((37, 131), generator=gen).bfloat16()
            for layer in ("pooler", "qkv_gate", "unpooler", "o")}}
        actual_state, expected_state = self.candidate.prepare(weights), self.parent.prepare(weights)
        for layer in weights:
            for m in (1, 7, 9, 16, 33, 128, 256):
                for strided in (False, True):
                    x = torch.randn((m, 262 if strided else 131), generator=gen).bfloat16()
                    if strided:
                        x = x[:, ::2]
                    data = {"x": x, "config": dict(m=m, n=29, k=131, layer=layer)}
                    torch.testing.assert_close(self.candidate.kernel_fn(data, actual_state),
                                               self.parent.kernel_fn(data, expected_state), rtol=0, atol=0)

    def test_public_addmm_arguments_and_nan_output_is_not_an_operand(self):
        gen = torch.Generator().manual_seed(50051)
        x = torch.randn((7, 31), generator=gen).bfloat16()
        weight = torch.randn((47, 31), generator=gen).bfloat16()
        bias = torch.zeros(47, dtype=torch.bfloat16)
        output = torch.full((7, 37), float("nan"), dtype=torch.bfloat16)
        actual_addmm = torch.addmm
        with patch.object(torch, "empty", return_value=output), patch.object(torch, "addmm", wraps=actual_addmm) as call:
            result = self.candidate._zero_bias_addmm(x, weight, bias, 37)
        self.assertIs(result, output)
        self.assertEqual(call.call_count, 1)
        args, kwargs = call.call_args
        self.assertEqual(args[0].data_ptr(), bias.data_ptr())
        self.assertEqual(args[0].shape, (37,))
        self.assertIs(args[1], x)
        self.assertEqual(args[2].stride(), (1, 31))
        self.assertEqual({k: v for k, v in kwargs.items() if k != "out"}, dict(beta=1, alpha=1))
        self.assertIs(kwargs["out"], output)
        self.assertNotIn(output.data_ptr(), (x.data_ptr(), weight.data_ptr(), bias.data_ptr()))
        data = {"x": x, "config": dict(m=7, n=37, k=31, layer="qkv_gate")}
        self.check_output(data, output, {"qkv_gate": weight})


if __name__ == "__main__":
    unittest.main()
