"""P049 CPU-only numerical/dispatch tests; no CUDA or backend-selection proof."""
import ast
import hashlib
import unittest
from unittest.mock import patch

import torch

from harness import ROOT, correctness_cases, load_module

PARENT = ROOT / "candidates/projection_pooler_column_batch2_hybrid.py"
SOURCE = ROOT / "candidates/projection_qkv_unpooler_zero_bias.py"
PARENT_SHA = "001ea12e8764564d507de51e433769f5bd9a9916bfc4c014f0f87522c278545e"
LAYERS = {"qkv_gate": (2304, 8192), "unpooler": (8192, 4096)}


class ProjectionZeroBias(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(4)
        cls.candidate = load_module(SOURCE, "p049_candidate")
        cls.parent = load_module(PARENT, "p049_parent")
        cls.oracle = load_module(ROOT / "official/reference.py", "p049_oracle")
        gen = torch.Generator().manual_seed(49001)
        cls.weights = {"router": torch.randn((7, 131), generator=gen)}
        for layer, (n, k) in LAYERS.items():
            cls.weights[layer] = (torch.randn((n, k), generator=gen) / k**.5).bfloat16()
        cls.state = cls.candidate.prepare(cls.weights)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def assert_output(self, data, out, weights=None):
        ok, message = self.oracle.check_implementation(data, out, weights or self.weights)
        self.assertTrue(ok, message)
        self.assertEqual(out.shape, (data["x"].shape[0], data["config"]["n"]))
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertTrue(out.is_contiguous())
        self.assertEqual(out.stride(), (out.shape[1], 1))

    def test_frozen_parent_and_only_intended_ast_changes(self):
        self.assertEqual(hashlib.sha256(PARENT.read_bytes()).hexdigest(), PARENT_SHA)
        parent, candidate = ast.parse(PARENT.read_text()), ast.parse(SOURCE.read_text())
        parent.body.pop(0)
        candidate.body.pop(0)
        candidate.body = [node for node in candidate.body if not (
            isinstance(node, ast.FunctionDef) and node.name in {
                "_zero_bias_addmm", "_zero_bias_addmm_compatible"})]
        for node in candidate.body:
            if isinstance(node, ast.FunctionDef) and node.name == "prepare":
                added = node.body[3].body[0].body.pop()  # zero allocation in existing layer loop
                self.assertIsInstance(added, ast.Assign)
                self.assertIn("_zero_bias", ast.unparse(added))
            if isinstance(node, ast.FunctionDef) and node.name == "kernel_fn":
                added = node.body.pop(3)
                self.assertIsInstance(added, ast.If)
                self.assertIn("_zero_bias_addmm_compatible", ast.unparse(added.test))
        self.assertEqual(ast.dump(candidate, include_attributes=False), ast.dump(parent, include_attributes=False))
        marker = "# Applicable project/parent Apache License 2.0 text for standalone upload:"
        self.assertEqual(SOURCE.read_text().split(marker)[1].split("import torch")[0],
                         PARENT.read_text().split(marker)[1].split("import torch")[0])

    def test_prepare_zero_metadata_original_values_and_no_activation_state(self):
        original = self.parent.prepare(self.weights)
        self.assertEqual(set(self.state) - set(original), {"qkv_gate_zero_bias", "unpooler_zero_bias"})
        total = 0
        for layer, (n, _) in LAYERS.items():
            bias = self.state[layer + "_zero_bias"]
            self.assertEqual(bias.shape, (n,))
            self.assertEqual(bias.dtype, torch.bfloat16)
            self.assertEqual(bias.device, self.weights[layer].device)
            self.assertTrue(bias.is_contiguous())
            self.assertEqual(torch.count_nonzero(bias), 0)
            total += bias.numel() * bias.element_size()
        self.assertEqual(total, 20992)
        for key in original:
            self.assertTrue(torch.equal(original[key], self.state[key]), key)

    def test_all_30_scored_and_nine_prerequisite_metadata_routes(self):
        count = targets = 0
        for m in (1, 16, 32, 8, 128, 256):
            for layer, (full_n, k) in self.oracle.LAYERS.items():
                n = self.oracle.out_features(layer, m)
                x = torch.empty((m, k), dtype=torch.bfloat16)
                actual = self.candidate._zero_bias_addmm_compatible(x, self.state, layer, n)
                expected = layer in LAYERS and m > 1
                self.assertEqual(actual, expected, (layer, m, n))
                count += 1
                targets += actual
        self.assertEqual((count, targets), (30, 10))
        for case in correctness_cases():
            layer, m = case["layer"], case["m"]
            k = self.oracle.LAYERS[layer][1]
            x = torch.empty((m, k), dtype=torch.bfloat16)
            self.assertEqual(self.candidate._zero_bias_addmm_compatible(
                x, self.state, layer, self.oracle.out_features(layer, m)), layer in LAYERS and m > 1)

    def test_nearby_rows_and_canonical_unpooler_width_transition(self):
        for layer, (_, k) in LAYERS.items():
            for m in (1, 2, 5, 7, 8, 9, 15, 17, 31, 33, 127, 129, 130, 255, 257, 512, 1000):
                x = torch.empty((m, k), dtype=torch.bfloat16)
                n = self.oracle.out_features(layer, m)
                self.assertEqual(self.candidate._zero_bias_addmm_compatible(x, self.state, layer, n), m > 1)
                if layer == "unpooler":
                    wrong_n = 8192 if n == 1024 else 1024
                    self.assertFalse(self.candidate._zero_bias_addmm_compatible(x, self.state, layer, wrong_n))

    def test_incompatible_dtype_stride_width_weight_bias_fallback_guards(self):
        for layer, (n, k) in LAYERS.items():
            m = 16
            x = torch.empty((m, k), dtype=torch.bfloat16)
            self.assertTrue(self.candidate._zero_bias_addmm_compatible(x, self.state, layer, n))
            for bad_x in (x.float(), torch.empty((m, k * 2), dtype=x.dtype)[:, ::2],
                          torch.empty((k, m), dtype=x.dtype).T,
                          torch.empty((m, k + 1), dtype=x.dtype), torch.empty((k,), dtype=x.dtype)):
                self.assertFalse(self.candidate._zero_bias_addmm_compatible(bad_x, self.state, layer, n))
            for bad_n in (0, 1, n - 1, n + 1):
                self.assertFalse(self.candidate._zero_bias_addmm_compatible(x, self.state, layer, bad_n))
            for value in (None, torch.empty((n, k), dtype=torch.float32),
                          torch.empty((k, n), dtype=x.dtype).T,
                          torch.empty((n - 1, k), dtype=x.dtype),
                          torch.empty((n, k), dtype=x.dtype, device="meta")):
                state = dict(self.state, **{layer: value})
                self.assertFalse(self.candidate._zero_bias_addmm_compatible(x, state, layer, n))
            for value in (None, torch.empty(n, dtype=torch.float32),
                          torch.empty(n * 2, dtype=x.dtype)[::2], torch.empty(n - 1, dtype=x.dtype),
                          torch.empty(n, dtype=x.dtype, device="meta")):
                state = dict(self.state, **{layer + "_zero_bias": value})
                self.assertFalse(self.candidate._zero_bias_addmm_compatible(x, state, layer, n))

    def test_kernel_calls_target_helper_and_never_m1(self):
        for layer, (_, k) in LAYERS.items():
            for m in (8, 16, 32, 128, 256):
                n = self.oracle.out_features(layer, m)
                x = torch.empty((m, k), dtype=torch.bfloat16)
                data = {"x": x, "config": {"m": m, "n": n, "k": k, "layer": layer}}
                sentinel = torch.empty((m, n), dtype=x.dtype)
                with patch.object(self.candidate, "_zero_bias_addmm", return_value=sentinel) as call:
                    self.assertIs(self.candidate.kernel_fn(data, self.state), sentinel)
                    self.assertEqual(call.call_count, 1)
                    self.assertIs(call.call_args.args[0], x)
            x = torch.zeros((1, k), dtype=torch.bfloat16)
            n = self.oracle.out_features(layer, 1)
            with patch.object(self.candidate, "_zero_bias_addmm", side_effect=AssertionError("m1 changed")):
                data = {"x": x, "config": {"m": 1, "n": n, "k": k, "layer": layer}}
                self.assertTrue(torch.equal(self.candidate.kernel_fn(data, self.state), self.parent.kernel_fn(data, self.state)))

    def test_full_k_m2_normal_outlier_sparse_zero(self):
        for layer, (_, k) in LAYERS.items():
            for dist in ("normal", "outliers", "sparse", "zero"):
                data = self.oracle.generate_input(2, layer, 49002, dist="outliers" if dist == "outliers" else "normal", device="cpu")
                if dist in ("sparse", "zero"):
                    data["x"].zero_()
                if dist == "sparse":
                    data["x"][0, 0], data["x"][1, k - 1] = 8, -128
                before = data["x"].clone()
                out = self.candidate.kernel_fn(data, self.state)
                self.assert_output(data, out)
                self.assertTrue(torch.equal(data["x"], before))
                if dist == "sparse":
                    self.assertTrue(torch.equal(out, self.oracle.ref_kernel(data, self.weights)))
                if dist == "zero":
                    self.assertEqual(torch.count_nonzero(out), 0)

    def test_full_unpooler_width_m9_full_k(self):
        data = self.oracle.generate_input(9, "unpooler", 49009, dist="outliers", device="cpu")
        out = self.candidate.kernel_fn(data, self.state)
        self.assert_output(data, out)

    def test_full_k_same_address_aba_fresh_output_and_state_immutability(self):
        before = {key: value.clone() for key, value in self.state.items()}
        for layer in LAYERS:
            a = self.oracle.generate_input(2, layer, 49011, device="cpu")
            b = self.oracle.generate_input(2, layer, 49012, dist="outliers", device="cpu")
            slot = torch.empty_like(a["x"])
            address = slot.data_ptr()
            outputs = []
            for current in (a, b, a):
                slot.copy_(current["x"])
                data = {"x": slot, "config": dict(current["config"])}
                out = self.candidate.kernel_fn(data, self.state)
                self.assert_output(data, out)
                self.assertEqual(slot.data_ptr(), address)
                self.assertTrue(torch.equal(slot, current["x"]))
                self.assertEqual(data["config"], current["config"])
                self.assertNotIn(out.data_ptr(), [slot.data_ptr()] + [t.data_ptr() for t in self.state.values()])
                outputs.append(out)
            self.assertEqual(len({out.data_ptr() for out in outputs}), 3)
            self.assertTrue(torch.equal(outputs[0], outputs[2]))
            self.assertFalse(torch.equal(outputs[0], outputs[1]))
        for key in before:
            self.assertTrue(torch.equal(self.state[key], before[key]), key)

    def test_larger_m_reduced_k_helper_models_and_fresh_weights(self):
        # Reduced K131 models exercise arithmetic/output layout, not production dispatch.
        for seed in (49021, 49022):
            gen = torch.Generator().manual_seed(seed)
            for layer, (n, _) in LAYERS.items():
                weight = (torch.randn((n, 131), generator=gen) / 131**.5).bfloat16()
                bias = torch.zeros(n, dtype=torch.bfloat16)
                for m in (7, 8, 16, 32, 128, 256):
                    used_n = 1024 if layer == "unpooler" and m <= 8 else n
                    x = torch.randn((m, 131), generator=gen).bfloat16()
                    x[0].zero_()
                    x[0, 130] = -64
                    data = {"x": x, "config": {"m": m, "n": used_n, "k": 131, "layer": layer}}
                    out = self.candidate._zero_bias_addmm(x, weight, bias, used_n)
                    self.assert_output(data, out, {layer: weight})
                self.assertEqual(torch.count_nonzero(bias), 0)

    def test_public_addmm_api_arguments_and_output_ownership(self):
        x = torch.randn((7, 31)).bfloat16()
        w = torch.randn((47, 31)).bfloat16()
        bias = torch.zeros(47, dtype=torch.bfloat16)
        actual = torch.addmm
        with patch.object(torch, "addmm", wraps=actual) as call:
            out = self.candidate._zero_bias_addmm(x, w, bias, 37)
        self.assertEqual(call.call_count, 1)
        args, kwargs = call.call_args
        self.assertEqual(args[0].shape, (37,))
        self.assertEqual(args[0].data_ptr(), bias.data_ptr())
        self.assertIs(args[1], x)
        self.assertEqual(args[2].shape, (31, 37))
        self.assertEqual(args[2].stride(), (1, 31))
        self.assertEqual({k: v for k, v in kwargs.items() if k != "out"}, {"beta": 1, "alpha": 1})
        self.assertIs(kwargs["out"], out)
        self.assertEqual(out.stride(), (37, 1))
        self.assertNotIn(out.data_ptr(), (x.data_ptr(), w.data_ptr(), bias.data_ptr()))

    def test_poisoned_fresh_output_is_not_an_input(self):
        gen = torch.Generator().manual_seed(49041)
        x = torch.randn((7, 31), generator=gen).bfloat16()
        w = torch.randn((47, 31), generator=gen).bfloat16()
        bias = torch.zeros(47, dtype=torch.bfloat16)
        out = torch.full((7, 37), float("nan"), dtype=torch.bfloat16)
        with patch.object(torch, "empty", return_value=out):
            result = self.candidate._zero_bias_addmm(x, w, bias, 37)
        self.assertIs(result, out)
        data = {"x": x, "config": {"m": 7, "n": 37, "k": 31, "layer": "qkv_gate"}}
        self.assert_output(data, result, {"qkv_gate": w})
        self.assertEqual(torch.count_nonzero(bias), 0)

    def test_noncanonical_and_strided_inputs_match_parent(self):
        gen = torch.Generator().manual_seed(49031)
        weights = {"router": torch.randn((31, 131), generator=gen), **{
            layer: torch.randn((37, 131), generator=gen).bfloat16()
            for layer in ("pooler", "qkv_gate", "unpooler", "o")}}
        state, parent_state = self.candidate.prepare(weights), self.parent.prepare(weights)
        for layer in weights:
            for m in (1, 7, 9, 16, 33, 128, 256):
                for strided in (False, True):
                    x = torch.randn((m, 262 if strided else 131), generator=gen).bfloat16()
                    if strided:
                        x = x[:, ::2]
                    data = {"x": x, "config": {"m": m, "n": 29, "k": 131, "layer": layer}}
                    torch.testing.assert_close(self.candidate.kernel_fn(data, state), self.parent.kernel_fn(data, parent_state), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
