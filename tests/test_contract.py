"""Focused tests for boundaries and the official oracle's rejection behavior."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from harness import ROOT, contract, correctness_cases, load_module

reference = load_module(ROOT / "official/reference.py", "reference_tests")
candidate = load_module(ROOT / "submission.py", "candidate_tests")


class ContractTests(unittest.TestCase):
    def test_official_shape_inventory(self):
        cases = contract()["benchmarks"]
        self.assertEqual(len(cases), 30)
        self.assertEqual(len(correctness_cases()), 9)
        self.assertEqual({c["m"] for c in cases}, {1, 8, 16, 32, 128, 256})
        self.assertEqual(reference.out_features("unpooler", 8), 1024)
        self.assertEqual(reference.out_features("unpooler", 9), 8192)

    def test_fresh_inputs_and_unpooler_slice(self):
        weights = {"unpooler": torch.arange(24).reshape(6, 4).to(torch.bfloat16)}
        saved = weights["unpooler"].clone()
        state = candidate.prepare(weights)
        for scale in [1, 2, -1]:
            x = torch.tensor([[scale, 0, 0, 0]], dtype=torch.bfloat16)
            data = dict(x=x, config=dict(layer="unpooler", m=1, n=3, k=4))
            out = candidate.kernel_fn(data, state)
            torch.testing.assert_close(out, (scale * weights["unpooler"][:3, 0])[None], rtol=0, atol=0)
        self.assertTrue(torch.equal(saved, weights["unpooler"]))

    def test_router_precision_and_negative_controls(self):
        # A weight-column probe is sensitive to rounding the router to BF16.
        w = torch.tensor([[1.001, 0.0], [2.003, 0.0]], dtype=torch.float32)
        data = dict(x=torch.tensor([[8.0, 0]], dtype=torch.bfloat16),
                    config=dict(layer="router", m=1, n=2, k=2))
        weights = {"router": w}
        good = candidate.kernel_fn(data, candidate.prepare(weights))
        self.assertTrue(reference.check_implementation(data, good, weights)[0])
        bad_precision = (data["x"] @ w.bfloat16().T).float()
        self.assertFalse(reference.check_implementation(data, bad_precision, weights)[0])
        for bad in [good.bfloat16(), good[:, :1], torch.full_like(good, float("nan")), good + 1]:
            self.assertFalse(reference.check_implementation(data, bad, weights)[0])

    def test_probe_rejects_error_below_general_tolerance(self):
        weights = {"pooler": torch.ones((3, 4), dtype=torch.bfloat16)}
        data = dict(x=torch.tensor([[8, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.bfloat16),
                    config=dict(layer="pooler", m=2, n=3, k=4))
        ref = reference.ref_kernel(data, weights)
        corrupted = ref.clone()
        corrupted[0] += 0.0625  # One BF16 step at 8: <1% general, >0.1% probe bound.
        self.assertTrue(reference.compare(corrupted, ref, reference.REL_L2_TOL)[0])
        ok, message = reference.check_implementation(data, corrupted, weights)
        self.assertFalse(ok)
        self.assertIn("probe row", message)

    def test_per_row_check_rejects_small_global_error(self):
        ref = torch.ones((4096, 4), dtype=torch.float32)
        corrupted = ref.clone()
        corrupted[123] += 0.06
        self.assertLess(float((corrupted - ref).norm() / ref.norm()), reference.REL_L2_TOL)
        ok, message = reference.compare(corrupted, ref, reference.REL_L2_TOL)
        self.assertFalse(ok)
        self.assertIn("but row", message)


if __name__ == "__main__":
    unittest.main()
