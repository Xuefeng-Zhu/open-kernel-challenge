"""CPU numerical gates; CUDA tensor-core paths are not exercised on this Mac."""
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from harness import ROOT, load_module

reference = load_module(ROOT / "official/reference.py", "router_candidate_reference")
requested = os.environ.get("ROUTER_CANDIDATE")
paths = [ROOT / requested] if requested else sorted((ROOT / "candidates").glob("router_*.py"))


class RouterCandidateMixin:
    def test_router_boundaries_fresh_inputs_and_probes(self):
        gen = torch.Generator().manual_seed(83629)
        w = torch.randn((31, 127), generator=gen) / 127**0.5
        original = w.clone()
        weights = {"router": w}
        state = self.candidate.prepare(weights)
        for m in [1, 2, 7, 8, 9, 16, 32, 128, 256]:
            with self.subTest(m=m):
                x = torch.randn((m, 127), generator=gen)
                x[:, :2] *= 30
                x[0].zero_()
                x[0, 17] = -128
                x = x.bfloat16()
                before = x.clone()
                data = dict(x=x, config=dict(m=m, n=31, k=127, layer="router"))
                out = self.candidate.kernel_fn(data, state)
                ok, msg = reference.check_implementation(data, out, weights)
                self.assertTrue(ok, msg)
                self.assertTrue(torch.equal(x, before))
        self.assertTrue(torch.equal(w, original))

    def test_router_lo_component_and_near_zero_rows(self):
        # BF16-only weights fail this oracle; the legal hi+lo path must retain the residual.
        w = torch.tensor([[1.001, 0.25, -0.5], [2.003, -0.25, 0.5]], dtype=torch.float32)
        x = torch.zeros((16, 3), dtype=torch.bfloat16)
        x[0, 0] = 8
        data = dict(x=x, config=dict(m=16, n=2, k=3, layer="router"))
        weights = {"router": w}
        out = self.candidate.kernel_fn(data, self.candidate.prepare(weights))
        self.assertTrue(reference.check_implementation(data, out, weights)[0])
        degraded = (x @ w.bfloat16().T).float()
        self.assertFalse(reference.check_implementation(data, degraded, weights)[0])
        self.assertTrue(torch.equal(out[1:], torch.zeros_like(out[1:])))

    def test_other_projection_and_n_slice(self):
        w = torch.arange(24).reshape(6, 4).bfloat16()
        weights = {"o": w, "router": torch.ones((2, 4))}
        state = self.candidate.prepare(weights)
        for scale in [1, -2, 8]:
            x = torch.tensor([[scale, 0, 0, 0]], dtype=torch.bfloat16)
            data = dict(x=x, config=dict(m=1, n=3, k=4, layer="o"))
            out = self.candidate.kernel_fn(data, state)
            torch.testing.assert_close(out, (scale * w[:3, 0])[None], rtol=0, atol=0)


for path in paths:
    if not path.is_file():
        raise FileNotFoundError(path)
    module = load_module(path, "candidate_" + path.stem)
    globals()["Test_" + path.stem] = type(
        "Test_" + path.stem, (RouterCandidateMixin, unittest.TestCase), {"candidate": module}
    )
