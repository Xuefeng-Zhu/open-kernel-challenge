"""CPU numerical gates for isolated BF16 projections; target kernels need their own H200 run."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from harness import ROOT, load_module

reference = load_module(ROOT / "official/reference.py", "pooler_reference")


class ProjectionCandidateMixin:
    def test_random_tails_and_batch_boundary(self):
        gen = torch.Generator().manual_seed(71823)
        w = (torch.randn((37, 131), generator=gen) / 131**0.5).bfloat16()
        saved = w.clone()
        weights = {self.layer: w, "router": torch.ones((2, 131))}
        state = self.candidate.prepare(weights)
        for m in [1, 2, 7, 8, 9]:
            with self.subTest(m=m):
                x = torch.randn((m, 131), generator=gen).bfloat16()
                data = dict(x=x, config=dict(layer=self.layer, m=m, n=37, k=131))
                out = self.candidate.kernel_fn(data, state)
                ok, msg = reference.check_implementation(data, out, weights)
                self.assertTrue(ok, msg)
        self.assertTrue(torch.equal(w, saved))

    def test_full_width_probe_columns(self):
        gen = torch.Generator().manual_seed(81937)
        w = (torch.randn((self.full_width, 8192), generator=gen) / 8192**0.5).bfloat16()
        x = torch.zeros((8, 8192), dtype=torch.bfloat16)
        cols = [0, 63, 64, 511, 512, 4095, 4096, 8191]
        for row, col in enumerate(cols):
            x[row, col] = 8 if row % 2 else -128
        data = dict(x=x, config=dict(layer=self.layer, m=8, n=self.full_width, k=8192))
        weights = {self.layer: w, "router": torch.ones((2, 8192))}
        out = self.candidate.kernel_fn(data, self.candidate.prepare(weights))
        expected = torch.stack([w[:, col] * (8 if row % 2 else -128) for row, col in enumerate(cols)])
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        # Exercise one-row dispatch with fresh different probe values each call.
        for row in range(8):
            single = dict(x=x[row:row + 1].clone(),
                          config=dict(layer=self.layer, m=1, n=self.full_width, k=8192))
            actual = self.candidate.kernel_fn(single, self.candidate.prepare(weights))
            torch.testing.assert_close(actual, expected[row:row + 1], rtol=0, atol=0)

    def test_router_control_preserved(self):
        gen = torch.Generator().manual_seed(71293)
        w = torch.randn((31, 127), generator=gen) / 127**0.5
        weights = {self.layer: torch.ones((2, 127), dtype=torch.bfloat16), "router": w}
        state = self.candidate.prepare(weights)
        for m in [1, 8, 9, 32]:
            x = torch.randn((m, 127), generator=gen).bfloat16()
            data = dict(x=x, config=dict(layer="router", m=m, n=31, k=127))
            ok, msg = reference.check_implementation(data, self.candidate.kernel_fn(data, state), weights)
            self.assertTrue(ok, msg)


for filename, layer, width in [
    ("pooler_triton_split8.py", "pooler", 4096),
    ("qkv_triton_split8.py", "qkv_gate", 2304),
    ("pooler_simt_row.py", "pooler", 4096),
    ("router_pooler_hybrid.py", "pooler", 4096),
]:
    module = load_module(ROOT / "candidates" / filename, "bf16_" + layer)
    globals()["Test_" + Path(filename).stem] = type(
        "Test_" + Path(filename).stem, (ProjectionCandidateMixin, unittest.TestCase),
        dict(candidate=module, layer=layer, full_width=width))
