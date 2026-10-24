"""CPU numerical gates for pooler; target kernels need their own H200 run."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from harness import ROOT, load_module

reference = load_module(ROOT / "official/reference.py", "pooler_reference")
candidate = load_module(ROOT / "candidates/pooler_triton_split8.py", "pooler_candidate")


class PoolerCandidateTests(unittest.TestCase):
    def test_random_tails_and_batch_boundary(self):
        gen = torch.Generator().manual_seed(71823)
        w = (torch.randn((37, 131), generator=gen) / 131**0.5).bfloat16()
        saved = w.clone()
        weights = {"pooler": w, "router": torch.ones((2, 131))}
        state = candidate.prepare(weights)
        for m in [1, 7, 8, 9]:
            with self.subTest(m=m):
                x = torch.randn((m, 131), generator=gen).bfloat16()
                data = dict(x=x, config=dict(layer="pooler", m=m, n=37, k=131))
                out = candidate.kernel_fn(data, state)
                ok, msg = reference.check_implementation(data, out, weights)
                self.assertTrue(ok, msg)
        self.assertTrue(torch.equal(w, saved))

    def test_full_width_probe_columns(self):
        gen = torch.Generator().manual_seed(81937)
        w = (torch.randn((4096, 8192), generator=gen) / 8192**0.5).bfloat16()
        x = torch.zeros((8, 8192), dtype=torch.bfloat16)
        cols = [0, 63, 64, 511, 512, 4095, 4096, 8191]
        for row, col in enumerate(cols):
            x[row, col] = 8 if row % 2 else -128
        data = dict(x=x, config=dict(layer="pooler", m=8, n=4096, k=8192))
        weights = {"pooler": w, "router": torch.ones((2, 8192))}
        out = candidate.kernel_fn(data, candidate.prepare(weights))
        expected = torch.stack([w[:, col] * (8 if row % 2 else -128) for row, col in enumerate(cols)])
        torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_router_control_preserved(self):
        gen = torch.Generator().manual_seed(71293)
        w = torch.randn((31, 127), generator=gen) / 127**0.5
        weights = {"pooler": torch.ones((2, 127), dtype=torch.bfloat16), "router": w}
        state = candidate.prepare(weights)
        for m in [1, 8, 9, 32]:
            x = torch.randn((m, 127), generator=gen).bfloat16()
            data = dict(x=x, config=dict(layer="router", m=m, n=31, k=127))
            ok, msg = reference.check_implementation(data, candidate.kernel_fn(data, state), weights)
            self.assertTrue(ok, msg)
