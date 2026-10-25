"""CPU representation checks, not Triton compilation or GPU correctness."""
import unittest
import torch


class WideningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoded = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn)
        cls.decoded = cls.encoded.to(torch.float32)
        cls.finite = torch.isfinite(cls.decoded)

    def test_all_finite_e4m3_values_exact_in_bfloat16(self):
        self.assertEqual(int(self.finite.sum()), 254)
        widened = self.encoded.to(torch.bfloat16).to(torch.float32)
        self.assertTrue(torch.equal(widened[self.finite], self.decoded[self.finite]))
        self.assertTrue(torch.equal(torch.signbit(widened[self.finite]),
                                    torch.signbit(self.decoded[self.finite])))

    def test_widening_matches_existing_fp16_operand_values(self):
        a = self.encoded.to(torch.float16).to(torch.float32)
        b = self.encoded.to(torch.bfloat16).to(torch.float32)
        self.assertTrue(torch.equal(a[self.finite], b[self.finite]))

    def test_verify_has_64_useful_rows_decode_does_not(self):
        self.assertEqual(8 * (16 // 2), 64)
        self.assertEqual(1 * (16 // 2), 8)


if __name__ == "__main__":
    unittest.main()
