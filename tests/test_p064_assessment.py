import unittest
from assess_p064 import evaluate,reconstructed_bf16_allocations,sass_instructions,ROOT


class StaticGate(unittest.TestCase):
    def test_saved_evidence_rejects_even_when_register_count_improves(self):
        result=evaluate(ROOT/'evidence/p063',ROOT/'evidence/p064')
        self.assertTrue(result['gates']['raw_instruction_text_identical'])
        self.assertTrue(result['gates']['register_reduction'])
        self.assertFalse(result['gates']['decoded_bf16_shared_staging_removed'])
        self.assertFalse(result['compiler_gate_pass'])

    def test_source_valued_shared_alloc_is_not_hidden_by_missing_transpose(self):
        ir='''%buf = ttg.local_alloc : () -> !ttg.memdesc<64x64xbf16, #shared>
%a = ttg.local_alloc %decoded : (tensor<64x64xbf16, #blocked>) -> !ttg.memdesc<64x64xbf16, #shared>
%b = ttg.local_alloc %acc : (tensor<16x64xf32, #blocked>) -> !ttg.memdesc<16x64xf32, #shared>'''
        self.assertEqual(reconstructed_bf16_allocations(ir),[('%a','%decoded','64x64')])

    def test_sass_normalization_keeps_predicates_registers_and_order(self):
        a='header\n/*0010*/ @!P0 LDS R1, [R2]; /*encoding*/\n/*0020*/ NOP;'
        b='different header\n/*0010*/ @!P0 LDS R1, [R2]; /*different encoding comment*/\n/*0020*/ NOP;'
        self.assertEqual(sass_instructions(a),sass_instructions(b))
        self.assertNotEqual(sass_instructions(a),sass_instructions(b.replace('R1','R3')))


if __name__=='__main__':unittest.main()
