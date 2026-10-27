import unittest
import hashlib
import json
from pathlib import Path
import tempfile
from assess_p063 import assess,parse_ptxas,sass_ops,score_scenario


class OfflineAssessment(unittest.TestCase):
    def test_assembler_log_must_be_bound_and_untampered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            manifest=dict(gpu_execution=False,benchmark=False,
                          variants=[dict(label='raw',hashes={})])
            (root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'not hash-bound'):
                assess(root)
            log=b'Used 38 registers'
            manifest['variants'][0]['hashes']['ptxas.txt']=hashlib.sha256(log).hexdigest()
            (root/'manifest.json').write_text(json.dumps(manifest))
            (root/'raw.ptxas.txt').write_bytes(b'Used 12 registers')
            with self.assertRaisesRegex(ValueError,'Artifact hash mismatch'):
                assess(root)

    def test_resource_parse_rejects_missing_data(self):
        with self.assertRaises(ValueError):parse_ptxas('no report')
        x=parse_ptxas('0 bytes stack frame, 8 bytes spill stores, 16 bytes spill loads\nUsed 48 registers')
        self.assertEqual(x['spill_store_bytes'],8)
        self.assertEqual(x['spill_load_bytes'],16)
        self.assertEqual(x['registers'],48)

    def test_static_opcode_counts_do_not_count_encoding_lines(self):
        x=sass_ops('/*0040*/ @!P5 LDG.E.U8 R1, [R2]; /* 0x0123 */\n/*0050*/ HMMA.16816.F32.BF16 R1, R2;\n /* 0x12345678 */\n/*0060*/ @!UP0 BRA 0x20;\n/*0070*/ NOP;')
        self.assertEqual(x,{'LDG.E.U8':1,'HMMA.16816.F32.BF16':1,'BRA':1,'NOP':1})

    def test_score_uses_six_weighted_loads(self):
        rows=[dict(event='benchmark',case=dict(load=str(i),layer=layer,m=8,weight=62),median_ns=ns)
              for i in range(6) for layer,ns in [('pooler',20),('o',80)]]
        s={'result':dict(records=rows,speedup=1)}
        a={'targets':[dict(m=8,raw_over_selected=1)]}
        self.assertAlmostEqual(score_scenario(a,s,.5)['score_factor'],1/.9)
        self.assertEqual(score_scenario(a,s,1)['modeled_score'],1)


if __name__=='__main__':unittest.main()
