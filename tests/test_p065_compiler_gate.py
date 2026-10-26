"""Source-bound offline evidence gate; these assertions are not GPU tests."""
import hashlib,json,re,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class Gate(unittest.TestCase):
 def test_artifacts_are_bound_to_owned_source(self):
  base=ROOT/'evidence/p065';d=json.loads((base/'manifest.json').read_text())
  for name,sha in d['manifest']['sources'].items():self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),sha)
  self.assertFalse(d['gpu_execution']);self.assertFalse(d['benchmark'])
  for v in d['variants']:
   for ext,sha in v['hashes'].items():
    if ext!='cubin':self.assertEqual(hashlib.sha256((base/(v['label']+'.'+ext)).read_bytes()).hexdigest(),sha)
 def test_explicit_decode_consumer_and_no_weight_staging(self):
  for name in ['raw','encoded']:
   text=(ROOT/'evidence/p065'/f'{name}.ttgir').read_text()
   for op in ['ttg.local_alloc','ttg.local_load','ttg.async_copy_global_to_local']:self.assertNotIn(op,text)
   self.assertEqual(text.count('ttg.convert_layout'),1)
   self.assertRegex(text,r'ttg.convert_layout.*tensor<16x64xf32')
   self.assertIn('tt.dot',text)
  encoded=(ROOT/'evidence/p065/encoded.ttgir').read_text()
  self.assertIn('kWidth = 2',encoded)
 def test_precision_no_spills_and_costs_retained(self):
  for name,registers in [('raw',28),('encoded',116)]:
   ptx=(ROOT/'evidence/p065'/f'{name}.ptx').read_text()
   self.assertIn('mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32',ptx)
   log=(ROOT/'evidence/p065'/f'{name}.ptxas.txt').read_text()
   self.assertIn(f'Used {registers} registers',log)
   self.assertIn('0 bytes spill stores, 0 bytes spill loads',log)
if __name__=='__main__':unittest.main()
