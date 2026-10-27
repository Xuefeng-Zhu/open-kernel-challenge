"""CPU source/index contracts, not execution of generated GPU code."""
import ast
import importlib.util
import math
from pathlib import Path
import struct
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('m64n8_offline',HERE/'offline_compile.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def bf16(value):
 bits=struct.unpack('>I',struct.pack('>f',value))[0]
 bits=(bits+0x7fff+((bits>>16)&1))&0xffff0000
 return struct.unpack('>f',struct.pack('>I',bits))[0]


class SourceContract(unittest.TestCase):
 def test_owned_hash_ABI_and_no_host_execution(self):
  raw=(HERE/'kernel.py').read_bytes()
  self.assertEqual(c.sha(raw),c.KERNEL_SHA)
  self.assertEqual(len(c.source_contract(raw)),64)
  with self.assertRaisesRegex(ValueError,'hash'):
   c.source_contract(raw+b'\n')

 def test_exact_small_shapes_precision_and_key_axis(self):
  tree=ast.parse((HERE/'kernel.py').read_text())
  calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
  named=lambda n:ast.unparse(n.func)
  mma=[n for n in calls if named(n)=='warpgroup_mma']
  self.assertEqual(len(mma),2)
  self.assertEqual([ast.unparse(n.args[0]) for n in mma],['k_buf','vt_buf'])
  self.assertEqual([ast.unparse(n.args[1]) for n in mma],['qt_buf','p_buf'])
  for n in mma:
   self.assertEqual({k.arg:ast.literal_eval(k.value) for k in n.keywords}['max_num_imprecise_acc'],0)
  for fn in ['gl.max','gl.sum']:
   n=next(n for n in calls if named(n)==fn)
   self.assertEqual(ast.literal_eval(n.args[1]),0)
  layout=next(n for n in calls if named(n)=='gl.NVMMADistributedLayout')
  self.assertEqual({k.arg:ast.literal_eval(k.value) for k in layout.keywords},dict(version=[3,0],warps_per_cta=[4,1],instr_shape=[16,8,16]))
  zeros=[n for n in calls if named(n)=='gl.zeros']
  self.assertEqual([ast.literal_eval(n.args[0]) for n in zeros],[(64,8),(64,8)])
  text=(HERE/'kernel.py').read_text()
  self.assertIn('q.to(gl.bfloat16).to(gl.float8e4nv).to(gl.bfloat16)',text)
  self.assertIn('p_buf.store(probability.to(gl.bfloat16))',text)
  self.assertIn('denominator = gl.sum(probability, 0)',text)

 def test_offline_driver_free_API_and_fixed_target(self):
  tree=ast.parse((HERE/'offline_compile.py').read_text())
  forbidden={'create_binder','_init_handles','load_binary','get_current_target','get_current_device','n_regs','cuda','synchronize'}
  attrs={n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)}
  self.assertFalse(attrs&forbidden)
  self.assertTrue(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='GluonASTSource' for n in ast.walk(tree)))
  self.assertNotIn('torch',{n.name for n in ast.walk(tree) if isinstance(n,ast.alias)})
  self.assertEqual(c.OPTIONS,dict(num_warps=4,num_stages=3))
  self.assertEqual(c.CONSTANTS,dict(QK_FACTOR=5.4730064863875,SM_SCALE=.125))
  self.assertIn("GPUTarget('cuda',90,32)",(HERE/'offline_compile.py').read_text())

 def test_prepare_binds_source_dependency_and_roundtrip(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);m=c.prepare(p)
   self.assertEqual(c.sha((p/'kernel.py').read_bytes()),c.KERNEL_SHA)
   self.assertEqual(m['helper_sha256'],c.sha((p/'offline_compile.py').read_bytes()))
   self.assertEqual(m['selected_sha256'],c.SELECTED_SHA)
   self.assertEqual(m['dependency']['sha256'],'a9c404c69ed4a39e8ec632eaf6b9fe058a060bf98979c177f6ef666f06bb8d50')
   self.assertEqual(m['target'],['cuda',90,32])

 def test_static_inventory_and_size_bound(self):
  ptx='wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 a;\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 b;\n'
  counts=c.instruction_summary(ptx,'HGMMA.64x8x16.F32;')
  self.assertEqual(counts['wgmma_m64n8k16_sites'],2)
  self.assertEqual(counts['mma_sync_sites'],0)
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaisesRegex(ValueError,'8MiB'):
    c.bounded_write(Path(d)/'large',b'x'*(8*1024*1024+1))

 def test_nonlinux_compile_refused_before_vendor_import(self):
  if __import__('platform').system()!='Linux':
   with self.assertRaisesRegex(RuntimeError,'Linux'):
    c.compile_payload(Path('/nonexistent'))


class MathematicalIndexContract(unittest.TestCase):
 def test_double_transpose_softmax_and_output_indices(self):
  # Inputs chosen exactly representable in E4M3/BF16. This tests indexing and
  # algebra only; Python arithmetic is not a GPU reduction-order oracle.
  q=[[float(((g*7+d*3)%9)-4)/2 for d in range(64)] for g in range(8)]
  k=[[float(((n*5+d)%7)-3)/2 for d in range(64)] for n in range(64)]
  v=[[float(((n+d*2)%13)-6)/4 for d in range(64)] for n in range(64)]
  for ks,vs in [(1.,1.),(1/64,1/64)]:
   scale=5.4730064863875*.125*ks*math.log2(math.e)
   normal=[[sum(q[g][d]*k[n][d] for d in range(64))*scale for n in range(64)] for g in range(8)]
   reverse=[[sum(k[n][d]*q[g][d] for d in range(64))*scale for g in range(8)] for n in range(64)]
   pn=[];ln=[]
   for g in range(8):
    maximum=max(normal[g]);p=[2**(x-maximum) for x in normal[g]]
    pn.append([bf16(x) for x in p]);ln.append(sum(p))
   maxima=[max(reverse[n][g] for n in range(64)) for g in range(8)]
   pt=[[2**(reverse[n][g]-maxima[g]) for g in range(8)] for n in range(64)]
   lt=[sum(pt[n][g] for n in range(64)) for g in range(8)]
   pt=[[bf16(x) for x in row] for row in pt]
   expected=[[sum(pn[g][n]*v[n][d] for n in range(64))*vs/ln[g] for d in range(64)] for g in range(8)]
   raw=[[sum(v[n][d]*pt[n][g] for n in range(64))*vs/lt[g] for g in range(8)] for d in range(64)]
   stored=[None]*512
   for d in range(64):
    for g in range(8):stored[g*64+d]=raw[d][g]
   self.assertTrue(all(x is not None for x in stored))
   for g in range(8):
    for d in range(64):self.assertAlmostEqual(stored[g*64+d],expected[g][d],places=12)


if __name__=='__main__':unittest.main()
