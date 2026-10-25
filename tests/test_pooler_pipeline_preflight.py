"""P038 CPU models, layout gates and source invariants; never native Triton tests."""
import ast
import hashlib
import unittest
import torch
from harness import ROOT, load_module

PARENT = ROOT / 'candidates/projection_full_composition.py'
FILES = ['projection_pooler_m256_tma.py', 'projection_pooler_m256_copy.py']


def dump(node):
    return ast.dump(node, include_attributes=False)


def functions(tree):
    return {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}


class PipelinePreflight(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.modules = [load_module(ROOT/'candidates'/f, 'pipeline_'+str(i)) for i,f in enumerate(FILES)]
        cls.best = load_module(PARENT, 'pipeline_parent')
        cls.reference = load_module(ROOT/'official/reference.py', 'pipeline_oracle')

    def test_parent_prepare_and_all_untouched_dispatch_are_identical(self):
        self.assertEqual(hashlib.sha256(PARENT.read_bytes()).hexdigest(),
                         'c34c497b8219b7c8e37ee7da754db7444db351a7eb27f04473796fdadc1a57f7')
        parent = functions(ast.parse(PARENT.read_text()))
        for f in FILES:
            candidate = functions(ast.parse((ROOT/'candidates'/f).read_text()))
            for name in parent.keys() - {'kernel_fn'}:
                self.assertEqual(dump(parent[name]), dump(candidate[name]), name)
            body = candidate['kernel_fn'].body
            # New target branch is the only insertion after x/config/m/k setup.
            self.assertIsInstance(body[3], ast.If)
            self.assertIn('_pooler_descriptor_compatible', ast.unparse(body[3].test))
            del body[3]
            self.assertEqual(dump(parent['kernel_fn']), dump(candidate['kernel_fn']))

    def test_tma_and_copy_sources_differ_only_by_switch(self):
        a,b = [(ROOT/'candidates'/f).read_text() for f in FILES]
        self.assertEqual(a.replace('USE_TMA = True', 'USE_TMA = False', 1), b)
        self.assertTrue(self.modules[0].USE_TMA)
        self.assertFalse(self.modules[1].USE_TMA)
        self.assertNotIn('set_allocator', a)
        tree = ast.parse(a)
        imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        self.assertEqual([(n.module, [(v.name,v.asname) for v in n.names]) for n in imports],
                         [('triton.tools.tensor_descriptor', [('TensorDescriptor', None)])])

    def test_metadata_gate_accepts_dense_and_rejects_tails_strides_alignment(self):
        module = self.modules[0]
        w = torch.randn((128,64)).bfloat16()
        x = torch.randn((256,64)).bfloat16()
        self.assertTrue(module._pooler_descriptor_compatible(x,w,64))
        for n in [0,63,65,129,192]:
            self.assertFalse(module._pooler_descriptor_compatible(x,w,n))
        for m in [1,8,16,32,127,128,129,255,257]:
            self.assertFalse(module._pooler_descriptor_compatible(x[:m] if m<=256 else torch.empty((m,64),dtype=x.dtype),w,64))
        for bad_x in [torch.empty((256,128),dtype=x.dtype)[:,::2],
                      torch.empty((256,65),dtype=x.dtype)[:,:64],
                      torch.empty(256*64+1,dtype=x.dtype)[1:].view(256,64)]:
            self.assertFalse(module._pooler_descriptor_compatible(bad_x,w,64))
        self.assertFalse(module._pooler_descriptor_compatible(x,w.float(),64))
        self.assertFalse(module._pooler_descriptor_compatible(x,torch.empty((128,128),dtype=w.dtype)[:,::2],64))
        self.assertFalse(module._pooler_descriptor_compatible(torch.empty((256,65),dtype=x.dtype),torch.empty((128,65),dtype=w.dtype),64))

    def test_fresh_cpu_model_probes_and_zero_rows(self):
        gen = torch.Generator().manual_seed(38001)
        weights = {'pooler':torch.randn((128,64),generator=gen).bfloat16(),
                   'router':torch.randn((31,64),generator=gen)}
        snapshots = {k:v.clone() for k,v in weights.items()}
        for module in self.modules:
            state = module.prepare(weights)
            for seed in [38002,38003]:
                gen.manual_seed(seed)
                x = torch.randn((256,64),generator=gen).bfloat16()
                x[0].zero_(); x[0,63] = 8
                x[1].zero_(); x[2,0] *= 64
                before = x.clone()
                data = dict(x=x,config=dict(m=256,n=64,k=64,layer='pooler'))
                out = module.kernel_fn(data,state)
                ok,message = self.reference.check_implementation(data,out,weights)
                self.assertTrue(ok,message)
                torch.testing.assert_close(out[0],weights['pooler'][:64,63]*8,rtol=0,atol=0)
                self.assertTrue(torch.equal(x,before))
                self.assertEqual(out.dtype,torch.bfloat16)
        for k,w in weights.items(): self.assertTrue(torch.equal(w,snapshots[k]))

    def test_nearby_tail_and_strided_fallback_matches_parent_exactly(self):
        gen = torch.Generator().manual_seed(38004)
        for m,n,k,strided in [(255,64,64,False),(257,64,64,False),
                              (256,63,64,False),(256,64,65,False),(256,64,64,True)]:
            weights = {'pooler':torch.randn((128,k),generator=gen).bfloat16(),
                       'router':torch.randn((31,k),generator=gen)}
            backing = torch.randn((m,k*(2 if strided else 1)),generator=gen).bfloat16()
            x = backing[:,::2] if strided else backing
            data = dict(x=x,config=dict(m=m,n=n,k=k,layer='pooler'))
            expected = self.best.kernel_fn(data,self.best.prepare(weights))
            for module in self.modules:
                torch.testing.assert_close(module.kernel_fn(data,module.prepare(weights)),expected,rtol=0,atol=0)
