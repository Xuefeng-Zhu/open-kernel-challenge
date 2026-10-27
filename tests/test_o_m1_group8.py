"""P040 CPU mathematical, metadata and source gates; not native Triton tests."""
import ast
import hashlib
import unittest
import torch
from harness import ROOT,load_module

PARENT=ROOT/'candidates/projection_router_m1_fp32_gemv.py'
SOURCE=ROOT/'candidates/projection_o_m1_group8.py'


def dump(node):
    return ast.dump(node,include_attributes=False)


class OGroup8(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.candidate=load_module(SOURCE,'p040_candidate')
        cls.best=load_module(PARENT,'p040_parent')
        cls.oracle=load_module(ROOT/'official/reference.py','p040_oracle')

    def test_preserves_parent_helpers_prepare_and_other_dispatch(self):
        self.assertEqual(hashlib.sha256(PARENT.read_bytes()).hexdigest(),
                         '0c0c41119ef925b03a33ccab230bc64176cc22c242e3fea77ce01e317376defc')
        def functions(path):
            return {n.name:n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.FunctionDef)}
        old,new=functions(PARENT),functions(SOURCE)
        for name in old.keys()-{'prepare','kernel_fn'}:
            self.assertEqual(dump(old[name]),dump(new[name]),name)
        added=new['prepare'].body.pop(1)
        self.assertEqual(ast.unparse(added.test),"'o' in weights")
        self.assertEqual(dump(old['prepare']),dump(new['prepare']))
        added=new['kernel_fn'].body.pop(3)
        self.assertEqual(ast.unparse(added.test),"layer == 'o' and m == 1")
        self.assertEqual(dump(old['kernel_fn']),dump(new['kernel_fn']))

    def test_lossless_original_bf16_weight_state(self):
        gen=torch.Generator().manual_seed(40001)
        w=torch.randn((37,262),generator=gen).bfloat16()[:,::2]
        before=w.clone();state=self.candidate.prepare({'o':w,'router':torch.ones((2,131))})
        self.assertEqual(state['o_rows'].dtype,torch.bfloat16)
        self.assertTrue(state['o_rows'].is_contiguous())
        self.assertTrue(torch.equal(state['o_rows'],before))
        self.assertTrue(torch.equal(w,before))
        self.assertIs(state['o'],w)

    def test_metadata_gate_and_exact_library_fallback(self):
        x=torch.randn((1,131)).bfloat16();w=torch.randn((37,131)).bfloat16()
        self.assertTrue(self.candidate._o_m1_compatible(x,w,34))
        self.assertFalse(self.candidate._o_m1_compatible(x,w,0))
        self.assertFalse(self.candidate._o_m1_compatible(x,w,38))
        self.assertFalse(self.candidate._o_m1_compatible(x,w.float(),34))
        self.assertFalse(self.candidate._o_m1_compatible(torch.empty((2,131),dtype=x.dtype),w,34))
        self.assertFalse(self.candidate._o_m1_compatible(x,torch.empty((37,262),dtype=w.dtype)[:,::2],34))
        for k,strided in [(1025,False),(2048,False),(131,True)]:
            backing=torch.randn((1,k*(2 if strided else 1))).bfloat16()
            x=backing[:,::2] if strided else backing
            w=torch.randn((37,k)).bfloat16();weights={'o':w,'router':torch.ones((2,k))}
            self.assertFalse(self.candidate._o_m1_compatible(x,w,34))
            data=dict(x=x,config=dict(m=1,n=34,k=k,layer='o'))
            out=self.candidate.kernel_fn(data,self.candidate.prepare(weights))
            expected=self.best.kernel_fn(data,self.best.prepare(weights))
            torch.testing.assert_close(out,expected,rtol=0,atol=0)

    def test_k_n_tails_fresh_probe_zero_and_outlier_rows(self):
        gen=torch.Generator().manual_seed(40002)
        for k in [1,3,63,64,65,127,511,512,513,1023,1024]:
            w=torch.randn((37,k),generator=gen).bfloat16();snapshot=w.clone()
            weights={'o':w,'router':torch.ones((2,k))};state=self.candidate.prepare(weights)
            for n in [1,7,8,9,34,37]:
                for kind in ['fresh','probe','zero','outlier']:
                    x=torch.randn((1,k),generator=gen).bfloat16()
                    if kind=='probe':x.zero_();x[0,k-1]=-128
                    if kind=='zero':x.zero_()
                    if kind=='outlier':x[0,k//2]*=30
                    before=x.clone();data=dict(x=x,config=dict(m=1,n=n,k=k,layer='o'))
                    out=self.candidate.kernel_fn(data,state)
                    ok,msg=self.oracle.check_implementation(data,out,weights)
                    self.assertTrue(ok,(k,n,kind,msg))
                    self.assertEqual(out.dtype,torch.bfloat16)
                    self.assertTrue(torch.equal(x,before))
            self.assertTrue(torch.equal(w,snapshot))

    def test_canonical_dense_outlier_zero_and_exact_sparse_probes(self):
        gen=torch.Generator().manual_seed(40003)
        w=(torch.randn((8192,1024),generator=gen)/32).bfloat16()
        weights={'o':w,'router':torch.ones((2,1024))};state=self.candidate.prepare(weights)
        for n in [8192,8189]:
            for kind in ['fresh','outlier','zero']:
                x=torch.randn((1,1024),generator=gen).bfloat16()
                if kind=='outlier':x[0,17]*=30
                if kind=='zero':x.zero_()
                data=dict(x=x,config=dict(m=1,n=n,k=1024,layer='o'))
                ok,msg=self.oracle.check_implementation(data,self.candidate.kernel_fn(data,state),weights)
                self.assertTrue(ok,(n,kind,msg))
            for col in [0,63,64,511,512,1023]:
                x=torch.zeros((1,1024),dtype=torch.bfloat16);x[0,col]=-128 if col%2 else 8
                data=dict(x=x,config=dict(m=1,n=n,k=1024,layer='o'))
                out=self.candidate.kernel_fn(data,state)
                torch.testing.assert_close(out,(w[:n,col]*x[0,col])[None,:],rtol=0,atol=0)
                self.assertTrue(self.oracle.check_implementation(data,out,weights)[0])

    def test_other_layers_and_o_m2_plus_match_p039_exactly(self):
        gen=torch.Generator().manual_seed(40004)
        weights={'router':torch.randn((31,127),generator=gen),
                 **{layer:torch.randn((37,127),generator=gen).bfloat16() for layer in ['o','pooler','qkv_gate','unpooler']}}
        state=self.candidate.prepare(weights);parent=self.best.prepare(weights)
        for layer in weights:
            for m in [1,2,7,8,9,16,32,128,256]:
                if layer=='o' and m==1:continue
                x=torch.randn((m,127),generator=gen).bfloat16();n=29 if layer=='router' else 34
                data=dict(x=x,config=dict(m=m,n=n,k=127,layer=layer))
                torch.testing.assert_close(self.candidate.kernel_fn(data,state),self.best.kernel_fn(data,parent),rtol=0,atol=0)
