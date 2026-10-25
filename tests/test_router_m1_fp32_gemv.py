"""P039 source invariants/CPU FP32 model; no native Triton execution."""
import ast
import hashlib
import unittest
import torch
from harness import ROOT,load_module

PARENT=ROOT/'candidates/projection_full_composition.py'
SOURCE=ROOT/'candidates/projection_router_m1_fp32_gemv.py'


def dump(node):
    return ast.dump(node,include_attributes=False)


class RouterM1FP32(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.candidate=load_module(SOURCE,'p039_candidate')
        cls.best=load_module(PARENT,'p039_best')
        cls.oracle=load_module(ROOT/'official/reference.py','p039_oracle')

    def test_parent_and_existing_kernels_unchanged(self):
        self.assertEqual(hashlib.sha256(PARENT.read_bytes()).hexdigest(),
                         'c34c497b8219b7c8e37ee7da754db7444db351a7eb27f04473796fdadc1a57f7')
        def functions(path):
            return {n.name:n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.FunctionDef)}
        old,new=functions(PARENT),functions(SOURCE)
        for name in old.keys()-{'prepare','kernel_fn'}:
            self.assertEqual(dump(old[name]),dump(new[name]),name)
        added=new['prepare'].body.pop(1)
        self.assertEqual(ast.unparse(added),'state[\'router_fp32_rows\'] = weights[\'router\'].contiguous()')
        self.assertEqual(dump(old['prepare']),dump(new['prepare']))
        body=new['kernel_fn'].body
        changed=[n for n in body if isinstance(n,ast.If) and ast.unparse(n.test)=='m == 1']
        self.assertEqual(len(changed),1)
        body.remove(changed[0])
        self.assertEqual(dump(old['kernel_fn']),dump(new['kernel_fn']))

    def test_original_fp32_contiguous_state_preserves_every_value(self):
        gen=torch.Generator().manual_seed(39001)
        w=torch.randn((37,262),generator=gen)[:,::2]
        original=w.clone();state=self.candidate.prepare({'router':w})
        self.assertEqual(state['router_fp32_rows'].dtype,torch.float32)
        self.assertTrue(state['router_fp32_rows'].is_contiguous())
        self.assertTrue(torch.equal(state['router_fp32_rows'],original))
        self.assertTrue(torch.equal(w,original))
        self.assertTrue(torch.equal(state['router'],original))

    def test_metadata_gate_and_strided_fallback(self):
        x=torch.randn((1,131)).bfloat16();w=torch.randn((37,131))
        self.assertTrue(self.candidate._router_m1_compatible(x,w,34))
        for bad_x,bad_w,n in [(torch.empty((2,131),dtype=x.dtype),w,34),
                               (torch.empty((1,262),dtype=x.dtype)[:,::2],w,34),
                               (x,w.bfloat16(),34),(x,w,0),(x,w,38),
                               (x,torch.empty((37,262))[:,::2],34)]:
            self.assertFalse(self.candidate._router_m1_compatible(bad_x,bad_w,n))
        backing=torch.randn((1,262)).bfloat16();x=backing[:,::2]
        state=self.candidate.prepare({'router':w})
        data=dict(x=x,config=dict(m=1,n=34,k=131,layer='router'))
        out=self.candidate.kernel_fn(data,state)
        torch.testing.assert_close(out,x.float()@w[:34].T,rtol=0,atol=0)
        self.assertTrue(self.oracle.check_implementation(data,out,{'router':w})[0])

    def test_full_width_sparse_probes_fp32_exactness(self):
        gen=torch.Generator().manual_seed(39002)
        w=torch.randn((256,8192),generator=gen)/8192**.5
        state=self.candidate.prepare({'router':w})
        for col in [0,63,64,511,512,4095,4096,8191]:
            x=torch.zeros((1,8192),dtype=torch.bfloat16);x[0,col]=-128 if col%2 else 8
            data=dict(x=x,config=dict(m=1,n=253,k=8192,layer='router'))
            out=self.candidate.kernel_fn(data,state)
            torch.testing.assert_close(out,(w[:253,col]*x[0,col].float())[None,:],rtol=0,atol=0)
            self.assertTrue(self.oracle.check_implementation(data,out,{'router':w})[0])
        # Canonical-width dense/outlier and zero inputs supplement sparse probes.
        for kind in ['fresh','outlier','zero']:
            x=torch.randn((1,8192),generator=gen).bfloat16()
            if kind=='outlier':x[0,127]*=128
            if kind=='zero':x.zero_()
            data=dict(x=x,config=dict(m=1,n=256,k=8192,layer='router'))
            out=self.candidate.kernel_fn(data,state)
            ok,msg=self.oracle.check_implementation(data,out,{'router':w})
            self.assertTrue(ok,(kind,msg))

    def test_fresh_outliers_zeros_mixed_sign_k_tails_n_prefixes(self):
        gen=torch.Generator().manual_seed(39003)
        for k in [1,3,63,64,65,127,131,8191,8192]:
            w=torch.randn((37,k),generator=gen)/k**.5
            # Preserve non-BF16 residuals: BF16-only weights must not silently pass.
            w[0,0]=1.001;state=self.candidate.prepare({'router':w});before=w.clone()
            for n in [1,34,37]:
                for kind in ['fresh','outlier','zero','mixed_sign']:
                    x=torch.randn((1,k),generator=gen).bfloat16()
                    if kind=='outlier':x[0,0]*=128
                    if kind=='zero':x.zero_()
                    if kind=='mixed_sign' and k>1:x[0,1]=-x[0,0]
                    snapshot=x.clone();data=dict(x=x,config=dict(m=1,n=n,k=k,layer='router'))
                    out=self.candidate.kernel_fn(data,state)
                    ok,msg=self.oracle.check_implementation(data,out,{'router':w})
                    self.assertTrue(ok,(k,n,kind,msg))
                    self.assertEqual(out.dtype,torch.float32)
                    self.assertTrue(torch.equal(x,snapshot))
            self.assertTrue(torch.equal(w,before))
        w=torch.tensor([[1.001,0.25,-0.5],[2.003,-0.25,0.5]],dtype=torch.float32)
        x=torch.tensor([[8,0,0]],dtype=torch.bfloat16)
        data=dict(x=x,config=dict(m=1,n=2,k=3,layer='router'))
        out=self.candidate.kernel_fn(data,self.candidate.prepare({'router':w}))
        self.assertTrue(self.oracle.check_implementation(data,out,{'router':w})[0])
        self.assertFalse(self.oracle.check_implementation(data,(x@w.bfloat16().T).float(),{'router':w})[0])

    def test_other_dispatch_matches_parent_exactly(self):
        gen=torch.Generator().manual_seed(39004)
        weights={'router':torch.randn((31,127),generator=gen),
                 **{layer:torch.randn((37,127),generator=gen).bfloat16() for layer in ['pooler','qkv_gate','unpooler','o']}}
        state=self.candidate.prepare(weights);best=self.best.prepare(weights)
        for layer in weights:
            for m in [1,2,7,8,9,16,32,128,256]:
                if layer=='router' and m==1:continue
                x=torch.randn((m,127),generator=gen).bfloat16();n=29 if layer=='router' else 34
                data=dict(x=x,config=dict(m=m,n=n,k=127,layer=layer))
                torch.testing.assert_close(self.candidate.kernel_fn(data,state),self.best.kernel_fn(data,best),rtol=0,atol=0)
