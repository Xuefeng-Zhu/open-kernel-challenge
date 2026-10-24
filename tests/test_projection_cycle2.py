"""CPU contract gates for cycle2 projections; does not execute CUDA/Triton."""
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from harness import ROOT,load_module
reference=load_module(ROOT/'official/reference.py','cycle2_reference')
control=load_module(ROOT/'candidates/router_pooler_hybrid.py','cycle2_best_control')
SPECS=[('projection_pooler_layout.py','pooler',8192,4096),
       ('projection_qkv_m1_simt.py','qkv_gate',8192,2304),
       ('projection_unpooler_m1_simt.py','unpooler',4096,1024),
       ('projection_native_composition.py','qkv_gate',8192,2304),
       ('projection_native_composition.py','unpooler',4096,1024),
       ('projection_native_composition.py','pooler',8192,4096),
       ('projection_full_composition.py','qkv_gate',8192,2304),
       ('projection_full_composition.py','unpooler',4096,1024),
       ('projection_full_composition.py','pooler',8192,4096),
       ('projection_qkv_layout.py','qkv_gate',8192,2304)]
requested=os.environ.get('PROJECTION_CANDIDATE')


class ProjectionCycle2Mixin:
    def test_fresh_inputs_tails_and_boundaries(self):
        gen=torch.Generator().manual_seed(39181)
        w=(torch.randn((37,131),generator=gen)/131**.5).bfloat16()
        original=w.clone()
        weights={self.layer:w,'router':torch.randn((31,131),generator=gen)/131**.5}
        state=self.candidate.prepare(weights)
        for m in [1,2,7,8,9,16,32,128,256]:
            with self.subTest(m=m):
                x=torch.randn((m,131),generator=gen).bfloat16()
                x[:,0]*=16
                x[0].zero_();x[0,64]=8
                if m>1:x[1].zero_()
                before=x.clone()
                data=dict(x=x,config=dict(layer=self.layer,m=m,n=34,k=131))
                out=self.candidate.kernel_fn(data,state)
                ok,message=reference.check_implementation(data,out,weights)
                self.assertTrue(ok,message)
                self.assertEqual(out.dtype,torch.bfloat16)
                self.assertTrue(torch.equal(x,before))
        self.assertTrue(torch.equal(w,original))

    def test_real_width_probes_and_output_slice(self):
        gen=torch.Generator().manual_seed(39182)
        w=(torch.randn((self.n,self.k),generator=gen)/self.k**.5).bfloat16()
        original=w.clone()
        weights={self.layer:w,'router':torch.ones((2,self.k))}
        state=self.candidate.prepare(weights)
        cols=[0,63,64,511,512,self.k//2-1,self.k//2,self.k-1]
        x=torch.zeros((8,self.k),dtype=torch.bfloat16)
        for row,col in enumerate(cols):x[row,col]=8 if row%2 else -128
        n=self.n-3
        expected=torch.stack([w[:n,col]*(8 if row%2 else -128) for row,col in enumerate(cols)])
        for rows in [8,1]:
            for first in (range(8) if rows==1 else [0]):
                value=x[first:first+rows].clone()
                data=dict(x=value,config=dict(layer=self.layer,m=rows,n=n,k=self.k))
                out=self.candidate.kernel_fn(data,state)
                torch.testing.assert_close(out,expected[first:first+rows],rtol=0,atol=0)
        self.assertTrue(torch.equal(w,original))

    def test_verified_router_and_other_projection_preserved(self):
        gen=torch.Generator().manual_seed(39183)
        weights={'router':torch.randn((31,127),generator=gen)/127**.5,
                 self.layer:torch.randn((37,127),generator=gen).bfloat16(),
                 'o':torch.randn((37,127),generator=gen).bfloat16()}
        state=self.candidate.prepare(weights);baseline=control.prepare(weights)
        for layer in ['router','o']:
            for m in [1,8,9,32]:
                x=torch.randn((m,127),generator=gen).bfloat16()
                n=31 if layer=='router' else 34
                data=dict(x=x,config=dict(layer=layer,m=m,n=n,k=127))
                out=self.candidate.kernel_fn(data,state)
                torch.testing.assert_close(out,control.kernel_fn(data,baseline),rtol=0,atol=0)
                ok,message=reference.check_implementation(data,out,weights)
                self.assertTrue(ok,message)


for filename,layer,k,n in SPECS:
    if requested and filename!=Path(requested).name:continue
    module=load_module(ROOT/'candidates'/filename,'cycle2_'+Path(filename).stem)
    globals()['Test_'+Path(filename).stem+'_'+layer]=type('Test_'+Path(filename).stem+'_'+layer,
        (ProjectionCycle2Mixin,unittest.TestCase),dict(candidate=module,layer=layer,k=k,n=n))
