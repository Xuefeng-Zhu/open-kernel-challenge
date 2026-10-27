"""CPU metadata boundaries and immutable controls, not Triton execution."""
import unittest
import torch
from harness import ROOT,load_module


class QKVDispatch(unittest.TestCase):
    def test_nearby_rows_and_noncontiguous_inputs(self):
        candidate=load_module(ROOT/'candidates/projection_qkv_m128_singlepass.py','m128_cpu_candidate')
        best=load_module(ROOT/'candidates/projection_full_composition.py','m128_cpu_best')
        reference=load_module(ROOT/'official/reference.py','m128_cpu_reference')
        gen=torch.Generator().manual_seed(3400011)
        weights={'qkv_gate':(torch.randn((37,131),generator=gen)/131**.5).bfloat16(),
                 'router':torch.randn((31,131),generator=gen)/131**.5}
        before_w=weights['qkv_gate'].clone()
        state=candidate.prepare(weights);control=best.prepare(weights)
        for m in [127,128,129,256]:
            backing=torch.randn((m,262),generator=gen).bfloat16()
            x=backing[:,::2];x[0].zero_();x[0,130]=8
            data=dict(x=x,config=dict(m=m,n=34,k=131,layer='qkv_gate'))
            before=x.clone();out=candidate.kernel_fn(data,state)
            ok,message=reference.check_implementation(data,out,weights)
            self.assertTrue(ok,message)
            if m!=128:torch.testing.assert_close(out,best.kernel_fn(data,control),rtol=0,atol=0)
            self.assertTrue(torch.equal(x,before))
        self.assertTrue(torch.equal(weights['qkv_gate'],before_w))
