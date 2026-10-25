"""P043 source/CPU correctness gates. No native compilation or CUDA evidence."""
import ast
import hashlib
import unittest
from unittest.mock import patch
import torch
from harness import ROOT, load_module

SOURCE=ROOT/'candidates/projection_pooler_lossless_codec.py'
CONTROL=ROOT/'candidates/projection_pooler_lossless_raw_control.py'
PARENT=ROOT/'candidates/projection_router_m1_fp32_gemv.py'

class PoolerCodec(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.c=load_module(SOURCE,'p043')
        cls.raw=load_module(CONTROL,'p043_raw')
        cls.parent=load_module(PARENT,'p043_parent')
        cls.oracle=load_module(ROOT/'official/reference.py','p043_oracle')

    def weights(self,k=1024,n=67,seed=43001):
        g=torch.Generator().manual_seed(seed)
        return {'router':torch.randn((31,k),generator=g)/k**.5,
                **{layer:(torch.randn((n,k),generator=g)/k**.5).bfloat16()
                   for layer in ['pooler','qkv_gate','unpooler','o']}}

    def test_parent_immutable_other_paths_exact_ast(self):
        self.assertEqual(hashlib.sha256(PARENT.read_bytes()).hexdigest(),
          '0c0c41119ef925b03a33ccab230bc64176cc22c242e3fea77ce01e317376defc')
        def funcs(p):return {n.name:n for n in ast.walk(ast.parse(p.read_text())) if isinstance(n,ast.FunctionDef)}
        old,new=funcs(PARENT),funcs(SOURCE)
        for key in old.keys()-{'prepare','kernel_fn'}:
            self.assertEqual(ast.dump(old[key]),ast.dump(new[key]),key)
        body=new['kernel_fn'].body
        changes=[v for v in body if isinstance(v,ast.If) and 'pooler_codec_efficient' in ast.unparse(v.test)]
        self.assertEqual(len(changes),1);body.remove(changes[0])
        self.assertEqual(ast.dump(old['kernel_fn']),ast.dump(new['kernel_fn']))
        self.assertEqual(SOURCE.read_text().replace('POOLER_COMPRESSED = True','POOLER_COMPRESSED = False',1),CONTROL.read_text())

    def test_metadata_gate(self):
        w=self.weights()['pooler'];x=torch.zeros((8,1024),dtype=torch.bfloat16)
        self.assertTrue(self.c._pooler_codec_compatible(x,w,65))
        for m in [1,33,128,256]:self.assertFalse(self.c._pooler_codec_compatible(torch.empty((m,1024),dtype=x.dtype),w,65))
        for xx,ww,n in [(x.float(),w,65),(x,w.float(),65),(x,w,68),(x,w,0),
                        (torch.zeros((8,2048),dtype=x.dtype)[:,::2],w,65)]:
            self.assertFalse(self.c._pooler_codec_compatible(xx,ww,n))

    def test_static_weight_only_guard_and_mutation(self):
        weights=self.weights();snapshot={k:v.clone() for k,v in weights.items()}
        state=self.c.prepare(weights)
        self.assertTrue(state['pooler_codec_efficient'])
        self.assertTrue(all(torch.equal(v,snapshot[k]) for k,v in weights.items()))
        # All exponent values: exact encoding is preserved, but no reduced-byte native dispatch.
        bits=torch.arange(65536,dtype=torch.int32).reshape(256,256).T.contiguous().to(torch.int16).reshape(64,1024)
        state2=self.c.prepare({'router':weights['router'],'pooler':bits.view(torch.bfloat16)})
        self.assertFalse(state2['pooler_codec_efficient'])
        enc=dict(state2['pooler_codec'])
        for k in ['exponent_base','escape_index']:enc[k]=enc[k].t().contiguous()
        self.assertTrue(torch.equal(self.c.decode_weights(enc).view(torch.int16),bits))
        # Finite, incompressible exponents also take parent's robust raw library path.
        finite=((bits.to(torch.int32)&0x807f)|(((torch.arange(65536).reshape(64,1024)%254)+1)<<7)).to(torch.int16).view(torch.bfloat16)
        state3=self.c.prepare({'router':weights['router'],'pooler':finite})
        self.assertFalse(state3['pooler_codec_efficient'])
        with patch.object(self.c,'_pooler_codec_cpu',side_effect=AssertionError('unexpected codec path')):
            x=torch.zeros((8,1024),dtype=torch.bfloat16)
            self.assertTrue(torch.equal(self.c.kernel_fn(dict(x=x,config=dict(layer='pooler',n=64)),state3),x@finite.T))

    def test_fresh_same_address_aba_sparse_outliers_and_state_immutable(self):
        weights=self.weights();state=self.c.prepare(weights);raw=self.raw.prepare(weights)
        enc_before={k:v.clone() for k,v in state['pooler_codec'].items() if isinstance(v,torch.Tensor)}
        g=torch.Generator().manual_seed(43002)
        for m in [2,7,8,9,16,17,31,32]:
            a=torch.randn((m,1024),generator=g).bfloat16();a[0].zero_();a[0,511]=8
            b=torch.randn((m,1024),generator=g).bfloat16();b[-1,513]*=128
            x=torch.empty_like(a);ptr=x.data_ptr();outputs=[]
            for values in [a,b,a]:
                x.copy_(values);before=x.clone();data=dict(x=x,config=dict(layer='pooler',n=65))
                out=self.c.kernel_fn(data,state);outputs.append(out.clone())
                ok,msg=self.oracle.check_implementation(data,out,weights)
                self.assertTrue(ok,(m,msg));self.assertEqual(x.data_ptr(),ptr)
                self.assertTrue(torch.equal(x,before))
                self.assertTrue(torch.equal(out,self.raw.kernel_fn(data,raw)))
            self.assertTrue(torch.equal(outputs[0],outputs[2]))
        self.assertTrue(all(torch.equal(state['pooler_codec'][k],v) for k,v in enc_before.items()))

    def test_tail_partition_model_and_zero_empty_splits(self):
        for k in [1,63,64,65,255,256,257,513,1025]:
            weights=self.weights(k=k,n=19,seed=43000+k);state=self.c.prepare(weights)
            for m in [2,7,17,32]:
                x=torch.randn((m,k)).bfloat16();x[0].zero_();x[0,k-1]=4
                data=dict(x=x,config=dict(layer='pooler',n=17))
                # Direct model tests codec arithmetic even if padding makes storage guard fail.
                out=self.c._pooler_codec_cpu(x,state,17)
                ok,msg=self.oracle.check_implementation(data,out,weights)
                self.assertTrue(ok,(m,k,msg))
                torch.testing.assert_close(out[0],weights['pooler'][:17,k-1]*4,rtol=0,atol=0)
                zero=self.c._pooler_codec_cpu(torch.zeros_like(x),state,17)
                self.assertEqual(torch.count_nonzero(zero),0)

    def test_untargeted_cpu_paths_identical_and_strided_fallback(self):
        weights=self.weights();state=self.c.prepare(weights);old=self.parent.prepare(weights)
        for layer in weights:
            for m in [1,8,16,32,33,128,256]:
                if layer=='pooler' and 2<=m<=32:continue
                x=torch.randn((m,1024)).bfloat16();n=29 if layer=='router' else 65
                data=dict(x=x,config=dict(layer=layer,n=n))
                self.assertTrue(torch.equal(self.c.kernel_fn(data,state),self.parent.kernel_fn(data,old)),(layer,m))
        x=torch.randn((8,2048)).bfloat16()[:,::2];data=dict(x=x,config=dict(layer='pooler',n=65))
        self.assertTrue(torch.equal(self.c.kernel_fn(data,state),self.parent.kernel_fn(data,old)))

    def test_no_host_work_in_timed_gpu_branch_and_invocation_local_scratch(self):
        tree=ast.parse(SOURCE.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='kernel_fn')
        text=ast.unparse(fn)
        for bad in ['encode_weights(','.item(','.cpu(','.synchronize(','.to(device=\'cpu\')']:
            self.assertNotIn(bad,text)
        self.assertIn('partial = torch.empty((8, m, n)',text)
        self.assertIn('out = torch.empty((m, n)',text)
        self.assertNotIn('state[\'pooler_codec\'] =',text)
