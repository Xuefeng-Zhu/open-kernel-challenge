"""CPU adapter/dispatch/oracle tests with stand-ins, not FA3 or CUDA execution."""
import ast
import copy
import importlib.util
from pathlib import Path
import sys
import unittest
import warnings

warnings.filterwarnings('ignore',message='Failed to initialize NumPy')
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE/'official'))
import modal_worker as buffers
spec=importlib.util.spec_from_file_location('mixed_reference',HERE/'official/reference.py')
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)


def functions():
    tree=ast.parse((HERE/'mixed_fa3.py').read_text())
    return [n for n in tree.body if isinstance(n,ast.FunctionDef)
            and n.name in {'_fa3_norm','_fa3_full_verify','kernel_fn'}]


def cpu_adapter(data,trace):
    """Exercise our host adapter using independent small CPU stand-ins."""
    cfg=data['config'];hq,hk,d=[cfg[k] for k in ['num_q_heads','num_kv_heads','head_dim']]
    class Norm:
        def __init__(self,**kw):
            assert kw==dict(head_dim=d,num_q_heads=hq,num_kv_heads=hk,eps=1e-5)
        def __call__(self,x):
            trace.append('norm')
            q=x[:,:hq*d].reshape(-1,hq,d).float()
            k=x[:,hq*d:(hq+hk)*d].reshape(-1,hk,d).float()
            rms=lambda y:(y*torch.rsqrt((y*y).mean(-1,keepdim=True)+1e-5)).to(torch.bfloat16)
            return rms(q),rms(k)
    def store(k,v,kc,vc,loc,*,reserved_skip_index):
        assert reserved_skip_index==-1
        assert k.dtype==v.dtype==kc.dtype==vc.dtype==torch.uint8
        kc[loc.long()]=k;vc[loc.long()]=v;trace.append('store')
    def attention(**kw):
        assert kw['q'].dtype==torch.float8_e4m3fn
        assert kw['causal'] and kw['window_size']==(-1,-1)
        assert kw['max_seqlen_q']==cfg['q_len']
        assert kw['softmax_scale']==cfg['sm_scale']*cfg['qk_scale_factor']
        for key in ['k_cache','v_cache','page_table','cache_seqlens','cu_seqlens_q','scheduler_metadata']:
            assert kw[key] is data[key]
        if cfg['kv_scale']!=1:
            assert torch.equal(kw['k_descale'],data['k_scale'].expand(cfg['batch'],hk))
            assert torch.equal(kw['v_descale'],data['v_scale'].expand(cfg['batch'],hk))
        else:assert 'k_descale' not in kw and 'v_descale' not in kw
        b,l=cfg['batch'],cfg['q_len'];page=cfg['page_size'];g=hq//hk
        q=kw['q'].float().reshape(b,l,hk,g,d)
        kc=kw['k_cache'].view(-1,hk,d).float()*data['k_scale']
        vc=kw['v_cache'].view(-1,hk,d).float()*data['v_scale']
        out=torch.empty((b,l,hq,d),dtype=torch.bfloat16)
        for bi in range(b):
            for j in range(l):
                end=int(data['cache_seqlens'][bi])-l+j+1
                pos=torch.arange(end)
                rows=data['page_table'][bi].long()[pos//page]*page+pos%page
                score=torch.einsum('hgd,thd->hgt',q[bi,j],kc[rows])*kw['softmax_scale']
                out[bi,j]=torch.einsum('hgt,thd->hgd',score.softmax(-1),vc[rows]).reshape(hq,d).to(torch.bfloat16)
        trace.append('attention')
        return (out.reshape(b*l,hq,d),None)
    def finish(a,gate,weight,**kw):
        assert a.dtype==torch.bfloat16 and weight is None
        assert kw==dict(head_dim=d,eps=1e-5,has_norm=True)
        x=a.reshape(-1,hq,d).float()
        x=x*torch.rsqrt((x*x).mean(-1,keepdim=True)+1e-5)
        trace.append('gate')
        return (x*torch.sigmoid(gate.float().reshape_as(x))).reshape_as(a).to(torch.bfloat16)
    ns=dict(torch=torch,Inputs=dict,RMS_EPS=1e-5,_FA3_NORMS={},FusedQKNorm=Norm,
            store_cache=store,flash_attn_with_kvcache=attention,rms_norm_gate=finish)
    exec(compile(ast.Module(body=functions(),type_ignores=[]),'own_adapter_cpu','exec'),ns)
    return ns


def zero_new_page(data):
    """Swap physical pages bijectively, making the first new cache slot zero."""
    page=data['config']['page_size'];old=int(data['cache_loc'][0])//page
    assert int(data['cache_loc'][0])%page==0
    for key in ['k_cache','v_cache']:
        v=data[key].view(torch.uint8);a=v[0].clone();v[0]=v[old];v[old]=a
    pt=data['page_table'];data['page_table']=torch.where(pt==old,0,torch.where(pt==0,old,pt)).to(pt.dtype)
    data['cache_loc']=reference.new_slots(data).reshape(-1)
    assert int(data['cache_loc'][0])==0


class MixedFA3(unittest.TestCase):
    def test_e2_all_functions_unchanged_except_entrypoint_name(self):
        base=ast.parse((HERE/'final_attention.py').read_text());new=ast.parse((HERE/'mixed_fa3.py').read_text())
        old={n.name:n for n in base.body if isinstance(n,ast.FunctionDef)}
        current={n.name:n for n in new.body if isinstance(n,ast.FunctionDef)}
        for name,node in old.items():
            actual=copy.deepcopy(current['_e2_kernel_fn' if name=='kernel_fn' else name]);actual.name=name
            self.assertEqual(ast.dump(node),ast.dump(actual))

    def test_dispatch_only_full_verify_16_32(self):
        entry=next(n for n in functions() if n.name=='kernel_fn')
        ns=dict(torch=torch,Inputs=dict,_fa3_full_verify=lambda _: 'fa3',_e2_kernel_fn=lambda _:'e2')
        exec(compile(ast.Module(body=[entry],type_ignores=[]),'dispatch_cpu','exec'),ns)
        for b in [1,4,8,16,32,33]:
            for l in [1,7,8]:
                for window,rope in [(0,0),(2048,500000),(0,500000)]:
                    cfg=dict(batch=b,q_len=l,window=window,rope_theta=rope)
                    expected='fa3' if b in (16,32) and l==8 and window==rope==0 else 'e2'
                    self.assertEqual(ns['kernel_fn']({'config':cfg}),expected)

    def test_official_call_contract_and_live_slot_zero(self):
        for page,scale in [(1,1),(64,1/64)]:
            fresh=reference.generate_input(batch=16,ctx=page,heads='tp8',layer='full',q_len=8,
                                           page=page,kv_scale=scale,seed=1290+page,device='cpu')
            zero_new_page(fresh);work=buffers.clone_inputs(fresh);trace=[];ns=cpu_adapter(work,trace)
            out=ns['kernel_fn'](work)
            self.assertEqual(trace,['norm','store','attention','gate'])
            self.assertEqual(out.shape,(16,8,1024));self.assertEqual(out.dtype,torch.bfloat16)
            ok,msg=reference.check_implementation(work,reference.collect(work,out));self.assertTrue(ok,msg)
            self.assertTrue(buffers.unchanged(work,fresh)[0])

    def test_cpu_in_place_A_B_A_with_changed_lengths(self):
        for b,page,scale in [(16,1,1),(32,64,1/64)]:
            args=dict(batch=b,ctx=32,heads='tp8',layer='full',q_len=8,page=page,kv_scale=scale,
                      spread='mixed',device='cpu')
            a=reference.generate_input(**args,seed=803);z=reference.generate_input(**args,seed=804)
            self.assertFalse(torch.equal(a['seq_lens'],z['seq_lens']))
            work=buffers.clone_inputs(a);ptr={k:v.data_ptr() for k,v in work.items() if isinstance(v,torch.Tensor)}
            trace=[];ns=cpu_adapter(work,trace);first=None
            for i,fresh in enumerate([a,z,a]):
                buffers.restore(work,fresh);out=ns['kernel_fn'](work);raw=reference.collect(work,out)
                ok,msg=reference.check_implementation(work,raw);self.assertTrue(ok,msg)
                self.assertTrue(buffers.unchanged(work,fresh)[0])
                self.assertTrue(all(work[k].data_ptr()==v for k,v in ptr.items()))
                if i==0:first=raw.clone()
                if i==2:self.assertTrue(torch.equal(first,raw))
            self.assertEqual(trace.count('norm'),3)


if __name__=='__main__':unittest.main(verbosity=2)
