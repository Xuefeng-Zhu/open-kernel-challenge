"""P064 actual integer-decode AST on CPU; no Triton/GPU execution."""
import ast
import copy
import hashlib
from pathlib import Path
import unittest
import torch
from projection_weight_codec import encode_weights
import projection_p064_offline_compile as probe

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'candidates/projection_pooler_lossless_codec.py'
CANDIDATE=ROOT/'candidates/projection_pooler_codec_kn.py'


def functions(path):
    return {n.name:n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.FunctionDef)}


def decoder(path):
    fn=functions(path)['_pooler_codec_partial']
    branch=next(n for n in ast.walk(fn) if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='COMPRESSED')
    # Execute source through uint16 reconstruction; bitcast is checked separately.
    body=[]
    for stmt in branch.body:
        body.append(copy.deepcopy(stmt))
        if isinstance(stmt,ast.Assign) and stmt.targets[0].id=='bits':break
    module=ast.fix_missing_locations(ast.Module(body=body,type_ignores=[]))
    return compile(module,str(path),'exec')


class Pointer:
    def __init__(self,values,offset=0):self.values,self.offset=values.flatten(),offset
    def __add__(self,offset):return Pointer(self.values,self.offset+offset)


class CPUOps:
    int32=torch.int32
    uint16=torch.uint16
    @staticmethod
    def load(pointer,mask,other=0):
        offsets,mask=torch.broadcast_tensors(torch.as_tensor(pointer.offset),mask)
        valid=offsets[mask]
        if valid.numel():
            assert int(valid.min())>=0 and int(valid.max())<pointer.values.numel()
        safe=torch.where(mask,offsets,0).long()
        return torch.where(mask,pointer.values[safe],torch.as_tensor(other,dtype=pointer.values.dtype))
    @staticmethod
    def maximum(values,minimum):return values.clamp_min(minimum)
    where=staticmethod(torch.where)


def source_decode(path,encoded,start_n,start_k,bn,bk,wn=None):
    n,k=encoded['original_shape'];wn=n if wn is None else wn
    escape=encoded['escape_exponents']
    if escape.numel()==0:escape=torch.zeros((1,256),dtype=torch.uint8)
    ns=start_n+torch.arange(bn);ks=start_k+torch.arange(bk)
    env=dict(tl=CPUOps(),ns=ns,k=ks,N=n,K=k,WN=wn,KP=((k+255)//256)*256,start=start_k,
             SM=Pointer(encoded['sign_mantissa']),DELTA=Pointer(encoded['exponent_delta']),
             BASE=Pointer(encoded['exponent_base'].t().contiguous()),
             ESC_IDX=Pointer(encoded['escape_index'].t().contiguous()),ESC_EXP=Pointer(escape),
             valid=(ns[:,None]<n)&(ks[None,:]<k))
    exec(decoder(path),{'__builtins__':{}},env)
    bits=torch.where(env['valid'],env['bits'].to(torch.int32),0)
    return bits,env['valid']


class LayoutRepair(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(4)

    def test_only_native_operand_expression_changes(self):
        old,new=functions(PARENT),functions(CANDIDATE)
        self.assertEqual(set(old),set(new))
        for name in old.keys()-{'_pooler_codec_partial'}:
            self.assertEqual(ast.dump(old[name]),ast.dump(new[name]),name)
        self.assertEqual(hashlib.sha256(PARENT.read_bytes()).hexdigest(),
            '9598c7d1ebe6373b56f9ced2c0405b855f3a6895628f771c2eb8af314003b837')
        f=new['_pooler_codec_partial']
        calls=[n for n in ast.walk(f) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='dot']
        self.assertEqual([ast.unparse(c) for c in calls],['tl.dot(x, w, acc)','tl.dot(x, tl.trans(w), acc)'])
        self.assertEqual(probe.CONSTANTS,dict(M=16,N=4096,K=8192,WN=4096,KP=8192,S=8,BM=16,BN=64,BK=64))
        self.assertEqual(probe.OPTIONS,dict(num_warps=4,num_stages=2))

    def test_actual_integer_decode_all_bits_normal_and_escape(self):
        bits=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(torch.bfloat16).reshape(256,256)
        for weights in (bits,bits.t()):
            encoded=encode_weights(weights)
            expected=weights.contiguous().view(torch.int16).to(torch.int32)&65535
            restored=torch.zeros((256,256),dtype=torch.int32)
            for ni in range(0,256,64):
                for ki in range(0,256,64):
                    old,_=source_decode(PARENT,encoded,ni,ki,64,64)
                    new,valid=source_decode(CANDIDATE,encoded,ni,ki,64,64)
                    self.assertTrue(torch.equal(old.t(),new))
                    self.assertTrue(torch.all(valid))
                    restored[ni:ni+64,ki:ki+64]=new.t()
            self.assertTrue(torch.equal(restored,expected))
            # View rather than conversion preserves every NaN payload and signed zero.
            self.assertTrue(torch.equal(restored.to(torch.int16).view(torch.bfloat16).view(torch.int16),weights.contiguous().view(torch.int16)))

    def test_rectangular_axes_mixed_escapes_tails_and_empty_partitions(self):
        gen=torch.Generator().manual_seed(64064)
        for n,k in ((3,1),(19,65),(67,257),(17,513)):
            values=torch.randint(0,65536,(n,k),generator=gen,dtype=torch.int32)
            # Include compressible lanes alongside arbitrary exponent escape lanes.
            values[::2]=(values[::2]&0x807f)|(((values[::2]%4)+120)<<7)
            weights=values.to(torch.int16).view(torch.bfloat16);encoded=encode_weights(weights)
            for bn,bk in ((16,64),(64,32)):
                for ni in range(0,n+bn,bn):
                    for ki in range(0,k+512,64):
                        old,_=source_decode(PARENT,encoded,ni,ki,bn,bk)
                        new,valid=source_decode(CANDIDATE,encoded,ni,ki,bn,bk)
                        self.assertEqual(tuple(new.shape),(bk,bn))
                        self.assertTrue(torch.equal(old.t(),new),(n,k,ni,ki,bn,bk))
                        self.assertTrue(torch.all(new[~valid]==0))

    def test_frozen_compile_payload_hash_and_no_driver_calls(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            manifest=probe.prepare(Path(tmp))
            self.assertEqual(manifest['sources'][CANDIDATE.name],hashlib.sha256(CANDIDATE.read_bytes()).hexdigest())
        tree=ast.parse(Path(probe.__file__).read_text())
        attrs=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        for forbidden in ('create_binder','_init_handles','warmup','get_current_target'):
            self.assertNotIn(forbidden,attrs)


if __name__=='__main__':unittest.main()
