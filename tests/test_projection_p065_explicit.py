"""P065 CPU source interpreter and exact-codec checks; no GPU evidence."""
import ast
import copy
import hashlib
import tempfile
from pathlib import Path
import unittest
import torch
from harness import load_module,ROOT
from projection_weight_codec import encode_weights
from tests.test_projection_p064_layout import CPUOps,Pointer,source_decode,CANDIDATE as KN_SOURCE
import projection_p065_offline_compile as probe

SOURCE=ROOT/'projection_p065_explicit_operand.py'


def function(path):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef))


class CPUProgram(CPUOps):
    float32=torch.float32
    bfloat16=torch.bfloat16
    def __init__(self,ids):self.ids=ids;self.weight_operands=[]
    def program_id(self,axis):return self.ids[axis]
    @staticmethod
    def arange(start,end,layout=None):return torch.arange(start,end)
    @staticmethod
    def full(shape,value,dtype,layout=None):return torch.full(shape,value,dtype=dtype)
    @staticmethod
    def cdiv(a,b):return (a+b-1)//b
    @staticmethod
    def convert_layout(tensor,layout):return tensor
    @staticmethod
    def bitcast(tensor,dtype):return tensor.view(dtype)
    def mma(self,a,b,acc):
        self.weight_operands.append(b.view(torch.int16).clone())
        return acc+a.float()@b.float()
    @staticmethod
    def store(pointer,values,mask):
        offsets,mask,values=torch.broadcast_tensors(torch.as_tensor(pointer.offset),mask,values)
        selected=offsets[mask].long()
        if selected.numel():
            assert int(selected.min())>=0 and int(selected.max())<pointer.values.numel()
            assert selected.unique().numel()==selected.numel()
        pointer.values[selected]=values[mask]


def interpreted(ops):
    class StripLayouts(ast.NodeTransformer):
        def visit_FunctionDef(self,node):
            node.decorator_list=[]
            for arg in node.args.args:arg.annotation=None
            return self.generic_visit(node)
        def visit_AnnAssign(self,node):
            # All annotated local values in this kernel are static layout descriptors.
            assert node.target.id in {'mma','a_layout','b_layout','store_layout'}
            return ast.Assign(targets=[node.target],value=ast.Constant(None))
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if isinstance(node.func,ast.Attribute) and node.func.attr=='to' and any(k.arg=='bitcast' for k in node.keywords):
                return ast.Call(func=ast.Attribute(ast.Name('gl',ast.Load()),'bitcast',ast.Load()),args=[node.func.value,node.args[0]],keywords=[])
            return node
    fn=StripLayouts().visit(copy.deepcopy(function(SOURCE)))
    env={'gl':ops,'mma_v2':ops.mma,'SliceLayout':lambda *args:None,'__builtins__':{'range':range}}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(SOURCE),'exec'),env)
    return env[probe.KERNEL]


def pointers(weights):
    encoded=encode_weights(weights);n,k=weights.shape
    escape=encoded['escape_exponents']
    if escape.numel()==0:escape=torch.zeros((1,256),dtype=torch.uint8)
    return [Pointer(weights),Pointer(encoded['sign_mantissa']),Pointer(encoded['exponent_delta']),
            Pointer(encoded['exponent_base'].t().contiguous()),Pointer(encoded['escape_index'].t().contiguous()),Pointer(escape)]


class ExplicitLayout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.oracle=load_module(ROOT/'official/reference.py','p065_reference')

    def test_compressed_integer_expression_matches_tested_kn_source(self):
        def branch(fn):
            return next(n for n in ast.walk(fn) if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='COMPRESSED')
        old_fn=next(n for n in ast.walk(ast.parse(KN_SOURCE.read_text())) if isinstance(n,ast.FunctionDef) and n.name==probe.KERNEL)
        old=copy.deepcopy(branch(old_fn).body)
        new=copy.deepcopy(branch(function(SOURCE)).body)
        # Old branch includes its valid definition and dot; P065 keeps both common.
        old=[n for n in old if not (isinstance(n,ast.Assign) and n.targets[0].id in {'valid','acc'})]
        class Alias(ast.NodeTransformer):
            def visit_Name(self,node):
                if node.id=='gl':node.id='tl'
                return node
        new=[Alias().visit(n) for n in new]
        self.assertEqual([ast.dump(n) for n in old],[ast.dump(n) for n in new])

    def test_source_interpreter_all_bits_normal_and_escape(self):
        bits=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(torch.bfloat16).reshape(256,256)
        for weights in (bits,bits.t()):
            args=pointers(weights);n,k=weights.shape
            # Rectangular BK32/BN64 checks mapping; numerical outputs may be NaN,
            # but we inspect stored operand bits only for these representation cases.
            for bn,bk in ((64,64),(64,32)):
                for ni in range(0,n,bn):
                    for split in range(8):
                        ops=CPUProgram((ni//bn,split,0));run=interpreted(ops)
                        out=torch.full((8,2,n),float('nan'))
                        run(Pointer(torch.zeros((2,k),dtype=torch.bfloat16)),*args,Pointer(out),2,n,k,n,256,8,16,bn,bk,True)
                        start=split*((k+8*bk-1)//(8*bk))*bk
                        for j,operand in enumerate(ops.weight_operands):
                            ki=start+j*bk;width=max(0,min(bk,k-ki));height=min(bn,n-ni)
                            expected=torch.zeros((bk,bn),dtype=torch.int16)
                            if width:expected[:width,:height]=weights[ni:ni+height,ki:ki+width].contiguous().view(torch.int16).t()
                            self.assertTrue(torch.equal(operand,expected))

    def test_matched_raw_encoded_full_partition_and_oracle(self):
        gen=torch.Generator().manual_seed(65065)
        for m,n,k in ((2,19,1),(7,67,65),(17,19,257),(32,67,513),(33,19,65)):
            weights=torch.randn((n,k),generator=gen).bfloat16()
            # Finite escape-heavy row plus positive sparse power-of-two activation.
            weights[0,::2]=torch.tensor(2.0**25).bfloat16();weights[0,1::2]=torch.tensor(2.0**-20).bfloat16()
            x=torch.randn((m,k),generator=gen).bfloat16();x[0].zero_();x[0,k-1]=4
            args=pointers(weights);results=[]
            for compressed in (False,True):
                out=torch.full((8,m,n),float('nan'))
                for ni in range((n+63)//64):
                    for split in range(8):
                        for mi in range((m+15)//16):
                            run=interpreted(CPUProgram((ni,split,mi)))
                            run(Pointer(x),*args,Pointer(out),m,n,k,n,((k+255)//256)*256,8,16,64,64,compressed)
                self.assertTrue(torch.isfinite(out).all())
                results.append(out.sum(0).bfloat16())
                data=dict(x=x,config=dict(layer='pooler',n=n))
                ok,message=self.oracle.check_implementation(data,results[-1],{'pooler':weights})
                self.assertTrue(ok,(m,n,k,message))
            self.assertTrue(torch.equal(results[0],results[1]))
            torch.testing.assert_close(results[0][0],weights[:,k-1]*4,rtol=0,atol=0)

    def test_payload_is_frozen_and_has_explicit_dot_layouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest=probe.prepare(Path(tmp))
            self.assertEqual(manifest['sources'][SOURCE.name],hashlib.sha256(SOURCE.read_bytes()).hexdigest())
        text=SOURCE.read_text()
        for required in ('DotOperandLayout(0, mma, 2)','DotOperandLayout(1, mma, 2)','mma_v2(x, w, acc)'):
            self.assertIn(required,text)
        self.assertNotIn('allocate_shared',text)
        self.assertNotIn('tl.trans',text)

    def test_offline_wrapper_selects_gluon_without_runtime_binder(self):
        tree=ast.parse(Path(probe.__file__).read_text())
        imports=[(n.module,a.name) for n in ast.walk(tree) if isinstance(n,ast.ImportFrom) for a in n.names]
        self.assertIn(('triton.experimental.gluon._runtime','GluonASTSource'),imports)
        self.assertNotIn(('triton.compiler','ASTSource'),imports)
        names=[n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
        self.assertIn('GluonASTSource',names)
        attrs=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        for forbidden in ('create_binder','_init_handles','get_current_target','warmup'):
            self.assertNotIn(forbidden,attrs)


if __name__=='__main__':unittest.main()
