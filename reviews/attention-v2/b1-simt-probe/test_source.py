"""CPU-only source/host gates; never import candidate or Triton/Torch."""
import ast
import copy
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'fusion-lookahead-probe/candidate.py'
CONTROL_SHA='3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8'
CONTROL=ast.parse(BASE.read_text())
TREE=ast.parse((HERE/'candidate.py').read_text())
CF={n.name:n for n in CONTROL.body if isinstance(n,ast.FunctionDef)}
F={n.name:n for n in TREE.body if isinstance(n,ast.FunctionDef)}


def is_call(node,name):
    return isinstance(node,ast.Call) and (
        isinstance(node.func,ast.Name) and node.func.id==name or
        isinstance(node.func,ast.Attribute) and node.func.attr==name)


def host_function(name,namespace):
    node=copy.deepcopy(F[name])
    node.decorator_list=[]
    node.returns=None
    for arg in node.args.args:
        arg.annotation=None
    code=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
    exec(compile(code,'<extracted pure host function>','exec'),namespace)
    return namespace[name]


def cfg():
    return dict(batch=1,q_len=1,num_q_heads=16,num_kv_heads=2,head_dim=64,
                page_size=1,window=2048,rope_theta=500000,
                qk_scale_factor=5.4730064863875,sm_scale=.125,kv_scale=1.)


class Tensor:
    def __init__(self,shape,dtype='bf16',device='fake_cpu',base=None):
        self.shape=tuple(shape);self.dtype=dtype;self.device=device;self.base=base

    def reshape(self,*shape):
        if len(shape)==1 and isinstance(shape[0],tuple):shape=shape[0]
        shape=list(shape)
        if -1 in shape:
            shape[shape.index(-1)]=math.prod(self.shape)//math.prod(x for x in shape if x!=-1)
        return Tensor(shape,self.dtype,self.device,self)

    view=reshape

    def stride(self,axis):
        return math.prod(self.shape[axis+1:])


class Launch:
    def __init__(self):self.calls=[]
    def __getitem__(self,grid):
        def invoke(*args,**kwargs):self.calls.append((grid,args,kwargs))
        return invoke


class RestoreProducer(ast.NodeTransformer):
    def visit_Name(self,node):
        if node.id=='PARTIAL_M':node.id='BLOCK_M'
        return node

    def visit_Call(self,node):
        self.generic_visit(node)
        if isinstance(node.func,ast.Name) and node.func.id=='_simt_qk':
            return ast.Call(func=ast.Attribute(value=ast.Name(id='tl',ctx=ast.Load()),attr='dot',ctx=ast.Load()),
                args=[node.args[0],ast.Call(func=ast.Attribute(value=ast.Name(id='tl',ctx=ast.Load()),attr='trans',ctx=ast.Load()),args=[node.args[1]],keywords=[])],
                keywords=[ast.keyword(arg='out_dtype',value=ast.Attribute(value=ast.Name(id='tl',ctx=ast.Load()),attr='float32',ctx=ast.Load())),ast.keyword(arg='max_num_imprecise_acc',value=ast.Constant(value=0))])
        if isinstance(node.func,ast.Name) and node.func.id=='_simt_pv':
            return ast.Call(func=ast.Attribute(value=ast.Name(id='tl',ctx=ast.Load()),attr='dot',ctx=ast.Load()),args=node.args,keywords=[])
        return node


class SourceTests(unittest.TestCase):
    def test_all_original_functions_and_imports_preserved(self):
        self.assertEqual(hashlib.sha256(BASE.read_bytes()).hexdigest(),CONTROL_SHA)
        for name,old in CF.items():
            new=copy.deepcopy(F['_baseline_kernel_fn' if name=='kernel_fn' else name])
            new.name=name
            self.assertEqual(ast.dump(old),ast.dump(new),name)
        imports=lambda tree:[ast.dump(n) for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
        self.assertEqual(imports(CONTROL),imports(TREE))

    def test_prologue_partition_softmax_tail_and_ownership_exact_reconstruction(self):
        producer=copy.deepcopy(F['_fused_b1_simt'])
        producer.name='_fused'
        producer.args.args=[a for a in producer.args.args if a.arg!='PARTIAL_M']
        producer.body=[n for n in producer.body if not (isinstance(n,ast.Expr) and is_call(n.value,'static_assert'))]
        producer=RestoreProducer().visit(producer)
        self.assertEqual(ast.dump(producer),ast.dump(CF['_fused']))

    def test_helpers_are_bounded_2d_gather_reductions_without_payload_loads(self):
        for name,indexshape,axis in (('_simt_qk','(1, D)',1),('_simt_pv','(1, N)',0)):
            helper=F[name];calls=[n for n in ast.walk(helper) if isinstance(n,ast.Call)]
            self.assertFalse(any(is_call(n,'load') or is_call(n,'dot') for n in calls))
            gathers=[n for n in calls if is_call(n,'gather')]
            self.assertEqual(len(gathers),1)
            self.assertEqual(ast.unparse(gathers[0].args[1]),f'tl.full({indexshape}, gi, tl.int32)')
            self.assertEqual(ast.literal_eval(gathers[0].keywords[0].value),0)
            sums=[n for n in calls if is_call(n,'sum')]
            self.assertEqual(len(sums),1)
            self.assertEqual(ast.literal_eval(sums[0].args[1]),axis)
            self.assertEqual(len([n for n in ast.walk(helper) if isinstance(n,ast.For)]),1)
            self.assertEqual(ast.unparse(next(n for n in ast.walk(helper) if is_call(n,'static_range'))),
                             'tl.static_range(0, G)')
            for n in ast.walk(helper):
                if isinstance(n,ast.Subscript) and isinstance(n.slice,ast.Tuple):
                    self.assertLessEqual(len(n.slice.elts),2)
            self.assertIn('.to(tl.bfloat16).to(tl.float32)',ast.unparse(helper))

    def test_exact_shape_guard_and_other_routes_fall_back(self):
        env={};guard=host_function('_is_b1_simt_target',env)
        for page in (1,64):
            value=cfg();value['page_size']=page;self.assertTrue(guard(value))
        negatives=[]
        for key,value in [('batch',16),('q_len',8),('num_q_heads',128),('num_kv_heads',16),
                          ('head_dim',128),('page_size',2),('window',0),('rope_theta',0)]:
            other=cfg();other[key]=value;negatives.append(other);self.assertFalse(guard(other))
        seen=[]
        env.update(_b1_simt_kernel_fn=lambda data:seen.append('new') or 'new',
                   _baseline_kernel_fn=lambda data:seen.append('old') or 'old')
        wrapper=host_function('kernel_fn',env)
        self.assertEqual(wrapper({'config':cfg()}),'new')
        self.assertTrue(all(wrapper({'config':value})=='old' for value in negatives))
        self.assertEqual(seen.count('new'),1)

    def test_host_launch_allocations_and_old_merge_interface(self):
        allocations=[]
        def empty(shape,dtype,device):
            tensor=Tensor(shape,dtype,device);allocations.append(tensor);return tensor
        producer,merge=Launch(),Launch()
        env=dict(torch=SimpleNamespace(empty=empty,bfloat16='bf16',float32='fp32'),
                 triton=SimpleNamespace(next_power_of_2=lambda n:1<<(n-1).bit_length()),
                 NUM_SMS=132,RMS_EPS=1e-5,_DOT='bf16',_QK_DOT='fp8',
                 _fused_b1_simt=producer,_merge=merge,
                 _tickets=lambda *_:self.fail('target must not allocate a counter'))
        host_function('_num_splits',env)
        run=host_function('_b1_simt_kernel_fn',env)
        data={name:object() for name in ('seq_lens','cache_loc','k_cache','v_cache','k_scale','v_scale')}
        data.update(config=cfg(),qkv=Tensor((1,1,2304)),page_table=Tensor((1,80001),'int32'))
        result=run(data)
        self.assertEqual(result.shape,(1,1,1024))
        self.assertEqual([(x.shape,x.dtype) for x in allocations],
            [((1,1024),'bf16'),((512,64),'fp32'),((512,),'fp32'),((512,),'fp32')])
        self.assertEqual(len(producer.calls),1);self.assertEqual(len(merge.calls),1)
        grid,args,kw=producer.calls[0]
        self.assertEqual(grid,(1,2,16))
        expected=dict(BLOCK_M=8,PARTIAL_M=16,BLOCK_N=64,BLOCK_L=16,SPLITS=16,
                      L=1,G=8,HQ=16,HKV=2,D=64,ROPE=True,WINDOWED=True,
                      CHUNK_SIZE=0,SEPARATE_MERGE=True,PAGE_LOOKAHEAD=False,
                      num_warps=4,num_stages=3)
        for key,value in expected.items():self.assertEqual(kw[key],value,key)
        self.assertIs(args[11],allocations[0])  # unused ticket pointer aliases output.
        merge_grid,merge_args,merge_kw=merge.calls[0]
        self.assertEqual(merge_grid,(1,16))
        self.assertEqual((merge_kw['BLOCK_M'],merge_kw['SPLITS'],merge_kw['BS']),(16,16,16))
        self.assertIs(merge_args[2],args[8]);self.assertIs(merge_args[3],args[9]);self.assertIs(merge_args[4],args[10])


if __name__=='__main__':unittest.main()
