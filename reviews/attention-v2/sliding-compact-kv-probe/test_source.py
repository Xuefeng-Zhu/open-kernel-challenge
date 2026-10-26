"""CPU AST/dispatch gates. Never imports Torch, Triton or candidate code."""
import ast
import copy
import hashlib
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
BASE_SHA = '3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8'
BASE_RAW = (HERE/'baseline.py').read_bytes()
CAND_RAW = (HERE/'candidate.py').read_bytes()
BASE = ast.parse(BASE_RAW.decode())
CAND = ast.parse(CAND_RAW.decode())
ADDED = {'_compact_window_kv_bytes', '_fa3_sliding_verify_compact', 'kernel_fn', '_compact_fa3_metadata'}

def dump(x):
    return ast.dump(x, include_attributes=False)

def function(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)

def named_call(node, name):
    return next(n for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name)

class SourceContract(unittest.TestCase):
    def test_exact_baseline_and_whole_ast_preserved(self):
        self.assertEqual(hashlib.sha256(BASE_RAW).hexdigest(), BASE_SHA)
        restored = copy.deepcopy(CAND)
        restored.body = [n for n in restored.body if not (isinstance(n, ast.FunctionDef) and n.name in ADDED)
                         and not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
                                  and isinstance(n.value.func,ast.Name)
                                  and n.value.func.id == '_compact_fa3_metadata')]
        old_dispatch = function(restored, '_compact_baseline_kernel_fn')
        old_dispatch.name = 'kernel_fn'
        self.assertEqual(dump(restored), dump(BASE))
        self.assertEqual(CAND_RAW.splitlines()[0], BASE_RAW.splitlines()[0])

    def test_fa3_kwargs_only_four_representation_fields_differ(self):
        old = named_call(function(BASE, '_fa3_sliding_verify_fused'), 'flash_attn_with_kvcache')
        new = named_call(function(CAND, '_fa3_sliding_verify_compact'), 'flash_attn_with_kvcache')
        old_kw = {n.arg: dump(n.value) for n in old.keywords}
        new_kw = {n.arg: dump(n.value) for n in new.keywords}
        self.assertEqual(old_kw.keys(), new_kw.keys())
        self.assertEqual({k for k in old_kw if old_kw[k] != new_kw[k]},
                         {'k_cache','v_cache','page_table','cache_seqlens'})
        self.assertEqual(new_kw['page_table'], dump(ast.Constant(value=None)))
        self.assertEqual(new_kw['causal'], dump(ast.Constant(value=True)))
        self.assertEqual(new_kw['k_cache'], dump(ast.Name(id='packed_k',ctx=ast.Load())))
        self.assertEqual(new_kw['v_cache'], dump(ast.Name(id='packed_v',ctx=ast.Load())))
        self.assertEqual(new_kw['cache_seqlens'], dump(ast.Name(id='packed_lens',ctx=ast.Load())))
        self.assertNotIn('num_splits', new_kw)
        self.assertNotIn('pack_gqa', new_kw)

    def test_adapter_reconstructs_exact_original_after_pack_removal(self):
        new = copy.deepcopy(function(CAND, '_fa3_sliding_verify_compact'))
        new.name = '_fa3_sliding_verify_fused'
        extra_assign = {'capacity','packed_k','packed_v','packed_lens'}
        new.body = [n for n in new.body if not (
            isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in extra_assign for t in n.targets))
            and not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                     and isinstance(n.value.func, ast.Subscript)
                     and isinstance(n.value.func.value, ast.Name)
                     and n.value.func.value.id == '_compact_window_kv_bytes')]
        new_call = named_call(new, 'flash_attn_with_kvcache')
        old = function(BASE, '_fa3_sliding_verify_fused')
        old_call = named_call(old, 'flash_attn_with_kvcache')
        old_values = {k.arg:k.value for k in old_call.keywords}
        for k in new_call.keywords:
            if k.arg in {'k_cache','v_cache','page_table','cache_seqlens'}:
                k.value = copy.deepcopy(old_values[k.arg])
        self.assertEqual(dump(new), dump(old))

    def test_gather_source_contract_bytecopy_and_live_metadata(self):
        fn = function(CAND, '_compact_window_kv_bytes')
        code = ast.unparse(fn)
        self.assertIn('tl.load(CACHE_LENS + batch)', code)
        self.assertIn('tl.minimum(total, CAP)', code)
        self.assertIn('tl.maximum(total - CAP, 0)', code)
        self.assertIn('page >= 0', code)
        self.assertNotIn('page > 0', code)
        stores = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Attribute) and n.func.attr == 'store']
        self.assertEqual(len(stores), 3)
        for n in stores:
            self.assertTrue(any(s in ast.unparse(n.args[0]) for s in ['PACKED_K','PACKED_V','PACKED_LENS']))
        self.assertNotIn('bfloat16', code)
        self.assertNotIn('float8', code)
        self.assertNotIn('tl.dot', code)
        adapter = function(CAND, '_fa3_sliding_verify_compact')
        call = next(n for n in ast.walk(adapter) if isinstance(n,ast.Call)
            and isinstance(n.func,ast.Subscript) and isinstance(n.func.value,ast.Name)
            and n.func.value.id=='_compact_window_kv_bytes')
        self.assertEqual([ast.unparse(call.args[i]) for i in [0,1,4,5]],
                         ['kc.view(torch.uint8)','vc.view(torch.uint8)',
                          'packed_k.view(torch.uint8)','packed_v.view(torch.uint8)'])
        self.assertEqual({k.arg:ast.unparse(k.value) for k in call.keywords},
                         {'CAP':'capacity','FEATURES':'hkv * d','BT':'32','num_warps':'4'})
        banned = {'item','cpu','numpy','synchronize','clone','copy_'}
        for n in ast.walk(adapter):
            if isinstance(n,ast.Attribute): self.assertNotIn(n.attr,banned)

    def test_shape_only_dispatch_and_unchanged_fallback(self):
        fn = copy.deepcopy(function(CAND,'kernel_fn'))
        fn.returns = None
        for arg in fn.args.args: arg.annotation = None
        mod = ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
        ns = {'_fa3_sliding_verify_compact':lambda data:'compact',
              '_compact_baseline_kernel_fn':lambda data:'baseline'}
        exec(compile(mod,'pure-shape-dispatch','exec'),ns)
        cfg = dict(batch=1,q_len=8,window=2048,rope_theta=500000,page_size=1,
                   num_q_heads=16,num_kv_heads=2,head_dim=64)
        for b in [1,3,5,16,32,65]:
            self.assertEqual(ns['kernel_fn']({'config':dict(cfg,batch=b)}),'compact')
        for key,value in [('q_len',1),('window',0),('rope_theta',0),('page_size',64),
                          ('num_q_heads',128),('num_kv_heads',16),('head_dim',128)]:
            self.assertEqual(ns['kernel_fn']({'config':dict(cfg,**{key:value})}),'baseline')
        names = {n.id for n in ast.walk(fn) if isinstance(n,ast.Name)}
        self.assertNotIn('torch',names)
        guard_fields = [n.slice.value for n in ast.walk(fn.body[1].test)
                        if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name)
                        and n.value.id == 'cfg' and isinstance(n.slice,ast.Constant)]
        self.assertNotIn('batch',guard_fields)


    def test_import_metadata_is_bounded_and_does_not_invoke_native(self):
        import contextlib
        import io
        from types import SimpleNamespace
        class Operator:
            _schema = 'fwd(Tensor q, Tensor k, Tensor v) -> Tensor[]'
            def __call__(self, *args, **kwargs):
                raise AssertionError('Diagnostic must never invoke native fwd')
        op = Operator()
        fake = SimpleNamespace(__version__='2.14.1+cu130', version=SimpleNamespace(cuda='13.0'),
            ops=SimpleNamespace(sgl_kernel=SimpleNamespace(fwd=SimpleNamespace(default=op))))
        fn = copy.deepcopy(function(CAND,'_compact_fa3_metadata'))
        mod = ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
        ns = {'torch':fake}
        exec(compile(mod,'scalar-metadata-only','exec'),ns)
        def observe():
            out=io.StringIO()
            with contextlib.redirect_stdout(out): ns['_compact_fa3_metadata']()
            text=out.getvalue();self.assertEqual(text.count('OKC_COMPACT_FA3_METADATA'),1)
            self.assertLessEqual(len(text.encode('utf-8')),4500)
            return text
        text=observe()
        self.assertIn('torch=2.14.1+cu130',text);self.assertIn('cuda=13.0',text)
        self.assertIn('fwd_schema='+op._schema,text)
        op._schema='é'*2049
        text=observe();self.assertIn('oversize_not_retained',text);self.assertNotIn('é',text)
        op._schema='x'*4096
        self.assertIn('x'*4096,observe())
        fake.__version__=object();fake.version.cuda=object()
        text=observe();self.assertIn('torch=unavailable',text);self.assertIn('cuda=unavailable',text)
        fake.ops=SimpleNamespace()
        self.assertEqual(observe(),'OKC_COMPACT_FA3_METADATA unavailable\n')
        calls=[n for n in CAND.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
               and isinstance(n.value.func,ast.Name) and n.value.func.id=='_compact_fa3_metadata']
        self.assertEqual(len(calls),1)
        attrs={n.attr for n in ast.walk(fn) if isinstance(n,ast.Attribute)}
        self.assertTrue({'_schema','__version__','cuda'} <= attrs)
        self.assertFalse({'item','cpu','numpy','synchronize','current_device','get_device_capability'} & attrs)
        names={n.id for n in ast.walk(fn) if isinstance(n,ast.Name)}
        self.assertFalse({'data','qkv','inputs','inspect','os','sys'} & names)

if __name__ == '__main__': unittest.main()
