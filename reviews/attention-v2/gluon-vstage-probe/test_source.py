"""Offline AST/configuration/schedule gates; never import the Gluon kernel."""
import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
TREE=ast.parse((HERE/'kernel.py').read_text())
FUNCTIONS={node.name:node for node in TREE.body if isinstance(node,ast.FunctionDef)}
KERNEL=FUNCTIONS['attention_vstage_kernel']
PREFETCH=FUNCTIONS['_prefetch_k']
PUBLISH=FUNCTIONS['_publish_kv']
SPEC=importlib.util.spec_from_file_location('gluon_vstage_config',HERE/'config.py')
config=importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name]=config
SPEC.loader.exec_module(config)


def called(node,name):
    return isinstance(node,ast.Call) and (
        isinstance(node.func,ast.Name) and node.func.id==name or
        isinstance(node.func,ast.Attribute) and node.func.attr==name)


def has_call(node,name):
    return any(called(item,name) for item in ast.walk(node))


def literal_kwargs(call):
    return {item.arg:ast.literal_eval(item.value) for item in call.keywords
            if isinstance(item.value,ast.Constant)}


class Specialize(ast.NodeTransformer):
    def __init__(self,before):
        self.before=before

    def visit_If(self,node):
        value=None
        if isinstance(node.test,ast.Name) and node.test.id=='V_BEFORE_WAIT':
            value=self.before
        elif (isinstance(node.test,ast.UnaryOp) and isinstance(node.test.op,ast.Not)
              and isinstance(node.test.operand,ast.Name)
              and node.test.operand.id=='V_BEFORE_WAIT'):
            value=not self.before
        if value is not None:
            return [self.visit(item) for item in (node.body if value else node.orelse)]
        return self.generic_visit(node)


def loop_for(before):
    function=Specialize(before).visit(copy.deepcopy(KERNEL))
    return next(node for node in function.body if isinstance(node,ast.For))


class SourceGates(unittest.TestCase):
    def test_configuration_labels_only_one_static_delta(self):
        value=config.ProbeConfig()
        after=value.compile_kwargs('v_after_wait')
        before=value.compile_kwargs('v_before_wait')
        self.assertEqual({key for key in after if after[key]!=before[key]}, {'V_BEFORE_WAIT'})
        self.assertFalse(after['V_BEFORE_WAIT'])
        self.assertTrue(before['V_BEFORE_WAIT'])
        self.assertEqual(value.grid,(32,2))
        self.assertEqual(config.output_shape(value),(32,8,16,64))
        self.assertEqual(config.page_table_shape(value),(32,86008))
        self.assertEqual((after['num_warps'],after['num_stages']),(4,1))
        self.assertEqual(after['QK_FACTOR'],5.4730064863875)
        for label in ('serial','overlap',True,1,None):
            with self.assertRaises(ValueError):
                value.compile_kwargs(label)

    def test_invalid_configuration_is_host_shape_only(self):
        for field,values in {'batch':(0,-1,True), 'hkv':(0,False),
                             'page_table_stride':(0,7,1.5),
                             'sm_scale':(0,-1,float('nan'),float('inf'),True,'0.1'),
                             'qk_factor':(0,float('inf'))}.items():
            for value in values:
                with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                    config.ProbeConfig(**{field:value})

    def test_exact_api_no_execution_or_forced_dependency(self):
        self.assertEqual([arg.arg for arg in KERNEL.args.args],
            ['Q','KCache','VCache','PageTable','HistoryLengths','KScale','VScale','Out',
             'HKV','HQ','PT_STRIDE','SM_SCALE','QK_FACTOR','V_BEFORE_WAIT'])
        imports=[n.module for n in TREE.body if isinstance(n,ast.ImportFrom)]
        self.assertEqual(imports,['triton.experimental','triton.experimental.gluon',
            'triton.experimental.gluon.language.nvidia.hopper'])
        self.assertEqual(set(FUNCTIONS),{'_prefetch_k','_publish_kv','attention_vstage_kernel'})
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Subscript)
                             for n in ast.walk(TREE)))
        for name in ('warmup','synchronize','cpu','item','atomic_add','inline_asm_elementwise',
                     'debug_barrier','associative_scan','store_volatile'):
            self.assertFalse(has_call(TREE,name),name)
        for node in ast.walk(TREE):
            if isinstance(node,ast.Call):
                self.assertNotIn('volatile',{item.arg for item in node.keywords})

    def test_common_raw_k_prefetch_before_wait_both_arms(self):
        bodies=[]
        for before in (False,True):
            loop=loop_for(before)
            issue=next(i for i,n in enumerate(loop.body) if has_call(n,'warpgroup_mma'))
            prefetch=next(i for i,n in enumerate(loop.body) if has_call(n,'_prefetch_k'))
            wait=next(i for i,n in enumerate(loop.body) if has_call(n,'warpgroup_mma_wait'))
            publish=next(i for i,n in enumerate(loop.body) if has_call(n,'_publish_kv'))
            self.assertLess(issue,prefetch)
            self.assertLess(prefetch,wait)
            self.assertLess(prefetch,publish)
            self.assertLess(publish,wait) if before else self.assertLess(wait,publish)
            self.assertEqual(sum(has_call(n,'_prefetch_k') for n in loop.body),1)
            self.assertEqual(sum(has_call(n,'_publish_kv') for n in loop.body),1)
            self.assertEqual(sum(has_call(n,'warpgroup_mma_wait') for n in loop.body),1)
            bodies.append([n for n in loop.body if not has_call(n,'warpgroup_mma_wait')])
        self.assertEqual(ast.dump(ast.Module(body=bodies[0],type_ignores=[])),
                         ast.dump(ast.Module(body=bodies[1],type_ignores=[])))

    def test_raw_fp8_k_and_live_mapping_not_published_early(self):
        loads=[n for n in ast.walk(PREFETCH) if called(n,'load')]
        self.assertEqual(len(loads),2)
        self.assertIn('PageTable',ast.unparse(loads[0].args[0]))
        self.assertIn('KCache',ast.unparse(loads[1].args[0]))
        raw=next(n for n in PREFETCH.body if isinstance(n,ast.Assign)
                 and has_call(n,'load') and 'KCache' in ast.unparse(n))
        self.assertTrue(called(raw.value,'load'))  # no BF16 cast or shared store in K prefetch.
        self.assertFalse(has_call(PREFETCH,'store'))
        self.assertNotIn('gl.bfloat16',ast.unparse(PREFETCH))
        for load in loads:
            self.assertEqual(literal_kwargs(load)['other'],0)
            self.assertIn('mask',{item.arg for item in load.keywords})

    def test_v_stage_zero_padding_and_only_existing_publication_fence(self):
        loads=[n for n in ast.walk(PUBLISH) if called(n,'load')]
        self.assertEqual(len(loads),1)
        self.assertIn('VCache',ast.unparse(loads[0].args[0]))
        self.assertEqual(literal_kwargs(loads[0])['other'],0)
        stores=[n for n in ast.walk(PUBLISH) if called(n,'store')]
        self.assertEqual(len(stores),2)
        self.assertEqual([ast.unparse(n.func) for n in stores],['KSlot.store','VSlot.store'])
        self.assertTrue(all(not n.keywords for n in stores))
        self.assertEqual(sum(called(n,'fence_async_shared') for n in ast.walk(TREE)),1)
        source=ast.unparse(PUBLISH)
        self.assertEqual(source.count('.to(gl.bfloat16)'),2)
        self.assertLess(source.index('gl.load'),source.index('.to(gl.bfloat16)'))

    def test_precision_zero_c_and_math_unchanged_from_prior_probe(self):
        calls=[n for n in ast.walk(KERNEL) if called(n,'warpgroup_mma')]
        self.assertEqual([literal_kwargs(n) for n in calls],
            [dict(use_acc=False,max_num_imprecise_acc=0,is_async=True),
             dict(use_acc=False,max_num_imprecise_acc=0,is_async=False)])
        self.assertIn('q.to(gl.bfloat16).to(gl.float8e4nv).to(gl.bfloat16)',ast.unparse(KERNEL))
        for wait in [n for n in ast.walk(KERNEL) if called(n,'warpgroup_mma_wait')]:
            self.assertEqual(ast.unparse(next(k.value for k in wait.keywords if k.arg=='deps')),
                             '(score_pending,)')
        old=HERE.parent/'gluon-overlap-probe/kernel.py'
        self.assertEqual(hashlib.sha256(old.read_bytes()).hexdigest(),
            '3d847a290d747c204419fd131c8149a21c4afddbdfb1d0a6130f5f522b81847c')
        old_tree=ast.parse(old.read_text())
        old_kernel=next(n for n in old_tree.body if isinstance(n,ast.FunctionDef)
                        and n.name=='attention_overlap_kernel')
        old_loop=next(n for n in old_kernel.body if isinstance(n,ast.For))
        def math_suffix(loop):
            start=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign)
                       and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='t')
            return ast.dump(ast.Module(body=loop.body[start:],type_ignores=[]))
        self.assertEqual(math_suffix(loop_for(True)),math_suffix(old_loop))

    def test_double_buffers_and_output_only_write_scope(self):
        buffers=[n for n in ast.walk(KERNEL) if called(n,'allocate_shared_memory')]
        self.assertEqual(len(buffers),2)
        self.assertTrue(all(ast.literal_eval(n.args[1])==[2,64,64] for n in buffers))
        self.assertTrue(all(ast.unparse(n.args[0])=='gl.bfloat16' for n in buffers))
        loop=loop_for(True)
        publish=next(n for n in ast.walk(loop) if called(n,'_publish_kv'))
        self.assertIn('k_buffers.index(next_slot)',ast.unparse(publish))
        self.assertIn('v_buffers.index(next_slot)',ast.unparse(publish))
        stores=[n for n in ast.walk(KERNEL) if called(n,'store')]
        self.assertEqual(len(stores),1)
        self.assertEqual(ast.unparse(stores[0].args[0]),'Out + q_offsets')

    def test_causal_boundaries_all_keys_and_slot_lifetime_model(self):
        source=ast.unparse(KERNEL)
        self.assertIn('load_j = load_m // 8',source)
        self.assertIn('load_g = load_m % 8',source)
        self.assertIn('q_position = seq + m // 8',source)
        self.assertIn('visible = (t[None, :] < total) & (t[None, :] <= q_position[:, None])',source)
        for history in (0,1,55,56,63,64,65,127,128,8191,8192):
            total=history+8
            visited=[key for start in range(0,total,64) for key in range(start,start+64) if key<total]
            self.assertEqual(visited,list(range(total)))
            for query in range(8):
                self.assertEqual([key for key in visited if key<=history+query],
                                 list(range(history+query+1)))
            slots={0:0,1:None}
            current=0
            for tile in range((total+63)//64):
                self.assertEqual(slots[current],tile)
                other=current^1
                self.assertNotEqual(current,other)
                # Publication touches only the other slot while current QK is pending.
                slots[other]=tile+1
                # QK wait and common synchronous PV drain precede the next iteration.
                current=other


if __name__=='__main__':
    unittest.main()
