"""Offline source/configuration gates. Never import the Gluon kernel."""
import ast
import copy
import importlib.util
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
TREE = ast.parse((HERE/'kernel.py').read_text())
KERNEL = next(n for n in TREE.body if isinstance(n, ast.FunctionDef)
              and n.name == 'attention_overlap_kernel')
STAGE = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == '_stage_kv')
SPEC = importlib.util.spec_from_file_location('gluon_probe_config', HERE/'config.py')
config = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = config
SPEC.loader.exec_module(config)


def called(node, name):
    return isinstance(node, ast.Call) and (
        isinstance(node.func, ast.Name) and node.func.id == name or
        isinstance(node.func, ast.Attribute) and node.func.attr == name)


def contains_call(node, name):
    return any(called(item, name) for item in ast.walk(node))


def keywords(call):
    return {item.arg: ast.literal_eval(item.value) for item in call.keywords
            if isinstance(item.value, ast.Constant)}


class Specialize(ast.NodeTransformer):
    def __init__(self, overlap):
        self.overlap = overlap

    def visit_If(self, node):
        value = None
        if isinstance(node.test, ast.Name) and node.test.id == 'OVERLAP':
            value = self.overlap
        elif (isinstance(node.test, ast.UnaryOp) and isinstance(node.test.op, ast.Not)
              and isinstance(node.test.operand, ast.Name) and node.test.operand.id == 'OVERLAP'):
            value = not self.overlap
        if value is not None:
            return [self.visit(item) for item in (node.body if value else node.orelse)]
        return self.generic_visit(node)


def specialized_loop(overlap):
    function = Specialize(overlap).visit(copy.deepcopy(KERNEL))
    return next(node for node in function.body if isinstance(node, ast.For))


class SourceAndConfiguration(unittest.TestCase):
    def test_fixed_compile_pair_has_only_overlap_delta(self):
        value = config.ProbeConfig()
        serial, overlap = value.compile_kwargs(False), value.compile_kwargs(True)
        self.assertEqual({key for key in serial if serial[key] != overlap[key]}, {'OVERLAP'})
        self.assertEqual(value.grid, (32, 2))
        self.assertEqual(config.output_shape(value), (32, 8, 16, 64))
        self.assertEqual(config.page_table_shape(value), (32, 86008))
        self.assertEqual((serial['num_warps'], serial['num_stages']), (4, 1))
        self.assertEqual((serial['HKV'], serial['HQ']), (2, 16))
        self.assertEqual(serial['PT_STRIDE'], 86008)
        self.assertEqual(serial['QK_FACTOR'],5.4730064863875)

    def test_invalid_configuration_rejected_without_input_inspection(self):
        for field, values in {'batch': (0,-1,True,1.5), 'hkv': (0,False),
                              'page_table_stride': (0,7),
                              'sm_scale': (0,-1,float('nan'),float('inf'),True,'0.1'),
                              'qk_factor': (0,-1,float('inf'))}.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    config.ProbeConfig(**{field:value})
        with self.assertRaises(ValueError):
            config.ProbeConfig().compile_kwargs(1)

    def test_exact_api_and_no_host_launch_or_service_code(self):
        self.assertEqual([arg.arg for arg in KERNEL.args.args],
            ['Q','KCache','VCache','PageTable','HistoryLengths','KScale','VScale','Out',
             'HKV','HQ','PT_STRIDE','SM_SCALE','QK_FACTOR','OVERLAP'])
        imports = [node.module for node in TREE.body if isinstance(node, ast.ImportFrom)]
        self.assertEqual(imports, ['triton.experimental', 'triton.experimental.gluon',
            'triton.experimental.gluon.language.nvidia.hopper'])
        self.assertFalse(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript)
                             for node in ast.walk(TREE)))
        self.assertFalse(any(contains_call(TREE, name) for name in
            ('warmup','synchronize','cpu','item','system','popen','socket','request','atomic_add')))
        self.assertEqual({node.name for node in TREE.body if isinstance(node, ast.FunctionDef)},
                         {'_stage_kv', 'attention_overlap_kernel'})

    def test_static_specializations_differ_only_by_wait_placement(self):
        loops = [specialized_loop(value) for value in (False, True)]
        bodies = []
        for overlap, loop in zip((False,True), loops):
            issue = next(i for i,n in enumerate(loop.body) if contains_call(n, 'warpgroup_mma'))
            wait = next(i for i,n in enumerate(loop.body) if contains_call(n, 'warpgroup_mma_wait'))
            stage = next(i for i,n in enumerate(loop.body) if contains_call(n, '_stage_kv'))
            self.assertLess(issue, wait)
            self.assertEqual(len([n for n in loop.body if contains_call(n, 'warpgroup_mma_wait')]),1)
            self.assertLess(stage, wait) if overlap else self.assertLess(wait, stage)
            first_score_use = next(i for i,n in enumerate(loop.body)
                if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id == 'scores'
                    and isinstance(x.ctx, ast.Load) for x in ast.walk(n)))
            self.assertLess(wait, first_score_use)
            bodies.append([n for n in loop.body if not contains_call(n,'warpgroup_mma_wait')])
        self.assertEqual(ast.dump(ast.Module(body=bodies[0],type_ignores=[])),
                         ast.dump(ast.Module(body=bodies[1],type_ignores=[])))

    def test_bf16_wgmma_and_round_before_scale_contract(self):
        q_cast = next(n.value for n in KERNEL.body if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'q'
            and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute)
            and n.value.func.attr == 'to')
        self.assertEqual(ast.unparse(q_cast), 'q.to(gl.bfloat16).to(gl.float8e4nv).to(gl.bfloat16)')
        calls = [n for n in ast.walk(KERNEL) if called(n,'warpgroup_mma')]
        self.assertEqual(len(calls),2)
        self.assertEqual([keywords(n) for n in calls],
                         [dict(use_acc=False,max_num_imprecise_acc=0,is_async=True),
                          dict(use_acc=False,max_num_imprecise_acc=0,is_async=False)])
        self.assertEqual(ast.unparse(calls[0].args[0]), 'q')
        self.assertEqual(ast.unparse(calls[0].args[1]), 'current_k.permute((1, 0))')
        self.assertEqual(ast.unparse(calls[1].args[0]), 'p')
        self.assertEqual(ast.unparse(calls[1].args[1]), 'current_v')
        self.assertIn('p.to(gl.bfloat16)', ast.unparse(KERNEL))
        for wait in [n for n in ast.walk(KERNEL) if called(n,'warpgroup_mma_wait')]:
            self.assertEqual(ast.unparse(next(k.value for k in wait.keywords if k.arg=='deps')),
                             '(score_pending,)')

    def test_live_gpu_metadata_and_full_zero_padded_shared_stores(self):
        loads = [n for n in ast.walk(STAGE) if called(n,'load')]
        self.assertEqual(len(loads),3)
        self.assertTrue(any('PageTable' in ast.unparse(n.args[0]) for n in loads))
        for load in loads:
            self.assertIn('mask', {key.arg for key in load.keywords})
            self.assertEqual(keywords(load)['other'],0)
        stores = [n for n in ast.walk(STAGE) if called(n,'store')]
        self.assertEqual([ast.unparse(n) for n in stores],['KSlot.store(k)','VSlot.store(v)'])
        self.assertFalse(any(store.keywords for store in stores))
        self.assertEqual(len([n for n in ast.walk(STAGE) if called(n,'fence_async_shared')]),1)
        source = ast.unparse(STAGE)
        self.assertIn('k = gl.load(KCache + offsets, mask=valid[:, None], other=0.0).to(gl.bfloat16)',source)
        self.assertIn('v = gl.load(VCache + offsets, mask=valid[:, None], other=0.0).to(gl.bfloat16)',source)

    def test_double_buffer_and_output_write_scope(self):
        allocations=[n for n in ast.walk(KERNEL) if called(n,'allocate_shared_memory')]
        self.assertEqual(len(allocations),2)
        self.assertTrue(all(ast.literal_eval(n.args[1])==[2,64,64] for n in allocations))
        self.assertTrue(all(ast.unparse(n.args[0])=='gl.bfloat16' for n in allocations))
        loop=specialized_loop(True)
        stage=next(n for n in ast.walk(loop) if called(n,'_stage_kv'))
        self.assertEqual(ast.unparse(stage.args[5]),'start + 64')
        self.assertEqual(ast.unparse(stage.args[7]),'k_buffers.index(next_slot)')
        self.assertEqual(ast.unparse(stage.args[8]),'v_buffers.index(next_slot)')
        global_stores=[n for n in ast.walk(KERNEL) if called(n,'store')]
        self.assertEqual(len(global_stores),1)
        self.assertEqual(ast.unparse(global_stores[0].args[0]),'Out + q_offsets')
        self.assertEqual(ast.unparse(global_stores[0].args[1]),'out')

    def test_m64_query_mapping_and_causal_history_tail_boundaries(self):
        source=ast.unparse(KERNEL)
        self.assertIn('load_j = load_m // 8', source)
        self.assertIn('load_g = load_m % 8', source)
        self.assertIn('q_position = seq + m // 8', source)
        self.assertIn('visible = (t[None, :] < total) & (t[None, :] <= q_position[:, None])',source)
        for history in (0,1,55,56,63,64,65,127,128,8191,8192):
            total=history+8
            visited=[key for start in range(0,total,64) for key in range(start,start+64) if key<total]
            self.assertEqual(visited,list(range(total)))
            for query in range(8):
                visible=[key for key in visited if key<=history+query]
                self.assertEqual(visible,list(range(history+query+1)))
            next_start=((total-1)//64+1)*64
            self.assertTrue(all(key>=total for key in range(next_start,next_start+64)))
        for hkv in (1,2,3):
            mapped={(j,kv*8+g) for kv in range(hkv) for j in range(8) for g in range(8)}
            self.assertEqual(len(mapped),8*hkv*8)

    def test_symbolic_slot_lifetimes_have_no_overwrite_while_consumed(self):
        # Source-derived schedule model only: native alias/synchronization remains a separate gate.
        for tiles in (1,2,3,4,129):
            buffers={0:0,1:None}
            slot=0
            pending=set()
            for tile in range(tiles):
                self.assertEqual(buffers[slot],tile)
                pending.add(slot)  # current asynchronous QK reads Kslot.
                other=slot^1
                self.assertNotIn(other,pending)
                buffers[other]=tile+1  # identical next-tile K/V stage in either arm.
                pending.remove(slot)  # QK wait completes before score math.
                pending.add(slot)     # synchronous PV reads the current Vslot.
                pending.remove(slot)  # PV completes before buffer flip/reuse.
                slot=other
            self.assertFalse(pending)


if __name__=='__main__':
    unittest.main()
