"""CPU-only B1 SIMT compile/resource contracts; no Torch or driver action."""
import ast
import importlib.util
from pathlib import Path
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('b1_simt_compile_worker',HERE/'worker.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)


def rows():
    return [dict(arm=arm,unit=unit,registers=128,driver_spills_raw=0,sass_available=True,
        local_instruction_count=0,sass_tensor_mma_count=0,ptx_instruction_counts={'families':{'mma':0,'wgmma':0}},
        cuda_resources=dict(status='complete',total_shared_bytes=16384,static_max_cta_per_sm=3,
            threads_per_block=128,function_attributes={'local_bytes':0,'registers_per_thread':128}))
        for arm,unit in w.UNITS]


class B1CompileContracts(unittest.TestCase):
    def test_source_binding_shared_merge_identity_and_exact_three_units(self):
        self.assertEqual(set(w.source_gate()),set(w.SOURCE_HASHES)|set(w.LICENSE_HASHES))
        self.assertEqual(len(w.same_merge()),64)
        self.assertEqual(w.UNITS,[('baseline','producer'),('candidate','producer'),('shared','merge')])
        self.assertEqual(len(w.ALLOWED),18)

    def test_fixed_geometry_unpadded_internal_state_and_unchanged_partial_shape(self):
        baseline,candidate=w.producer_options(),w.producer_options(True)
        self.assertEqual({key for key in baseline if baseline[key]!=candidate[key]},{'BLOCK_M'})
        self.assertEqual((baseline['BLOCK_M'],candidate['BLOCK_M'],candidate['PARTIAL_M']),(16,8,16))
        self.assertEqual((candidate['SPLITS'],candidate['BLOCK_N'],candidate['num_warps'],candidate['num_stages']),(16,64,4,3))
        self.assertTrue(candidate['ROPE'] and candidate['WINDOWED'] and candidate['SEPARATE_MERGE'])
        self.assertFalse(candidate['PAGE_LOOKAHEAD']);self.assertEqual(candidate['CHUNK_SIZE'],0)
        self.assertEqual((w.merge_options()['BLOCK_M'],w.merge_options()['BS']),(16,16))

    def test_official_pointer_types_and_compile_only_calls(self):
        tree=ast.parse((HERE/'worker.py').read_text())
        assignment=next(node for node in ast.walk(tree) if isinstance(node,ast.Assign)
            and any(isinstance(target,ast.Name) and target.id=='dtypes' for target in node.targets))
        dtypes={key.arg:ast.unparse(key.value) for key in assignment.value.keywords}
        self.assertEqual((dtypes['seq'],dtypes['pt'],dtypes['loc']),('torch.int32','torch.int32','torch.int64'))
        self.assertEqual((dtypes['kc'],dtypes['vc']),('torch.float8_e4m3fn','torch.float8_e4m3fn'))
        self.assertEqual((dtypes['po'],dtypes['pm'],dtypes['pl']),('torch.float32',)*3)
        self.assertFalse(any(isinstance(node,ast.Call) and isinstance(node.func,ast.Subscript) for node in ast.walk(tree)))
        self.assertFalse(any(isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in {'remote','synchronize'} for node in ast.walk(tree)))
        self.assertIn("parser.add_argument('--compile'",(HERE/'worker.py').read_text())

    def test_shape_budget_is_192_not_inherited_96_and_unknowns_do_not_pass(self):
        sample=rows();sample[1]['registers']=192;sample[1]['cuda_resources']['function_attributes']['registers_per_thread']=192
        self.assertEqual(w.resource_screen(sample)['status'],'pass')
        self.assertFalse(w.resource_screen(sample)['execution_admitted'])
        for field in ('sass','resources','registers'):
            sample=rows()
            if field=='sass':sample[1]['sass_available']=False
            elif field=='resources':sample[1]['cuda_resources']['status']='inconclusive'
            else:sample[1]['registers']=None
            with self.subTest(field=field):self.assertEqual(w.resource_screen(sample)['status'],'inconclusive')

    def test_prospective_resource_and_no_mma_failures_reject(self):
        for field in ('registers','spills','local','shared','cta','ptx_mma','sass_mma'):
            sample=rows();candidate=sample[1]
            if field=='registers':candidate['registers']=193;candidate['cuda_resources']['function_attributes']['registers_per_thread']=193
            elif field=='spills':candidate['driver_spills_raw']=1
            elif field=='local':candidate['cuda_resources']['function_attributes']['local_bytes']=16
            elif field=='shared':candidate['cuda_resources']['total_shared_bytes']=65537
            elif field=='cta':candidate['cuda_resources']['static_max_cta_per_sm']=1
            elif field=='ptx_mma':candidate['ptx_instruction_counts']['families']['mma']=1
            else:candidate['sass_tensor_mma_count']=1
            with self.subTest(field=field):self.assertEqual(w.resource_screen(sample)['status'],'reject')

    def test_bounded_three_unit_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            for name in w.ALLOWED:(folder/name).write_bytes(b'fixture')
            bundle=w.bundle_artifacts(folder)
            self.assertEqual(set(bundle['files']),w.ALLOWED)
            self.assertLessEqual(bundle['uncompressed_size'],8*1024*1024)
            (folder/'other.ptx').write_bytes(b'not allowed')
            with self.assertRaises(ValueError):w.bundle_artifacts(folder)


if __name__=='__main__':unittest.main(verbosity=2)
