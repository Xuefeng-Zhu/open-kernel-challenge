"""CPU-only compile-contract/resource tests; no Torch/Gluon/driver execution."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
def load(name):
    spec=importlib.util.spec_from_file_location('gluon_probe_test_'+name,HERE/(name+'.py'))
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module
w,r=load('worker'),load('resources')


def resource_rows():
    return [dict(arm=arm,registers=128,driver_spills_raw=0,sass_available=True,
        instruction_inventory=dict(ptx_local_access=0,sass_local_access=0),
        cuda_resources=dict(status='complete',device_limits={'warp_size':32,'max_registers_sm':65536},
            function_attributes=dict(local_bytes=0,registers_per_thread=128),
            total_shared_bytes=32768,threads_per_block=128,static_max_cta_per_sm=3))
        for arm in ('serial','overlap')]


class CompileContracts(unittest.TestCase):
    def test_fixed_pair_only_overlap_changes(self):
        config=w.compile_config()
        a,b=config.compile_kwargs(False),config.compile_kwargs(True)
        self.assertEqual(config.grid,(32,2))
        self.assertEqual({key for key in a if a[key]!=b[key]},{'OVERLAP'})
        self.assertEqual((a['num_warps'],a['num_stages'],a['QK_FACTOR']),(4,1,5.4730064863875))
        self.assertEqual(w.ARMS,[('serial',False),('overlap',True)])

    def test_actual_local_source_hashes_and_no_gpu_default(self):
        observed=w.source_gate()
        self.assertEqual(set(observed),set(w.SOURCE_HASHES)|set(w.LICENSE_HASHES))
        tree=ast.parse((HERE/'worker.py').read_text())
        top_imports=[alias.name for node in tree.body if isinstance(node,ast.Import) for alias in node.names]
        self.assertFalse(set(top_imports)&{'torch','triton','ctypes','modal'})
        warmups=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='warmup']
        self.assertEqual(len(warmups),1)
        self.assertFalse(any(isinstance(node,ast.Call) and isinstance(node.func,ast.Subscript) for node in ast.walk(tree)))
        self.assertFalse(any(isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
            and (node.func.attr in {'remote','synchronize'} or node.func.attr=='run'
                 and not (isinstance(node.func.value,ast.Name) and node.func.value.id=='subprocess')) for node in ast.walk(tree)))
        self.assertIn("parser.add_argument('--compile'",(HERE/'worker.py').read_text())

    def test_installed_gate_exact_bytes_version_no_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'vendor.py'; path.write_bytes(b'fixed installed API\n')
            expected={str(path):hashlib.sha256(path.read_bytes()).hexdigest()}
            self.assertEqual(w.installed_gate(expected,'3.8.0')['source_sha256'],expected)
            for version in ('3.7.0','3.8.1'):
                with self.assertRaises(ValueError): w.installed_gate(expected,version)
            path.write_bytes(b'different public tag\n')
            with self.assertRaises(ValueError): w.installed_gate(expected,'3.8.0')

    def test_bundle_allowlist_bounds_and_cpu_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            for arm,_ in w.ARMS:
                for kind in w.REQUIRED_FORMATS: (folder/f'{arm}.core.{kind}').write_bytes(b'fixture')
            bundle=w.bundle_artifacts(folder)
            self.assertEqual(len(bundle['files']),6)
            self.assertLess(bundle['uncompressed_size'],w.MAX_UNCOMPRESSED)
            (folder/'bad.txt').write_bytes(b'bad')
            with self.assertRaises(ValueError): w.bundle_artifacts(folder)

    def test_header_values_come_from_actual_named_enums(self):
        names=[*r.DEVICE_ATTRIBUTES.values(),*r.FUNCTION_ATTRIBUTES.values()]
        text='\n'.join(name+' = '+str(index)+',' for index,name in enumerate(names))
        self.assertEqual(r.header_enums(text),{name:index for index,name in enumerate(names)})
        self.assertEqual(r.header_enums(''),{})

    def test_resource_pass_has_no_old_96_register_cap(self):
        result=r.resource_gate(resource_rows())
        self.assertEqual(result['status'],'pass')
        self.assertFalse(result['execution_admitted'])

    def test_resource_rejects_spill_local_shared_and_reduced_max_cta(self):
        changes=[('driver_spills_raw',1),('local',16),('shared',65537),('cta',2),('ptx',1),('sass',1)]
        for field,value in changes:
            rows=resource_rows(); row=rows[1]
            if field=='local': row['cuda_resources']['function_attributes']['local_bytes']=value
            elif field=='shared': row['cuda_resources']['total_shared_bytes']=value
            elif field=='cta': row['cuda_resources']['static_max_cta_per_sm']=value
            elif field in ('ptx','sass'): row['instruction_inventory'][field+'_local_access']=value
            else: row[field]=value
            with self.subTest(field=field): self.assertEqual(r.resource_gate(rows)['status'],'reject')

    def test_missing_isa_or_exact_metadata_is_inconclusive(self):
        for field in ('sass','limits','spills','registers'):
            rows=resource_rows()
            if field=='sass': rows[1]['sass_available']=False
            elif field=='limits': rows[1]['cuda_resources']['status']='inconclusive'
            elif field=='spills': rows[1]['driver_spills_raw']=None
            else: rows[1]['registers']=None
            with self.subTest(field=field): self.assertEqual(r.resource_gate(rows)['status'],'inconclusive')

    def test_instruction_inventory_is_presence_not_dataflow_proof(self):
        ptx='wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16;\nwgmma.commit_group.sync.aligned;\nwgmma.wait_group.sync.aligned 0;\nld.local.u32;'
        sass='HGMMA.64x64x16.F32\nLDG.E\nSTSM.16\nSTL.64\n'
        counts=r.instruction_inventory(ptx,sass)
        self.assertEqual((counts['ptx_wgmma_async'],counts['ptx_wgmma_commit'],counts['ptx_wgmma_wait']),(1,1,1))
        self.assertEqual((counts['ptx_local_access'],counts['sass_local_access']),(1,1))
        self.assertEqual(counts['sass_stsm'],1)


if __name__=='__main__': unittest.main(verbosity=2)
