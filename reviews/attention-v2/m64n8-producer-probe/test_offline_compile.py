"""CPU contracts for full producer offline compilation, never GPU execution."""
import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
import platform
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location('producer_offline_under_test',HERE/'offline_compile.py')
c=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(c)
SELECTED=HERE.parent/'fusion-lookahead-probe/candidate.py'
RUNTIME=('qkv_ptr','seq_ptr','pt_ptr','loc_ptr','kc_ptr','vc_ptr','ks_ptr','vs_ptr',
         'o_ptr','m_ptr','l_ptr','ticket_ptr','out_ptr','theta','eps','qk_factor',
         'sm_scale','window','page','stride_tok','stride_pt')
CONSTEXPR=('L','G','HQ','HKV','D','BLOCK_M','BLOCK_N','BLOCK_L','SPLITS','ROPE',
           'WINDOWED','DOT','QK_DOT','CHUNK_SIZE','SEPARATE_MERGE','PAGE_LOOKAHEAD')
POINTER_TYPES=('*bf16','*i32','*i32','*i64','*fp8e4nv','*fp8e4nv','*fp32',
               '*fp32','*fp32','*fp32','*fp32','*bf16','*bf16')


def sha(raw):return hashlib.sha256(raw).hexdigest()


def toy_candidate(extra=''):
    # Synthetic ABI/surface model only; not an attention algorithm or compile result.
    args=list(RUNTIME)+[name+': gl.constexpr' for name in CONSTEXPR]
    return ('from triton.experimental import gluon\n'
            'from triton.experimental.gluon import language as gl\n'
            '@gluon.jit\n'
            'def _m64n8_fused('+', '.join(args)+'):\n    pass\n'+extra).encode()


def alignment_indexes(attrs):
    """Normalize only index serialization, never relax required attribute values."""
    result={}
    for index,value in attrs.items():
        if isinstance(index,tuple):index=index[0]
        if isinstance(index,str):index=int(index.strip('(), '))
        result[index]=value
    return result


class ControlExtraction(unittest.TestCase):
    def test_exact_selected_functions_and_LOG2E_survive_without_host_setup(self):
        raw=SELECTED.read_bytes()
        self.assertEqual(sha(raw),'3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8')
        self.assertIsInstance(c.selected_contract(raw),dict)
        extracted=c.extract_control(raw)
        if isinstance(extracted,bytes):extracted=extracted.decode()
        original=ast.parse(raw);small=ast.parse(extracted)
        expected={n.name:n for n in original.body if isinstance(n,ast.FunctionDef) and n.name in ('_norm_rope','_fused')}
        observed={n.name:n for n in small.body if isinstance(n,ast.FunctionDef)}
        self.assertEqual(set(observed),set(expected))
        for name in expected:
            self.assertEqual(ast.dump(observed[name],include_attributes=False),ast.dump(expected[name],include_attributes=False))
            self.assertIn(ast.get_source_segment(raw.decode(),expected[name]),extracted)
        globals_old=[n for n in original.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='LOG2E' for t in n.targets)]
        globals_new=[n for n in small.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='LOG2E' for t in n.targets)]
        self.assertEqual(ast.dump(globals_old[0],include_attributes=False),ast.dump(globals_new[0],include_attributes=False))
        aliases={a.name for n in small.body if isinstance(n,(ast.Import,ast.ImportFrom)) for a in n.names}
        self.assertFalse(aliases&{'torch','os','TypedDict','flash_attn_with_kvcache','store_cache'})
        self.assertFalse(any(isinstance(n,ast.ClassDef) for n in small.body))
        for n in small.body:
            if isinstance(n,ast.Expr):self.assertIsInstance(n.value,ast.Constant)

    def test_control_entry_has_exact_21_runtime_and16_explicit_constants(self):
        tree=ast.parse(SELECTED.read_bytes())
        entry=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_fused')
        self.assertEqual([a.arg for a in entry.args.args],list(RUNTIME+CONSTEXPR))
        self.assertTrue(all(a.annotation is None for a in entry.args.args[:21]))
        self.assertTrue(all(ast.unparse(a.annotation)=='tl.constexpr' for a in entry.args.args[21:]))
        self.assertEqual([ast.unparse(d) for d in entry.decorator_list],['triton.jit'])
        self.assertEqual([ast.unparse(d) for d in next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_norm_rope').decorator_list],['triton.jit'])

    def test_changed_selected_bytes_are_refused_even_if_valid_python(self):
        with self.assertRaises(ValueError):c.selected_contract(SELECTED.read_bytes()+b'\nopen("should-not-run","w")\n')
        with self.assertRaises(ValueError):c.extract_control(SELECTED.read_bytes().replace(b'max_num_imprecise_acc=0',b'max_num_imprecise_acc=32'))


class FixedSpecializations(unittest.TestCase):
    def test_all_four_split_geometries_derive_from_actual_selected_host_rule(self):
        cases=c.case_plans()
        self.assertEqual(len(cases),4)
        seen=[]
        for case in cases:
            b=case['batch'];constants=case['constants_tokens'];scalars=case['scalar_values']
            windowed=bool(constants['WINDOWED'])
            splits=min((264+2*b-1)//(2*b),2048//128) if windowed else (86001+8191)//8192
            seen.append((b,windowed,splits))
            self.assertEqual(case['splits'],splits)
            self.assertEqual(constants['SPLITS'],splits)
            self.assertEqual(tuple(case['grid']),(b,2,splits))
            self.assertEqual({k:constants[k] for k in ['L','G','HQ','HKV','D','BLOCK_M','BLOCK_N','BLOCK_L']},
                dict(L=1,G=8,HQ=16,HKV=2,D=64,BLOCK_M=16,BLOCK_N=64,BLOCK_L=16))
            self.assertIs(constants['ROPE'],windowed)
            self.assertEqual(constants['CHUNK_SIZE'],0 if windowed else 8192)
            self.assertTrue(constants['SEPARATE_MERGE'])
            self.assertEqual(constants['PAGE_LOOKAHEAD'],not windowed)
            self.assertEqual(scalars['theta'],500000. if windowed else 1.)
            self.assertEqual(scalars['eps'],1e-5)
            self.assertEqual(scalars['qk_factor'],5.4730064863875)
            self.assertEqual(scalars['sm_scale'],.125)
            self.assertEqual(scalars['window'],2048 if windowed else 0)
            self.assertEqual(scalars['stride_pt'],80001 if b==1 else 86001)
            self.assertEqual(scalars['stride_tok'],2304)
        self.assertEqual(set(seen),{(1,True,16),(16,True,9),(32,True,5),(32,False,11)})

    def test_page_one_autospecialization_types_and_alignment_are_matched(self):
        for case in c.case_plans():
            signature=case['signature_tokens'];constants=case['constants_tokens']
            self.assertEqual(list(signature),list(RUNTIME+CONSTEXPR))
            self.assertEqual([signature[k] for k in RUNTIME[:13]],list(POINTER_TYPES))
            self.assertEqual([signature[k] for k in RUNTIME[13:17]],['fp32']*4)
            self.assertEqual([signature[k] for k in ['window','stride_tok','stride_pt']],['i32']*3)
            self.assertEqual(signature['page'],'constexpr')
            self.assertEqual(constants['page'],1)
            self.assertEqual(constants['DOT'],'dtype:bf16')
            self.assertEqual(constants['QK_DOT'],'dtype:fp8e4nv')
            attrs=alignment_indexes(case['alignment_attrs'])
            self.assertEqual(set(attrs),set(range(13))|{17,19})
            self.assertTrue(all(value==[['tt.divisibility',16]] for value in attrs.values()))
            self.assertNotIn(18,attrs)
            self.assertNotIn(20,attrs)
        self.assertEqual(c.OPTIONS,dict(num_warps=4,num_stages=3))


class CandidateSurface(unittest.TestCase):
    def test_pure_jit_ABI_is_accepted_but_wrong_hash_or_ABI_is_refused(self):
        raw=toy_candidate()
        self.assertIsInstance(c.candidate_contract(raw,sha(raw)),dict)
        with self.assertRaises(ValueError):c.candidate_contract(raw,'0'*64)
        bad=raw.replace(b'qkv_ptr,',b'wrong_ptr,',1)
        with self.assertRaises(ValueError):c.candidate_contract(bad,sha(bad))
        bad=raw.replace(b'PAGE_LOOKAHEAD: gl.constexpr',b'PAGE_LOOKAHEAD')
        with self.assertRaises(ValueError):c.candidate_contract(bad,sha(bad))

    def test_hash_matched_host_sideeffects_are_still_refused(self):
        variants=[
            b'import torch\n'+toy_candidate(),
            b'from os import system\n'+toy_candidate(),
            toy_candidate('open("should-not-run","w")\n'),
            toy_candidate('class HostObject:\n    pass\n'),
            toy_candidate().replace(b'@gluon.jit',b'@gluon.jit(some_host_call())'),
            toy_candidate('def host_launcher():\n    return 1\n'),
        ]
        for raw in variants:
            with self.subTest(source=sha(raw)):
                with self.assertRaises(ValueError):c.candidate_contract(raw,sha(raw))

    def test_helper_constructs_sources_directly_without_runtime_driver_binder(self):
        tree=ast.parse((HERE/'offline_compile.py').read_text())
        attrs={n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)}
        self.assertFalse(attrs&{'create_binder','_init_handles','load_binary','get_current_target','get_current_device','n_regs','synchronize'})
        names={a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names}
        self.assertFalse(names&{'torch','sglang','sgl_kernel','modal'})
        calls={n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
        self.assertTrue({'ASTSource','GluonASTSource'}<=calls)


class ArtifactEvidence(unittest.TestCase):
    def fixture(self,regs=96):
        ptxas=("ptxas info : Function properties for _m64n8_fused\n"
               " 0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"
               f"ptxas info : Used {regs} registers, used 1 barriers\n")
        cubin=("Resource usage:\n Function _m64n8_fused:\n"
               f" REG:{regs} STACK:0 SHARED:1024 LOCAL:0 CONSTANT[0]:608\n")
        return ptxas,cubin

    def test_resources_keep_static_shared_separate_and_missing_unknown(self):
        resources=c.parse_resources(*self.fixture())
        expected=dict(registers_ptxas=96,registers_cuobjdump=96,stack_frame_bytes=0,stack_cuobjdump_bytes=0,
            spill_store_bytes=0,spill_load_bytes=0,local_bytes=0,static_shared_bytes=1024)
        for k,v in expected.items():self.assertEqual(resources[k],v)
        missing=c.parse_resources('', '')
        for k in expected:self.assertIsNone(missing[k])

    def test_resource_gate_enforces_no_spill_local_and_exact_predeclared_limits(self):
        resources=c.parse_resources(*self.fixture())
        inventory=c.instruction_summary('', '')
        self.assertTrue(c.resource_gate({'shared':40960},resources,inventory)['passed'])
        self.assertFalse(c.resource_gate({'shared':40961},resources,inventory)['passed'])
        self.assertFalse(c.resource_gate({'shared':40960},c.parse_resources(*self.fixture(97)),inventory)['passed'])
        for field in ['stack_frame_bytes','stack_cuobjdump_bytes','spill_store_bytes','spill_load_bytes','local_bytes']:
            bad=dict(resources);bad[field]=1
            self.assertFalse(c.resource_gate({'shared':0},bad,inventory)['passed'])
        self.assertFalse(c.resource_gate({'shared':0},c.parse_resources('',''),inventory)['passed'])
        self.assertFalse(c.resource_gate({},resources,inventory)['passed'])
        self.assertFalse(c.resource_gate({'shared':-1},resources,inventory)['passed'])
        mismatch=dict(resources);mismatch['registers_cuobjdump']=95
        self.assertFalse(c.resource_gate({'shared':0},mismatch,inventory)['passed'])

    def test_instruction_inventory_is_static_and_local_access_blocks_gate(self):
        ptx=('wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 a;\n'
             'mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 b;\n'
             'cp.async.ca.shared.global [x],[y],16;\n'
             'cp.async.commit_group;\ncp.async.wait_group 1;\nld.local.u32 x,[y];\n')
        inv=c.instruction_summary(ptx,'HGMMA.64x8x16.F32;\nLDL R1,[R2];')
        self.assertEqual(inv['wgmma_m64n8k16_sites'],1)
        self.assertEqual(inv['mma_sync_sites'],1)
        self.assertEqual(inv['local_load_sites'],1)
        self.assertFalse(c.resource_gate({'shared':0},c.parse_resources(*self.fixture()),inv)['passed'])

    def test_prepare_binds_exact_sources_and_refuses_nonempty_output(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);kernel=root/'toy.py';kernel.write_bytes(toy_candidate())
            dest=root/'payload'
            manifest=c.prepare(dest,sha(kernel.read_bytes()),kernel_path=kernel,selected_path=SELECTED)
            self.assertEqual(manifest['kernel_sha256'],sha(kernel.read_bytes()))
            self.assertEqual(manifest['selected_sha256'],sha(SELECTED.read_bytes()))
            self.assertEqual(manifest['helper_sha256'],sha((dest/'offline_compile.py').read_bytes()))
            self.assertEqual((dest/'kernel.py').read_bytes(),kernel.read_bytes())
            self.assertEqual(manifest['target'],['cuda',90,32])
            self.assertEqual(manifest['dependency']['sha256'],'a9c404c69ed4a39e8ec632eaf6b9fe058a060bf98979c177f6ef666f06bb8d50')
            self.assertEqual(len(manifest['cases']),4)
            with self.assertRaises(ValueError):c.prepare(dest,sha(kernel.read_bytes()),kernel_path=kernel,selected_path=SELECTED)

    def test_nonlinux_compile_refused_before_toolchain_import(self):
        if platform.system()!='Linux':
            with self.assertRaisesRegex(RuntimeError,'Linux'):
                c.compile_payload(Path('/not-present'))

    def test_manifest_cannot_skip_reorder_or_relabel_modified_control(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);kernel=root/'toy.py';kernel.write_bytes(toy_candidate())
            dest=root/'payload';m=c.prepare(dest,sha(kernel.read_bytes()),kernel_path=kernel,selected_path=SELECTED)
            control=(dest/'control.py').read_bytes();helper=(dest/'offline_compile.py').read_bytes()
            c.validate_manifest(m,control,kernel.read_bytes(),helper)
            changes={'control_order':m['control_order'][:-1],
                'candidate_order':list(reversed(m['candidate_order'])),
                'selected_sha256':'0'*64,'dependency':{},'expected_tool_sha256':{}}
            for field,value in changes.items():
                changed=copy.deepcopy(m);changed[field]=value
                with self.subTest(field=field):
                    with self.assertRaises(ValueError):c.validate_manifest(changed,control,kernel.read_bytes(),helper)
            edited=control+b'\n'
            changed=copy.deepcopy(m);changed['control_sha256']=sha(edited)
            with self.assertRaises(ValueError):c.validate_manifest(changed,edited,kernel.read_bytes(),helper)

    def test_full_exception_text_survives_summary_limit_or_explicit_oversize_refusal(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);message='failure '+('x'*12001)
            result=c.exception_record(root,'compile.error.txt',ValueError(message))
            raw=(root/'compile.error.txt').read_bytes()
            self.assertTrue(result['reason_summary_truncated'])
            self.assertEqual(result['error_artifact_status'],'full_text_saved')
            self.assertIn(message.encode(),raw)
            self.assertEqual(result['error_text_sha256'],sha(raw))
            with patch.object(c,'MAX_ARTIFACT',32):
                refused=c.exception_record(root,'too-big.error.txt',ValueError(message))
            self.assertEqual(refused['error_artifact_status'],'refused_oversize_no_silent_truncation')
            self.assertFalse((root/'too-big.error.txt').exists())

    def test_artifact_bound_keeps_exact_8MiB_and_refuses_larger_without_partial_write(self):
        self.assertEqual(c.MAX_ARTIFACT,8*1024*1024)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);raw=b'x'*c.MAX_ARTIFACT
            self.assertEqual(c.bounded_write(root/'boundary.bin',raw),sha(raw))
            self.assertEqual((root/'boundary.bin').stat().st_size,len(raw))
            with self.assertRaises(ValueError):c.bounded_write(root/'over.bin',raw+b'x')
            self.assertFalse((root/'over.bin').exists())


if __name__=='__main__':unittest.main()
