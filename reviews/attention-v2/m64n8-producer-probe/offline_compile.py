"""Same-toolchain full-producer compilation only; no driver, handles or execution."""
from __future__ import annotations
import argparse
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
SELECTED=HERE.parent/'fusion-lookahead-probe/candidate.py'
SELECTED_SHA='3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8'
CONTROL_SHA='cfefc9b9a9b2b175945fe7e6990d7c5fe46efb6e752acf88683c0dee4bb05356'
CONTROL_ORDER=['D1slide','D16slide','D32slide','D32full']
CANDIDATE_ORDER=['D32full','D1slide','D16slide','D32slide']
ENTRY='_m64n8_fused'
RUNTIME_ARGS=('qkv_ptr','seq_ptr','pt_ptr','loc_ptr','kc_ptr','vc_ptr','ks_ptr','vs_ptr',
              'o_ptr','m_ptr','l_ptr','ticket_ptr','out_ptr','theta','eps','qk_factor',
              'sm_scale','window','page','stride_tok','stride_pt')
CONSTEXPR_ARGS=('L','G','HQ','HKV','D','BLOCK_M','BLOCK_N','BLOCK_L','SPLITS','ROPE',
                'WINDOWED','DOT','QK_DOT','CHUNK_SIZE','SEPARATE_MERGE','PAGE_LOOKAHEAD')
POINTER_TYPES=('*bf16','*i32','*i32','*i64','*fp8e4nv','*fp8e4nv','*fp32','*fp32',
               '*fp32','*fp32','*fp32','*bf16','*bf16')
OPTIONS=dict(num_warps=4,num_stages=3)
TARGET=['cuda',90,32]
MAX_ARTIFACT=8*1024*1024
DEPENDENCY=dict(filename='triton-3.8.0-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl',
                sha256='a9c404c69ed4a39e8ec632eaf6b9fe058a060bf98979c177f6ef666f06bb8d50',size=226474486)
API_SHA={
 'triton/compiler/compiler.py':'28ff26b7e752df455c410418d72fd0afc5b3bfa952ca4bca95b1cfd5a31a0e2f',
 'triton/runtime/jit.py':'38b8151fcceec1a0210a696775b7c64d7fac5e9b69eee2dda7e93c1e2a377509',
 'triton/backends/compiler.py':'c5d3403b88b5bb728f8ac78c55163198e418430f126c842892c68cb4b6baaa78',
 'triton/backends/nvidia/compiler.py':'e898d9a654e9e51aecb43b263fe61102f7162375c4275307fd1830d18b814eb8',
 'triton/experimental/gluon/_runtime.py':'739a9c208c3c73ff5d4ec2d38a4051c20140e4f149c442a05cc37010443d5684',
 'triton/experimental/gluon/language/_layouts.py':'817880af6cd4814150a7312b5fc5484da75c1e876013a95afdc2e045a3d225bb',
 'triton/experimental/gluon/language/nvidia/hopper/__init__.py':'c427a1f7fa5277d701d6cced0dc05ac3af4e5a32f9dc2675699fa254023a99dc',
 'triton/experimental/gluon/language/nvidia/ampere/__init__.py':'4d9b5bf15e328cdfde7bb47904f4dba7b431edf3de61c68906097a10f947d1e5',
 'triton/experimental/gluon/language/nvidia/ampere/async_copy.py':'86e23dfd1231d2709e93a126c91349254440a930ef2196c6d5e3fcefc7e8f623',
}
TOOL_SHA={'ptxas':'d07db8b5fd19692ad6fc9643ad3449cc27190fad1f1ad6c6b8338fdec95ea72e',
          'cuobjdump':'892ae8d0a7bd7cee42db000bbc9be72681952e73a2b2921d1ba3de51b4c472f4'}
IMPORTS={
 'triton.experimental':{'gluon'},
 'triton.experimental.gluon':{'language'},
 'triton.experimental.gluon.language.nvidia.hopper':{'fence_async_shared','warpgroup_mma','warpgroup_mma_wait'},
 'triton.experimental.gluon.language.nvidia.ampere':{'async_copy','mma_v2'},
}


def sha(raw):return hashlib.sha256(raw).hexdigest()


def read_source(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>256*1024:
        raise ValueError('Source path/size guard')
    return path.read_bytes()


def save(path,value):
    raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
    if len(raw)>MAX_ARTIFACT:raise ValueError('Manifest exceeds8MiB')
    temp=path.with_suffix('.tmp');temp.write_bytes(raw);temp.replace(path)


def bounded_write(path,value):
    raw=value if isinstance(value,bytes) else value.encode()
    if len(raw)>MAX_ARTIFACT:raise ValueError('Artifact exceeds8MiB; no truncation')
    path.write_bytes(raw);return sha(raw)


def _module_contract(raw,mode):
    """Small owned JIT-module surface; not a sandbox for arbitrary Python."""
    tree=ast.parse(raw);imports={};functions={};language_aliases=set();jit_aliases=set()
    for node in tree.body:
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):continue
        if isinstance(node,ast.Import):
            for alias in node.names:
                if alias.name=='triton' and alias.asname in (None,'triton'):
                    imports[alias.asname or alias.name]=alias.name;jit_aliases.add('triton')
                elif alias.name=='triton.language' and alias.asname=='tl':
                    imports['tl']=alias.name;language_aliases.add('tl')
                else:raise ValueError('Nonallowlisted JIT module import')
        elif isinstance(node,ast.ImportFrom):
            if node.level or node.module not in IMPORTS:raise ValueError('Nonallowlisted JIT module import')
            for alias in node.names:
                if alias.name not in IMPORTS[node.module]:raise ValueError('Nonallowlisted imported API')
                name=alias.asname or alias.name;imports[name]=node.module+'.'+alias.name
                if alias.name=='language':language_aliases.add(name)
                if alias.name=='gluon':jit_aliases.add(name)
        elif isinstance(node,ast.FunctionDef):functions[node.name]=node
        elif isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            name=node.targets[0].id
            if name=='LOG2E' and isinstance(node.value,ast.Call) and ast.unparse(node.value.func) in ('tl.constexpr','gl.constexpr','_m64n8_gl.constexpr'):
                if len(node.value.args)!=1 or node.value.keywords or ast.literal_eval(node.value.args[0])!=1.4426950408889634:
                    raise ValueError('LOG2E constant changed')
            else:
                if not re.fullmatch(r'[A-Z][A-Z0-9_]*',name):raise ValueError('No host binding in JIT module')
                value=ast.literal_eval(node.value)
                if not isinstance(value,(int,float,bool,tuple)):raise ValueError('No host object construction')
        else:raise ValueError('No host side effects or binder in JIT module')
    if len(functions)!=sum(isinstance(n,ast.FunctionDef) for n in tree.body):raise ValueError('Duplicate JIT function')
    entry='_fused' if mode=='control' else ENTRY
    expected_names={'_norm_rope','_fused'} if mode=='control' else set(functions)
    if entry not in functions or set(functions)!=expected_names:raise ValueError('JIT entry/dependencies missing')
    if mode=='candidate' and any(not name.startswith('_m64n8_') for name in functions):raise ValueError('Owned JIT helper prefix required')
    for fn in functions.values():
        if len(fn.decorator_list)!=1 or not isinstance(fn.decorator_list[0],ast.Attribute):raise ValueError('Direct JIT decorator required')
        d=fn.decorator_list[0]
        if d.attr!='jit' or not isinstance(d.value,ast.Name) or d.value.id not in jit_aliases:
            raise ValueError('Direct allowlisted JIT decorator required')
        for value in [*fn.args.defaults,*[x for x in fn.args.kw_defaults if x is not None]]:
            if any(isinstance(n,ast.Call) for n in ast.walk(value)):raise ValueError('No host call in function default')
        for node in ast.walk(fn):
            if isinstance(node,(ast.Import,ast.ImportFrom,ast.Global,ast.Nonlocal)):
                raise ValueError('No runtime import/global manipulation in owned kernel')
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in {'open','exec','eval','__import__','input','print'}:
                raise ValueError('No host side-effect call in owned kernel')
    fn=functions[entry];args=[a.arg for a in fn.args.args]
    if args!=list(RUNTIME_ARGS+CONSTEXPR_ARGS) or fn.args.posonlyargs or fn.args.kwonlyargs or fn.args.vararg or fn.args.kwarg or fn.args.defaults:
        raise ValueError('Exact full producer ABI required')
    for a in fn.args.args[:len(RUNTIME_ARGS)]:
        if a.annotation is not None:raise ValueError('Runtime argument annotation changed')
    for a in fn.args.args[len(RUNTIME_ARGS):]:
        ann=a.annotation
        if not(isinstance(ann,ast.Attribute) and ann.attr=='constexpr' and isinstance(ann.value,ast.Name) and ann.value.id in language_aliases):
            raise ValueError('Exact constexpr annotation required')
    return dict(entry=entry,functions={name:sha(ast.dump(fn,include_attributes=False).encode()) for name,fn in functions.items()},
                imports=imports,runtime_args=list(RUNTIME_ARGS),constexpr_args=list(CONSTEXPR_ARGS))


def _control_text(raw):
    text=raw.decode();tree=ast.parse(text);lines=text.splitlines(keepends=True)
    functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    pieces=['import triton\nimport triton.language as tl\n']
    assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='LOG2E' for t in n.targets))
    pieces.append(ast.get_source_segment(text,assignment)+'\n\n')
    for name in ('_norm_rope','_fused'):
        node=functions[name];start=min([node.lineno]+[d.lineno for d in node.decorator_list])
        pieces.append(''.join(lines[start-1:node.end_lineno])+'\n\n')
    return ''.join(pieces)


def selected_contract(raw):
    if sha(raw)!=SELECTED_SHA:raise ValueError('Selected source hash mismatch')
    code=_control_text(raw)
    if sha(code.encode())!=CONTROL_SHA:raise ValueError('Known selected extraction hash mismatch')
    contract=_module_contract(code.encode(),'control')
    source={n.name:n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef)}
    extracted={n.name:n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef)}
    for name in ('_norm_rope','_fused'):
        if ast.dump(source[name],include_attributes=False)!=ast.dump(extracted[name],include_attributes=False):
            raise ValueError('Control function AST changed during extraction')
    return dict(contract,selected_sha256=SELECTED_SHA,extracted_source_sha256=sha(code.encode()),
                executed_selected_host_code=False,qualification='Exact selected JIT functions plus LOG2E; all host/torch/sgl code excluded')


def extract_control(raw):
    selected_contract(raw)
    return _control_text(raw)


def candidate_contract(raw,expected_sha):
    if not re.fullmatch(r'[0-9a-f]{64}',expected_sha or '') or sha(raw)!=expected_sha:raise ValueError('Owned candidate hash mismatch')
    return _module_contract(raw,'candidate')


def case_plans():
    cases=[]
    for name,batch,window in [('D1slide',1,2048),('D16slide',16,2048),('D32slide',32,2048),('D32full',32,0)]:
        width=80001 if batch==1 else 86001;chunk=8192 if not window else 0
        splits=math.ceil(width/chunk) if chunk else min(math.ceil(264/(batch*2)),2048//128)
        signature=dict(zip(RUNTIME_ARGS,(*POINTER_TYPES,*('fp32',)*4,'i32','constexpr','i32','i32')))
        signature.update({name:'constexpr' for name in CONSTEXPR_ARGS})
        constants=dict(page=1,L=1,G=8,HQ=16,HKV=2,D=64,BLOCK_M=16,BLOCK_N=64,BLOCK_L=16,
            SPLITS=splits,ROPE=bool(window),WINDOWED=bool(window),DOT='dtype:bf16',QK_DOT='dtype:fp8e4nv',
            CHUNK_SIZE=chunk,SEPARATE_MERGE=True,PAGE_LOOKAHEAD=not bool(window))
        attrs={str(i):[['tt.divisibility',16]] for i in [*range(13),17,19]}
        cases.append(dict(name=name,batch=batch,splits=splits,grid=[batch,2,splits],
            scalar_values=dict(theta=500000. if window else 1.,eps=1e-5,qk_factor=5.4730064863875,
                sm_scale=.125,window=window,page=1,stride_tok=2304,stride_pt=width),
            signature_tokens=signature,constants_tokens=constants,alignment_attrs=attrs,
            qualification='page=1 is specialized constexpr; floats remain dynamic. Integer/pointer attrs match saved selected full-producer TTGIR; no runtime binder/launch used.'))
    return cases


def parse_resources(ptxas_text,cu_text):
    def one(pattern,text):
        values=re.findall(pattern,text,re.I)
        return int(values[0]) if len(values)==1 else None
    return dict(registers_ptxas=one(r'Used\s+(\d+)\s+registers',ptxas_text),
        registers_cuobjdump=one(r'\bREG:(\d+)',cu_text),
        stack_frame_bytes=one(r'(\d+)\s+bytes stack frame',ptxas_text),
        stack_cuobjdump_bytes=one(r'\bSTACK:(\d+)',cu_text),
        spill_store_bytes=one(r'(\d+)\s+bytes spill stores',ptxas_text),
        spill_load_bytes=one(r'(\d+)\s+bytes spill loads',ptxas_text),
        local_bytes=one(r'\bLOCAL:(\d+)',cu_text),static_shared_bytes=one(r'\bSHARED:(\d+)',cu_text))


def instruction_summary(ptx,sass):
    return dict(wgmma_m64n8k16_sites=len(re.findall(r'wgmma\.mma_async[^;\n]*\.m64n8k16\.',ptx)),
        mma_sync_sites=len(re.findall(r'\bmma\.sync\.',ptx)),
        async_copy_sites=len(re.findall(r'\bcp\.async\.(?:ca|cg)\.',ptx)),
        async_commit_sites=len(re.findall(r'\bcp\.async\.commit_group',ptx)),
        async_wait_sites=len(re.findall(r'\bcp\.async\.wait_group',ptx)),
        wgmma_commit_sites=len(re.findall(r'\bwgmma\.commit_group',ptx)),
        wgmma_wait_sites=len(re.findall(r'\bwgmma\.wait_group',ptx)),
        local_load_sites=len(re.findall(r'\bld\.local\.',ptx)),local_store_sites=len(re.findall(r'\bst\.local\.',ptx)),
        sass_local_load_sites=len(re.findall(r'\bLDL(?:\.|\b)',sass)),
        sass_local_store_sites=len(re.findall(r'\bSTL(?:\.|\b)',sass)),
        sass_hgmma_sites=len(re.findall(r'\bHGMMA\.',sass)),
        qualification='Static sites only; mainloop/tail/pipeline and QK/PV dataflow require independent native review, not performance or executed counts')


def resource_gate(metadata,resources,inventory):
    reasons=[]
    for key in ('registers_ptxas','registers_cuobjdump'):
        value=resources.get(key)
        if type(value) is not int or not 0<value<=96:reasons.append(key+'_unknown_or_above96')
    if resources.get('registers_ptxas')!=resources.get('registers_cuobjdump'):reasons.append('register_sources_disagree')
    for key in ('stack_frame_bytes','stack_cuobjdump_bytes','spill_store_bytes','spill_load_bytes','local_bytes'):
        if type(resources.get(key)) is not int or resources[key]!=0:reasons.append(key+'_unknown_or_nonzero')
    for key in ('local_load_sites','local_store_sites','sass_local_load_sites','sass_local_store_sites'):
        if type(inventory.get(key)) is not int or inventory[key]!=0:reasons.append(key+'_unknown_or_nonzero')
    shared=metadata.get('shared');static=resources.get('static_shared_bytes')
    if type(shared) is not int or not 0<=shared<=40*1024:reasons.append('compiler_launch_shared_unknown_or_above40KiB')
    if type(static) is not int or static<0:reasons.append('cubin_static_shared_unknown')
    return dict(passed=not reasons,reasons=reasons,compiler_launch_shared_bytes=shared,cubin_static_shared_bytes=static,
        conservative_shared_sum_bytes=shared+static if type(shared) is int and type(static) is int else None,
        shared_model_with_at_least1024_static_reserve_bytes=shared+max(1024,static) if type(shared) is int and type(static) is int else None,
        qualification='Declared compile-resource screen only. Launch shared and cubin static shared are recorded separately; no driver loading, achieved occupancy or native mechanism pass.')


def prepare(dest,kernel_sha,kernel_path=None,selected_path=None):
    raw=read_source(kernel_path or HERE/'kernel.py');control_raw=read_source(selected_path or SELECTED)
    candidate=candidate_contract(raw,kernel_sha);control=selected_contract(control_raw)
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    if any(dest.iterdir()):raise ValueError('Preparation directory must be empty')
    control_text=extract_control(control_raw).encode()
    (dest/'kernel.py').write_bytes(raw);(dest/'control.py').write_bytes(control_text)
    (dest/'offline_compile.py').write_bytes(Path(__file__).read_bytes())
    manifest=dict(status='prepared_compile_only_not_executed',selected_sha256=SELECTED_SHA,
        kernel_sha256=kernel_sha,helper_sha256=sha(Path(__file__).read_bytes()),
        control_sha256=sha(control_text),control_contract=control,candidate_contract=candidate,
        cases=case_plans(),options=OPTIONS,target=TARGET,dependency=DEPENDENCY,API_source_sha256=API_SHA,
        expected_tool_sha256=TOOL_SHA,control_order=CONTROL_ORDER,candidate_order=CANDIDATE_ORDER,
        stop_rule='Compile four fixed controls first; stop candidate branch on first compiler/resource failure. No variant retries or changed source/options.',
        precision_scope='Control E4M3 Q/K source unchanged; candidate BF16 MMA after same E4M3 encoding. FP32 state/accumulation and BF16 PV, no performance or correctness claim.',
        scalar_policy='page1 constexpr +pointer/window/stride_tok divisibility16; from saved selected full-producer native TTGIR. Not an official judge binary identity.')
    save(dest/'manifest.json',manifest);return manifest


def validate_manifest(m,control_raw,kernel_raw,helper_raw):
    """Validate declarations before loading owned JIT modules or compiler APIs."""
    expected=dict(selected_sha256=SELECTED_SHA,control_sha256=CONTROL_SHA,cases=case_plans(),
        options=OPTIONS,target=TARGET,dependency=DEPENDENCY,API_source_sha256=API_SHA,
        expected_tool_sha256=TOOL_SHA,control_order=CONTROL_ORDER,candidate_order=CANDIDATE_ORDER)
    for key,value in expected.items():
        if m.get(key)!=value:raise ValueError('Frozen manifest field changed: '+key)
    if sha(control_raw)!=CONTROL_SHA:raise ValueError('Known extracted control changed')
    if sha(helper_raw)!=m.get('helper_sha256'):raise ValueError('Frozen helper differs')
    actual_control=_module_contract(control_raw,'control')
    if actual_control!={k:m['control_contract'][k] for k in actual_control}:
        raise ValueError('Control contract changed')
    if candidate_contract(kernel_raw,m['kernel_sha256'])!=m.get('candidate_contract'):
        raise ValueError('Candidate contract changed')


def exception_record(directory,name,exc):
    """Keep full diagnostics within the hard artifact bound; mark refused oversize."""
    message=str(exc);raw=(type(exc).__name__+'\n'+message+'\n').encode()
    record=dict(reason_type=type(exc).__name__,reason=message[:12000],
        reason_summary_truncated=len(message)>12000,error_text_bytes=len(raw),error_text_sha256=sha(raw))
    if len(raw)>MAX_ARTIFACT:
        record['error_artifact_status']='refused_oversize_no_silent_truncation'
    else:
        record.update(error_artifact_status='full_text_saved',
            error_artifact_sha256={name:bounded_write(directory/name,raw)})
    return record


def _load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def compile_payload(directory,wheel_archive=None,deadline=230):
    if platform.system()!='Linux':raise RuntimeError('Pinned Linux CPU-only container required')
    if list(Path('/dev').glob('nvidia*')):raise RuntimeError('NVIDIA devices must be absent')
    if not 1<=deadline<=230:raise ValueError('Compile deadline must be1..230seconds')
    started=time.monotonic();directory=Path(directory)
    def remaining(cap=30):
        value=deadline-(time.monotonic()-started)
        if value<=0:raise TimeoutError('Offline producer compile deadline')
        return min(cap,value)
    m=json.loads((directory/'manifest.json').read_text())
    control_raw=read_source(directory/'control.py');kernel_raw=read_source(directory/'kernel.py')
    validate_manifest(m,control_raw,kernel_raw,read_source(directory/'offline_compile.py'))
    if sha(Path(__file__).read_bytes())!=m['helper_sha256']:raise ValueError('Executing helper differs')
    wheel=Path(wheel_archive) if wheel_archive else directory/DEPENDENCY['filename']
    if wheel.is_symlink() or wheel.stat().st_size!=DEPENDENCY['size']:raise ValueError('Exact cached wheel required')
    wheel_digest=hashlib.sha256()
    with wheel.open('rb') as stream:
        while chunk:=stream.read(1024*1024):remaining();wheel_digest.update(chunk)
    if wheel_digest.hexdigest()!=DEPENDENCY['sha256']:raise ValueError('Pinned compiler wheel hash mismatch')
    import triton
    import triton.language as tl
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource
    from triton.experimental.gluon._runtime import GluonASTSource
    if triton.__version__!='3.8.0':raise ValueError('Pinned Triton3.8.0 required')
    package=Path(triton.__file__).resolve().parent
    for name,expected in API_SHA.items():
        if sha((package.parent/name).read_bytes())!=expected:raise ValueError('Pinned compiler API differs: '+name)
    tools={name:Path(getattr(triton.knobs.nvidia,name).path) for name in TOOL_SHA}
    tools_hash={name:sha(path.read_bytes()) for name,path in tools.items()}
    if tools_hash!=TOOL_SHA:raise ValueError('Pinned assembler/disassembler tool bytes differ')
    versions={name:subprocess.run([str(path),'--version'],capture_output=True,text=True,check=True,
                                  timeout=remaining(10)).stdout for name,path in tools.items()}
    control=_load(directory/'control.py','m64n8_exact_selected_control')
    candidate=_load(directory/'kernel.py','m64n8_owned_full_producer')
    out=directory/'artifacts';out.mkdir(exist_ok=True)
    report=dict(status='compiling',manifest=m,python=platform.python_version(),host_machine=platform.machine(),
        triton=triton.__version__,wheel_archive_sha256=wheel_digest.hexdigest(),compiler_API_source_sha256=API_SHA,
        tool_sha256=tools_hash,tool_versions=versions,units=[],gpu_execution=False,driver_or_handles_initialized=False,
        numerical_correctness=False,performance=False,variant_retries=0)
    save(out/'compiler-manifest.json',report)
    triton.knobs.nvidia.dump_ptxas_log=True
    by_name={c['name']:c for c in case_plans()}
    try:
        for arm in ('control','candidate'):
            names=m['control_order'] if arm=='control' else m['candidate_order']
            for name in names:
                remaining();case=by_name[name];prefix=arm+'.'+name;log=io.StringIO()
                row=dict(arm=arm,case=name,status='starting',signature=case['signature_tokens'],constants=case['constants_tokens'],
                         alignment_attrs=case['alignment_attrs'],options=OPTIONS,grid_not_launched=case['grid'])
                report['units'].append(row)
                signature=case['signature_tokens']
                constants={k:tl.bfloat16 if v=='dtype:bf16' else tl.float8e4nv if v=='dtype:fp8e4nv' else v
                           for k,v in case['constants_tokens'].items()}
                attrs={(int(i),):value for i,value in case['alignment_attrs'].items()}
                fn=control._fused if arm=='control' else getattr(candidate,ENTRY)
                src=ASTSource(fn,signature,constants,attrs) if arm=='control' else GluonASTSource(fn,signature,constants,attrs)
                try:
                    with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                        compiled=triton.compile(src,target=GPUTarget('cuda',90,32),options=OPTIONS)
                except Exception as exc:
                    row.update(exception_record(out,prefix+'.error.txt',exc))
                    row.update(status='compile_failed',
                        artifact_sha256={'compile.log':bounded_write(out/(prefix+'.compile.log'),log.getvalue())})
                    report['status']='stopped_on_compile_failure';save(out/'compiler-manifest.json',report);return report
                hashes={}
                for extension in ('ttgir','llir','ptx','cubin'):
                    if extension not in compiled.asm:raise ValueError('Required compiler artifact missing: '+extension)
                    hashes[extension]=bounded_write(out/(prefix+'.'+extension),compiled.asm[extension])
                if 'ttir' in compiled.asm:hashes['ttir']=bounded_write(out/(prefix+'.ttir'),compiled.asm['ttir'])
                hashes['ptxas.txt']=bounded_write(out/(prefix+'.ptxas.txt'),log.getvalue())
                texts={}
                for flag,extension in [('--dump-resource-usage','resources.txt'),('--dump-sass','sass')]:
                    completed=subprocess.run([str(tools['cuobjdump']),flag,str(out/(prefix+'.cubin'))],
                        capture_output=True,text=True,check=True,timeout=remaining(30))
                    texts[extension]=completed.stdout
                    hashes[extension]=bounded_write(out/(prefix+'.'+extension),completed.stdout)
                raw_meta=compiled.metadata._asdict()
                metadata={k:raw_meta[k] for k in ('name','num_warps','num_ctas','shared','global_scratch_size','global_scratch_align','tmem_size','arch','enable_fp_fusion') if k in raw_meta}
                resources=parse_resources(log.getvalue(),texts['resources.txt'])
                inventory=instruction_summary((out/(prefix+'.ptx')).read_text(),texts['sass'])
                row.update(status='compiled_pending_native_review',metadata=metadata,resources=resources,
                    instruction_inventory=inventory,artifact_sha256=hashes)
                if arm=='candidate':
                    row['resource_gate']=resource_gate(metadata,resources,inventory)
                    if not row['resource_gate']['passed']:
                        report['status']='stopped_on_candidate_resource_gate';save(out/'compiler-manifest.json',report);return report
                save(out/'compiler-manifest.json',report)
        report['status']='offline_matrix_complete_pending_native_dataflow_review'
    except Exception as exc:
        report.update(exception_record(out,'failure.error.txt',exc));report['status']='failed'
        raise
    finally:
        report['wall_seconds']=time.monotonic()-started;save(out/'compiler-manifest.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--preflight',action='store_true');group.add_argument('--prepare',type=Path);group.add_argument('--compile',type=Path)
    parser.add_argument('--kernel-sha');parser.add_argument('--wheel-archive',type=Path);parser.add_argument('--deadline',type=int,default=230)
    args=parser.parse_args()
    if not 1<=args.deadline<=230:parser.error('deadline must be1..230seconds')
    if args.prepare and not args.kernel_sha:parser.error('--prepare requires reviewed --kernel-sha')
    def alarm(*_):raise TimeoutError('230second offline compile alarm')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(args.deadline)
    try:
        if args.preflight:
            result=dict(status='CPU_source_preflight',control=selected_contract(read_source(SELECTED)),cases=case_plans(),
                        candidate_not_admitted=True,gpu_execution=False,driver_or_handles_initialized=False)
        elif args.prepare:result=prepare(args.prepare,args.kernel_sha)
        else:result=compile_payload(args.compile,args.wheel_archive,args.deadline)
        print(json.dumps(result))
        return 0 if result.get('status') not in ('failed','stopped_on_compile_failure','stopped_on_candidate_resource_gate') else 1
    finally:signal.alarm(0)


if __name__=='__main__':raise SystemExit(main())
