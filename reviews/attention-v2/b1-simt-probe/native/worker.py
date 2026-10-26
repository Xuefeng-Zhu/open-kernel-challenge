"""Bounded H200 compile/disassembly artifact probe; no candidate kernel launch.

One-element typed allocator pointers are deliberately undersized. They may only
be passed to Triton warmup and module-handle initialization, never execution.
No Modal, credentials, networking, profiler or benchmark code is present.
"""
import argparse
import ast
import ctypes
import base64
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PROBE = HERE.parent
REPO = HERE.parents[3]
SOURCE_HASHES = {
    'baseline.py': '3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8',
    'candidate.py': 'f5161be2e91b05fceb01e28564f96eecff471244704d036fa4cc7950dbe0ef34',
}
LICENSE_HASHES = {
    'LICENSE': '1495e1e757ef4d0925a2350563cf5754bb23c51701a8ec4fb3c5cdcbedae6747',
    'THIRD_PARTY_NOTICES.md': 'c0417e70786e1094b0741a8b569927d8469cc742bf16e63a2fda2d59ad01a88b',
}
UNITS = [('baseline', 'producer'), ('candidate', 'producer'), ('shared', 'merge')]
FORMATS = ['ttir', 'ttgir', 'llir', 'ptx', 'cubin', 'sass']
ALLOWED = {f'{arm}.{unit}.{kind}' for arm,unit in UNITS for kind in FORMATS}
MAX_REPORT = 2 * 1024 * 1024
MAX_UNCOMPRESSED = 8 * 1024 * 1024
MAX_FILE = 4 * 1024 * 1024


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def source_gate(probe=PROBE, repo=REPO):
    observed = {}
    for root, hashes in [(probe, SOURCE_HASHES), (repo, LICENSE_HASHES)]:
        for name, expected in hashes.items():
            path = root/name
            if path.is_symlink():
                raise ValueError('Symlink source/license refused: ' + name)
            got = digest(path.read_bytes())
            if got != expected:
                raise ValueError('Frozen source/license mismatch: ' + name)
            observed[name] = got
    return observed


def producer_options(candidate=False):
    options = dict(L=1,G=8,HQ=16,HKV=2,D=64,BLOCK_M=8 if candidate else 16,
        BLOCK_N=64,BLOCK_L=16,SPLITS=16,ROPE=True,WINDOWED=True,CHUNK_SIZE=0,
        SEPARATE_MERGE=True,PAGE_LOOKAHEAD=False,num_warps=4,num_stages=3)
    if candidate: options['PARTIAL_M']=16
    return options


def merge_options():
    return dict(L=1,G=8,HQ=16,HKV=2,D=64,BLOCK_M=16,SPLITS=16,BS=16,num_warps=4,num_stages=3)


def same_merge(probe=PROBE):
    segments=[]
    for arm in ('baseline','candidate'):
        text=(probe/(arm+'.py')).read_text(); tree=ast.parse(text)
        node=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='_merge')
        segments.append(ast.get_source_segment(text,node))
    if segments[0]!=segments[1]: raise ValueError('Identical shared merge required')
    return digest(segments[0].encode())


def ptx_instruction_counts(text):
    """Static textual instruction inventory, not executed counts or bottlenecks."""
    counts = {}
    for line in text.splitlines():
        line = line.split('//', 1)[0]
        match = re.match(r'^\s*(?:@!?[%\w]+\s+)?([a-z][a-z0-9_.]*)(?=\s|;)', line)
        if match:
            opcode = match.group(1)
            counts[opcode] = counts.get(opcode, 0) + 1
    families = {
        'mma': r'^mma\.', 'wgmma': r'^wgmma\.',
        'fp32_mul': r'^mul(?:\.[a-z0-9]+)*\.f32$',
        'fp32_fma': r'^fma(?:\.[a-z0-9]+)*\.f32$',
        'fp32_add': r'^add(?:\.[a-z0-9]+)*\.f32$',
        'exp2': r'^ex2\.', 'branch': r'^bra(?:\.|$)', 'select': r'^selp\.',
        'shuffle': r'^shfl\.', 'vote': r'^(?:vote|redux)\.',
        'global_load': r'^ld\.global', 'shared_load': r'^ld\.shared',
        'shared_store': r'^st\.shared', 'local_load': r'^ld\.local',
        'local_store': r'^st\.local', 'async_copy': r'^cp\.async',
        'barrier': r'^(?:bar|mbarrier)\.',
    }
    return dict(opcodes=dict(sorted(counts.items())),
        families={name: sum(value for opcode, value in counts.items()
                            if re.search(pattern, opcode))
                  for name, pattern in families.items()},
        qualification='Static textual inventory; no execution frequency, causal bottleneck or numerical-equivalence inference')


def bundle_artifacts(folder):
    """Strict relative-name JSON archive; deterministic gzip, bounded transport."""
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError('Artifact directory is not a regular private directory')
    entries = sorted(folder.iterdir())
    if len(entries) > len(ALLOWED):
        raise ValueError('Too many artifact files')
    contents, manifest = {}, {}
    for path in entries:
        if path.name not in ALLOWED or path.is_symlink() or not path.is_file():
            raise ValueError('Artifact path/type outside allowlist')
        if path.stat().st_size > MAX_FILE:
            raise ValueError('Artifact exceeds per-file bound')
        raw = path.read_bytes()
        contents[path.name] = base64.b64encode(raw).decode('ascii')
        manifest[path.name] = dict(sha256=digest(raw), size=len(raw))
    raw = json.dumps(dict(files=contents), sort_keys=True, separators=(',', ':')).encode()
    if len(raw) > MAX_UNCOMPRESSED:
        raise ValueError('Artifact archive exceeds uncompressed bound')
    packed = gzip.compress(raw, compresslevel=9, mtime=0)
    encoded = base64.b64encode(packed).decode('ascii')
    if len(encoded) > MAX_REPORT - 65536:
        raise ValueError('Artifact archive exceeds encoded transport bound')
    return dict(encoding='gzip+base64', data=encoded,
        compressed_sha256=digest(packed), compressed_size=len(packed),
        uncompressed_sha256=digest(raw), uncompressed_size=len(raw), files=manifest)


def metadata_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): metadata_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [metadata_value(item) for item in value]
    return repr(value)[:1024]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def find_cuobjdump(triton):
    found = shutil.which('cuobjdump')
    candidates = [Path(found)] if found else []
    candidates += [Path('/usr/local/cuda/bin/cuobjdump'),
                   Path(triton.__file__).parent/'backends/nvidia/bin/cuobjdump']
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    return None


DEVICE_ATTRIBUTES = {
    'warp_size': 'CU_DEVICE_ATTRIBUTE_WARP_SIZE',
    'max_threads_block': 'CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK',
    'max_threads_sm': 'CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR',
    'max_registers_block': 'CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK',
    'max_registers_sm': 'CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR',
    'max_shared_sm': 'CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR',
    'max_shared_block_optin': 'CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN',
    'max_blocks_sm': 'CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR',
}
FUNCTION_ATTRIBUTES = {
    'static_shared_bytes': 'CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES',
    'local_bytes': 'CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES',
    'registers_per_thread': 'CU_FUNC_ATTRIBUTE_NUM_REGS',
}


def header_enums(text):
    result = {}
    for name in [*DEVICE_ATTRIBUTES.values(), *FUNCTION_ATTRIBUTES.values()]:
        match = re.search(r'\b'+re.escape(name)+r'\s*=\s*(\d+)\b', text)
        if match:
            result[name] = int(match.group(1))
    return result


def inspect_resources(kernel, num_warps, triton_root):
    """Query already-installed driver APIs after warmup/module load; never launch."""
    result = dict(status='inconclusive', kernel_launched=False,
                  qualification='CUDA occupancy legal maximum, not achieved residency')
    try:
        headers = [Path('/usr/local/cuda/include/cuda.h'),
            Path('/usr/local/cuda/targets/x86_64-linux/include/cuda.h'),
            Path(triton_root)/'backends/nvidia/include/cuda.h']
        selected = next((path for path in headers if path.is_file() and path.stat().st_size <= 4*1024*1024), None)
        if selected is None:
            raise ValueError('Existing CUDA enum header unavailable')
        raw = selected.read_bytes()
        enums = header_enums(raw.decode('utf-8'))
        required = set(DEVICE_ATTRIBUTES.values()) | set(FUNCTION_ATTRIBUTES.values())
        if set(enums) != required:
            raise ValueError('Exact existing CUDA enum values unavailable')
        result.update(header=str(selected), header_sha256=hashlib.sha256(raw).hexdigest())
        driver = ctypes.CDLL('libcuda.so.1')
        current = driver.cuCtxGetDevice
        current.argtypes, current.restype = [ctypes.POINTER(ctypes.c_int)], ctypes.c_int
        device = ctypes.c_int()
        if current(ctypes.byref(device)) != 0:
            raise ValueError('Current CUDA device metadata unavailable')
        get_device = driver.cuDeviceGetAttribute
        get_device.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        get_device.restype = ctypes.c_int
        limits = {}
        for key, name in DEVICE_ATTRIBUTES.items():
            value = ctypes.c_int()
            if get_device(ctypes.byref(value), enums[name], device.value) != 0 or value.value <= 0:
                raise ValueError('Actual CUDA device limit unavailable: '+key)
            limits[key] = value.value
        handle = ctypes.c_void_p(int(kernel.function))
        get_function = driver.cuFuncGetAttribute
        get_function.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_void_p]
        get_function.restype = ctypes.c_int
        attributes = {}
        for key, name in FUNCTION_ATTRIBUTES.items():
            value = ctypes.c_int()
            if get_function(ctypes.byref(value), enums[name], handle) != 0 or value.value < 0:
                raise ValueError('Actual CUDA function attribute unavailable: '+key)
            attributes[key] = value.value
        dynamic_shared = int(kernel.metadata.shared)
        threads = num_warps*limits['warp_size']
        occupancy = driver.cuOccupancyMaxActiveBlocksPerMultiprocessor
        occupancy.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        occupancy.restype = ctypes.c_int
        blocks = ctypes.c_int()
        code = occupancy(ctypes.byref(blocks), handle, threads, dynamic_shared)
        if code != 0 or blocks.value <= 0 or blocks.value > limits['max_blocks_sm']:
            raise ValueError('Compiled maximum CTA/SM unavailable')
        result.update(status='complete', device_limits=limits, function_attributes=attributes,
            threads_per_block=threads, dynamic_shared_bytes=dynamic_shared,
            total_shared_bytes=attributes['static_shared_bytes']+dynamic_shared,
            static_max_cta_per_sm=blocks.value,
            dynamic_shared_basis='Triton launch metadata.shared; CUDA API accounts for static function shared allocation')
    except Exception as exc:
        result.update(error_type=type(exc).__name__, reason=str(exc)[:500])
    return result




def resource_screen(rows):
    reasons,unknown=[],[]
    if {(row.get('arm'),row.get('unit')) for row in rows} != set(UNITS) or len(rows)!=3:
        unknown.append('Exactly control producer, SIMT producer and one shared merge required')
    for row in rows:
        label=row.get('arm','?')+'.'+row.get('unit','?')
        metadata=row.get('cuda_resources',{})
        if row.get('driver_spills_raw') is None: unknown.append(label+': spill metadata unavailable')
        elif row['driver_spills_raw']!=0: reasons.append(label+': nonzero spills')
        if not row.get('sass_available'): unknown.append(label+': readable exact SASS unavailable')
        if row.get('local_instruction_count') is None: unknown.append(label+': local instruction inventory unavailable')
        elif row['local_instruction_count']!=0: reasons.append(label+': local-memory instructions')
        if metadata.get('status')!='complete': unknown.append(label+': actual CUDA metadata unavailable')
        else:
            if metadata['total_shared_bytes']>65536: reasons.append(label+': shared exceeds64KiB')
            if metadata['function_attributes']['local_bytes']!=0: reasons.append(label+': local bytes nonzero')
            if metadata['static_max_cta_per_sm']<2: reasons.append(label+': legal maximum below2CTA/SM')
            if metadata['threads_per_block']!=128 or metadata['function_attributes']['registers_per_thread']!=row.get('registers'):
                unknown.append(label+': exact block/register metadata mismatch')
        if row.get('arm')=='candidate' and row.get('unit')=='producer':
            if row.get('registers') is None: unknown.append('SIMT register count unavailable')
            elif row['registers']>192: reasons.append('SIMT producer exceeds prospective192register budget')
            counts=row.get('ptx_instruction_counts',{}).get('families',{})
            if counts.get('mma',0) or counts.get('wgmma',0) or row.get('sass_tensor_mma_count',0):
                reasons.append('Tensor MMA present in SIMT producer')
    return dict(status='reject' if reasons else 'inconclusive' if unknown else 'pass',reasons=reasons+unknown,
        execution_admitted=False,qualification='Resource screen only; 8-row state/no3D/no8xreload/ownership need independent native review; no performance claim')


def main(args):
    started=time.monotonic()
    report=dict(status='starting',passed=False,compile_complete=False,kernel_launches=0,
        source_sha256=SOURCE_HASHES,worker_sha256=digest(Path(__file__).read_bytes()),kernels=[],artifact_bundle=None,
        correctness_tested=False,performance_tested=False,execution_admitted=False,
        protocol=dict(compile_only=True,fixed_units=3,units=UNITS,producer_grid=[1,2,16],merge_grid=[1,16],
            target=dict(batch=1,q_len=1,heads='tp8',D=64,window=2048,page=1,kv_scale=1.,ROPE=True),
            strict_math='FP8QKencoding, explicitBF16 boundaries, FP32SIMTreductions, PBF16/PVFP32, originaltoleranceslater',
            resource_budget=dict(candidate_max_registers=192,max_shared_bytes=65536,min_legal_cta_per_sm=2,
                spills=0,local_memory=0,block_threads=128),
            predeclared='Rank2[N,D] only; true8headstate; noTensorMMA/no8xKVreload; partialM16/newKVlastsplit/mergeunchanged',
            prospective_runtime_gate='Originaloracle/replay/caches first, then >=15% actual-best complete-call reduction per fresh fixture; not current evidence'))
    folder=args.artifact_dir or args.output.with_name('b1-simt-compile-artifacts')
    def save():
        report['worker_wall_seconds']=time.monotonic()-started
        raw=json.dumps(report,separators=(',',':')).encode()+b'\n'
        if len(raw)>MAX_REPORT: raise ValueError('Bounded2MiB report exceeded')
        temporary=args.output.with_suffix('.tmp');temporary.write_bytes(raw);temporary.replace(args.output)
    def gate():
        save()
        if time.monotonic()-started>=args.deadline: raise TimeoutError('SIMT compile deadline')
    def alarm(*_): raise TimeoutError('SIMT compile alarm')
    if not 0<args.deadline<=90: raise ValueError('Deadline must be positive and <=90s')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(math.ceil(args.deadline));save()
    try:
        report['verified_hashes']=source_gate();report['shared_merge_sha256']=same_merge()
        folder.mkdir(parents=True,exist_ok=False)
        if os.environ.get('TRITON_INTERPRET'): raise ValueError('Native compilation required')
        import torch
        import triton
        import triton.language as tl
        if triton.__version__!='3.8.0' or not torch.cuda.is_available() or 'H200' not in torch.cuda.get_device_name():
            raise ValueError('ExistingH200/Triton3.8.0 required')
        report['environment']=dict(torch=torch.__version__,triton=triton.__version__,cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),python=platform.python_version())
        tool=find_cuobjdump(triton);report['cuobjdump_path']=tool
        dtypes=dict(qkv=torch.bfloat16,seq=torch.int32,pt=torch.int32,loc=torch.int64,kc=torch.float8_e4m3fn,
            vc=torch.float8_e4m3fn,ks=torch.float32,vs=torch.float32,po=torch.float32,pm=torch.float32,pl=torch.float32,out=torch.bfloat16)
        tensors={name:torch.empty(1,dtype=dtype,device='cuda') for name,dtype in dtypes.items()}
        report['pointer_metadata']={name:dict(dtype=str(value.dtype),elements=value.numel(),alignment_mod16=value.data_ptr()%16) for name,value in tensors.items()}
        if any(value.data_ptr()%16 for value in tensors.values()): raise ValueError('Compile pointer alignment changed')
        t=tensors
        arguments=[t[name] for name in ['qkv','seq','pt','loc','kc','vc','ks','vs','po','pm','pl']]
        arguments += [t['out'],t['out'],500000.,1e-5,5.4730064863875,.125,2048,1,2304,80001]
        report['producer_runtime_scalars']=arguments[-8:]
        modules={arm:load(PROBE/(arm+'.py'),'b1_simt_compile_'+arm) for arm in ['baseline','candidate']}
        for arm,unit in UNITS:
            gate();before=time.monotonic()
            if unit=='producer':
                options=producer_options(arm=='candidate')
                function=modules[arm]._fused_b1_simt if arm=='candidate' else modules[arm]._fused
                kernel=function.warmup(*arguments,DOT=tl.bfloat16,QK_DOT=tl.float8e4nv,grid=(1,2,16),**options)
            else:
                options=merge_options()
                kernel=modules['baseline']._merge.warmup(t['qkv'],t['vs'],t['po'],t['pm'],t['pl'],t['out'],1e-5,2304,
                    grid=(1,16),**options)
            kernel._init_handles()
            row=dict(arm=arm,unit=unit,options=options,compile_and_load_seconds=time.monotonic()-before,
                registers=getattr(kernel,'n_regs',None),driver_spills_raw=getattr(kernel,'n_spills',None),
                shared_memory_bytes=getattr(kernel.metadata,'shared',None),driver_module_loaded=True,kernel_launched=False,
                metadata=metadata_value(kernel.metadata._asdict() if hasattr(kernel.metadata,'_asdict') else str(kernel.metadata)),
                artifacts={},sass_available=False)
            report['kernels'].append(row)
            for kind in FORMATS[:-1]:
                value=kernel.asm.get(kind)
                if value is None: raise ValueError('Compiler artifact missing: '+kind)
                raw=value.encode() if isinstance(value,str) else bytes(value)
                if len(raw)>MAX_FILE: raise ValueError('Artifact file exceeds bound')
                name=f'{arm}.{unit}.{kind}';(folder/name).write_bytes(raw)
                row['artifacts'][kind]=dict(file=name,size=len(raw),sha256=digest(raw))
                if kind=='ptx': row['ptx_instruction_counts']=ptx_instruction_counts(raw.decode())
            if tool is not None:
                dis=subprocess.run([tool,'--dump-sass',str(folder/f'{arm}.{unit}.cubin')],capture_output=True,timeout=10)
                row['cuobjdump_exit_code']=dis.returncode
                if dis.returncode==0 and 0<len(dis.stdout)<=MAX_FILE:
                    name=f'{arm}.{unit}.sass';(folder/name).write_bytes(dis.stdout)
                    row['artifacts']['sass']=dict(file=name,size=len(dis.stdout),sha256=digest(dis.stdout));row['sass_available']=True
            ptx=(folder/f'{arm}.{unit}.ptx').read_text()
            sass=(folder/f'{arm}.{unit}.sass').read_text() if row['sass_available'] else ''
            row['local_instruction_count']=len(re.findall(r'\b(?:ld|st)\.local\b',ptx))+len(re.findall(r'\b(?:LDL|STL)(?:\.|\s)',sass)) if row['sass_available'] else None
            row['sass_tensor_mma_count']=len(re.findall(r'\b(?:HGMMA|HMMA|IMMA|BMMA)(?:\.|\s)',sass)) if row['sass_available'] else None
            row['cuda_resources']=inspect_resources(kernel,4,Path(triton.__file__).parent)
            gate()
        report['resource_gate']=resource_screen(report['kernels'])
        report['structural_status']='reject' if report['resource_gate']['status']=='reject' else 'review_pending' if report['resource_gate']['status']=='pass' else 'inconclusive'
        report['artifact_bundle']=bundle_artifacts(folder)
        report.update(status='compiled',passed=True,compile_complete=True,
            qualification='Compile-only; independent actualIR/native8head/no3D/no8xreload/ownership review mandatory; no execution/performance admission')
        save();return 0
    except Exception as exc:
        report.update(status='incomplete_or_failed',passed=False,error_type=type(exc).__name__,error=str(exc)[:1500])
        try:
            if folder.is_dir():report['artifact_bundle']=bundle_artifacts(folder)
        except Exception as packaging: report['artifact_packaging_error']=type(packaging).__name__
        save();return 1
    finally:signal.alarm(0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compile',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--artifact-dir',type=Path)
    parser.add_argument('--deadline',type=float,default=90.)
    args=parser.parse_args()
    if args.compile:
        if args.output is None:parser.error('--compile requires --output')
        raise SystemExit(main(args))
    print(json.dumps(dict(status='offline-preflight',verified_hashes=source_gate(),shared_merge_sha256=same_merge(),kernel_launches=0,native_compilation=False),indent=2))
