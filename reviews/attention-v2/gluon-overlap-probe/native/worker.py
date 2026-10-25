"""Matched Gluon compile-only probe; default local action is CPU preflight.

Typed one-element pointers are undersized and must NEVER be executed. Native
warmup/module load is available only behind explicit --compile, after external
admission. No Modal, credentials, network, profiler or benchmark code.
"""
import argparse
import base64
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PROBE = HERE.parent
REPO = HERE.parents[3]
SOURCE_HASHES = {
    'kernel.py': '3d847a290d747c204419fd131c8149a21c4afddbdfb1d0a6130f5f522b81847c',
    'config.py': 'ecaf6588893ad4ce6bc57944a9932018f511113aeb13a6ee266edcf536d701ec',
    'resources.py': '48adfe47cdb9ca72890b93c816772e454552cf634e5f771cd23e3a14234fbdee',
}
LICENSE_HASHES = {
    'LICENSE': '1495e1e757ef4d0925a2350563cf5754bb23c51701a8ec4fb3c5cdcbedae6747',
    'THIRD_PARTY_NOTICES.md': 'c0417e70786e1094b0741a8b569927d8469cc742bf16e63a2fda2d59ad01a88b',
}
INSTALLED_HASHES = {
    '/opt/sglang/lib/python3.12/site-packages/triton/experimental/gluon/__init__.py': 'd7b607f028fed01715550c5e5ef708b0bd71881ccda220fb7390248ec29ca904',
    '/opt/sglang/lib/python3.12/site-packages/triton/experimental/gluon/language/nvidia/hopper/__init__.py': 'c427a1f7fa5277d701d6cced0dc05ac3af4e5a32f9dc2675699fa254023a99dc',
    '/opt/sglang/lib/python3.12/site-packages/triton/experimental/gluon/language/nvidia/ampere/async_copy.py': '86e23dfd1231d2709e93a126c91349254440a930ef2196c6d5e3fcefc7e8f623',
}
ARMS = [('serial', False), ('overlap', True)]
FORMATS = ['ttir', 'ttgir', 'llir', 'ptx', 'cubin', 'sass']
REQUIRED_FORMATS = {'ttgir', 'ptx', 'cubin'}
ALLOWED = {f'{arm}.core.{kind}' for arm, _ in ARMS for kind in FORMATS}
MAX_REPORT, MAX_UNCOMPRESSED, MAX_FILE = 2*1024*1024, 8*1024*1024, 4*1024*1024

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

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

def source_gate(probe=PROBE, repo=REPO):
    observed = {}
    for root, hashes in [(probe, {name:value for name,value in SOURCE_HASHES.items() if name!='resources.py'}),
                         (HERE, {'resources.py':SOURCE_HASHES['resources.py']}), (repo, LICENSE_HASHES)]:
        for name, expected in hashes.items():
            path = root/name
            if path.is_symlink() or digest(path.read_bytes()) != expected:
                raise ValueError('Exact reviewed source/license mismatch: '+name)
            compile(path.read_bytes(), name, 'exec') if path.suffix == '.py' else None
            observed[name] = expected
    return observed


def installed_gate(hashes=INSTALLED_HASHES, version=None):
    version = version or importlib.metadata.version('triton')
    if version != '3.8.0':
        raise ValueError('Pinned installed Triton3.8.0 required; no fallback')
    observed = {}
    for name, expected in hashes.items():
        path = Path(name)
        if not path.is_file() or path.stat().st_size > 1024*1024 or digest(path.read_bytes()) != expected:
            raise ValueError('Pinned installed Gluon source mismatch; no fallback: '+name)
        observed[name] = expected
    return dict(triton_version=version, source_sha256=observed,
                qualification='Actual CPU-inventoried installed bytes; public-tag source is not assumed identical')


def compile_config():
    config = load(PROBE/'config.py', 'gluon_probe_fixed_config').ProbeConfig()
    kwargs = config.compile_kwargs(False)
    if config.grid != (32, 2) or kwargs != dict(HKV=2, HQ=16, PT_STRIDE=86008,
            SM_SCALE=.125, QK_FACTOR=5.4730064863875, OVERLAP=False, num_warps=4, num_stages=1):
        raise ValueError('One fixed matched configuration required')
    return config


def native_compile(args):
    started = time.monotonic()
    report = dict(status='starting', passed=False, compile_complete=False, kernel_launches=0,
        correctness_tested=False, performance_tested=False, execution_admitted=False,
        source_sha256=SOURCE_HASHES, worker_sha256=digest(Path(__file__).read_bytes()),
        resources_sha256=digest((HERE/'resources.py').read_bytes()),
        started_utc=datetime.now(timezone.utc).isoformat(), kernels=[], artifact_bundle=None,
        protocol=dict(compile_only=True, arms=['serial','overlap'], only_delta='OVERLAP constexpr',
            grid=[32,2], M=64,N=64,D=64,L=8,G=8,num_warps=4,num_stages=1,page=1,
            QK_FACTOR=5.4730064863875, max_num_imprecise_acc=0,
            required_artifacts=sorted(REQUIRED_FORMATS), optional_artifacts=['ttir','llir','sass'],
            resource_gate='Zero spills/local memory; shared<=64KiB; overlap maximum CTA/SM >=serial using actual CUDA limits',
            mechanism_gate='Readable exact ISA and next-tile gather/convert/shared-store before current QK wait with safe buffer ownership; independent review required',
            precision='Unscaled post-RMS/RoPE BF16Q ->E4M3->BF16; E4M3K/V->BF16; FP32softmax/state; scale once afterQK',
            accumulation='Both arms use zero-PV then acc*alpha+PV; old TL carry-C bitwise identity is NOT claimed',
            standalone=True, production_entrypoint=False))
    folder = args.artifact_dir or args.output.with_name('gluon-matched-compile-artifacts')
    def save():
        report['worker_wall_seconds'] = time.monotonic()-started
        raw = json.dumps(report,separators=(',', ':')).encode()+b'\n'
        if len(raw)>MAX_REPORT:
            raise ValueError('Worker report exceeds bounded2MiB transport')
        temp=args.output.with_suffix('.tmp'); temp.write_bytes(raw); temp.replace(args.output)
    def gate():
        save()
        if time.monotonic()-started>=args.deadline:
            raise TimeoutError('Gluon compile deadline')
    def alarm(*_):
        raise TimeoutError('Gluon compile alarm')
    if not 0<args.deadline<=90:
        raise ValueError('Deadline must be positive and <=90s')
    signal.signal(signal.SIGALRM,alarm); signal.alarm(math.ceil(args.deadline))
    save()
    try:
        report['verified_hashes']=source_gate()
        config=compile_config()
        report['installed_source_gate']=installed_gate()  # BEFORE loading the probe/Torch/Gluon.
        resources=load(HERE/'resources.py','gluon_probe_resource_gates')
        folder.mkdir(parents=True,exist_ok=False)
        if os.environ.get('TRITON_INTERPRET'):
            raise ValueError('Native compilation required')
        import torch
        import triton
        if triton.__version__!='3.8.0' or not torch.cuda.is_available() or 'H200' not in torch.cuda.get_device_name():
            raise ValueError('Pinned native H200/Triton required')
        report['environment']=dict(torch=torch.__version__,triton=triton.__version__,cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),python=platform.python_version())
        module=load(PROBE/'kernel.py','gluon_matched_source')
        tool=find_cuobjdump(triton); report['cuobjdump_path']=tool
        dtypes=[torch.bfloat16,torch.float8_e4m3fn,torch.float8_e4m3fn,torch.int64,
                torch.int32,torch.float32,torch.float32,torch.bfloat16]
        tensors=[torch.empty(1,dtype=dtype,device='cuda') for dtype in dtypes]
        report['pointer_metadata']=[dict(dtype=str(value.dtype),elements=value.numel(),alignment_mod16=value.data_ptr()%16) for value in tensors]
        if any(value.data_ptr()%16 for value in tensors):
            raise ValueError('One-element compile pointers must be aligned')
        for arm,overlap in ARMS:
            gate(); before=time.monotonic(); options=config.compile_kwargs(overlap)
            kernel=module.attention_overlap_kernel.warmup(*tensors,grid=config.grid,**options)
            kernel._init_handles()  # Driver module loading/resource query; no invocation.
            row=dict(arm=arm,unit='core',options=options,compile_and_load_seconds=time.monotonic()-before,
                registers=getattr(kernel,'n_regs',None),driver_spills_raw=getattr(kernel,'n_spills',None),
                shared_memory_bytes=getattr(kernel.metadata,'shared',None),
                metadata=metadata_value(kernel.metadata._asdict() if hasattr(kernel.metadata,'_asdict') else str(kernel.metadata)),
                driver_module_loaded=True,kernel_launched=False,artifacts={},sass_available=False)
            report['kernels'].append(row)
            for kind in FORMATS[:-1]:
                value=kernel.asm.get(kind)
                if value is None:
                    if kind in REQUIRED_FORMATS: raise ValueError('Required compiler artifact missing: '+kind)
                    continue
                raw=value.encode() if isinstance(value,str) else bytes(value)
                if len(raw)>MAX_FILE: raise ValueError('Compiler artifact exceeds per-file bound')
                name=f'{arm}.core.{kind}'; (folder/name).write_bytes(raw)
                row['artifacts'][kind]=dict(file=name,size=len(raw),sha256=digest(raw))
            if tool is not None:
                dis=subprocess.run([tool,'--dump-sass',str(folder/f'{arm}.core.cubin')],capture_output=True,timeout=10)
                row['cuobjdump_exit_code']=dis.returncode
                if dis.returncode==0 and 0<len(dis.stdout)<=MAX_FILE:
                    name=f'{arm}.core.sass'; (folder/name).write_bytes(dis.stdout)
                    row['artifacts']['sass']=dict(file=name,size=len(dis.stdout),sha256=digest(dis.stdout))
                    row['sass_available']=True
                else:
                    row['sass_unavailable_reason']='Existing cuobjdump failed or output out of bounds'
            else:
                row['sass_unavailable_reason']='Existing cuobjdump unavailable; no installation/fallback'
            ptx=(folder/f'{arm}.core.ptx').read_text()
            sass=(folder/f'{arm}.core.sass').read_text() if row['sass_available'] else ''
            row['instruction_inventory']=resources.instruction_inventory(ptx,sass)
            row['cuda_resources']=resources.inspect_resources(kernel,4,Path(triton.__file__).parent)
            gate()
        report['resource_gate']=resources.resource_gate(report['kernels'])
        primitives=all(row['instruction_inventory']['ptx_wgmma_async']>0 and row['instruction_inventory']['ptx_wgmma_commit']>0
            and row['instruction_inventory']['ptx_wgmma_wait']>0 and row['instruction_inventory']['sass_hgmma']>0 for row in report['kernels'])
        readable=all(row['sass_available'] for row in report['kernels'])
        status=report['resource_gate']['status']
        report['mechanism_status']='reject' if status=='reject' or (readable and not primitives) else 'review_pending' if status=='pass' and readable and primitives else 'inconclusive'
        report['mechanism_qualification']='Mnemonic presence is not next-tile dataflow or wait/ownership proof; no execution/performance admission'
        report['artifact_bundle']=bundle_artifacts(folder)
        report.update(status='compiled',passed=True,compile_complete=True)
        save(); return 0
    except Exception as exc:
        report.update(status='incomplete_or_failed',passed=False,error_type=type(exc).__name__,error=str(exc)[:1500])
        try:
            if folder.is_dir(): report['artifact_bundle']=bundle_artifacts(folder)
        except Exception as packaging:
            report['artifact_packaging_error']=type(packaging).__name__
        save(); return 1
    finally:
        signal.alarm(0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compile',action='store_true',help='Explicit native compile only after external admission; never executes kernel')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--artifact-dir',type=Path)
    parser.add_argument('--deadline',type=float,default=90.)
    args=parser.parse_args()
    if args.compile:
        if args.output is None: parser.error('--compile requires --output')
        raise SystemExit(native_compile(args))
    print(json.dumps(dict(status='offline-preflight',verified_hashes=source_gate(),
        grid=list(compile_config().grid),native_compilation=False,kernel_launches=0),indent=2))
