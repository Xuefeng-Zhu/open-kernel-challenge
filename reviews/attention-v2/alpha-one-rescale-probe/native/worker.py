"""Bounded H200 compile/disassembly artifact probe; no candidate kernel launch.

One-element typed allocator pointers are deliberately undersized. They may only
be passed to Triton warmup and module-handle initialization, never execution.
No Modal, credentials, networking, profiler or benchmark code is present.
"""
import argparse
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
    'candidate.py': 'e66f502812f10f070170efcc9a41a7ef566576799d7461d27cd860a10c94f764',
}
LICENSE_HASHES = {
    'LICENSE': '1495e1e757ef4d0925a2350563cf5754bb23c51701a8ec4fb3c5cdcbedae6747',
    'THIRD_PARTY_NOTICES.md': 'c0417e70786e1094b0741a8b569927d8469cc742bf16e63a2fda2d59ad01a88b',
}
ARMS = ['baseline', 'candidate']
UNITS = ['producer', 'merge']
FORMATS = ['ttir', 'ttgir', 'llir', 'ptx', 'cubin', 'sass']
ALLOWED = {f'{arm}.{unit}.{kind}' for arm in ARMS for unit in UNITS for kind in FORMATS}
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


def producer_options():
    return dict(L=1, G=8, HQ=16, HKV=2, D=64, BLOCK_M=16,
        BLOCK_N=64, BLOCK_L=16, SPLITS=11, ROPE=False, WINDOWED=False,
        CHUNK_SIZE=8192, SEPARATE_MERGE=True, PAGE_LOOKAHEAD=True,
        num_warps=4, num_stages=3)


def merge_options():
    return dict(L=1, G=8, HQ=16, HKV=2, D=64, BLOCK_M=16,
        SPLITS=11, BS=16, num_warps=4, num_stages=3)


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


def main(args):
    started = time.monotonic()
    report = dict(status='starting', passed=False, compile_complete=False,
        source_sha256=SOURCE_HASHES, worker_sha256=digest(Path(__file__).read_bytes()),
        kernel_launches=0, correctness_tested=False, performance_tested=False,
        started_utc=datetime.now(timezone.utc).isoformat(), kernels=[],
        artifact_bundle=None,
        protocol=dict(compile_only=True, kernel_launches=0, fixed_units=4,
            compiler_method='Triton warmup followed by driver module-handle initialization; no kernel launch',
            cuobjdump_timeout_seconds=10, deadline_seconds=args.deadline,
            logical_fixture=dict(batch=32, q_len=1, heads='tp8', layer='full', ctx=80000,
                spread='agentic', page=1, page_table_columns=86001,
                qkv_width=2304, producer_grid=[32, 2, 11], merge_grid=[32, 16]),
            precision='E4M3 Q/K encoding, max_num_imprecise_acc=0 in exact source, BF16 PV, FP32 softmax/partials',
            false_specialization='Not compiled in this bounded first probe; existing constexpr-false AST identity is separate source evidence',
            semantic_admission=False, artifact_formats=FORMATS,
            transport='Deterministic gzip JSON containing base64 file values; whole report <=2MiB'))
    folder = args.artifact_dir or args.output.with_name('alpha-one-native-artifacts')

    def save():
        report['worker_wall_seconds'] = time.monotonic()-started
        encoded = json.dumps(report, separators=(',', ':')).encode() + b'\n'
        if len(encoded) > MAX_REPORT:
            raise ValueError('Whole worker report exceeds 2MiB bound')
        temporary = args.output.with_suffix('.tmp')
        temporary.write_bytes(encoded)
        temporary.replace(args.output)

    def gate():
        save()
        if time.monotonic()-started >= args.deadline:
            raise TimeoutError('Compile-only worker deadline')

    def alarm(*_):
        raise TimeoutError('Compile-only worker alarm')

    if not 0 < args.deadline <= 90:
        raise ValueError('Worker deadline must be positive and at most 90 seconds')
    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(math.ceil(args.deadline))
    save()
    try:
        report['verified_hashes'] = source_gate()
        folder.mkdir(parents=True, exist_ok=False)
        if os.environ.get('TRITON_INTERPRET'):
            raise RuntimeError('Native compilation required')
        import torch
        import triton
        import triton.language as tl
        if not torch.cuda.is_available() or 'H200' not in torch.cuda.get_device_name():
            raise RuntimeError('Existing native H200 required')
        driver = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                capture_output=True, text=True, timeout=10)
        report['environment'] = dict(python=platform.python_version(), torch=torch.__version__,
            triton=triton.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
            capability=torch.cuda.get_device_capability(),
            driver=driver.stdout.strip() if driver.returncode == 0 else None)
        tool = find_cuobjdump(triton)
        report['cuobjdump_path'] = tool
        if tool is None:
            raise RuntimeError('Already-installed cuobjdump unavailable; no installation attempted')
        dtypes = dict(qkv=torch.bfloat16, seq=torch.int32, pt=torch.int32,
            loc=torch.int64, kc=torch.float8_e4m3fn, vc=torch.float8_e4m3fn,
            ks=torch.float32, vs=torch.float32, po=torch.float32,
            pm=torch.float32, pl=torch.float32, out=torch.bfloat16)
        tensors = {name: torch.empty(1, dtype=dtype, device='cuda') for name, dtype in dtypes.items()}
        report['pointer_metadata'] = {name: dict(dtype=str(value.dtype), elements=value.numel(),
            alignment_mod16=value.data_ptr()%16) for name, value in tensors.items()}
        if not all(value.data_ptr()%16 == 0 for value in tensors.values()):
            raise AssertionError('Compile pointer alignment differs from frozen specialization')
        t = tensors
        arguments = [t[name] for name in ['qkv', 'seq', 'pt', 'loc', 'kc', 'vc', 'ks', 'vs', 'po', 'pm', 'pl']]
        arguments += [t['out'], t['out'], 1.0, 1e-5, 5.4730064863875, .125, 0, 1, 2304, 86001]
        report['producer_runtime_scalars'] = arguments[-8:]
        for arm in ARMS:
            gate()
            module = load(PROBE/(arm+'.py'), 'alpha_compile_'+arm)
            for unit in UNITS:
                gate()
                before = time.monotonic()
                if unit == 'producer':
                    options = producer_options()
                    kernel = module._fused.warmup(*arguments, DOT=tl.bfloat16, QK_DOT=tl.float8e4nv,
                                                 grid=(32, 2, 11), **options)
                else:
                    options = merge_options()
                    kernel = module._merge.warmup(t['qkv'], t['vs'], t['po'], t['pm'], t['pl'],
                        t['out'], 1e-5, 2304, grid=(32, 16), **options)
                kernel._init_handles()
                row = dict(arm=arm, unit=unit, options=options,
                    compile_and_load_seconds=time.monotonic()-before,
                    registers=getattr(kernel, 'n_regs', None),
                    driver_spills_raw=getattr(kernel, 'n_spills', None),
                    shared_memory_bytes=getattr(kernel.metadata, 'shared', None),
                    metadata=metadata_value(kernel.metadata._asdict() if hasattr(kernel.metadata, '_asdict') else str(kernel.metadata)),
                    driver_module_loaded=True, kernel_launched=False, artifacts={})
                report['kernels'].append(row)
                for kind in FORMATS[:-1]:
                    value = kernel.asm.get(kind)
                    if value is None:
                        raise RuntimeError('Required compiler artifact missing: '+kind)
                    raw = value.encode() if isinstance(value, str) else bytes(value)
                    if len(raw) > MAX_FILE:
                        raise ValueError('Compiler artifact exceeds per-file bound')
                    name = f'{arm}.{unit}.{kind}'
                    (folder/name).write_bytes(raw)
                    row['artifacts'][kind] = dict(file=name, size=len(raw), sha256=digest(raw))
                    if kind == 'ptx':
                        row['ptx_instruction_counts'] = ptx_instruction_counts(raw.decode())
                gate()
                disassembly = subprocess.run([tool, '--dump-sass', str(folder/f'{arm}.{unit}.cubin')],
                                             capture_output=True, timeout=10)
                row['cuobjdump_exit_code'] = disassembly.returncode
                if disassembly.returncode:
                    row['cuobjdump_stderr_tail'] = disassembly.stderr.decode(errors='replace')[-2048:]
                    raise RuntimeError('Already-installed cuobjdump failed')
                if len(disassembly.stdout) > MAX_FILE:
                    raise ValueError('SASS artifact exceeds per-file bound')
                name = f'{arm}.{unit}.sass'
                (folder/name).write_bytes(disassembly.stdout)
                row['artifacts']['sass'] = dict(file=name, size=len(disassembly.stdout), sha256=digest(disassembly.stdout))
                gate()
        if {path.name for path in folder.iterdir()} != ALLOWED:
            raise AssertionError('Complete four-kernel/six-format artifact set required')
        report['artifact_bundle'] = bundle_artifacts(folder)
        report.update(status='compiled', passed=True, compile_complete=True,
            semantic_admission=False,
            qualification='Compilation/artifacts only; independent generated-dataflow and numerical review required before any execution or benefit claim')
        save()
        return 0
    except Exception as exc:
        report.update(status='incomplete_or_failed', passed=False, compile_complete=False,
            error_type=type(exc).__name__, error=str(exc)[:2000])
        try:
            if folder.is_dir():
                report['artifact_bundle'] = bundle_artifacts(folder)
        except Exception as packaging:
            report['artifact_bundle'] = None
            report['artifact_packaging_error'] = type(packaging).__name__+': '+str(packaging)[:1000]
        save()
        return 1
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deadline', type=float, default=90.)
    parser.add_argument('--artifact-dir', type=Path)
    raise SystemExit(main(parser.parse_args()))
