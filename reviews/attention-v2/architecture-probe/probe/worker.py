"""Compile exact A2-13 L8 specializations; never launch their invalid buffers.

Only the QK_DOT compile-time operand selector changes between fp8 and bf16.
The candidate module's bytes and runtime dispatch remain exactly A2-13.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback

SOURCE_SHA256 = 'e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e'
DEFAULT_DEST = '/tmp/architecture-probe-native'


def now():
    return datetime.now(timezone.utc).isoformat()


def specialization():
    # Matches _e2_kernel_fn for full verify TP8 B16/B32, page-table capacity
    # 86008: ceil((86000 + L8) / CHUNK8192) = 11. L8 uses integrated merge.
    return dict(L=8, G=8, HQ=16, HKV=2, D=64, BLOCK_M=64,
                BLOCK_N=64, BLOCK_L=16, SPLITS=11, ROPE=False,
                WINDOWED=False, CHUNK_SIZE=8192, SEPARATE_MERGE=False,
                num_warps=4, num_stages=3)


def metadata_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): metadata_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [metadata_value(v) for v in value]
    return repr(value)


def installed_fa3_evidence():
    """Read wrapper source only; this does not prove a dispatched CUDA scheduler."""
    result = dict(actual_dispatch_verified=False,
        limitation='No FA3 launch. Wrapper source and available files identify interfaces, not executed specialization.')
    try:
        import sgl_kernel.flash_attn as fa
    except Exception as exc:
        result['wrapper_unavailable'] = type(exc).__name__
        return result
    for name in ['flash_attn_with_kvcache', 'get_scheduler_metadata']:
        fn = getattr(fa, name, None)
        if fn is None:
            result[name] = None
            continue
        try:
            text = inspect.getsource(fn)
            lines = text.splitlines()
            relevant = set()
            for i, line in enumerate(lines):
                if re.search(r'num_splits|scheduler|varlen|dynamic|persistent|pack_gqa|page_size|headdim', line, re.I):
                    relevant.update(range(max(0, i-1), min(len(lines), i+2)))
            snippet = '\n'.join(f'{i+1}: {lines[i]}' for i in sorted(relevant))[:12000]
            result[name] = dict(module=fn.__module__, source_file=inspect.getsourcefile(fn),
                source_sha256=hashlib.sha256(text.encode()).hexdigest(), snippet=snippet)
        except (OSError, TypeError) as exc:
            result[name] = dict(unavailable=type(exc).__name__)
    try:
        wrapper = Path(fa.__file__).resolve()
        result['package_file'] = str(wrapper)
        result['package_file_sha256'] = hashlib.sha256(wrapper.read_bytes()).hexdigest()
        result['shared_objects_inspected'] = False
    except Exception as exc:
        result['package_inspection_error'] = type(exc).__name__
    return result


def worker(dest, deadline):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parent
    report = dict(kind='M64 BF16 QK compile/disassembly only', status='starting',
        candidate_kernel_launches=0, correctness_tested=False, performance_tested=False,
        source_sha256=SOURCE_SHA256, worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        records=[], started_utc=now(), deadline_seconds=deadline,
        specialization_difference={'QK_DOT': ['float8e4nv', 'bfloat16']},
        merge_semantics='L8 integrated final-CTA ticket merge; no separate merge specialization',
        logical_fixture=dict(batches=[16, 32], q_len=8, heads='tp8', layer='full',
            spread='agentic', ctx=80000, page=1, page_table_columns=86008,
            qkv_width=2304, splits=11, chunk=8192,
            qk_factor=5.4730064863875, sm_scale=.125))

    def save():
        tmp = dest/'summary.tmp'
        tmp.write_text(json.dumps(report, indent=2)+'\n')
        tmp.replace(dest/'summary.json')

    def alarm(*_):
        raise TimeoutError('Compile-only worker deadline')

    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(deadline)
    started = time.monotonic()
    save()
    try:
        import torch
        import triton
        import triton.language as tl
        assert not os.environ.get('TRITON_INTERPRET'), 'Native compiler required'
        assert torch.cuda.is_available() and 'H200' in torch.cuda.get_device_name()
        raw = (root/'candidate.py').read_bytes()
        assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA256
        spec = importlib.util.spec_from_file_location('architecture_candidate', root/'candidate.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        driver = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                capture_output=True, text=True, timeout=5)
        report['environment'] = dict(python=platform.python_version(), torch=torch.__version__,
            triton=triton.__version__, triton_package_file=str(Path(triton.__file__).resolve()),
            cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
            capability=torch.cuda.get_device_capability(),
            driver=driver.stdout.strip() if driver.returncode == 0 else None,
            triton_git_revision=None)
        compiler_source = Path(triton.__file__).parent/'compiler/compiler.py'
        if compiler_source.is_file():
            report['environment']['triton_compiler_source_sha256'] = hashlib.sha256(compiler_source.read_bytes()).hexdigest()
        for pkg in ['triton', 'torch', 'sglang', 'sgl-kernel']:
            try:
                dist = importlib.metadata.distribution(pkg)
                direct = json.loads(dist.read_text('direct_url.json') or '{}')
                report['environment'][pkg+'_distribution'] = dict(version=dist.version,
                    vcs_commit_id=direct.get('vcs_info', {}).get('commit_id'),
                    metadata_sha256=hashlib.sha256((dist.read_text('METADATA') or '').encode()).hexdigest(),
                    record_sha256=hashlib.sha256((dist.read_text('RECORD') or '').encode()).hexdigest())
                if pkg == 'triton':
                    report['environment']['triton_git_revision'] = direct.get('vcs_info', {}).get('commit_id')
            except importlib.metadata.PackageNotFoundError:
                report['environment'][pkg+'_distribution'] = None
        # These deliberately undersized, allocator-aligned buffers reproduce JIT
        # pointer types/alignment. They are NEVER valid data buffers for a launch.
        dtypes = dict(qkv=torch.bfloat16, seq=torch.int32, pt=torch.int32,
            loc=torch.int64, kc=torch.float8_e4m3fn, vc=torch.float8_e4m3fn,
            ks=torch.float32, vs=torch.float32, po=torch.float32,
            pm=torch.float32, pl=torch.float32, ticket=torch.int32, out=torch.bfloat16)
        tensors = {k: torch.empty(1, dtype=v, device='cuda') for k, v in dtypes.items()}
        report['pointer_metadata'] = {k: dict(dtype=str(v.dtype), elements=v.numel(),
            alignment_mod16=v.data_ptr() % 16) for k, v in tensors.items()}
        assert all(v.data_ptr() % 16 == 0 for v in tensors.values())
        args = [tensors[k] for k in ['qkv', 'seq', 'pt', 'loc', 'kc', 'vc', 'ks', 'vs', 'po', 'pm', 'pl', 'ticket', 'out']]
        args += [1.0, 1e-5, 5.4730064863875, .125, 0, 1, 2304, 86008]
        report['runtime_scalars'] = args[-8:]
        save()
        tool = shutil.which('cuobjdump')
        if tool is None:
            for p in [Path('/usr/local/cuda/bin/cuobjdump'), Path(triton.__file__).parent/'backends/nvidia/bin/cuobjdump']:
                if p.is_file():
                    tool = str(p)
                    break
        for operand, qk_dot in [('fp8', tl.float8e4nv), ('bf16', tl.bfloat16)]:
            for batch in [16, 32]:
                before = time.monotonic()
                options = specialization()
                kernel = module._fused.warmup(*args, DOT=tl.bfloat16, QK_DOT=qk_dot,
                                              grid=(batch, 2, 11), **options)
                kernel._init_handles()
                label = f'{operand}.b{batch}.fused'
                row = dict(label=label, batch=batch, kernel='fused producer/integrated merge',
                    QK_DOT=operand, DOT='bfloat16', options=options, grid=[batch, 2, 11],
                    compile_and_load_seconds=time.monotonic()-before,
                    kernel_hash=getattr(kernel, 'hash', None),
                    registers=getattr(kernel, 'n_regs', None),
                    driver_spills_raw=getattr(kernel, 'n_spills', None),
                    shared_memory_bytes=getattr(kernel.metadata, 'shared', None),
                    metadata=metadata_value(kernel.metadata._asdict() if hasattr(kernel.metadata, '_asdict') else str(kernel.metadata)),
                    artifacts={})
                report['records'].append(row)
                for kind in ['ttir', 'ttgir', 'llir', 'ptx', 'cubin']:
                    value = kernel.asm.get(kind)
                    if value is None:
                        row['artifacts'][kind] = None
                        continue
                    value = value.encode() if isinstance(value, str) else bytes(value)
                    name = f'{label}.{kind}'
                    (dest/name).write_bytes(value)
                    row['artifacts'][kind] = dict(file=name, bytes=len(value), sha256=hashlib.sha256(value).hexdigest())
                save()
                if tool and (dest/f'{label}.cubin').exists():
                    sass = subprocess.run([tool, '--dump-sass', str(dest/f'{label}.cubin')], capture_output=True, timeout=5)
                    row['sass_tool_exit_code'] = sass.returncode
                    if sass.returncode == 0:
                        name = f'{label}.sass'
                        (dest/name).write_bytes(sass.stdout)
                        row['artifacts']['sass'] = dict(file=name, bytes=len(sass.stdout), sha256=hashlib.sha256(sass.stdout).hexdigest())
                else:
                    row['sass_unavailable_reason'] = 'cuobjdump absent; PTX is not final machine scheduling'
                save()
        report['installed_fa3'] = installed_fa3_evidence()
        report['status'] = 'compiled'
    except Exception as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc)[:2000],
            traceback=traceback.format_exc()[-6000:])
    finally:
        signal.alarm(0)
        report['worker_seconds'] = time.monotonic()-started
        save()
    return 0 if report['status'] == 'compiled' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default=DEFAULT_DEST)
    parser.add_argument('--deadline', type=int, default=55)
    args = parser.parse_args()
    assert 1 <= args.deadline <= 55
    raise SystemExit(worker(args.output_dir, args.deadline))
