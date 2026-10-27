"""CPU-only exact-source Triton SM90 lowering; never discovers or loads a GPU."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import time


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    root = Path(__file__).resolve().parent
    out = root / 'artifacts'
    out.mkdir(exist_ok=True)
    expected = json.loads((root / 'source.json').read_text())
    raw = (root / 'kernel_source.py').read_bytes()
    if sha(raw) != expected['kernel_sha256']:
        raise RuntimeError('Frozen kernel changed')
    if platform.system() != 'Linux' or list(Path('/dev').glob('nvidia*')):
        raise RuntimeError('CPU-only Linux container required')
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    if triton.__version__ != '3.8.0':
        raise RuntimeError('Pinned Triton required')
    spec = importlib.util.spec_from_file_location('owned_dualhead', root/'kernel_source.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    tools = {key: Path(getattr(triton.knobs.nvidia, key).path)
             for key in ('ptxas', 'cuobjdump')}
    result = dict(source=expected, target=['cuda', 90, 32],
                  triton=triton.__version__, gpu_execution=False,
                  versions={k: subprocess.check_output([str(p), '--version'], text=True)
                            for k,p in tools.items()}, variants=[])
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    triton.knobs.nvidia.dump_ptxas_log = True
    variants = [('full_l1_s5', 1, 5, 4), ('full_l8_s5', 8, 5, 8), ('merge_l1_s5', 1, 5, 4), ('merge_l8_s132', 8, 132, 4)]
    for name, length, splits, warps in variants:
        start = time.monotonic()
        merge = name.startswith('merge')
        fn = module._dual_merge if merge else module._dual_full
        hkv = 16 if '_dp_' in name else 2
        constants = dict(L=length, G=8, HQ=hkv*8, HKV=hkv, D=64,
                         BLOCK_M=16 if length == 1 else 64, SPLITS=splits)
        if merge:
            constants['BS'] = triton.next_power_of_2(splits)
            pointers = dict(qkv_ptr='*bf16', vs_ptr='*fp32', o_ptr='*fp32',
                            m_ptr='*fp32', l_ptr='*fp32', out_ptr='*bf16')
            scalars = dict(eps='fp32', stride_tok='i32')
        else:
            constants.update(BLOCK_N=64, BLOCK_L=16,
                             QK_DOT=triton.language.float8e4nv,
                             DOT=triton.language.bfloat16)
            pointers = dict(qkv_ptr='*bf16', seq_ptr='*i32', pt_ptr='*i32',
                            loc_ptr='*i64', kc_ptr='*fp8e4nv', vc_ptr='*fp8e4nv',
                            ks_ptr='*fp32', vs_ptr='*fp32', o_ptr='*fp32',
                            m_ptr='*fp32', l_ptr='*fp32')
            scalars = dict(eps='fp32', qk_factor='fp32', sm_scale='fp32',
                           page='i32', stride_tok='i32', stride_pt='i32')
        signature = dict(pointers, **scalars, **{k: 'constexpr' for k in constants})
        attrs = {(fn.arg_names.index(k),): [['tt.divisibility', 16]] for k in pointers}
        src = ASTSource(fn=fn, signature=signature, constexprs=constants, attrs=attrs)
        capture = io.StringIO()
        item = dict(name=name, options=dict(num_warps=warps, num_stages=1),
                    constants={k: str(v) for k,v in constants.items()},
                    signature=signature, status='starting')
        result['variants'].append(item)
        try:
            with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
                compiled = triton.compile(src, target=GPUTarget('cuda',90,32),
                                          options=item['options'])
            hashes = {}
            for ext in ('ttir','ttgir','llir','ptx','cubin'):
                value = compiled.asm[ext]
                data = value if isinstance(value, bytes) else value.encode()
                (out/f'{name}.{ext}').write_bytes(data)
                hashes[ext] = sha(data)
            for flag,ext in (('--dump-resource-usage','resources.txt'),('--dump-sass','sass')):
                data = subprocess.check_output([str(tools['cuobjdump']), flag,
                                                str(out/f'{name}.cubin')], text=True, timeout=30)
                (out/f'{name}.{ext}').write_text(data)
                hashes[ext] = sha(data.encode())
            item.update(status='compiled', metadata=compiled.metadata._asdict(), hashes=hashes)
        except Exception as exc:
            item.update(status='failed', error_type=type(exc).__name__, error=str(exc)[:8000])
        log = capture.getvalue()
        (out/f'{name}.ptxas.txt').write_text(log)
        item['wall_seconds'] = time.monotonic()-start
        (out/'result.json').write_text(json.dumps(result, indent=2, default=str)+'\n')
        print(json.dumps(dict(name=name, status=item['status'],
                              metadata=item.get('metadata'), error=item.get('error')), default=str), flush=True)
        if item['status'] != 'compiled':
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
