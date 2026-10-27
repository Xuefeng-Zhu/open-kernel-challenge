"""Assemble the isolated full-attention experiment without editing its champion."""
import ast
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'fusion-lookahead-probe/candidate.py'
BASE_SHA = '3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8'

ADAPTER = '''
def _dual_full_call(data):
    cfg = data['config']
    b, length = cfg['batch'], cfg['q_len']
    hq, hkv, d = cfg['num_q_heads'], cfg['num_kv_heads'], cfg['head_dim']
    g, page = hq // hkv, cfg['page_size']
    qkv = data['qkv'].view(b * length, -1)
    bm, bn = max(16, triton.next_power_of_2(length * g)), 64
    # One initial wave of paired-head CTAs. This is a hypothesis, not occupancy.
    pairs = b * (hkv // 2)
    maximum_span = data['page_table'].shape[1] * page
    splits = min(max(1, triton.cdiv(NUM_SMS, pairs)),
                 max(1, maximum_span // (2 * bn)))
    n = b * hkv * splits * bm
    partial_o = torch.empty((n, d), device=qkv.device, dtype=torch.float32)
    partial_m = torch.empty((n,), device=qkv.device, dtype=torch.float32)
    partial_l = torch.empty((n,), device=qkv.device, dtype=torch.float32)
    out = torch.empty((b * length, hq * d), device=qkv.device, dtype=torch.bfloat16)
    _dual_full[(b, hkv // 2, splits)](
        qkv, data['seq_lens'], data['page_table'], data['cache_loc'],
        data['k_cache'], data['v_cache'], data['k_scale'], data['v_scale'],
        partial_o, partial_m, partial_l, RMS_EPS,
        cfg['qk_scale_factor'], cfg['sm_scale'], page,
        qkv.stride(0), data['page_table'].stride(0),
        L=length, G=g, HQ=hq, HKV=hkv, D=d, BLOCK_M=bm, BLOCK_N=bn,
        BLOCK_L=16, SPLITS=splits, QK_DOT=_QK_DOT, DOT=_DOT,
        num_warps=8 if length == 8 else 4, num_stages=1)
    _dual_merge[(b * length, hq)](
        qkv, data['v_scale'], partial_o, partial_m, partial_l, out,
        RMS_EPS, qkv.stride(0), L=length, G=g, HQ=hq, HKV=hkv, D=d,
        BLOCK_M=bm, SPLITS=splits, BS=triton.next_power_of_2(splits),
        num_warps=4, num_stages=1)
    return out.view(b, length, hq * d)


def kernel_fn(data: Inputs) -> torch.Tensor:
    cfg = data['config']
    if (cfg['window'] == 0 and cfg['rope_theta'] == 0
            and cfg['q_len'] in (1, 8) and cfg['head_dim'] == 64
            and cfg['num_kv_heads'] % 2 == 0
            and cfg['num_q_heads'] == 8 * cfg['num_kv_heads']):
        return _dual_full_call(data)
    return _champion_kernel_fn(data)
'''


def assemble():
    raw = BASE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASE_SHA:
        raise ValueError('Champion source changed')
    base = raw.decode()
    if base.count('def kernel_fn(data: Inputs)') != 1:
        raise ValueError('Expected one champion entrypoint')
    base = base.replace('def kernel_fn(data: Inputs)', 'def _champion_kernel_fn(data: Inputs)', 1)
    kernel = (HERE / 'kernel.py').read_text()
    tree = ast.parse(kernel)
    bodies = [ast.get_source_segment(kernel, node) for node in tree.body
              if isinstance(node, ast.FunctionDef)]
    # AST source segments exclude decorators, so restore the two known JITs.
    if [node.name for node in tree.body if isinstance(node, ast.FunctionDef)] != ['_dual_full', '_dual_merge']:
        raise ValueError('Unexpected kernel exports')
    combined = base + '\n\n' + '\n\n'.join('@triton.jit\n' + body for body in bodies) + '\n\n' + ADAPTER
    ast.parse(combined)
    (HERE / 'candidate.py').write_text(combined)
    return hashlib.sha256(combined.encode()).hexdigest()


if __name__ == '__main__':
    print(assemble())
