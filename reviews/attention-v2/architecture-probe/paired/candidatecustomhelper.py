"""Experimental BF16 QK adapter; original candidate module stays immutable.

Derived from exact A2-13 _e2_kernel_fn. Only global qualification, function
name and QK_DOT=bfloat16 differ. No formal route or candidate global is edited.
SPDX-License-Identifier: Apache-2.0; source license retained in candidate.py.
"""
import candidate
import triton.language as tl

def bf16_custom_fn(data: candidate.Inputs) -> candidate.torch.Tensor:
    cfg = data['config']
    batch, length = (cfg['batch'], cfg['q_len'])
    hq, hkv, d, page = (cfg['num_q_heads'], cfg['num_kv_heads'], cfg['head_dim'], cfg['page_size'])
    g = hq // hkv
    tokens = batch * length
    qkv = data['qkv'].reshape(tokens, -1)
    window = cfg['window']
    rope = bool(cfg['rope_theta'])
    theta = float(cfg['rope_theta']) if rope else 1.0
    page_table = data['page_table']
    dev = qkv.device
    block_n = 64
    block_m = max(16, candidate.triton.next_power_of_2(length * g))
    span_max = window if window else page_table.shape[1] * page
    chunk_size = 8192 if window == 0 and batch in (16, 32) else 0
    splits = (span_max + chunk_size - 1) // chunk_size if chunk_size else candidate._num_splits(batch, hkv, span_max, block_n)
    out = candidate.torch.empty((tokens, hq * d), dtype=candidate.torch.bfloat16, device=dev)
    if splits > 1:
        n = batch * hkv * splits * block_m
        part_o = candidate.torch.empty((n, d), dtype=candidate.torch.float32, device=dev)
        part_m = candidate.torch.empty((n,), dtype=candidate.torch.float32, device=dev)
        part_l = candidate.torch.empty((n,), dtype=candidate.torch.float32, device=dev)
    else:
        part_o = part_m = part_l = out
    separate_merge = length == 1
    tickets = out if separate_merge else candidate._tickets(dev, batch * hkv)
    candidate._fused[batch, hkv, splits](qkv, data['seq_lens'], page_table, data['cache_loc'], data['k_cache'], data['v_cache'], data['k_scale'], data['v_scale'], part_o, part_m, part_l, tickets, out, theta, candidate.RMS_EPS, float(cfg['qk_scale_factor']), float(cfg['sm_scale']), window, page, qkv.stride(0), page_table.stride(0), L=length, G=g, HQ=hq, HKV=hkv, D=d, BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_L=16, SPLITS=splits, ROPE=rope, WINDOWED=bool(window), DOT=candidate._DOT, QK_DOT=tl.bfloat16, CHUNK_SIZE=chunk_size, SEPARATE_MERGE=separate_merge, num_warps=4, num_stages=3)
    if separate_merge and splits > 1:
        candidate._merge[batch, hq](qkv, data['v_scale'], part_o, part_m, part_l, out, candidate.RMS_EPS, qkv.stride(0), L=length, G=g, HQ=hq, HKV=hkv, D=d, BLOCK_M=block_m, SPLITS=splits, BS=candidate.triton.next_power_of_2(splits), num_warps=4, num_stages=3)
    return out.view(batch, length, hq * d)

kernel_fn = bf16_custom_fn
