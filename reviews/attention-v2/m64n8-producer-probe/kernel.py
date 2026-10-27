"""Unexecuted L1/G8/D64 page-1 producer integration, compile admission only.

History uses reversed BF16 WGMMA M64/N8; the new-token block remains N16
and uses BF16 Ampere MMA. Original split-major M16 partial strides and the
selected merge remain unchanged. Unsupported contracts must use the selected
producer. Native buffering, resources and numerical correctness are gates,
not established by this source or its CPU contract tests.
"""
from triton.experimental import gluon as _m64n8_gluon
from triton.experimental.gluon import language as _m64n8_gl
from triton.experimental.gluon.language.nvidia.ampere import (
    async_copy as _m64n8_async_copy, mma_v2 as _m64n8_mma_v2,
)
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared as _m64n8_fence_async_shared,
    warpgroup_mma as _m64n8_warpgroup_mma,
    warpgroup_mma_wait as _m64n8_warpgroup_mma_wait,
)


@_m64n8_gluon.jit
def _m64n8_norm_rope(x_ptr, offs_d, pos, theta, eps,
                      ROPE: _m64n8_gl.constexpr, D: _m64n8_gl.constexpr, mask):
    # Match selected FP32 RMS/RoPE; rounding belongs to each caller below.
    x = _m64n8_gl.load(x_ptr + offs_d[None, :], mask=mask, other=0.0).to(_m64n8_gl.float32)
    inv = _m64n8_gl.rsqrt(_m64n8_gl.sum(x * x, 1) / D + eps)[:, None]
    x = x * inv
    if ROPE:
        partner = _m64n8_gl.load(x_ptr + (offs_d ^ 1)[None, :], mask=mask, other=0.0).to(_m64n8_gl.float32) * inv
        i = (offs_d // 2).to(_m64n8_gl.float32)
        inv_freq = _m64n8_gl.exp2(-(2.0 * i / D) * _m64n8_gl.log2(theta))
        ang = pos.to(_m64n8_gl.float32) * inv_freq[None, :]
        sign = _m64n8_gl.where((offs_d % 2) == 0, -1.0, 1.0)[None, :]
        x = x * _m64n8_gl.cos(ang) + sign * partner * _m64n8_gl.sin(ang)
    return x


@_m64n8_gluon.jit
def _m64n8_page_ids(pt_ptr, b, stride_pt, start, offs_n, s_hi):
    # Page size is exactly one. Preserve live, masked PT reads and clamp.
    t = start + offs_n
    pg = _m64n8_gl.load(pt_ptr + b * stride_pt + t, mask=t < s_hi, other=0)
    return _m64n8_gl.maximum(pg, 0)


@_m64n8_gluon.jit
def _m64n8_issue_tile(raw_k, raw_v, kc_ptr, vc_ptr, pg, start, offs_n,
                       offs_d, s_hi, kvh, HKV: _m64n8_gl.constexpr,
                       D: _m64n8_gl.constexpr):
    # One committed group contains BOTH raw FP8 payloads for one real tile.
    # No masked-only future groups are committed.
    t_ok = start + offs_n < s_hi
    off = pg.to(_m64n8_gl.int64)[:, None] * (HKV * D) + kvh * D + offs_d[None, :]
    _m64n8_async_copy.async_load(raw_k, kc_ptr + off, mask=t_ok[:, None])
    _m64n8_async_copy.async_load(raw_v, vc_ptr + off, mask=t_ok[:, None])
    _m64n8_async_copy.commit_group()


@_m64n8_gluon.jit
def _m64n8_fused(qkv_ptr, seq_ptr, pt_ptr, loc_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr,
                  o_ptr, m_ptr, l_ptr, ticket_ptr, out_ptr,
                  theta, eps, qk_factor, sm_scale, window, page, stride_tok, stride_pt,
                  L: _m64n8_gl.constexpr, G: _m64n8_gl.constexpr,
                  HQ: _m64n8_gl.constexpr, HKV: _m64n8_gl.constexpr,
                  D: _m64n8_gl.constexpr, BLOCK_M: _m64n8_gl.constexpr,
                  BLOCK_N: _m64n8_gl.constexpr, BLOCK_L: _m64n8_gl.constexpr,
                  SPLITS: _m64n8_gl.constexpr, ROPE: _m64n8_gl.constexpr,
                  WINDOWED: _m64n8_gl.constexpr, DOT: _m64n8_gl.constexpr,
                  QK_DOT: _m64n8_gl.constexpr, CHUNK_SIZE: _m64n8_gl.constexpr,
                  SEPARATE_MERGE: _m64n8_gl.constexpr,
                  PAGE_LOOKAHEAD: _m64n8_gl.constexpr):
    _m64n8_gl.static_assert(L == 1 and G == 8 and HQ == 16 and HKV == 2 and D == 64)
    _m64n8_gl.static_assert(BLOCK_M == 16 and BLOCK_N == 64 and BLOCK_L == 16)
    _m64n8_gl.static_assert(SPLITS > 1 and SEPARATE_MERGE)
    _m64n8_gl.static_assert(page == 1)
    _m64n8_gl.static_assert(kc_ptr.dtype.element_ty == _m64n8_gl.float8e4nv)
    _m64n8_gl.static_assert(vc_ptr.dtype.element_ty == _m64n8_gl.float8e4nv)

    mma: _m64n8_gl.constexpr = _m64n8_gl.NVMMADistributedLayout(
        version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 8, 16])
    load: _m64n8_gl.constexpr = _m64n8_gl.BlockedLayout(
        size_per_thread=[1, 4], threads_per_warp=[4, 8],
        warps_per_cta=[4, 1], order=[1, 0])
    raw_shared: _m64n8_gl.constexpr = _m64n8_gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=8, order=[1, 0])
    k_shared: _m64n8_gl.constexpr = _m64n8_gl.NVMMASharedLayout.get_default_for(
        [64, 64], _m64n8_gl.bfloat16)
    b_shared: _m64n8_gl.constexpr = _m64n8_gl.NVMMASharedLayout.get_default_for(
        [64, 8], _m64n8_gl.bfloat16, transposed=True)
    vt_shared: _m64n8_gl.constexpr = _m64n8_gl.NVMMASharedLayout.get_default_for(
        [64, 64], _m64n8_gl.bfloat16, transposed=True)
    tail_mma: _m64n8_gl.constexpr = _m64n8_gl.NVMMADistributedLayout(
        version=[2, 0], warps_per_cta=[4, 1], instr_shape=[16, 8])
    tail_a: _m64n8_gl.constexpr = _m64n8_gl.DotOperandLayout(
        operand_index=0, parent=tail_mma, k_width=2)
    tail_b: _m64n8_gl.constexpr = _m64n8_gl.DotOperandLayout(
        operand_index=1, parent=tail_mma, k_width=2)

    b = _m64n8_gl.program_id(0)
    kvh = _m64n8_gl.program_id(1)
    split = _m64n8_gl.program_id(2)
    seq = _m64n8_gl.load(seq_ptr + b).to(_m64n8_gl.int64)
    n = _m64n8_gl.arange(0, 64, layout=_m64n8_gl.SliceLayout(1, load))
    d = _m64n8_gl.arange(0, 64, layout=_m64n8_gl.SliceLayout(0, load))
    g = _m64n8_gl.arange(0, 8, layout=_m64n8_gl.SliceLayout(1, load))
    q_rows = qkv_ptr + b.to(_m64n8_gl.int64) * stride_tok + ((kvh * G + g) * D)[:, None]
    q = _m64n8_norm_rope(q_rows, d, _m64n8_gl.full((8, 1), 0, _m64n8_gl.int64, load) + seq,
                           theta, eps, ROPE, D, _m64n8_gl.full((8, 1), True, _m64n8_gl.int1, load))
    q = q.to(_m64n8_gl.bfloat16).to(_m64n8_gl.float8e4nv).to(_m64n8_gl.bfloat16)
    ks = _m64n8_gl.load(ks_ptr).to(_m64n8_gl.float32)
    vs = _m64n8_gl.load(vs_ptr).to(_m64n8_gl.float32)
    qk_scale = (sm_scale * qk_factor) * ks * 1.4426950408889634

    if WINDOWED:
        lo = _m64n8_gl.maximum(seq - window + 1, 0)
    else:
        lo = seq * 0
    lo = _m64n8_gl.minimum(lo, seq)
    span = seq - lo
    if CHUNK_SIZE:
        s_lo = split * CHUNK_SIZE
        s_hi = _m64n8_gl.minimum(s_lo + CHUNK_SIZE, seq)
    else:
        per = ((span + SPLITS - 1) // SPLITS + BLOCK_N - 1) // BLOCK_N * BLOCK_N
        s_lo = lo + split * per
        s_hi = _m64n8_gl.minimum(s_lo + per, seq)

    raw_k = _m64n8_gl.allocate_shared_memory(_m64n8_gl.float8e4nv, [2, 64, 64], raw_shared)
    raw_v = _m64n8_gl.allocate_shared_memory(_m64n8_gl.float8e4nv, [2, 64, 64], raw_shared)
    k_buf = _m64n8_gl.allocate_shared_memory(_m64n8_gl.bfloat16, [64, 64], k_shared)
    qt_buf = _m64n8_gl.allocate_shared_memory(_m64n8_gl.bfloat16, [64, 8], b_shared)
    vt_buf = _m64n8_gl.allocate_shared_memory(_m64n8_gl.bfloat16, [64, 64], vt_shared)
    p_buf = _m64n8_gl.allocate_shared_memory(_m64n8_gl.bfloat16, [64, 8], b_shared)
    qt_buf.store(_m64n8_gl.permute(q, (1, 0)))
    _m64n8_gl.barrier()

    if s_lo < s_hi:
        pg0 = _m64n8_page_ids(pt_ptr, b, stride_pt, s_lo, n, s_hi)
        _m64n8_issue_tile(raw_k.index(0), raw_v.index(0), kc_ptr, vc_ptr,
                            pg0, s_lo, n, d, s_hi, kvh, HKV, D)
    if s_lo + BLOCK_N < s_hi:
        pg1 = _m64n8_page_ids(pt_ptr, b, stride_pt, s_lo + BLOCK_N, n, s_hi)
        _m64n8_issue_tile(raw_k.index(1), raw_v.index(1), kc_ptr, vc_ptr,
                            pg1, s_lo + BLOCK_N, n, d, s_hi, kvh, HKV, D)
    if PAGE_LOOKAHEAD:
        # Address-only lookahead stays live separately from payload buffering.
        next_pg = _m64n8_page_ids(pt_ptr, b, stride_pt, s_lo + 2 * BLOCK_N, n, s_hi)

    m_i = _m64n8_gl.full((8,), float("-inf"), _m64n8_gl.float32,
                            _m64n8_gl.SliceLayout(0, mma))
    l_i = _m64n8_gl.zeros((8,), _m64n8_gl.float32, _m64n8_gl.SliceLayout(0, mma))
    acc = _m64n8_gl.zeros((64, 8), _m64n8_gl.float32, mma)
    score_n = _m64n8_gl.arange(0, 64, layout=_m64n8_gl.SliceLayout(1, mma))
    slot = 0
    for start in range(s_lo, s_hi, BLOCK_N):
        if start + BLOCK_N < s_hi:
            _m64n8_async_copy.wait_group(1)
        else:
            _m64n8_async_copy.wait_group(0)
        _m64n8_gl.barrier()
        # Explicit mask: async_load's Python API does not promise other=0.
        t_ok = start + n < s_hi
        k = _m64n8_gl.where(t_ok[:, None], raw_k.index(slot).load(load), 0).to(_m64n8_gl.bfloat16)
        v = _m64n8_gl.where(t_ok[:, None], raw_v.index(slot).load(load), 0).to(_m64n8_gl.bfloat16)
        k_buf.store(k)
        vt_buf.store(_m64n8_gl.permute(v, (1, 0)))
        # All warps finish raw reads before this slot can be overwritten.
        _m64n8_gl.barrier()
        if start + 2 * BLOCK_N < s_hi:
            if PAGE_LOOKAHEAD:
                future_pg = next_pg
            else:
                future_pg = _m64n8_page_ids(pt_ptr, b, stride_pt, start + 2 * BLOCK_N, n, s_hi)
            _m64n8_issue_tile(raw_k.index(slot), raw_v.index(slot), kc_ptr, vc_ptr,
                                future_pg, start + 2 * BLOCK_N, n, d, s_hi, kvh, HKV, D)
        if PAGE_LOOKAHEAD:
            next_pg = _m64n8_page_ids(pt_ptr, b, stride_pt, start + 3 * BLOCK_N, n, s_hi)
        _m64n8_fence_async_shared()
        zero = _m64n8_gl.zeros((64, 8), _m64n8_gl.float32, mma)
        pending = _m64n8_warpgroup_mma(k_buf, qt_buf, zero, use_acc=False,
                                          max_num_imprecise_acc=0, is_async=True)
        score = _m64n8_warpgroup_mma_wait(num_outstanding=0, deps=(pending,)) * qk_scale
        score_t = start + score_n
        vis = score_t[:, None] < s_hi
        if WINDOWED:
            vis = vis & (score_t[:, None] > seq - window)
        score = _m64n8_gl.where(vis, score, float("-inf"))
        m_new = _m64n8_gl.maximum(m_i, _m64n8_gl.max(score, 0))
        m_safe = _m64n8_gl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = _m64n8_gl.exp2(m_i - m_safe)
        p = _m64n8_gl.exp2(score - m_safe[None, :])
        l_i = l_i * alpha + _m64n8_gl.sum(p, 0)
        p_buf.store(p.to(_m64n8_gl.bfloat16))
        _m64n8_gl.barrier()
        _m64n8_fence_async_shared()
        pv = _m64n8_warpgroup_mma(vt_buf, p_buf, zero, use_acc=False,
                                    max_num_imprecise_acc=0, is_async=False)
        acc = acc * alpha[None, :] + pv
        m_i = m_new
        slot = slot ^ 1

    # Includes zero-history and final partial tiles. Nothing async may outlive
    # raw-buffer storage or be recycled into the separate N16 tail's scratch.
    _m64n8_async_copy.wait_group(0)
    _m64n8_gl.barrier()

    if split == SPLITS - 1:
        nl = _m64n8_gl.arange(0, 16, layout=_m64n8_gl.SliceLayout(1, load))
        l_ok = nl < L
        ntok = (b * L + _m64n8_gl.where(l_ok, nl, 0)).to(_m64n8_gl.int64)
        npos = seq + nl
        k_rows = qkv_ptr + ntok[:, None] * stride_tok + (HQ * D + kvh * D)
        kn = _m64n8_norm_rope(k_rows, d, npos[:, None], theta, eps, ROPE, D, l_ok[:, None])
        vn = _m64n8_gl.load(qkv_ptr + ntok[:, None] * stride_tok
                              + ((HQ + HKV) * D + kvh * D) + d[None, :],
                              mask=l_ok[:, None], other=0.0).to(_m64n8_gl.float32)
        # New K has NO extra BF16 rounding before E4M3 cache quantization.
        kq = (kn / ks).to(kc_ptr.dtype.element_ty)
        vq = (vn / vs).to(vc_ptr.dtype.element_ty)
        loc = _m64n8_gl.load(loc_ptr + ntok, mask=l_ok, other=0)
        dst = loc[:, None] * (HKV * D) + kvh * D + d[None, :]
        _m64n8_gl.store(kc_ptr + dst, kq, mask=l_ok[:, None])
        _m64n8_gl.store(vc_ptr + dst, vq, mask=l_ok[:, None])

        # Reload the eight encoded Q rows from existing shared storage. Repeat
        # in-bounds indices, then zero ignored query rows, retaining M16/N16.
        q8 = qt_buf.permute((1, 0)).load(load)
        q_index = (nl % G)[:, None] + _m64n8_gl.zeros((16, 64), _m64n8_gl.int32, load)
        q16 = _m64n8_gl.gather(q8, q_index, axis=0)
        q16 = _m64n8_gl.where((nl < G)[:, None], q16, 0)
        tail_q = _m64n8_gl.convert_layout(q16, tail_a)
        tail_k = _m64n8_gl.convert_layout(_m64n8_gl.permute(kq.to(_m64n8_gl.bfloat16), (1, 0)), tail_b)
        tail_zero = _m64n8_gl.zeros((16, 16), _m64n8_gl.float32, tail_mma)
        # mma_v2 BF16 operands select strict max_num_imprecise_acc=0 in the
        # pinned semantic implementation; its API has no explicit max arg.
        s16 = _m64n8_mma_v2(tail_q, tail_k, tail_zero) * qk_scale
        s16 = _m64n8_gl.convert_layout(s16, load)
        first8 = g[:, None] + _m64n8_gl.zeros((8, 16), _m64n8_gl.int32, load)
        s = _m64n8_gl.gather(s16, first8, axis=0)
        tail_n = _m64n8_gl.arange(0, 16, layout=_m64n8_gl.SliceLayout(0, load))
        vis_tail = (tail_n < L)[None, :] & ((seq + tail_n)[None, :] <= seq)
        s = _m64n8_gl.where(vis_tail, s, float("-inf"))
        m_old = _m64n8_gl.convert_layout(m_i, _m64n8_gl.SliceLayout(1, load))
        l_old = _m64n8_gl.convert_layout(l_i, _m64n8_gl.SliceLayout(1, load))
        m_new = _m64n8_gl.maximum(m_old, _m64n8_gl.max(s, 1))
        m_safe = _m64n8_gl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = _m64n8_gl.exp2(m_old - m_safe)
        p8 = _m64n8_gl.exp2(s - m_safe[:, None])
        l_new = l_old * alpha + _m64n8_gl.sum(p8, 1)

        p_index = (nl % G)[:, None] + _m64n8_gl.zeros((16, 16), _m64n8_gl.int32, load)
        p16 = _m64n8_gl.gather(p8, p_index, axis=0)
        p16 = _m64n8_gl.where((nl < G)[:, None], p16, 0).to(_m64n8_gl.bfloat16)
        tail_p = _m64n8_gl.convert_layout(p16, tail_a)
        tail_v = _m64n8_gl.convert_layout(vq.to(_m64n8_gl.float32).to(_m64n8_gl.bfloat16), tail_b)
        pv16 = _m64n8_mma_v2(tail_p, tail_v,
                                 _m64n8_gl.zeros((16, 64), _m64n8_gl.float32, tail_mma))
        pv16 = _m64n8_gl.convert_layout(pv16, load)
        first8d = g[:, None] + _m64n8_gl.zeros((8, 64), _m64n8_gl.int32, load)
        pv8 = _m64n8_gl.gather(pv16, first8d, axis=0)
        tail_pv = _m64n8_gl.convert_layout(_m64n8_gl.permute(pv8, (1, 0)), mma)
        alpha_hist = _m64n8_gl.convert_layout(alpha, _m64n8_gl.SliceLayout(0, mma))
        acc = acc * alpha_hist[None, :] + tail_pv
        m_i = _m64n8_gl.convert_layout(m_new, _m64n8_gl.SliceLayout(0, mma))
        l_i = _m64n8_gl.convert_layout(l_new, _m64n8_gl.SliceLayout(0, mma))

    # Preserve allocated M16 ABI; the unchanged merge reads only valid G8.
    od = _m64n8_gl.arange(0, 64, layout=_m64n8_gl.SliceLayout(1, mma))
    og = _m64n8_gl.arange(0, 8, layout=_m64n8_gl.SliceLayout(0, mma))
    pbase = ((b * HKV + kvh) * SPLITS + split) * BLOCK_M
    _m64n8_gl.store(o_ptr + (pbase + og[None, :]) * D + od[:, None], acc)
    _m64n8_gl.store(m_ptr + pbase + og, m_i)
    _m64n8_gl.store(l_ptr + pbase + og, l_i)
    # These ignored padding values intentionally need not equal the selected
    # producer's ignored history rows; stride and all merge-consumed rows do.
    _m64n8_gl.store(o_ptr + (pbase + og[None, :] + G) * D + od[:, None], 0.0)
    _m64n8_gl.store(m_ptr + pbase + og + G, float("-inf"))
    _m64n8_gl.store(l_ptr + pbase + og + G, 0.0)
