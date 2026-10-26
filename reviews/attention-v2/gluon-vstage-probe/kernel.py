"""Uncompiled revised V-stage mechanism probe; no production route change.

Q is prepared BF16 [B, 8, HQ, 64] after common RMS/position processing.
E4M3 K/V caches [physical_rows, HKV, 64] already contain history and eight new
keys. PageTable is the live page1 map; HistoryLengths excludes those eight.
Out is raw attention BF16 [B, 8, HQ, 64], before output normalization/gating.
There is no cache append, preparation, host wrapper or launch in this module.

Both arms use grid (B, HKV), num_warps=4, fixed M=N=D=64 and L=G=8. Both
prefetch next raw E4M3 K before the QK wait. V_BEFORE_WAIT moves only that wait
across the identical V gather, K then V BF16 widening, full shared stores and
publication fence. No cp.async/TMA, forced dependency or extra barrier is used.
Public layout/API precedent: Triton v3.8.0 python/tutorials/gluon/05-wgmma.py.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    warpgroup_mma,
    warpgroup_mma_wait,
)


@gluon.jit
def _prefetch_k(KCache, PageTable, b, kvh, start, total,
                HKV: gl.constexpr, PT_STRIDE: gl.constexpr):
    """Return masked raw E4M3 K and the common live page1 payload mapping."""
    load_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 4], threads_per_warp=[4, 8],
        warps_per_cta=[4, 1], order=[1, 0],
    )
    n = start + gl.arange(0, 64, layout=gl.SliceLayout(1, load_layout))
    d = gl.arange(0, 64, layout=gl.SliceLayout(0, load_layout))
    valid = n < total
    physical = gl.load(PageTable + b * PT_STRIDE + n, mask=valid, other=0)
    physical = gl.maximum(physical.to(gl.int64), 0)
    offsets = physical[:, None] * (HKV * 64) + kvh * 64 + d[None, :]
    raw_k = gl.load(KCache + offsets, mask=valid[:, None], other=0.0)
    return raw_k, offsets, valid


@gluon.jit
def _publish_kv(VCache, raw_k, offsets, valid, KSlot, VSlot):
    """Gather raw V, widen K then V, and fully publish both BF16 shared tiles."""
    raw_v = gl.load(VCache + offsets, mask=valid[:, None], other=0.0)
    k = raw_k.to(gl.bfloat16)
    KSlot.store(k)
    v = raw_v.to(gl.bfloat16)
    VSlot.store(v)
    fence_async_shared()


@gluon.jit
def attention_vstage_kernel(
    Q, KCache, VCache, PageTable, HistoryLengths, KScale, VScale, Out,
    HKV: gl.constexpr, HQ: gl.constexpr, PT_STRIDE: gl.constexpr,
    SM_SCALE: gl.constexpr, QK_FACTOR: gl.constexpr, V_BEFORE_WAIT: gl.constexpr,
):
    gl.static_assert(HKV > 0)
    gl.static_assert(HQ == HKV * 8)
    gl.static_assert(PT_STRIDE >= 8)
    b = gl.program_id(0)
    kvh = gl.program_id(1)
    seq = gl.load(HistoryLengths + b).to(gl.int64)
    total = seq + 8
    ks = gl.load(KScale).to(gl.float32)
    vs = gl.load(VScale).to(gl.float32)
    qk_scale = (SM_SCALE * QK_FACTOR) * ks * 1.4426950408889634

    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 64, 16],
    )
    dot_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mma_layout, k_width=2,
    )
    shared_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [64, 64], gl.bfloat16,
    )
    load_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 4], threads_per_warp=[4, 8],
        warps_per_cta=[4, 1], order=[1, 0],
    )
    row_layout: gl.constexpr = gl.SliceLayout(1, mma_layout)
    col_layout: gl.constexpr = gl.SliceLayout(0, mma_layout)

    load_m = gl.arange(0, 64, layout=gl.SliceLayout(1, load_layout))
    load_d = gl.arange(0, 64, layout=gl.SliceLayout(0, load_layout))
    load_j = load_m // 8
    load_g = load_m % 8
    q_offsets = ((b * 8 + load_j[:, None]) * HQ + kvh * 8
                 + load_g[:, None]) * 64 + load_d[None, :]
    q = gl.load(Q + q_offsets)
    # Required query rounding occurs before widening and before scale use.
    q = q.to(gl.bfloat16).to(gl.float8e4nv).to(gl.bfloat16)
    q = gl.convert_layout(q, dot_layout)

    k_buffers = gl.allocate_shared_memory(gl.bfloat16, [2, 64, 64], shared_layout)
    v_buffers = gl.allocate_shared_memory(gl.bfloat16, [2, 64, 64], shared_layout)
    slot = 0
    raw_k, offsets, valid = _prefetch_k(
        KCache, PageTable, b, kvh, 0, total, HKV, PT_STRIDE)
    _publish_kv(VCache, raw_k, offsets, valid,
                k_buffers.index(slot), v_buffers.index(slot))

    m_i = gl.full((64,), float("-inf"), gl.float32, layout=row_layout)
    l_i = gl.zeros((64,), gl.float32, layout=row_layout)
    acc = gl.zeros((64, 64), gl.float32, layout=mma_layout)
    m = gl.arange(0, 64, layout=row_layout)
    n = gl.arange(0, 64, layout=col_layout)
    q_position = seq + m // 8

    for start in range(0, total, 64):
        current_k = k_buffers.index(slot)
        current_v = v_buffers.index(slot)
        score_zero = gl.zeros((64, 64), gl.float32, layout=mma_layout)
        score_pending = warpgroup_mma(
            q, current_k.permute((1, 0)), score_zero,
            use_acc=False, max_num_imprecise_acc=0, is_async=True,
        )
        # Both arms prefetch the same raw K bytes before the score wait.
        next_raw_k, next_offsets, next_valid = _prefetch_k(
            KCache, PageTable, b, kvh, start + 64, total, HKV, PT_STRIDE)
        if not V_BEFORE_WAIT:
            scores = warpgroup_mma_wait(num_outstanding=0, deps=(score_pending,))

        # Only the V gather and common K/V widening/publication move across
        # the wait. The other slot is free after the prior synchronous PV.
        # The final iteration publishes a fully zero-filled masked tile.
        next_slot = slot ^ 1
        _publish_kv(VCache, next_raw_k, next_offsets, next_valid,
                    k_buffers.index(next_slot), v_buffers.index(next_slot))

        if V_BEFORE_WAIT:
            scores = warpgroup_mma_wait(num_outstanding=0, deps=(score_pending,))

        t = start + n
        visible = (t[None, :] < total) & (t[None, :] <= q_position[:, None])
        scores = gl.where(visible, scores * qk_scale, float("-inf"))
        m_new = gl.maximum(m_i, gl.max(scores, 1))
        m_safe = gl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = gl.exp2(m_i - m_safe)
        p = gl.exp2(scores - m_safe[:, None])
        l_i = l_i * alpha + gl.sum(p, 1)
        p = gl.convert_layout(p.to(gl.bfloat16), dot_layout)
        pv_zero = gl.zeros((64, 64), gl.float32, layout=mma_layout)
        pv = warpgroup_mma(
            p, current_v, pv_zero,
            use_acc=False, max_num_imprecise_acc=0, is_async=False,
        )
        acc = acc * alpha[:, None] + pv
        m_i = m_new
        slot = next_slot

    out = (acc / l_i[:, None] * vs).to(gl.bfloat16)
    out = gl.convert_layout(out, load_layout)
    gl.store(Out + q_offsets, out)
