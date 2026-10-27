# okc: problem=muse_attention_decode
"""v3 (Frank Zhu): one Triton launch for every shape, FA3 nowhere.

Changes vs v2 (the per-KV-head kernel the earlier repo shipped as `_fused`):
  1. PAIRED KV HEADS. One CTA owns two KV heads (all 16 query heads on TP8). In the cache a
     token's two heads are one contiguous 128 B row, so each K (and V) gather reads a whole
     row once, instead of two CTAs each reading 64 B halves at different times. That halves
     the page-table reads and the number of CTAs that walk the same rows.
  2. SPLITS SIZED TO THE GPU. split count = enough CTAs for ~3 per SM (132 SMs), but never
     fewer than MIN_TOK keys per split. Each request divides its own length, so ragged
     ("agentic") batches stay balanced. Shape-only, so it is graph-safe.
  3. TWO-LEVEL TICKET MERGE. Splits are grouped (about sqrt(splits) per group). The last
     CTA of a group merges that group; the last group-merger merges the groups and writes
     the output. No CTA ever reads more than ~sqrt(splits) partials, so a b1 step with
     ~170 splits doesn't stall on one serial merger, and there is still no second launch.
  4. EXPLICIT PREFETCH (PREFETCH=1). The next tile's page indices and K/V rows are loaded
     into registers before the current tile is computed, so two tiles are always in
     flight, whatever Triton's pipeliner decides. PREFETCH=0 leaves it to num_stages.

Kept from v2: QK-norm + RoPE in registers, the last split computes/stores/attends the
block's own new tokens, BF16 tensor-core dots, FP32 online softmax, RMSNorm*sigmoid(gate).
Every key in the window/context is attended.
"""

import math
import os
from typing import TypedDict

import torch
import triton
import triton.language as tl


class Config(TypedDict):
    batch: int
    q_len: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    window: int
    rope_theta: float
    qk_scale_factor: float
    sm_scale: float
    kv_scale: float


class Inputs(TypedDict):
    qkv: torch.Tensor
    positions: torch.Tensor
    cache_loc: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    scheduler_metadata: torch.Tensor | None
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    config: Config


RMS_EPS = 1e-5
LOG2E = 1.4426950408889634
NUM_SMS = 132  # H200 SXM

# ---- tunables (the experiment knobs) -------------------------------------------------
CTAS_PER_SM = 3
MIN_TOK_FULL = 512
MIN_TOK_SLIDING = 128
PREFETCH = 1
BLOCK_N = 64
# ---------------------------------------------------------------------------------------

_INTERP = os.environ.get("TRITON_INTERPRET") == "1"
# Tensor-core operands are BF16 on the GPU. Triton's CPU interpreter (local testing only)
# can't matmul BF16, so there the same BF16-rounded values are multiplied in FP32.
_DOT = tl.float32 if _INTERP else tl.bfloat16
_TICKETS = {}


def _tickets(device, n):
    """Zeroed int32 counters. Every merger resets its counter to 0, so the buffer (not
    derived from inputs) is reusable across calls and graph replays without a memset."""
    buf = _TICKETS.get(device)
    if buf is None or buf.numel() < n:
        buf = torch.zeros(max(n, 1 << 16), dtype=torch.int32, device=device)
        _TICKETS[device] = buf
    return buf


@triton.jit
def _norm_rope(x_ptr, offs_d, pos, theta, eps, ROPE: tl.constexpr, D: tl.constexpr, mask):
    """RMSNorm (no weight) then interleaved RoPE of rows x[..., D] (fp32).
    x_ptr: [R, 1] row base pointers; pos: [R, 1] positions; mask: [R, 1]."""
    x = tl.load(x_ptr + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 1) / D + eps)[:, None]
    x = x * inv
    if ROPE:
        partner = tl.load(x_ptr + (offs_d ^ 1)[None, :], mask=mask, other=0.0).to(tl.float32) * inv
        i = (offs_d // 2).to(tl.float32)
        inv_freq = tl.exp2(-(2.0 * i / D) * tl.log2(theta))
        ang = pos.to(tl.float32) * inv_freq[None, :]
        sign = tl.where((offs_d % 2) == 0, -1.0, 1.0)[None, :]
        x = x * tl.cos(ang) + sign * partner * tl.sin(ang)
    return x


@triton.jit
def _softmax_step(q, k, v, vis, m_i, l_i, acc, qk_scale, DOT: tl.constexpr):
    """One online-softmax tile for one KV head. k, v: [BLOCK_N, D] FP8; scores in log2."""
    s = tl.dot(q, tl.trans(k.to(DOT))) * qk_scale
    s = tl.where(vis, s, float("-inf"))
    m_new = tl.maximum(m_i, tl.max(s, 1))
    m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
    alpha = tl.exp2(m_i - m_safe)
    p = tl.exp2(s - m_safe[:, None])
    l_i = l_i * alpha + tl.sum(p, 1)
    acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16).to(DOT), v.to(DOT))
    return m_new, l_i, acc


@triton.jit
def _rows_of(pt_ptr, b, stride_pt, t, t_ok, page, PAGE1: tl.constexpr):
    if PAGE1:
        row = tl.load(pt_ptr + b * stride_pt + t, mask=t_ok, other=0).to(tl.int64)
    else:
        pg = tl.load(pt_ptr + b * stride_pt + t // page, mask=t_ok, other=0).to(tl.int64)
        row = pg * page + t % page
    return tl.maximum(row, 0)


@triton.jit
def _new_tokens(q, m_i, l_i, acc, qkv_ptr, loc_ptr, kc_ptr, vc_ptr, b, kvh, seq, qpos,
                ks, vs, qk_scale, theta, eps, stride_tok, offs_d,
                L: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
                BLOCK_L: tl.constexpr, ROPE: tl.constexpr, DOT: tl.constexpr):
    """The block's own L tokens for KV head kvh: norm/RoPE k, round to E4M3 exactly as
    stored, write the cache rows, attend causally from registers."""
    offs_l = tl.arange(0, BLOCK_L)
    l_ok = offs_l < L
    ntok = (b * L + tl.where(l_ok, offs_l, 0)).to(tl.int64)
    npos = seq + offs_l
    k_rows = qkv_ptr + ntok[:, None] * stride_tok + (HQ * D + kvh * D)
    kn = _norm_rope(k_rows, offs_d, npos[:, None], theta, eps, ROPE, D, l_ok[:, None])
    vn = tl.load(qkv_ptr + ntok[:, None] * stride_tok + ((HQ + HKV) * D + kvh * D)
                 + offs_d[None, :], mask=l_ok[:, None], other=0.0).to(tl.float32)
    kq = (kn / ks).to(kc_ptr.dtype.element_ty)
    vq = (vn / vs).to(vc_ptr.dtype.element_ty)
    loc = tl.load(loc_ptr + ntok, mask=l_ok, other=0)
    dst = loc[:, None] * (HKV * D) + kvh * D + offs_d[None, :]
    tl.store(kc_ptr + dst, kq, mask=l_ok[:, None])
    tl.store(vc_ptr + dst, vq, mask=l_ok[:, None])
    vis = l_ok[None, :] & (npos[None, :] <= qpos[:, None])
    return _softmax_step(q, kq.to(tl.float32), vq.to(tl.float32), vis, m_i, l_i, acc,
                         qk_scale, DOT)


@triton.jit
def _merge(m_ptr, l_ptr, o_ptr, first, count, offs_m, offs_d, row_ok,
           R: tl.constexpr, BLOCK_M: tl.constexpr, D: tl.constexpr):
    """Log-sum-exp merge of `count` partial row-sets starting at row-set index `first`."""
    m_t = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_t = tl.zeros([BLOCK_M], tl.float32)
    a = tl.zeros([BLOCK_M, D], tl.float32)
    for s_i in range(0, count):
        qb = (first + s_i) * R + offs_m
        m_s = tl.load(m_ptr + qb, mask=row_ok, other=float("-inf"), cache_modifier=".cg")
        l_s = tl.load(l_ptr + qb, mask=row_ok, other=0.0, cache_modifier=".cg")
        o_s = tl.load(o_ptr + qb[:, None] * D + offs_d[None, :], mask=row_ok[:, None],
                      other=0.0, cache_modifier=".cg")
        m_n = tl.maximum(m_t, m_s)
        m_z = tl.where(m_n == float("-inf"), 0.0, m_n)
        w_t = tl.exp2(m_t - m_z)
        w_s = tl.exp2(m_s - m_z)
        l_t = l_t * w_t + l_s * w_s
        a = a * w_t[:, None] + o_s * w_s[:, None]
        m_t = m_n
    return m_t, l_t, a


@triton.jit
def _put(m_ptr, l_ptr, o_ptr, idx, m_i, l_i, acc, offs_m, offs_d, row_ok,
         R: tl.constexpr, D: tl.constexpr):
    qb = idx * R + offs_m
    tl.store(o_ptr + qb[:, None] * D + offs_d[None, :], acc, mask=row_ok[:, None])
    tl.store(m_ptr + qb, m_i, mask=row_ok)
    tl.store(l_ptr + qb, l_i, mask=row_ok)


@triton.jit
def _finish(qkv_ptr, out_ptr, acc, l_i, vs, eps, tok, head, offs_d, row_ok, stride_tok,
            HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr):
    a = acc * vs / l_i[:, None]
    a = a * tl.rsqrt(tl.sum(a * a, 1) / D + eps)[:, None]
    gate_off = tok[:, None] * stride_tok + ((HQ + 2 * HKV) * D) + head[:, None] * D + offs_d[None, :]
    gate = tl.load(qkv_ptr + gate_off, mask=row_ok[:, None], other=0.0).to(tl.float32)
    out_off = tok[:, None] * (HQ * D) + head[:, None] * D + offs_d[None, :]
    tl.store(out_ptr + out_off, (a * tl.sigmoid(gate)).to(tl.bfloat16), mask=row_ok[:, None])


@triton.jit
def _attn_pair(qkv_ptr, seq_ptr, pt_ptr, loc_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr,
               o1_ptr, m1_ptr, l1_ptr, o2_ptr, m2_ptr, l2_ptr, tk1_ptr, tk2_ptr, out_ptr,
               theta, eps, qk_factor, sm_scale, window, page, stride_tok, stride_pt,
               L: tl.constexpr, G: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr,
               D: tl.constexpr, R: tl.constexpr, BLOCK_M: tl.constexpr,
               BLOCK_N: tl.constexpr, BLOCK_L: tl.constexpr,
               SPLITS: tl.constexpr, GS: tl.constexpr, NG: tl.constexpr,
               ROPE: tl.constexpr, WINDOWED: tl.constexpr, PAGE1: tl.constexpr,
               PREFETCH: tl.constexpr, DOT: tl.constexpr):
    b = tl.program_id(0)
    pair = tl.program_id(1)
    split = tl.program_id(2)
    h0 = pair * 2
    h1 = h0 + 1
    NP: tl.constexpr = HKV // 2

    seq = tl.load(seq_ptr + b).to(tl.int64)
    # Query rows r = j * G + g  (block token j, query head kvh*G + g), per KV head.
    offs_m = tl.arange(0, BLOCK_M)
    row_ok = offs_m < R
    j = tl.where(row_ok, offs_m // G, 0)
    g = offs_m % G
    qpos = seq + j
    offs_d = tl.arange(0, D)
    tok = (b * L + j).to(tl.int64)
    tok_base = qkv_ptr + tok[:, None] * stride_tok
    q0 = _norm_rope(tok_base + ((h0 * G + g) * D)[:, None], offs_d, qpos[:, None], theta, eps,
                    ROPE, D, row_ok[:, None])
    q1 = _norm_rope(tok_base + ((h1 * G + g) * D)[:, None], offs_d, qpos[:, None], theta, eps,
                    ROPE, D, row_ok[:, None])
    # As the reference: RMSNorm * qk_scale_factor (RoPE is linear), rounded to BF16.
    q0 = (q0 * qk_factor).to(tl.bfloat16).to(DOT)
    q1 = (q1 * qk_factor).to(tl.bfloat16).to(DOT)
    ks = tl.load(ks_ptr)
    vs = tl.load(vs_ptr)
    qk_scale = sm_scale * ks * LOG2E

    # This split's slice [s_lo, s_hi) of the cached keys [lo, seq).
    if WINDOWED:
        lo = tl.maximum(seq - window + 1, 0)   # earliest key the first block row sees
    else:
        lo = seq * 0
    lo = tl.minimum(lo, seq)
    span = seq - lo
    per = ((span + SPLITS - 1) // SPLITS + BLOCK_N - 1) // BLOCK_N * BLOCK_N
    s_lo = lo + split * per
    s_hi = tl.minimum(s_lo + per, seq)
    n_tiles = (s_hi - s_lo + BLOCK_N - 1) // BLOCK_N   # <= 0 for an empty slice

    m0 = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l0 = tl.zeros([BLOCK_M], tl.float32)
    a0 = tl.zeros([BLOCK_M, D], tl.float32)
    m1 = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l1 = tl.zeros([BLOCK_M], tl.float32)
    a1 = tl.zeros([BLOCK_M, D], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    ROW: tl.constexpr = HKV * D   # cache row stride: all KV heads of one token
    c0 = h0 * D + offs_d[None, :]
    c1 = h1 * D + offs_d[None, :]

    if PREFETCH:
        t = s_lo + offs_n
        t_ok = t < s_hi
        row = _rows_of(pt_ptr, b, stride_pt, t, t_ok, page, PAGE1)
        off = row[:, None] * ROW
        k0n = tl.load(kc_ptr + off + c0, mask=t_ok[:, None], other=0.0)
        k1n = tl.load(kc_ptr + off + c1, mask=t_ok[:, None], other=0.0)
        v0n = tl.load(vc_ptr + off + c0, mask=t_ok[:, None], other=0.0)
        v1n = tl.load(vc_ptr + off + c1, mask=t_ok[:, None], other=0.0)
        for it in range(0, n_tiles):
            k0 = k0n
            k1 = k1n
            v0 = v0n
            v1 = v1n
            tc = t
            tc_ok = t_ok
            # Issue the next tile's gathers before computing this one.
            t = s_lo + (it + 1) * BLOCK_N + offs_n
            t_ok = t < s_hi
            row = _rows_of(pt_ptr, b, stride_pt, t, t_ok, page, PAGE1)
            off = row[:, None] * ROW
            k0n = tl.load(kc_ptr + off + c0, mask=t_ok[:, None], other=0.0)
            k1n = tl.load(kc_ptr + off + c1, mask=t_ok[:, None], other=0.0)
            v0n = tl.load(vc_ptr + off + c0, mask=t_ok[:, None], other=0.0)
            v1n = tl.load(vc_ptr + off + c1, mask=t_ok[:, None], other=0.0)
            vis = tc_ok[None, :]
            if WINDOWED:
                vis = vis & (tc[None, :] > qpos[:, None] - window)
            m0, l0, a0 = _softmax_step(q0, k0, v0, vis, m0, l0, a0, qk_scale, DOT)
            m1, l1, a1 = _softmax_step(q1, k1, v1, vis, m1, l1, a1, qk_scale, DOT)
    else:
        for start in range(s_lo, s_hi, BLOCK_N):
            t = start + offs_n
            t_ok = t < s_hi
            row = _rows_of(pt_ptr, b, stride_pt, t, t_ok, page, PAGE1)
            off = row[:, None] * ROW
            k0 = tl.load(kc_ptr + off + c0, mask=t_ok[:, None], other=0.0)
            k1 = tl.load(kc_ptr + off + c1, mask=t_ok[:, None], other=0.0)
            v0 = tl.load(vc_ptr + off + c0, mask=t_ok[:, None], other=0.0)
            v1 = tl.load(vc_ptr + off + c1, mask=t_ok[:, None], other=0.0)
            vis = t_ok[None, :]
            if WINDOWED:
                vis = vis & (t[None, :] > qpos[:, None] - window)
            m0, l0, a0 = _softmax_step(q0, k0, v0, vis, m0, l0, a0, qk_scale, DOT)
            m1, l1, a1 = _softmax_step(q1, k1, v1, vis, m1, l1, a1, qk_scale, DOT)

    if split == SPLITS - 1:
        m0, l0, a0 = _new_tokens(q0, m0, l0, a0, qkv_ptr, loc_ptr, kc_ptr, vc_ptr, b, h0, seq,
                                 qpos, ks, vs, qk_scale, theta, eps, stride_tok, offs_d,
                                 L, HQ, HKV, D, BLOCK_L, ROPE, DOT)
        m1, l1, a1 = _new_tokens(q1, m1, l1, a1, qkv_ptr, loc_ptr, kc_ptr, vc_ptr, b, h1, seq,
                                 qpos, ks, vs, qk_scale, theta, eps, stride_tok, offs_d,
                                 L, HQ, HKV, D, BLOCK_L, ROPE, DOT)

    head0 = h0 * G + g
    head1 = h1 * G + g
    if SPLITS == 1:
        _finish(qkv_ptr, out_ptr, a0, l0, vs, eps, tok, head0, offs_d, row_ok, stride_tok, HQ, HKV, D)
        _finish(qkv_ptr, out_ptr, a1, l1, vs, eps, tok, head1, offs_d, row_ok, stride_tok, HQ, HKV, D)
    else:
        bh0 = b * HKV + h0
        bh1 = bh0 + 1
        # Level 1: partials [B*HKV, SPLITS, R, (D)].
        _put(m1_ptr, l1_ptr, o1_ptr, bh0 * SPLITS + split, m0, l0, a0, offs_m, offs_d, row_ok, R, D)
        _put(m1_ptr, l1_ptr, o1_ptr, bh1 * SPLITS + split, m1, l1, a1, offs_m, offs_d, row_ok, R, D)
        tl.debug_barrier()  # every warp's partial stores precede the ticket (release below)
        grp = split // GS
        gsz = tl.minimum(GS, SPLITS - grp * GS)
        bp = b * NP + pair
        t1 = tk1_ptr + bp * NG + grp
        prev = tl.atomic_add(t1, 1, sem="acq_rel", scope="gpu")
        if prev == gsz - 1:
            gm0, gl0, ga0 = _merge(m1_ptr, l1_ptr, o1_ptr, bh0 * SPLITS + grp * GS, gsz,
                                   offs_m, offs_d, row_ok, R, BLOCK_M, D)
            gm1, gl1, ga1 = _merge(m1_ptr, l1_ptr, o1_ptr, bh1 * SPLITS + grp * GS, gsz,
                                   offs_m, offs_d, row_ok, R, BLOCK_M, D)
            tl.atomic_xchg(t1, 0, sem="relaxed", scope="gpu")
            if NG == 1:
                _finish(qkv_ptr, out_ptr, ga0, gl0, vs, eps, tok, head0, offs_d, row_ok, stride_tok, HQ, HKV, D)
                _finish(qkv_ptr, out_ptr, ga1, gl1, vs, eps, tok, head1, offs_d, row_ok, stride_tok, HQ, HKV, D)
            else:
                # Level 2: group partials [B*HKV, NG, R, (D)].
                _put(m2_ptr, l2_ptr, o2_ptr, bh0 * NG + grp, gm0, gl0, ga0, offs_m, offs_d, row_ok, R, D)
                _put(m2_ptr, l2_ptr, o2_ptr, bh1 * NG + grp, gm1, gl1, ga1, offs_m, offs_d, row_ok, R, D)
                tl.debug_barrier()
                t2 = tk2_ptr + bp
                prev2 = tl.atomic_add(t2, 1, sem="acq_rel", scope="gpu")
                if prev2 == NG - 1:
                    fm0, fl0, fa0 = _merge(m2_ptr, l2_ptr, o2_ptr, bh0 * NG, NG,
                                           offs_m, offs_d, row_ok, R, BLOCK_M, D)
                    fm1, fl1, fa1 = _merge(m2_ptr, l2_ptr, o2_ptr, bh1 * NG, NG,
                                           offs_m, offs_d, row_ok, R, BLOCK_M, D)
                    _finish(qkv_ptr, out_ptr, fa0, fl0, vs, eps, tok, head0, offs_d, row_ok, stride_tok, HQ, HKV, D)
                    _finish(qkv_ptr, out_ptr, fa1, fl1, vs, eps, tok, head1, offs_d, row_ok, stride_tok, HQ, HKV, D)
                    tl.atomic_xchg(t2, 0, sem="relaxed", scope="gpu")


def _plan(batch, pairs, span_max, min_tok):
    """(splits, group size, groups) from shapes only."""
    ctas = batch * pairs
    want = max(1, -(-CTAS_PER_SM * NUM_SMS // ctas))
    most = max(1, span_max // min_tok)
    splits = min(want, most)
    gs = splits if splits <= 8 else math.ceil(math.sqrt(splits))
    ng = -(-splits // gs)
    return splits, gs, ng


def kernel_fn(data: Inputs) -> torch.Tensor:
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d, page = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"], cfg["page_size"]
    assert hkv % 2 == 0, "paired kernel needs an even number of KV heads"
    g = hq // hkv
    pairs = hkv // 2
    tokens = batch * length
    qkv = data["qkv"].reshape(tokens, -1)
    window = cfg["window"]
    rope = bool(cfg["rope_theta"])
    theta = float(cfg["rope_theta"]) if rope else 1.0
    page_table = data["page_table"]
    dev = qkv.device

    r = length * g
    block_m = max(16, triton.next_power_of_2(r))
    span_max = window if window else page_table.shape[1] * page  # shape-only bound
    min_tok = MIN_TOK_SLIDING if window else MIN_TOK_FULL
    splits, gs, ng = _plan(batch, pairs, span_max, min_tok)
    out = torch.empty((tokens, hq * d), dtype=torch.bfloat16, device=dev)
    if splits > 1:
        n1 = batch * hkv * splits * r
        o1 = torch.empty((n1, d), dtype=torch.float32, device=dev)
        m1 = torch.empty((n1,), dtype=torch.float32, device=dev)
        l1 = torch.empty((n1,), dtype=torch.float32, device=dev)
        n2 = batch * hkv * ng * r if ng > 1 else 1
        o2 = torch.empty((n2, d), dtype=torch.float32, device=dev)
        m2 = torch.empty((n2,), dtype=torch.float32, device=dev)
        l2 = torch.empty((n2,), dtype=torch.float32, device=dev)
    else:
        o1 = m1 = l1 = o2 = m2 = l2 = out
    n_t1 = batch * pairs * ng
    tk = _tickets(dev, n_t1 + batch * pairs)
    tk1, tk2 = tk[:n_t1], tk[n_t1:]
    warps = 8 if block_m >= 64 else 4
    _attn_pair[(batch, pairs, splits)](
        qkv, data["seq_lens"], page_table, data["cache_loc"], data["k_cache"], data["v_cache"],
        data["k_scale"], data["v_scale"], o1, m1, l1, o2, m2, l2, tk1, tk2, out,
        theta, RMS_EPS, float(cfg["qk_scale_factor"]), float(cfg["sm_scale"]),
        window, page, qkv.stride(0), page_table.stride(0),
        L=length, G=g, HQ=hq, HKV=hkv, D=d, R=r, BLOCK_M=block_m, BLOCK_N=BLOCK_N,
        BLOCK_L=16, SPLITS=splits, GS=gs, NG=ng, ROPE=rope, WINDOWED=bool(window),
        PAGE1=(page == 1), PREFETCH=PREFETCH, DOT=_DOT,
        num_warps=warps, num_stages=1 if PREFETCH else 3)
    return out.view(batch, length, hq * d)
