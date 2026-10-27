"""Two independent KV heads per CTA for exact full attention.

Adapted from the selected fusion/lookahead producer's normalization, FP8 QK,
BF16 PV, FP32 online state and cache-write semantics. The leading dot dimension
is a batch of two heads, never a concatenation that mixes the two attentions.
Physical rows retain the input's [row, HKV, D] layout and live page mapping.
"""

import triton
import triton.language as tl


@triton.jit
def _dual_full(qkv_ptr, seq_ptr, pt_ptr, loc_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr,
               o_ptr, m_ptr, l_ptr, eps, qk_factor, sm_scale,
               page, stride_tok, stride_pt,
               L: tl.constexpr, G: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr,
               D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
               BLOCK_L: tl.constexpr, SPLITS: tl.constexpr,
               QK_DOT: tl.constexpr, DOT: tl.constexpr):
    tl.static_assert(HKV % 2 == 0)
    tl.static_assert(HQ == HKV * G)
    tl.static_assert(D == 64)
    tl.static_assert(L * G <= BLOCK_M)
    b = tl.program_id(0)
    pair = tl.program_id(1)
    split = tl.program_id(2)
    kvh = pair * 2 + tl.arange(0, 2)
    seq = tl.load(seq_ptr + b).to(tl.int64)
    rm = tl.arange(0, BLOCK_M)
    rd = tl.arange(0, D)
    valid_m = rm < L * G
    j = tl.where(valid_m, rm // G, 0)
    g = rm % G
    tok = (b * L + j).to(tl.int64)
    head = kvh[:, None] * G + g[None, :]
    qoff = tok[None, :, None] * stride_tok + head[:, :, None] * D + rd[None, None, :]
    qf = tl.load(qkv_ptr + qoff, mask=valid_m[None, :, None], other=0.0).to(tl.float32)
    qf = qf * tl.rsqrt(tl.sum(qf * qf, 2) / D + eps)[:, :, None]
    # Scale placement and both rounding boundaries match the selected producer.
    q = qf.to(tl.bfloat16).to(tl.float8e4nv).to(QK_DOT)
    ks = tl.load(ks_ptr)
    qk_scale = sm_scale * qk_factor * ks * 1.4426950408889634

    # Balanced, N-aligned history chunks consume GPU-live lengths on each replay.
    per = ((seq + SPLITS - 1) // SPLITS + BLOCK_N - 1) // BLOCK_N * BLOCK_N
    s_lo = split * per
    s_hi = tl.minimum(s_lo + per, seq)
    rn = tl.arange(0, BLOCK_N)
    m = tl.full([2, BLOCK_M], float("-inf"), tl.float32)
    mass = tl.zeros([2, BLOCK_M], tl.float32)
    acc = tl.zeros([2, BLOCK_M, D], tl.float32)
    for start in range(s_lo, s_hi, BLOCK_N):
        t = start + rn
        valid_n = t < s_hi
        pg = tl.load(pt_ptr + b * stride_pt + t // page,
                     mask=valid_n, other=0).to(tl.int64)
        physical_row = tl.maximum(pg, 0) * page + t % page
        # Both heads share page IDs; their payload addresses and math stay separate.
        koff = physical_row[None, :, None] * (HKV * D) + kvh[:, None, None] * D + rd[None, None, :]
        k = tl.load(kc_ptr + koff, mask=valid_n[None, :, None], other=0.0).to(QK_DOT)
        score = tl.dot(q, tl.permute(k, (0, 2, 1)), out_dtype=tl.float32,
                       max_num_imprecise_acc=0) * qk_scale
        score = tl.where(valid_n[None, None, :], score, float("-inf"))
        m_new = tl.maximum(m, tl.max(score, 2))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m - m_safe)
        p = tl.exp2(score - m_safe[:, :, None])
        mass = mass * alpha + tl.sum(p, 2)
        v = tl.load(vc_ptr + koff, mask=valid_n[None, :, None], other=0.0).to(DOT)
        acc = acc * alpha[:, :, None] + tl.dot(p.to(tl.bfloat16).to(DOT), v)
        m = m_new

    if split == SPLITS - 1:
        # The final split alone owns new KV writes and the causal new-token tail.
        # This branch remains necessary even when its historical interval is empty.
        rl = tl.arange(0, BLOCK_L)
        valid_l = rl < L
        ntok = (b * L + tl.where(valid_l, rl, 0)).to(tl.int64)
        noff = ntok[None, :, None] * stride_tok + kvh[:, None, None] * D + rd[None, None, :]
        kn = tl.load(qkv_ptr + noff + HQ * D,
                     mask=valid_l[None, :, None], other=0.0).to(tl.float32)
        kn = kn * tl.rsqrt(tl.sum(kn * kn, 2) / D + eps)[:, :, None]
        vn = tl.load(qkv_ptr + noff + (HQ + HKV) * D,
                     mask=valid_l[None, :, None], other=0.0).to(tl.float32)
        vs = tl.load(vs_ptr)
        kq = (kn / ks).to(kc_ptr.dtype.element_ty)
        vq = (vn / vs).to(vc_ptr.dtype.element_ty)
        loc = tl.load(loc_ptr + ntok, mask=valid_l, other=0).to(tl.int64)
        dst = loc[None, :, None] * (HKV * D) + kvh[:, None, None] * D + rd[None, None, :]
        tl.store(kc_ptr + dst, kq, mask=valid_l[None, :, None])
        tl.store(vc_ptr + dst, vq, mask=valid_l[None, :, None])
        score = tl.dot(q, tl.permute(kq.to(QK_DOT), (0, 2, 1)),
                       out_dtype=tl.float32, max_num_imprecise_acc=0) * qk_scale
        causal = valid_l[None, :] & (rl[None, :] <= j[:, None])
        score = tl.where(causal[None, :, :], score, float("-inf"))
        m_new = tl.maximum(m, tl.max(score, 2))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m - m_safe)
        p = tl.exp2(score - m_safe[:, :, None])
        mass = mass * alpha + tl.sum(p, 2)
        acc = acc * alpha[:, :, None] + tl.dot(p.to(tl.bfloat16).to(DOT),
                                               vq.to(tl.float32).to(DOT))
        m = m_new

    # Preserve [B, HKV, SPLITS, BLOCK_M, D] partial ownership for both heads.
    base = ((b * HKV + kvh[:, None]) * SPLITS + split) * BLOCK_M + rm[None, :]
    tl.store(o_ptr + base[:, :, None] * D + rd[None, None, :], acc)
    tl.store(m_ptr + base, m)
    tl.store(l_ptr + base, mass)


@triton.jit
def _dual_merge(qkv_ptr, vs_ptr, o_ptr, m_ptr, l_ptr, out_ptr, eps, stride_tok,
                L: tl.constexpr, G: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr,
                D: tl.constexpr, BLOCK_M: tl.constexpr, SPLITS: tl.constexpr,
                BS: tl.constexpr):
    # One valid (token, query head) per CTA; no reduction combines KV heads.
    tok = tl.program_id(0)
    b = tok // L
    j = tok % L
    head = tl.program_id(1)
    kvh = head // G
    qrow = j * G + head % G
    rs = tl.arange(0, BS)
    rd = tl.arange(0, D)
    valid_s = rs < SPLITS
    base = ((b * HKV + kvh) * SPLITS + rs) * BLOCK_M + qrow
    ms = tl.load(m_ptr + base, mask=valid_s, other=float("-inf"), cache_modifier=".cg")
    ls = tl.load(l_ptr + base, mask=valid_s, other=0.0, cache_modifier=".cg")
    os = tl.load(o_ptr + base[:, None] * D + rd[None, :],
                 mask=valid_s[:, None], other=0.0, cache_modifier=".cg")
    m = tl.max(ms, 0)
    safe_m = tl.where(m == float("-inf"), 0.0, m)
    weight = tl.exp2(ms - safe_m)
    mass = tl.sum(ls * weight, 0)
    acc = tl.sum(os * weight[:, None], 0)
    vs = tl.load(vs_ptr)
    a = acc * vs / mass
    a = a * tl.rsqrt(tl.sum(a * a, 0) / D + eps)
    out_off = tok * HQ * D + head * D + rd
    gate_off = tok * stride_tok + (HQ + 2 * HKV) * D + head * D + rd
    gate = tl.load(qkv_ptr + gate_off).to(tl.float32)
    tl.store(out_ptr + out_off, (a * tl.sigmoid(gate)).to(tl.bfloat16))
