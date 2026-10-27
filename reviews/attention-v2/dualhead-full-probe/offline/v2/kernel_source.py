"""Two independent KV heads per CTA for exact full attention.

Adapted from the selected fusion/lookahead producer's normalization, FP8 QK,
BF16 PV, FP32 online state and cache-write semantics. Each head has an explicit
2D dot and independent state. Contiguous paired loads share live page mapping;
the physical stride remains HKV*D, including DP configurations.
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
    kvh0 = pair * 2
    seq = tl.load(seq_ptr + b).to(tl.int64)
    rm = tl.arange(0, BLOCK_M)
    rd = tl.arange(0, D)
    rc = tl.arange(0, 2 * D)
    valid_m = rm < L * G
    j = tl.where(valid_m, rm // G, 0)
    g = rm % G
    tok = (b * L + j).to(tl.int64)
    # Adjacent D-wide column groups select heads kvh0 and kvh0+1. Query heads
    # are G*D apart; cache heads below are D apart. Never conflate those strides.
    qhead = kvh0 * G + g[:, None] + (rc[None, :] // D) * G
    qoff = tok[:, None] * stride_tok + qhead * D + rc[None, :] % D
    q01 = tl.load(qkv_ptr + qoff, mask=valid_m[:, None], other=0.0).to(tl.float32)
    q0, q1 = tl.split(tl.permute(tl.reshape(q01, (BLOCK_M, 2, D)), (0, 2, 1)))
    q0 = q0 * tl.rsqrt(tl.sum(q0 * q0, 1) / D + eps)[:, None]
    q1 = q1 * tl.rsqrt(tl.sum(q1 * q1, 1) / D + eps)[:, None]
    # Scale placement and both rounding boundaries match the selected producer.
    q0 = q0.to(tl.bfloat16).to(tl.float8e4nv).to(QK_DOT)
    q1 = q1.to(tl.bfloat16).to(tl.float8e4nv).to(QK_DOT)
    ks = tl.load(ks_ptr)
    qk_scale = sm_scale * qk_factor * ks * 1.4426950408889634

    # Balanced, N-aligned history chunks consume GPU-live lengths on each replay.
    per = ((seq + SPLITS - 1) // SPLITS + BLOCK_N - 1) // BLOCK_N * BLOCK_N
    s_lo = split * per
    s_hi = tl.minimum(s_lo + per, seq)
    rn = tl.arange(0, BLOCK_N)
    m0 = tl.full([BLOCK_M], float("-inf"), tl.float32)
    m1 = tl.full([BLOCK_M], float("-inf"), tl.float32)
    mass0 = tl.zeros([BLOCK_M], tl.float32)
    mass1 = tl.zeros([BLOCK_M], tl.float32)
    acc0 = tl.zeros([BLOCK_M, D], tl.float32)
    acc1 = tl.zeros([BLOCK_M, D], tl.float32)
    for start in range(s_lo, s_hi, BLOCK_N):
        t = start + rn
        valid_n = t < s_hi
        pg = tl.load(pt_ptr + b * stride_pt + t // page,
                     mask=valid_n, other=0).to(tl.int64)
        physical_row = tl.maximum(pg, 0) * page + t % page
        # One [N,128] physical load per cache; each dot remains a 2D operation.
        koff = physical_row[:, None] * (HKV * D) + kvh0 * D + rc[None, :]
        k01 = tl.load(kc_ptr + koff, mask=valid_n[:, None], other=0.0).to(QK_DOT)
        k0, k1 = tl.split(tl.permute(tl.reshape(k01, (BLOCK_N, 2, D)), (0, 2, 1)))
        score0 = tl.dot(q0, tl.trans(k0), out_dtype=tl.float32,
                        max_num_imprecise_acc=0) * qk_scale
        score0 = tl.where(valid_n[None, :], score0, float("-inf"))
        mn0 = tl.maximum(m0, tl.max(score0, 1))
        safe0 = tl.where(mn0 == float("-inf"), 0.0, mn0)
        alpha0 = tl.exp2(m0 - safe0)
        p0 = tl.exp2(score0 - safe0[:, None])
        mass0 = mass0 * alpha0 + tl.sum(p0, 1)
        v01 = tl.load(vc_ptr + koff, mask=valid_n[:, None], other=0.0).to(DOT)
        v0, v1 = tl.split(tl.permute(tl.reshape(v01, (BLOCK_N, 2, D)), (0, 2, 1)))
        acc0 = acc0 * alpha0[:, None] + tl.dot(p0.to(tl.bfloat16).to(DOT), v0)
        m0 = mn0

        score1 = tl.dot(q1, tl.trans(k1), out_dtype=tl.float32,
                        max_num_imprecise_acc=0) * qk_scale
        score1 = tl.where(valid_n[None, :], score1, float("-inf"))
        mn1 = tl.maximum(m1, tl.max(score1, 1))
        safe1 = tl.where(mn1 == float("-inf"), 0.0, mn1)
        alpha1 = tl.exp2(m1 - safe1)
        p1 = tl.exp2(score1 - safe1[:, None])
        mass1 = mass1 * alpha1 + tl.sum(p1, 1)
        acc1 = acc1 * alpha1[:, None] + tl.dot(p1.to(tl.bfloat16).to(DOT), v1)
        m1 = mn1

    if split == SPLITS - 1:
        # The final split alone owns new KV writes and the causal new-token tail.
        # This branch remains necessary even when its historical interval is empty.
        rl = tl.arange(0, BLOCK_L)
        valid_l = rl < L
        ntok = (b * L + tl.where(valid_l, rl, 0)).to(tl.int64)
        noff = ntok[:, None] * stride_tok + kvh0 * D + rc[None, :]
        kn01 = tl.load(qkv_ptr + noff + HQ * D,
                       mask=valid_l[:, None], other=0.0).to(tl.float32)
        kn0, kn1 = tl.split(tl.permute(tl.reshape(kn01, (BLOCK_L, 2, D)), (0, 2, 1)))
        kn0 = kn0 * tl.rsqrt(tl.sum(kn0 * kn0, 1) / D + eps)[:, None]
        kn1 = kn1 * tl.rsqrt(tl.sum(kn1 * kn1, 1) / D + eps)[:, None]
        vn01 = tl.load(qkv_ptr + noff + (HQ + HKV) * D,
                       mask=valid_l[:, None], other=0.0).to(tl.float32)
        vn0, vn1 = tl.split(tl.permute(tl.reshape(vn01, (BLOCK_L, 2, D)), (0, 2, 1)))
        vs = tl.load(vs_ptr)
        kq0 = (kn0 / ks).to(kc_ptr.dtype.element_ty)
        kq1 = (kn1 / ks).to(kc_ptr.dtype.element_ty)
        vq0 = (vn0 / vs).to(vc_ptr.dtype.element_ty)
        vq1 = (vn1 / vs).to(vc_ptr.dtype.element_ty)
        loc = tl.load(loc_ptr + ntok, mask=valid_l, other=0).to(tl.int64)
        dst = loc[:, None] * (HKV * D) + kvh0 * D + rc[None, :]
        kq01 = tl.reshape(tl.permute(tl.join(kq0, kq1), (0, 2, 1)), (BLOCK_L, 2 * D))
        vq01 = tl.reshape(tl.permute(tl.join(vq0, vq1), (0, 2, 1)), (BLOCK_L, 2 * D))
        tl.store(kc_ptr + dst, kq01, mask=valid_l[:, None])
        tl.store(vc_ptr + dst, vq01, mask=valid_l[:, None])
        causal = valid_l[None, :] & (rl[None, :] <= j[:, None])
        score0 = tl.dot(q0, tl.trans(kq0.to(QK_DOT)), out_dtype=tl.float32,
                        max_num_imprecise_acc=0) * qk_scale
        score0 = tl.where(causal, score0, float("-inf"))
        mn0 = tl.maximum(m0, tl.max(score0, 1))
        safe0 = tl.where(mn0 == float("-inf"), 0.0, mn0)
        alpha0 = tl.exp2(m0 - safe0)
        p0 = tl.exp2(score0 - safe0[:, None])
        mass0 = mass0 * alpha0 + tl.sum(p0, 1)
        acc0 = acc0 * alpha0[:, None] + tl.dot(p0.to(tl.bfloat16).to(DOT), vq0.to(tl.float32).to(DOT))
        m0 = mn0

        score1 = tl.dot(q1, tl.trans(kq1.to(QK_DOT)), out_dtype=tl.float32,
                        max_num_imprecise_acc=0) * qk_scale
        score1 = tl.where(causal, score1, float("-inf"))
        mn1 = tl.maximum(m1, tl.max(score1, 1))
        safe1 = tl.where(mn1 == float("-inf"), 0.0, mn1)
        alpha1 = tl.exp2(m1 - safe1)
        p1 = tl.exp2(score1 - safe1[:, None])
        mass1 = mass1 * alpha1 + tl.sum(p1, 1)
        acc1 = acc1 * alpha1[:, None] + tl.dot(p1.to(tl.bfloat16).to(DOT), vq1.to(tl.float32).to(DOT))
        m1 = mn1

    # Preserve [B, HKV, SPLITS, BLOCK_M, D] partial ownership for both heads.
    base0 = ((b * HKV + kvh0) * SPLITS + split) * BLOCK_M + rm
    base1 = ((b * HKV + kvh0 + 1) * SPLITS + split) * BLOCK_M + rm
    tl.store(o_ptr + base0[:, None] * D + rd[None, :], acc0)
    tl.store(o_ptr + base1[:, None] * D + rd[None, :], acc1)
    tl.store(m_ptr + base0, m0)
    tl.store(m_ptr + base1, m1)
    tl.store(l_ptr + base0, mass0)
    tl.store(l_ptr + base1, mass1)


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
