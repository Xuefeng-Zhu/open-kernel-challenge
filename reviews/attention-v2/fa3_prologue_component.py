"""Offline embedding component, not an OKC entrypoint.

RMSNorm/RoPE indexing adapted from official SGLang FusedQKNorm at commit
a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f, Apache-2.0. The standalone parent
retains full license/attribution. Added direct FP8 Q and scaled live-row KV stores.
Cache clamp follows the official reference's finite E4M3 quantization bound.
"""


@triton.jit
def _fa3_prepare_fused(QKV, POS, COS_SIN, LOC, KC, VC, KS, VS, Q,
                       stride_qkv, capacity, eps,
                       T: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr,
                       D: tl.constexpr, BD: tl.constexpr, BM: tl.constexpr,
                       ROPE: tl.constexpr, SCALED: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    heads: tl.constexpr = HQ + 2 * HKV
    token = rows // heads
    head = rows % heads
    row_ok = rows < T * heads
    cols = tl.arange(0, BD)
    mask = row_ok[:, None] & (cols[None, :] < D)
    is_q = head < HQ
    is_k = (head >= HQ) & (head < HQ + HKV)
    is_v = head >= HQ + HKV
    src = token.to(tl.int64)[:, None] * stride_qkv + head[:, None] * D
    raw = tl.load(QKV + src + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(raw * raw, axis=1) / D + eps)
    # Exact official dtype boundaries: RMSNorm BF16, then RoPE BF16, then FP8.
    x = (raw * inv[:, None]).to(tl.bfloat16).to(tl.float32)
    if ROPE:
        partner = tl.load(QKV + src + (cols ^ 1)[None, :], mask=mask, other=0.0).to(tl.float32)
        partner = (partner * inv[:, None]).to(tl.bfloat16).to(tl.float32)
        pos = tl.load(POS + token, mask=row_ok, other=0).to(tl.int64)
        freq = cols // 2
        cosine = tl.load(COS_SIN + pos[:, None] * D + freq[None, :], mask=mask, other=1.0)
        sine = tl.load(COS_SIN + pos[:, None] * D + D // 2 + freq[None, :], mask=mask, other=0.0)
        x = tl.where((cols % 2)[None, :] == 0,
                     x * cosine - partner * sine, x * cosine + partner * sine)
        x = x.to(tl.bfloat16).to(tl.float32)
    x = tl.where(is_v[:, None], raw, x)
    q_head = tl.where(is_q, head, 0)
    q_dst = (token.to(tl.int64)[:, None] * HQ + q_head[:, None]) * D + cols[None, :]
    tl.store(Q + q_dst, x.to(Q.dtype.element_ty), mask=mask & is_q[:, None])
    k = x
    v = raw
    if SCALED:
        ks = tl.load(KS).to(tl.float32)
        vs = tl.load(VS).to(tl.float32)
        # Baseline Tensor(BF16)/scalar_tensor(FP32) produces BF16 before FP8.
        k = (k / ks).to(tl.bfloat16).to(tl.float32)
        v = (v / vs).to(tl.bfloat16).to(tl.float32)
    k = tl.minimum(tl.maximum(k, -448.0), 448.0)
    v = tl.minimum(tl.maximum(v, -448.0), 448.0)
    loc = tl.load(LOC + token, mask=row_ok, other=0).to(tl.int64)
    kv_head = tl.where(is_k, head - HQ, tl.maximum(head - HQ - HKV, 0))
    dst = (loc[:, None] * HKV + kv_head[:, None]) * D + cols[None, :]
    valid = mask & (loc[:, None] >= 0) & (loc[:, None] < capacity)
    # Slot zero is LIVE in OKC; the vendor's optional fused KV store skips zero.
    tl.store(KC + dst, k.to(KC.dtype.element_ty), mask=valid & is_k[:, None])
    tl.store(VC + dst, v.to(VC.dtype.element_ty), mask=valid & is_v[:, None])


def _fa3_sliding_verify_fused(data):
    cfg = data['config']
    batch, length = cfg['batch'], cfg['q_len']
    hq, hkv, d = cfg['num_q_heads'], cfg['num_kv_heads'], cfg['head_dim']
    tokens = batch * length
    qkv = data['qkv'].view(tokens, -1)
    q = torch.empty((tokens, hq, d), dtype=torch.float8_e4m3fn, device=qkv.device)
    cosine_sine = _fa3_cos_sin(cfg['rope_theta'], d, qkv.device)
    kc, vc = data['k_cache'], data['v_cache']
    scaled = cfg['kv_scale'] != 1.0
    _fa3_prepare_fused[(triton.cdiv(tokens * (hq + 2 * hkv), 4),)](
        qkv, data['positions'], cosine_sine, data['cache_loc'], kc, vc,
        data['k_scale'], data['v_scale'], q, qkv.stride(0), kc.numel() // (hkv * d), RMS_EPS,
        T=tokens, HQ=hq, HKV=hkv, D=d, BD=triton.next_power_of_2(d), BM=4,
        ROPE=True, SCALED=scaled, num_warps=4)
    descale = {}
    if scaled:
        descale = dict(k_descale=data['k_scale'].reshape(1, 1).expand(batch, hkv),
                       v_descale=data['v_scale'].reshape(1, 1).expand(batch, hkv))
    out = flash_attn_with_kvcache(
        q=q, k_cache=kc, v_cache=vc, page_table=data['page_table'],
        cache_seqlens=data['cache_seqlens'], cu_seqlens_q=data['cu_seqlens_q'],
        max_seqlen_q=length, softmax_scale=cfg['sm_scale'] * cfg['qk_scale_factor'],
        causal=True, window_size=(cfg['window'] - 1, 0),
        scheduler_metadata=data['scheduler_metadata'], **descale)
    if isinstance(out, tuple):
        out = out[0]
    gate = qkv[:, (hq + 2 * hkv) * d:]
    out = rms_norm_gate(out.reshape(tokens, hq * d).to(torch.bfloat16), gate, None,
                        head_dim=d, eps=RMS_EPS, has_norm=True)
    return out.view(batch, length, hq * d)
