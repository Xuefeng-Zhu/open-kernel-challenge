"""Offline full-attention host adapter, not an OKC entrypoint.

Intended integration is only full B1/L8; this file changes no dispatch. It reuses
the existing _fa3_prepare_fused body with ROPE=False and preserves the official
SGLang FA3 and output-gate arguments. No RoPE table is allocated or consulted.
Official-baseline adaptation: SGLang a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f,
Apache-2.0; parent source contains the attribution and applicable license text.
"""


def _fa3_full_verify_fused(data):
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    tokens = batch * length
    qkv = data["qkv"].view(tokens, -1)
    q = torch.empty((tokens, hq, d), dtype=torch.float8_e4m3fn, device=qkv.device)
    kc, vc = data["k_cache"], data["v_cache"]
    scaled = cfg["kv_scale"] != 1.0
    # COS_SIN is an unused pointer when the constexpr ROPE branch is false.
    # Reuse qkv as that placeholder; no table or extra allocation is needed.
    _fa3_prepare_fused[(triton.cdiv(tokens * (hq + 2 * hkv), 4),)](
        qkv, data["positions"], qkv, data["cache_loc"], kc, vc,
        data["k_scale"], data["v_scale"], q, qkv.stride(0), kc.numel() // (hkv * d), RMS_EPS,
        T=tokens, HQ=hq, HKV=hkv, D=d, BD=triton.next_power_of_2(d), BM=4,
        ROPE=False, SCALED=scaled, num_warps=4)
    descale = {}
    if scaled:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))
    out = flash_attn_with_kvcache(
        q=q, k_cache=kc, v_cache=vc,
        page_table=data["page_table"], cache_seqlens=data["cache_seqlens"],
        cu_seqlens_q=data["cu_seqlens_q"], max_seqlen_q=length,
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"], causal=True,
        window_size=(-1, -1), scheduler_metadata=data["scheduler_metadata"], **descale)
    if isinstance(out, tuple):
        out = out[0]
    gate = qkv[:, (hq + 2 * hkv) * d:]
    out = rms_norm_gate(out.reshape(tokens, hq * d).to(torch.bfloat16), gate, None,
                        head_dim=d, eps=RMS_EPS, has_norm=True)
    return out.view(batch, length, hq * d)
