# okc: problem=muse_attention_decode
# A2-13: fallback QK operands use E4M3; PV remains BF16, sums FP32.
# Official-baseline adaptation: SGLang a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f.
# SPDX-License-Identifier: Apache-2.0; applicable license text is included below.
"""A2-13: FP8 E4M3 QK encoding on the existing Triton fallback only.

Parent A2-11 is preserved. Normalized/rotated Q is rounded BF16 then E4M3;
qk_scale_factor moves into the logit scale exactly once. K stays E4M3 for QK.
PV stays BF16, softmax/partials stay FP32. Geometry, ownership, merge and FA3
routes are unchanged. Native compilation and accuracy require official checks.
"""

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
LOG2E = tl.constexpr(1.4426950408889634)
NUM_SMS = 132  # H200 SXM
_INTERP = os.environ.get("TRITON_INTERPRET") == "1"
# PV operands stay BF16 on the GPU. Optional CPU interpreter operands use FP32
# after BF16 rounding; QK has its own E4M3 encoding and selector below.
_DOT = tl.float32 if _INTERP else tl.bfloat16
# Encode E4M3 first even in the optional CPU interpreter; no interpreter run claimed.
_QK_DOT = tl.float32 if _INTERP else tl.float8e4nv
def _tickets(device, n):
    # Timed and captured per invocation. Separate graph captures use private pools;
    # callers must not overlap graphs whose allocation pools they explicitly share.
    return torch.zeros(max(n, 4096), dtype=torch.int32, device=device)


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
def _fused(qkv_ptr, seq_ptr, pt_ptr, loc_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr,
           o_ptr, m_ptr, l_ptr, ticket_ptr, out_ptr,
           theta, eps, qk_factor, sm_scale, window, page, stride_tok, stride_pt,
           L: tl.constexpr, G: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr,
           D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_L: tl.constexpr,
           SPLITS: tl.constexpr, ROPE: tl.constexpr, WINDOWED: tl.constexpr,
           DOT: tl.constexpr, QK_DOT: tl.constexpr, CHUNK_SIZE: tl.constexpr, SEPARATE_MERGE: tl.constexpr):
    b = tl.program_id(0)
    kvh = tl.program_id(1)
    split = tl.program_id(2)

    seq = tl.load(seq_ptr + b).to(tl.int64)
    # Query rows r = j * G + g  (block token j, query head kvh*G + g).
    offs_m = tl.arange(0, BLOCK_M)
    row_ok = offs_m < L * G
    j = tl.where(row_ok, offs_m // G, 0)
    g = offs_m % G
    qpos = seq + j
    offs_d = tl.arange(0, D)
    tok = (b * L + j).to(tl.int64)
    q_rows = qkv_ptr + tok[:, None] * stride_tok + ((kvh * G + g) * D)[:, None]
    q = _norm_rope(q_rows, offs_d, qpos[:, None], theta, eps, ROPE, D, row_ok[:, None])
    # FP8 query encoding follows the official scale placement: round unscaled Q.
    # Parent BF16(factor*Q) becomes E4M3(BF16(Q)); this changes rounding.
    # Factor is applied exactly once in qk_scale, never inside this FP8 cast.
    q = q.to(tl.bfloat16).to(tl.float8e4nv).to(QK_DOT)
    ks = tl.load(ks_ptr)
    vs = tl.load(vs_ptr)
    qk_scale = (sm_scale * qk_factor) * ks * LOG2E

    # Cached keys [lo, seq); new block keys [seq, seq + L).
    if WINDOWED:
        lo = tl.maximum(seq - window + 1, 0)   # earliest key the first block row sees
    else:
        lo = seq * 0
    lo = tl.minimum(lo, seq)
    span = seq - lo
    if CHUNK_SIZE:
        s_lo = split * CHUNK_SIZE
        s_hi = tl.minimum(s_lo + CHUNK_SIZE, seq)
    else:
        per = ((span + SPLITS - 1) // SPLITS + BLOCK_N - 1) // BLOCK_N * BLOCK_N
        s_lo = lo + split * per
        s_hi = tl.minimum(s_lo + per, seq)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    for start in range(s_lo, s_hi, BLOCK_N):
        t = start + offs_n
        t_ok = t < s_hi
        pg = tl.load(pt_ptr + b * stride_pt + t // page, mask=t_ok, other=0).to(tl.int64)
        pg = tl.maximum(pg, 0)
        row = pg * page + t % page
        kv_off = row[:, None] * (HKV * D) + kvh * D + offs_d[None, :]
        k = tl.load(kc_ptr + kv_off, mask=t_ok[:, None], other=0.0).to(QK_DOT)
        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32, max_num_imprecise_acc=0) * qk_scale  # log2 domain
        vis = t_ok[None, :]
        if WINDOWED:
            vis = vis & (t[None, :] > qpos[:, None] - window)
        s = tl.where(vis, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(vc_ptr + kv_off, mask=t_ok[:, None], other=0.0).to(DOT)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16).to(DOT), v)
        m_i = m_new

    if split == SPLITS - 1:
        # The block's own L tokens: compute, quantize, store, attend.
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
        s = tl.dot(q, tl.trans(kq.to(QK_DOT)), out_dtype=tl.float32, max_num_imprecise_acc=0) * qk_scale
        vis = l_ok[None, :] & (npos[None, :] <= qpos[:, None])
        s = tl.where(vis, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16).to(DOT), vq.to(tl.float32).to(DOT))
        m_i = m_new

    head = kvh * G + g
    out_off = tok[:, None] * (HQ * D) + head[:, None] * D + offs_d[None, :]
    gate_off = tok[:, None] * stride_tok + ((HQ + 2 * HKV) * D) + head[:, None] * D + offs_d[None, :]
    if SPLITS == 1:
        a = acc * vs / l_i[:, None]
        a = a * tl.rsqrt(tl.sum(a * a, 1) / D + eps)[:, None]
        gate = tl.load(qkv_ptr + gate_off, mask=row_ok[:, None], other=0.0).to(tl.float32)
        tl.store(out_ptr + out_off, (a * tl.sigmoid(gate)).to(tl.bfloat16), mask=row_ok[:, None])
    else:
        # Partials [B, HKV, SPLITS, BLOCK_M, (D)].
        pbase = ((b * HKV + kvh) * SPLITS + split) * BLOCK_M + offs_m
        tl.store(o_ptr + pbase[:, None] * D + offs_d[None, :], acc)
        tl.store(m_ptr + pbase, m_i)
        tl.store(l_ptr + pbase, l_i)
        if not SEPARATE_MERGE:
            tl.debug_barrier()  # every warp's partial stores precede the ticket (release below)
            tk = ticket_ptr + b * HKV + kvh
            prev = tl.atomic_add(tk, 1, sem="acq_rel", scope="gpu")
            if prev == SPLITS - 1:
                m_t = tl.full([BLOCK_M], float("-inf"), tl.float32)
                l_t = tl.zeros([BLOCK_M], tl.float32)
                a = tl.zeros([BLOCK_M, D], tl.float32)
                for s_i in range(0, SPLITS):
                    qb = ((b * HKV + kvh) * SPLITS + s_i) * BLOCK_M + offs_m
                    m_s = tl.load(m_ptr + qb, cache_modifier=".cg")
                    l_s = tl.load(l_ptr + qb, cache_modifier=".cg")
                    o_s = tl.load(o_ptr + qb[:, None] * D + offs_d[None, :], cache_modifier=".cg")
                    m_n = tl.maximum(m_t, m_s)
                    m_z = tl.where(m_n == float("-inf"), 0.0, m_n)
                    w_t = tl.exp2(m_t - m_z)
                    w_s = tl.exp2(m_s - m_z)
                    l_t = l_t * w_t + l_s * w_s
                    a = a * w_t[:, None] + o_s * w_s[:, None]
                    m_t = m_n
                a = a * vs / l_t[:, None]
                a = a * tl.rsqrt(tl.sum(a * a, 1) / D + eps)[:, None]
                gate = tl.load(qkv_ptr + gate_off, mask=row_ok[:, None], other=0.0).to(tl.float32)
                tl.store(out_ptr + out_off, (a * tl.sigmoid(gate)).to(tl.bfloat16), mask=row_ok[:, None])
                tl.atomic_xchg(tk, 0, sem="relaxed", scope="gpu")


@triton.jit
def _merge(qkv_ptr, vs_ptr, o_ptr, m_ptr, l_ptr, out_ptr, eps, stride_tok,
           L: tl.constexpr, G: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr,
           D: tl.constexpr, BLOCK_M: tl.constexpr, SPLITS: tl.constexpr,
           BS: tl.constexpr):
    tl.static_assert(L == 1)
    # L1 only: one CTA per valid query head, using the unchanged split-major
    # FP32 partials. Parallel reduction changes FP32 summation/rescaling order.
    b = tl.program_id(0)
    head = tl.program_id(1)
    kvh = head // G
    g = head % G
    offs_s = tl.arange(0, BS)
    offs_d = tl.arange(0, D)
    split_ok = offs_s < SPLITS
    # SPLITS is the actual allocation stride; BS only pads reduction lanes.
    qb = ((b * HKV + kvh) * SPLITS + offs_s) * BLOCK_M + g
    m_s = tl.load(m_ptr + qb, mask=split_ok, other=float("-inf"), cache_modifier=".cg")
    l_s = tl.load(l_ptr + qb, mask=split_ok, other=0.0, cache_modifier=".cg")
    o_s = tl.load(o_ptr + qb[:, None] * D + offs_d[None, :],
                  mask=split_ok[:, None], other=0.0, cache_modifier=".cg")
    m_t = tl.max(m_s, 0)
    m_z = tl.where(m_t == float("-inf"), 0.0, m_t)
    w_s = tl.exp2(m_s - m_z)
    l_t = tl.sum(l_s * w_s, 0)
    a = tl.sum(o_s * w_s[:, None], 0)
    vs = tl.load(vs_ptr)
    a = a * vs / l_t
    a = a * tl.rsqrt(tl.sum(a * a, 0) / D + eps)
    tok = (b * L).to(tl.int64)
    out_off = tok * (HQ * D) + head * D + offs_d
    gate_off = tok * stride_tok + ((HQ + 2 * HKV) * D) + head * D + offs_d
    gate = tl.load(qkv_ptr + gate_off).to(tl.float32)
    tl.store(out_ptr + out_off, (a * tl.sigmoid(gate)).to(tl.bfloat16))


def _num_splits(batch, hkv, span_max, block_n):
    ctas = batch * hkv
    want = max(1, (2 * NUM_SMS + ctas - 1) // ctas)
    most = max(1, span_max // (2 * block_n))  # >= 2 tiles per split
    return min(want, most)


def _e2_kernel_fn(data: Inputs) -> torch.Tensor:
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d, page = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"], cfg["page_size"]
    g = hq // hkv
    tokens = batch * length
    qkv = data["qkv"].reshape(tokens, -1)
    window = cfg["window"]
    rope = bool(cfg["rope_theta"])
    theta = float(cfg["rope_theta"]) if rope else 1.0
    page_table = data["page_table"]
    dev = qkv.device

    block_n = 64
    block_m = max(16, triton.next_power_of_2(length * g))
    span_max = window if window else page_table.shape[1] * page  # shape-only bound
    chunk_size = 8192 if window == 0 and batch in (16, 32) else 0
    splits = ((span_max + chunk_size - 1) // chunk_size if chunk_size
              else _num_splits(batch, hkv, span_max, block_n))
    out = torch.empty((tokens, hq * d), dtype=torch.bfloat16, device=dev)
    if splits > 1:
        n = batch * hkv * splits * block_m
        part_o = torch.empty((n, d), dtype=torch.float32, device=dev)
        part_m = torch.empty((n,), dtype=torch.float32, device=dev)
        part_l = torch.empty((n,), dtype=torch.float32, device=dev)
    else:
        part_o = part_m = part_l = out
    separate_merge = length == 1
    tickets = out if separate_merge else _tickets(dev, batch * hkv)
    _fused[(batch, hkv, splits)](
        qkv, data["seq_lens"], page_table, data["cache_loc"], data["k_cache"], data["v_cache"],
        data["k_scale"], data["v_scale"], part_o, part_m, part_l, tickets, out,
        theta, RMS_EPS, float(cfg["qk_scale_factor"]), float(cfg["sm_scale"]),
        window, page, qkv.stride(0), page_table.stride(0),
        L=length, G=g, HQ=hq, HKV=hkv, D=d, BLOCK_M=block_m, BLOCK_N=block_n,
        BLOCK_L=16, SPLITS=splits, ROPE=rope, WINDOWED=bool(window), DOT=_DOT, QK_DOT=_QK_DOT, CHUNK_SIZE=chunk_size, SEPARATE_MERGE=separate_merge,
        num_warps=4, num_stages=3)
    if separate_merge and splits > 1:
        _merge[(batch, hq)](qkv, data["v_scale"], part_o, part_m, part_l, out,
            RMS_EPS, qkv.stride(0), L=length, G=g, HQ=hq, HKV=hkv, D=d,
            BLOCK_M=block_m, SPLITS=splits, BS=triton.next_power_of_2(splits), num_warps=4, num_stages=3)
    return out.view(batch, length, hq * d)


# This adapter follows the public official muse_attention_decode/baseline.py:
# https://openkernelchallenge.dev/api/v1/problems/muse_attention_decode
# SGLang is Apache-2.0. No third-party competitor source is copied or imported.
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sglang.kernels.ops.kvcache.kvcache import store_cache
from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm
from sglang.kernels.ops.muse_spark_v12.norm_gate import rms_norm_gate

_FA3_NORMS = {}


def _fa3_norm(hq, hkv, d):
    key = (hq, hkv, d)
    if key not in _FA3_NORMS:
        _FA3_NORMS[key] = FusedQKNorm(head_dim=d, num_q_heads=hq, num_kv_heads=hkv, eps=RMS_EPS)
    return _FA3_NORMS[key]


def _fa3_full_verify(data):
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    tokens = batch * length
    qkv = data["qkv"].view(tokens, -1)
    # Full attention has rope_theta=0 by contract. Other configurations retain E2.
    q, k = _fa3_norm(hq, hkv, d)(qkv)
    v = qkv[:, (hq + hkv) * d:(hq + 2 * hkv) * d]
    k = k.reshape(tokens, hkv * d)
    scaled = cfg["kv_scale"] != 1.0
    if scaled:
        k, v = k / data["k_scale"], v / data["v_scale"]
    fp8 = torch.float8_e4m3fn
    kc, vc = data["k_cache"], data["v_cache"]
    # Slot zero is valid in this problem. Store FP8 bytes, including row zero.
    store_cache(k.to(fp8).view(torch.uint8), v.to(fp8).view(torch.uint8),
                kc.view(-1, hkv * d).view(torch.uint8), vc.view(-1, hkv * d).view(torch.uint8),
                data["cache_loc"], reserved_skip_index=-1)
    descale = {}
    if scaled:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))
    # FA3 with an FP8 cache expects FP8 E4M3 queries. This is the official
    # conversion, after BF16 QK normalization and before folding qk_scale_factor
    # into the softmax scale. No precision change is made to E2's other paths.
    out = flash_attn_with_kvcache(
        q=q.view(tokens, hq, d).to(fp8), k_cache=kc, v_cache=vc,
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


# Config-only table, matching the official baseline. No input values or outputs cached.
_FA3_ROPE = {}
_FA3_ROPE_POSITIONS = 1 << 21


def _fa3_cos_sin(theta, d, device):
    key = (theta, d, device)
    cache = _FA3_ROPE.get(key)
    if cache is None:
        inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float, device=device) / d))
        freqs = torch.einsum("i,j -> ij", torch.arange(_FA3_ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
        cache = _FA3_ROPE[key] = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    return cache


def _fa3_sliding_verify(data):
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    tokens = batch * length
    qkv = data["qkv"].view(tokens, -1)
    # Official sliding path: fused BF16 Q/K RMSNorm and interleaved RoPE.
    q, k = _fa3_norm(hq, hkv, d)(
        qkv, data["positions"], _fa3_cos_sin(cfg["rope_theta"], d, qkv.device))
    v = qkv[:, (hq + hkv) * d:(hq + 2 * hkv) * d]
    k = k.reshape(tokens, hkv * d)
    scaled = cfg["kv_scale"] != 1.0
    if scaled:
        k, v = k / data["k_scale"], v / data["v_scale"]
    fp8 = torch.float8_e4m3fn
    kc, vc = data["k_cache"], data["v_cache"]
    # Slot zero is valid in this problem. Store FP8 bytes, including row zero.
    store_cache(k.to(fp8).view(torch.uint8), v.to(fp8).view(torch.uint8),
                kc.view(-1, hkv * d).view(torch.uint8), vc.view(-1, hkv * d).view(torch.uint8),
                data["cache_loc"], reserved_skip_index=-1)
    descale = {}
    if scaled:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))
    # FA3 with an FP8 cache expects FP8 E4M3 queries. This is the official
    # conversion, after BF16 QK normalization and before folding qk_scale_factor
    # into the softmax scale. No precision change is made to E2's other paths.
    out = flash_attn_with_kvcache(
        q=q.view(tokens, hq, d).to(fp8), k_cache=kc, v_cache=vc,
        page_table=data["page_table"], cache_seqlens=data["cache_seqlens"],
        cu_seqlens_q=data["cu_seqlens_q"], max_seqlen_q=length,
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"], causal=True,
        window_size=(cfg["window"] - 1, 0), scheduler_metadata=data["scheduler_metadata"], **descale)
    if isinstance(out, tuple):
        out = out[0]
    gate = qkv[:, (hq + 2 * hkv) * d:]
    out = rms_norm_gate(out.reshape(tokens, hq * d).to(torch.bfloat16), gate, None,
                        head_dim=d, eps=RMS_EPS, has_norm=True)
    return out.view(batch, length, hq * d)


"""Offline embedding component, not an OKC entrypoint.

RMSNorm/RoPE indexing adapted from official SGLang FusedQKNorm at commit
a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f, Apache-2.0. The standalone parent
retains full license/attribution. Added direct FP8 Q and scaled live-row KV stores.
Use direct FP8 casts, preserving the parent conversion semantics without an added clamp.
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
        scheduler_metadata=data['scheduler_metadata'],
        **({'num_splits': 1} if batch in (16, 32) and length == 8 else {}), **descale)
    if isinstance(out, tuple):
        out = out[0]
    gate = qkv[:, (hq + 2 * hkv) * d:]
    out = rms_norm_gate(out.reshape(tokens, hq * d).to(torch.bfloat16), gate, None,
                        head_dim=d, eps=RMS_EPS, has_norm=True)
    return out.view(batch, length, hq * d)


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


def kernel_fn(data: Inputs) -> torch.Tensor:
    cfg = data["config"]
    if ((cfg["batch"], cfg["q_len"]) in ((1, 1), (1, 8), (16, 1), (16, 8), (32, 8))
            and cfg["window"] == 0 and cfg["rope_theta"] == 0):
        return _fa3_full_verify_fused(data)
    if (cfg["q_len"] == 8
            and cfg["window"] == 2048 and cfg["rope_theta"] == 500000):
        return _fa3_sliding_verify_fused(data)
    if (cfg["batch"] in (1, 16) and cfg["q_len"] == 1
            and cfg["window"] == 0 and cfg["rope_theta"] == 0):
        return _fa3_full_verify(data)
    if (cfg["batch"] in (1, 16, 32) and cfg["q_len"] == 8
            and cfg["window"] == 2048 and cfg["rope_theta"] == 500000):
        return _fa3_sliding_verify(data)
    if (cfg["window"] == 0 and cfg["rope_theta"] == 0 and cfg["q_len"] == 8
            and cfg["batch"] in (1, 16, 32)):
        return _fa3_full_verify(data)
    return _e2_kernel_fn(data)

# Apache-2.0 text for the official-baseline-adapted portions:
#                                  Apache License
#                            Version 2.0, January 2004
#                         http://www.apache.org/licenses/
#
#    TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
#
#    1. Definitions.
#
#       "License" shall mean the terms and conditions for use, reproduction,
#       and distribution as defined by Sections 1 through 9 of this document.
#
#       "Licensor" shall mean the copyright owner or entity authorized by
#       the copyright owner that is granting the License.
#
#       "Legal Entity" shall mean the union of the acting entity and all
#       other entities that control, are controlled by, or are under common
#       control with that entity. For the purposes of this definition,
#       "control" means (i) the power, direct or indirect, to cause the
#       direction or management of such entity, whether by contract or
#       otherwise, or (ii) ownership of fifty percent (50%) or more of the
#       outstanding shares, or (iii) beneficial ownership of such entity.
#
#       "You" (or "Your") shall mean an individual or Legal Entity
#       exercising permissions granted by this License.
#
#       "Source" form shall mean the preferred form for making modifications,
#       including but not limited to software source code, documentation
#       source, and configuration files.
#
#       "Object" form shall mean any form resulting from mechanical
#       transformation or translation of a Source form, including but
#       not limited to compiled object code, generated documentation,
#       and conversions to other media types.
#
#       "Work" shall mean the work of authorship, whether in Source or
#       Object form, made available under the License, as indicated by a
#       copyright notice that is included in or attached to the work
#       (an example is provided in the Appendix below).
#
#       "Derivative Works" shall mean any work, whether in Source or Object
#       form, that is based on (or derived from) the Work and for which the
#       editorial revisions, annotations, elaborations, or other modifications
#       represent, as a whole, an original work of authorship. For the purposes
#       of this License, Derivative Works shall not include works that remain
#       separable from, or merely link (or bind by name) to the interfaces of,
#       the Work and Derivative Works thereof.
#
#       "Contribution" shall mean any work of authorship, including
#       the original version of the Work and any modifications or additions
#       to that Work or Derivative Works thereof, that is intentionally
#       submitted to Licensor for inclusion in the Work by the copyright owner
#       or by an individual or Legal Entity authorized to submit on behalf of
#       the copyright owner. For the purposes of this definition, "submitted"
#       means any form of electronic, verbal, or written communication sent
#       to the Licensor or its representatives, including but not limited to
#       communication on electronic mailing lists, source code control systems,
#       and issue tracking systems that are managed by, or on behalf of, the
#       Licensor for the purpose of discussing and improving the Work, but
#       excluding communication that is conspicuously marked or otherwise
#       designated in writing by the copyright owner as "Not a Contribution."
#
#       "Contributor" shall mean Licensor and any individual or Legal Entity
#       on behalf of whom a Contribution has been received by Licensor and
#       subsequently incorporated within the Work.
#
#    2. Grant of Copyright License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       copyright license to reproduce, prepare Derivative Works of,
#       publicly display, publicly perform, sublicense, and distribute the
#       Work and such Derivative Works in Source or Object form.
#
#    3. Grant of Patent License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       (except as stated in this section) patent license to make, have made,
#       use, offer to sell, sell, import, and otherwise transfer the Work,
#       where such license applies only to those patent claims licensable
#       by such Contributor that are necessarily infringed by their
#       Contribution(s) alone or by combination of their Contribution(s)
#       with the Work to which such Contribution(s) was submitted. If You
#       institute patent litigation against any entity (including a
#       cross-claim or counterclaim in a lawsuit) alleging that the Work
#       or a Contribution incorporated within the Work constitutes direct
#       or contributory patent infringement, then any patent licenses
#       granted to You under this License for that Work shall terminate
#       as of the date such litigation is filed.
#
#    4. Redistribution. You may reproduce and distribute copies of the
#       Work or Derivative Works thereof in any medium, with or without
#       modifications, and in Source or Object form, provided that You
#       meet the following conditions:
#
#       (a) You must give any other recipients of the Work or
#           Derivative Works a copy of this License; and
#
#       (b) You must cause any modified files to carry prominent notices
#           stating that You changed the files; and
#
#       (c) You must retain, in the Source form of any Derivative Works
#           that You distribute, all copyright, patent, trademark, and
#           attribution notices from the Source form of the Work,
#           excluding those notices that do not pertain to any part of
#           the Derivative Works; and
#
#       (d) If the Work includes a "NOTICE" text file as part of its
#           distribution, then any Derivative Works that You distribute must
#           include a readable copy of the attribution notices contained
#           within such NOTICE file, excluding those notices that do not
#           pertain to any part of the Derivative Works, in at least one
#           of the following places: within a NOTICE text file distributed
#           as part of the Derivative Works; within the Source form or
#           documentation, if provided along with the Derivative Works; or,
#           within a display generated by the Derivative Works, if and
#           wherever such third-party notices normally appear. The contents
#           of the NOTICE file are for informational purposes only and
#           do not modify the License. You may add Your own attribution
#           notices within Derivative Works that You distribute, alongside
#           or as an addendum to the NOTICE text from the Work, provided
#           that such additional attribution notices cannot be construed
#           as modifying the License.
#
#       You may add Your own copyright statement to Your modifications and
#       may provide additional or different license terms and conditions
#       for use, reproduction, or distribution of Your modifications, or
#       for any such Derivative Works as a whole, provided Your use,
#       reproduction, and distribution of the Work otherwise complies with
#       the conditions stated in this License.
#
#    5. Submission of Contributions. Unless You explicitly state otherwise,
#       any Contribution intentionally submitted for inclusion in the Work
#       by You to the Licensor shall be under the terms and conditions of
#       this License, without any additional terms or conditions.
#       Notwithstanding the above, nothing herein shall supersede or modify
#       the terms of any separate license agreement you may have executed
#       with Licensor regarding such Contributions.
#
#    6. Trademarks. This License does not grant permission to use the trade
#       names, trademarks, service marks, or product names of the Licensor,
#       except as required for reasonable and customary use in describing the
#       origin of the Work and reproducing the content of the NOTICE file.
#
#    7. Disclaimer of Warranty. Unless required by applicable law or
#       agreed to in writing, Licensor provides the Work (and each
#       Contributor provides its Contributions) on an "AS IS" BASIS,
#       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
#       implied, including, without limitation, any warranties or conditions
#       of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
#       PARTICULAR PURPOSE. You are solely responsible for determining the
#       appropriateness of using or redistributing the Work and assume any
#       risks associated with Your exercise of permissions under this License.
#
#    8. Limitation of Liability. In no event and under no legal theory,
#       whether in tort (including negligence), contract, or otherwise,
#       unless required by applicable law (such as deliberate and grossly
#       negligent acts) or agreed to in writing, shall any Contributor be
#       liable to You for damages, including any direct, indirect, special,
#       incidental, or consequential damages of any character arising as a
#       result of this License or out of the use or inability to use the
#       Work (including but not limited to damages for loss of goodwill,
#       work stoppage, computer failure or malfunction, or any and all
#       other commercial damages or losses), even if such Contributor
#       has been advised of the possibility of such damages.
#
#    9. Accepting Warranty or Additional Liability. While redistributing
#       the Work or Derivative Works thereof, You may choose to offer,
#       and charge a fee for, acceptance of support, warranty, indemnity,
#       or other liability obligations and/or rights consistent with this
#       License. However, in accepting such obligations, You may act only
#       on Your own behalf and on Your sole responsibility, not on behalf
#       of any other Contributor, and only if You agree to indemnify,
#       defend, and hold each Contributor harmless for any liability
#       incurred by, or claims asserted against, such Contributor by reason
#       of your accepting any such warranty or additional liability.
#
#    END OF TERMS AND CONDITIONS
#
#    APPENDIX: How to apply the Apache License to your work.
#
#       To apply the Apache License to your work, attach the following
#       boilerplate notice, with the fields enclosed by brackets "[]"
#       replaced with your own identifying information. (Don't include
#       the brackets!)  The text should be enclosed in the appropriate
#       comment syntax for the file format. We also recommend that a
#       file or class name and description of purpose be included on the
#       same "printed page" as the copyright notice for easier
#       identification within third-party archives.
#
#    Copyright 2023-2024 SGLang Team
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
