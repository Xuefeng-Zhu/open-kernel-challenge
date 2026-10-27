# okc: problem=muse_bf16_proj
"""Experimental native composition plus separately measured BF16 pooler layout.

Only pooler m>1 adds contiguous[K,N] BF16 library GEMM. Single-token QKV/unpooler,
pooler and router paths retain the isolated native composition. Original weights
and all independent controls are preserved. Layout gains require same-app validation.
"""
import torch

SPLITS = 32
BLOCK_N = 32
BLOCK_K = 64

if torch.cuda.is_available():
    import triton
    import triton.language as tl

    @triton.jit
    def _gemv(X, W, OUT, K: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        ks = tl.arange(0, BLOCK)
        # Within-call reuse of the current 16-KiB x through hardware L1 is legal;
        # no input-derived persistent state or cross-call result cache exists.
        x = tl.load(X + ks, ks < K, other=0.0, cache_modifier=".ca").to(tl.float32)
        w = tl.load(W + row * K + ks, ks < K, other=0.0,
                    cache_modifier=".cg").to(tl.float32)
        value = tl.sum(x * w, axis=0)
        tl.store(OUT + row, value)


    @triton.jit
    def _partial(X, HI, LO, P, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                 S: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        ns = tl.program_id(0) * BN + tl.arange(0, BN)
        split = tl.program_id(1)
        ms = tl.arange(0, 16)
        ks = tl.arange(0, BK)
        part_k = tl.cdiv(K, S * BK) * BK
        acc_hi = tl.full((16, BN), 0.0, tl.float32)
        acc_lo = tl.full((16, BN), 0.0, tl.float32)
        for step in range(0, part_k, BK):
            k = split * part_k + step + ks
            x = tl.load(X + ms[:, None] * K + k[None, :],
                        (ms[:, None] < M) & (k[None, :] < K), other=0.0)
            hi = tl.load(HI + ns[:, None] * K + k[None, :],
                         (ns[:, None] < N) & (k[None, :] < K), other=0.0)
            lo = tl.load(LO + ns[:, None] * K + k[None, :],
                         (ns[:, None] < N) & (k[None, :] < K), other=0.0)
            acc_hi = tl.dot(x, tl.trans(hi), acc_hi)
            acc_lo = tl.dot(x, tl.trans(lo), acc_lo)
        offsets = (split * M + ms[:, None]) * N + ns[None, :]
        tl.store(P + offsets, acc_hi + acc_lo,
                 (ms[:, None] < M) & (ns[None, :] < N))

    @triton.jit
    def _reduce(P, OUT, COUNT: tl.constexpr, S: tl.constexpr, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        splits = tl.arange(0, S)
        values = tl.load(P + splits[:, None] * COUNT + offsets[None, :],
                         offsets[None, :] < COUNT, other=0.0)
        tl.store(OUT + offsets, tl.sum(values, axis=0), offsets < COUNT)


def prepare(weights):
    state = dict(weights)
    if "pooler" in weights:
        state["pooler"] = weights["pooler"].contiguous()
        state["pooler_t"] = weights["pooler"].t().contiguous()
    for layer in ("qkv_gate", "unpooler"):
        if layer in weights:
            state[layer] = weights[layer].contiguous()
    hi = weights["router"].to(torch.bfloat16).contiguous()
    lo = (weights["router"] - hi.float()).to(torch.bfloat16).contiguous()
    state["router_hi"] = hi
    state["router_lo"] = lo
    # Same packed layout and one-GEMM computation as measured R02.
    state["router_pair_t"] = torch.cat((hi.t(), lo.t()), dim=1).contiguous()
    return state


def _mm_fp32(x, w):
    if x.device.type == "cuda":
        return torch.mm(x, w, out_dtype=torch.float32)
    return torch.mm(x.float(), w.float())


def kernel_fn(data, state):
    x, cfg = data["x"], data["config"]
    layer, n = cfg["layer"], cfg["n"]
    m, k = x.shape
    if layer == "pooler" and m > 1:
        return x @ state["pooler_t"][:, :n]
    if layer in {"pooler", "qkv_gate", "unpooler"} and m == 1:
        if x.device.type == "cuda":
            out = torch.empty((1, n), dtype=torch.bfloat16, device=x.device)
            _gemv[(n,)](x, state[layer], out, k, triton.next_power_of_2(k), num_warps=4)
            return out
        return (x.float() @ state[layer][:n].float().T).bfloat16()
    if layer != "router":
        return x @ state[layer][:n].T
    if m <= 8 and x.device.type == "cuda":
        partial = torch.empty((SPLITS, m, n), dtype=torch.float32, device=x.device)
        out = torch.empty((m, n), dtype=torch.float32, device=x.device)
        _partial[(triton.cdiv(n, BLOCK_N), SPLITS)](
            x, state["router_hi"], state["router_lo"], partial, m, n, k,
            SPLITS, BLOCK_N, BLOCK_K, num_warps=4, num_stages=2)
        _reduce[(triton.cdiv(m * n, 128),)](partial, out, m * n, SPLITS, 128, num_warps=4)
        return out
    if m <= 8:
        # CPU numerical model of the split boundaries/FP32 partial reduction only.
        part_k = ((k + SPLITS * BLOCK_K - 1) // (SPLITS * BLOCK_K)) * BLOCK_K
        pieces = []
        for split in range(SPLITS):
            start, end = split * part_k, min((split + 1) * part_k, k)
            if start >= k:
                pieces.append(torch.zeros((m, n), dtype=torch.float32, device=x.device))
            else:
                hi = x[:, start:end].float() @ state["router_hi"][:n, start:end].float().T
                lo = x[:, start:end].float() @ state["router_lo"][:n, start:end].float().T
                pieces.append(hi + lo)
        return torch.stack(pieces).sum(0)
    width = state["router"].shape[0]
    pair = _mm_fp32(x, state["router_pair_t"])
    return pair[:, :n] + pair[:, width:width + n]
