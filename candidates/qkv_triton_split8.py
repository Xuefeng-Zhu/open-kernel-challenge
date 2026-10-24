# okc: problem=muse_bf16_proj
"""Isolate a low-batch BF16 qkv_gate split-K kernel on top of validated R01.

Only qkv_gate M<=8 changes. Every original BF16 weight is retained without a new
quantization. CPU explicitly models FP32 partials; it never executes Triton.
"""
import torch

SPLITS = 8
BLOCK_N = 64
BLOCK_K = 64

if torch.cuda.is_available():
    import triton
    import triton.language as tl

    @triton.jit
    def _partial(X, W, P, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                 S: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        ns = tl.program_id(0) * BN + tl.arange(0, BN)
        split = tl.program_id(1)
        ms = tl.arange(0, 16)
        ks = tl.arange(0, BK)
        part_k = tl.cdiv(K, S * BK) * BK
        acc = tl.full((16, BN), 0.0, tl.float32)
        for step in range(0, part_k, BK):
            k = split * part_k + step + ks
            x = tl.load(X + ms[:, None] * K + k[None, :],
                        (ms[:, None] < M) & (k[None, :] < K), other=0.0)
            w = tl.load(W + ns[:, None] * K + k[None, :],
                        (ns[:, None] < N) & (k[None, :] < K), other=0.0)
            acc = tl.dot(x, tl.trans(w), acc)
        offsets = (split * M + ms[:, None]) * N + ns[None, :]
        tl.store(P + offsets, acc, (ms[:, None] < M) & (ns[None, :] < N))

    @triton.jit
    def _reduce(P, OUT, COUNT: tl.constexpr, S: tl.constexpr, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        splits = tl.arange(0, S)
        values = tl.load(P + splits[:, None] * COUNT + offsets[None, :],
                         offsets[None, :] < COUNT, other=0.0)
        tl.store(OUT + offsets, tl.sum(values, axis=0), offsets < COUNT)


def prepare(weights):
    state = dict(weights)
    state["qkv_gate"] = weights["qkv_gate"].contiguous()
    hi = weights["router"].to(torch.bfloat16)
    lo = (weights["router"] - hi.float()).to(torch.bfloat16)
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    return state


def _mm_fp32(x, w):
    if x.device.type == "cuda":
        return torch.mm(x, w, out_dtype=torch.float32)
    return torch.mm(x.float(), w.float())


def kernel_fn(data, state):
    x, cfg = data["x"], data["config"]
    layer, n = cfg["layer"], cfg["n"]
    m, k = x.shape
    if layer == "qkv_gate" and m <= 8:
        if x.device.type == "cuda":
            partial = torch.empty((SPLITS, m, n), dtype=torch.float32, device=x.device)
            out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
            _partial[(triton.cdiv(n, BLOCK_N), SPLITS)](
                x, state["qkv_gate"], partial, m, n, k, SPLITS, BLOCK_N, BLOCK_K,
                num_warps=4, num_stages=2)
            _reduce[(triton.cdiv(m * n, 128),)](partial, out, m * n, SPLITS, 128, num_warps=4)
            return out
        part_k = ((k + SPLITS * BLOCK_K - 1) // (SPLITS * BLOCK_K)) * BLOCK_K
        pieces = []
        for split in range(SPLITS):
            start, end = split * part_k, min((split + 1) * part_k, k)
            if start >= k:
                pieces.append(torch.zeros((m, n), dtype=torch.float32, device=x.device))
            else:
                pieces.append(x[:, start:end].float() @ state["qkv_gate"][:n, start:end].float().T)
        return torch.stack(pieces).sum(0).bfloat16()
    if layer != "router":
        return x @ state[layer][:n].T
    if m <= 8:
        return x.float() @ state["router"][:n].T
    acc = _mm_fp32(x, state["router_hi_t"][:, :n])
    acc += _mm_fp32(x, state["router_lo_t"][:, :n])
    return acc
