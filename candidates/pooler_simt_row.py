# okc: problem=muse_bf16_proj
"""Isolate a single-row pooler SIMT GEMV, preserving original BF16 operands.

Only pooler M==1 changes versus validated R01. One CTA computes one output row
using FP32 multiplication/reduction, avoiding a padded 16-row tensor-core tile and
separate partial-output reduction. All other paths retain R01. CPU is an explicit
numerical model; native GPU behavior requires its own official check.
"""
import torch

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


def prepare(weights):
    state = dict(weights)
    state["pooler"] = weights["pooler"].contiguous()
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
    if layer == "pooler" and m == 1:
        if x.device.type == "cuda":
            out = torch.empty((1, n), dtype=torch.bfloat16, device=x.device)
            _gemv[(n,)](x, state["pooler"], out, k, triton.next_power_of_2(k), num_warps=4)
            return out
        return (x.float() @ state["pooler"][:n].float().T).bfloat16()
    if layer != "router":
        return x @ state[layer][:n].T
    if m <= 8:
        return x.float() @ state["router"][:n].T
    acc = _mm_fp32(x, state["router_hi_t"][:, :n])
    acc += _mm_fp32(x, state["router_lo_t"][:, :n])
    return acc
