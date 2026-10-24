# okc: problem=muse_bf16_proj
"""Experiment 002: SGLang-style hi+lo router above eight tokens.

Derived from the official baseline's permitted FP32-router decomposition.
CUDA uses BF16 operands and FP32 output/accumulation via the supplied torch.mm API.
The CPU branch is a numerical model using FP32 products of the same BF16-rounded
operands; it does not validate the CUDA tensor-core path or establish performance.
Every other projection and the small-M FP32 router retain the proven starter math.
"""
import torch

ROUTER_FP32_MAX_TOKENS = 8


def prepare(weights):
    state = dict(weights)
    hi = weights["router"].to(torch.bfloat16)
    lo = (weights["router"] - hi.float()).to(torch.bfloat16)
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    return state


def _mm_fp32(x, w):
    if x.device.type == "cuda":
        return torch.mm(x, w, out_dtype=torch.float32)
    # Explicit CPU numerical emulation, not the CUDA implementation.
    return torch.mm(x.float(), w.float())


def kernel_fn(data, state):
    x, cfg = data["x"], data["config"]
    layer, n = cfg["layer"], cfg["n"]
    if layer != "router":
        return x @ state[layer][:n].T
    if x.shape[0] <= ROUTER_FP32_MAX_TOKENS:
        return x.float() @ state["router"][:n].T
    acc = _mm_fp32(x, state["router_hi_t"][:, :n])
    acc += _mm_fp32(x, state["router_lo_t"][:, :n])
    return acc
