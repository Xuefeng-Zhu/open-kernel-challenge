# okc: problem=muse_bf16_proj
"""Experiment 004: one wide-N router GEMM, then an FP32 hi+lo sum.

Only the m>8 router computation changes relative to experiment 002. Weight-only
preparation concatenates the two BF16 components across output columns [K,2N].
The original BF16 activation is multiplied once; both output halves remain FP32
before their sum. No activation work is hoisted out of the timed call.
CPU is an explicit numerical model; the CUDA path/performance are unverified.
"""
import torch

ROUTER_FP32_MAX_TOKENS = 8


def prepare(weights):
    state = dict(weights)
    hi = weights["router"].to(torch.bfloat16)
    lo = (weights["router"] - hi.float()).to(torch.bfloat16)
    state["router_pair_t"] = torch.cat((hi.t(), lo.t()), dim=1).contiguous()
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
    width = state["router"].shape[0]
    pair = _mm_fp32(x, state["router_pair_t"])
    return pair[:, :n] + pair[:, width:width + n]
