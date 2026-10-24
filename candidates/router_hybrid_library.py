# okc: problem=muse_bf16_proj
"""Combine only the GPU-verified R02/R03 router shape regions.

M=1 retains original FP32; 2<=M<=8 uses R03's two BF16 hi+lo products;
M>8 uses R02's one wide-N product. All accumulation/outputs are FP32.
Other projections and original weights remain unchanged. CPU is a numerical model.
"""
import torch


def prepare(weights):
    state = dict(weights)
    hi = weights["router"].to(torch.bfloat16)
    lo = (weights["router"] - hi.float()).to(torch.bfloat16)
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    state["router_pair_t"] = torch.cat((hi.t(), lo.t()), dim=1).contiguous()
    return state


def _mm_fp32(x, w):
    if x.device.type == "cuda":
        return torch.mm(x, w, out_dtype=torch.float32)
    return torch.mm(x.float(), w.float())


def kernel_fn(data, state):
    x, cfg = data["x"], data["config"]
    layer, n = cfg["layer"], cfg["n"]
    if layer != "router":
        return x @ state[layer][:n].T
    m = x.shape[0]
    if m <= 1:
        return x.float() @ state["router"][:n].T
    if m <= 8:
        acc = _mm_fp32(x, state["router_hi_t"][:, :n])
        acc += _mm_fp32(x, state["router_lo_t"][:, :n])
        return acc
    width = state["router"].shape[0]
    pair = _mm_fp32(x, state["router_pair_t"])
    return pair[:, :n] + pair[:, width:width + n]
