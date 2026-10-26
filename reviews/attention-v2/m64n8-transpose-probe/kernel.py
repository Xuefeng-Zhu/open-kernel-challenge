"""One full historical tile, compile mechanism only; not a contest candidate.

Q is post-norm/RoPE BF16[8,64]; K/V are encoded E4M3[64,64]. There is no
new-token tail, cache write, online carry, split/merge, norm/gate or launch.
Both tensor operations keep the eight query heads on native N8, with keys
or output features on M64. Numerical GPU correctness remains untested.
"""
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared, warpgroup_mma, warpgroup_mma_wait,
)


@gluon.jit
def m64n8_history_tile(Q, K, V, KScale, VScale, Out, RowMax, RowSum,
                      QK_FACTOR: gl.constexpr, SM_SCALE: gl.constexpr):
    mma: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 8, 16])
    load: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 4], threads_per_warp=[4, 8],
        warps_per_cta=[4, 1], order=[1, 0])
    k_shared: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [64, 64], gl.bfloat16)
    b_shared: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [64, 8], gl.bfloat16, transposed=True)
    vt_shared: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [64, 64], gl.bfloat16, transposed=True)
    r = gl.arange(0, 64, layout=gl.SliceLayout(1, load))
    d = gl.arange(0, 64, layout=gl.SliceLayout(0, load))
    g = gl.arange(0, 8, layout=gl.SliceLayout(1, load))
    q = gl.load(Q + g[:, None] * 64 + d[None, :])
    q = q.to(gl.bfloat16).to(gl.float8e4nv).to(gl.bfloat16)
    k = gl.load(K + r[:, None] * 64 + d[None, :]).to(gl.bfloat16)
    v = gl.load(V + r[:, None] * 64 + d[None, :]).to(gl.bfloat16)
    k_buf = gl.allocate_shared_memory(gl.bfloat16, [64, 64], k_shared)
    qt_buf = gl.allocate_shared_memory(gl.bfloat16, [64, 8], b_shared)
    vt_buf = gl.allocate_shared_memory(gl.bfloat16, [64, 64], vt_shared)
    p_buf = gl.allocate_shared_memory(gl.bfloat16, [64, 8], b_shared)
    k_buf.store(k)
    qt_buf.store(gl.permute(q, (1, 0)))
    vt_buf.store(gl.permute(v, (1, 0)))
    fence_async_shared()
    qk_zero = gl.zeros((64, 8), gl.float32, layout=mma)
    pending = warpgroup_mma(k_buf, qt_buf, qk_zero, use_acc=False,
                           max_num_imprecise_acc=0, is_async=True)
    score = warpgroup_mma_wait(num_outstanding=0, deps=(pending,))
    ks = gl.load(KScale).to(gl.float32)
    vs = gl.load(VScale).to(gl.float32)
    score = score * ((SM_SCALE * QK_FACTOR) * ks * 1.4426950408889634)
    maximum = gl.max(score, 0)
    probability = gl.exp2(score - maximum[None, :])
    denominator = gl.sum(probability, 0)
    p_buf.store(probability.to(gl.bfloat16))
    fence_async_shared()
    pv_zero = gl.zeros((64, 8), gl.float32, layout=mma)
    pv = warpgroup_mma(vt_buf, p_buf, pv_zero, use_acc=False,
                      max_num_imprecise_acc=0, is_async=False)
    raw_attention = pv * vs / denominator[None, :]
    out_d = gl.arange(0, 64, layout=gl.SliceLayout(1, mma))
    out_g = gl.arange(0, 8, layout=gl.SliceLayout(0, mma))
    gl.store(Out + out_g[None, :] * 64 + out_d[:, None], raw_attention)
    gl.store(RowMax + out_g, maximum)
    gl.store(RowSum + out_g, denominator)
