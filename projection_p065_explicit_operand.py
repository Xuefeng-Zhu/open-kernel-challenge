"""P065 explicit MMA-v2 operand layouts; compile-only until native validation.

Owned code derived from P043/P064's exact codec arithmetic. The raw/encoded
specializations share this Gluon load/mainloop/output structure. Gluon does not
automatically preserve ordinary Triton's software pipeline; no such claim is made.
"""
import triton
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language import DotOperandLayout, NVMMADistributedLayout, BlockedLayout, SliceLayout
from triton.experimental.gluon.language.nvidia.ampere import mma_v2


@gluon.jit
def _pooler_codec_partial(X, W, SM, DELTA, BASE, ESC_IDX, ESC_EXP, P,
                          M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
                          WN: gl.constexpr, KP: gl.constexpr,
                          S: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr,
                          BK: gl.constexpr, COMPRESSED: gl.constexpr):
    mma: gl.constexpr = NVMMADistributedLayout(version=[2, 0], warps_per_cta=[1, 4], instr_shape=[16, 8])
    a_layout: gl.constexpr = DotOperandLayout(0, mma, 2)
    b_layout: gl.constexpr = DotOperandLayout(1, mma, 2)
    store_layout: gl.constexpr = BlockedLayout([1, 4], [2, 16], [4, 1], [1, 0])
    split = gl.program_id(1)
    ms = gl.program_id(2) * BM + gl.arange(0, BM, layout=SliceLayout(1, a_layout))
    kx = gl.arange(0, BK, layout=SliceLayout(0, a_layout))
    kw = gl.arange(0, BK, layout=SliceLayout(1, b_layout))
    ns = gl.program_id(0) * BN + gl.arange(0, BN, layout=SliceLayout(0, b_layout))
    part_k = gl.cdiv(K, S * BK) * BK
    acc = gl.full((BM, BN), 0.0, gl.float32, layout=mma)
    for step in range(0, part_k, BK):
        start = split * part_k + step
        xa = start + kx
        k = start + kw
        x = gl.load(X + ms[:, None] * K + xa[None, :],
                    (ms[:, None] < M) & (xa[None, :] < K), other=0.0)
        valid = (k[:, None] < K) & (ns[None, :] < N)
        if COMPRESSED:
            meta = (start // 256) * WN + ns
            base = gl.load(BASE + meta, (ns < N) & (start < K), other=0).to(gl.int32)
            esc = gl.load(ESC_IDX + meta, (ns < N) & (start < K), other=-1)
            sm = gl.load(SM + ns[None, :] * KP + k[:, None], valid, other=0).to(gl.int32)
            packed = gl.load(DELTA + ns[None, :] * (KP // 2) + (k // 2)[:, None],
                             valid & (esc[None, :] < 0), other=0).to(gl.int32)
            small = (packed >> ((k % 2) * 4)[:, None]) & 15
            full = gl.load(ESC_EXP + gl.maximum(esc, 0)[None, :] * 256 + (k % 256)[:, None],
                           valid & (esc[None, :] >= 0), other=0).to(gl.int32)
            exponent = gl.where(esc[None, :] >= 0, full, base[None, :] + small)
            bits = ((sm & 127) | ((sm & 128) << 8) | (exponent << 7)).to(gl.uint16)
            w = bits.to(gl.bfloat16, bitcast=True)
            w = gl.where(valid, w, 0.0).to(gl.bfloat16)
        else:
            w = gl.load(W + ns[None, :] * K + k[:, None], valid, other=0.0)
        acc = mma_v2(x, w, acc)
    values = gl.convert_layout(acc, store_layout)
    om = gl.program_id(2) * BM + gl.arange(0, BM, layout=SliceLayout(1, store_layout))
    on = gl.program_id(0) * BN + gl.arange(0, BN, layout=SliceLayout(0, store_layout))
    offsets = (split * M + om[:, None]) * N + on[None, :]
    gl.store(P + offsets, values, (om[:, None] < M) & (on[None, :] < N))
