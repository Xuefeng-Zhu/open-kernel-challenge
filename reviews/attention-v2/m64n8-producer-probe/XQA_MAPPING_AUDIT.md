# XQA / FA3 mapping audit before producer integration

Read-only source audit, 2026-10-26, by `/root/offline_transpose_review`, with an
independent FA3 source check by `/root/offline_transpose_review/fa3_mapping_source`.
Only this document was written. No GPU, container, service, authentication,
candidate-source or Git operation was performed.

**Finding:** reversed key-row/query-column attention is established XQA design.
Our question is whether a strict BF16, live page-1 adaptation can outperform the
actual selected complete call. Public XQA geometry alone cannot answer that.

## Reproducible source identities

- NVIDIA TensorRT-LLM pin:
  `cfae4519c3194c9d2088333a4028965ecb58ca7c`. The public commit/main page
  returned this revision during the audit; its full SHA was resolved from the
  [official commit patch](https://github.com/NVIDIA/TensorRT-LLM/commit/cfae4519c3194c9d2088333a4028965ecb58ca7c.patch).
  All XQA links below use that immutable pin. This is an observed public
  snapshot, not a guarantee that a moving branch has not advanced.
- FA3 comparison pin: `060c9188beec3a8b62b33a3bfa6d5d2d44975fab`, official
  v2.8.3 reference. Public main was also readable, but its current HEAD was
  not resolved. This pin is neither current-HEAD nor installed-judge proof.
- Selected local control:
  `../fusion-lookahead-probe/candidate.py`, SHA256
  `3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`.
- Reviewed one-tile prototype: `../m64n8-transpose-probe/kernel.py`, SHA256
  `118cdd776cf0f4668a2e3a31539c56f3f7d449748ae3881f9c0adf500344d033`.
  Its actual compile/dataflow evidence is in `../m64n8-transpose-probe/ARTIFACT_REVIEW.md`.
- The integrated producer was **not yet written/frozen** when this audit was
  prepared. Its proposed buffering/tail mapping is design intent, not audited
  source or native evidence.

## XQA: what actually transfers

The following statements concern the pinned Hopper implementation, not all
TensorRT-LLM dispatches. Source line numbers below are one-based.

| Ownership or mechanism | XQA source fact | Consequence for this experiment |
|---|---|---|
| CTA coordinates | Request/KV-head, query subsequence, KV subsequence are distinct coordinates; KV tiles are interleaved across subsequences. | Preserve selected contiguous split/chunk ownership; do not silently import XQA scheduling. |
| Small-query mapping | Ordinary decode uses SWAP_AB. Fixed speculative use requires uniform length with `L*G<=32`; query columns round to8/16/32. G8/L1 gives64 key rows x8 columns. | The M64N8 idea is established, not a novel discovery. G8/L8 is outside that fixed-small-query branch. |
| Work division | Separate four-warp QK, four-warp PV and four-warp IO groups, with explicit shared barriers/buffers. | This differs from our four-warp probe and intended producer; overlap/occupancy is not transferable. |
| QK / PV | SWAP_AB computes K x Q-transpose, then V-transpose x P-transpose; FP8 PV uses register-A fragments. | Orientation matches algebraically; operand storage, reduction and precision do not. |
| Multi-block ownership | A last-finishing CTA is elected by semaphore and combines the participating partials. | Preserve selected merge ABI and synchronization; do not equate elected finisher with last-index split/new-KV ownership. |

These facts are in pinned [mha_sm90.cu](https://github.com/NVIDIA/TensorRT-LLM/blob/cfae4519c3194c9d2088333a4028965ecb58ca7c/cpp/kernels/xqa/mha_sm90.cu)
lines40–109,320–322,676–745,899–915,1184–1272 and1627–1682.
Short excerpt: `#define SWAP_AB (!SPEC_DEC)`.

The [GMMA interface](https://github.com/NVIDIA/TensorRT-LLM/blob/cfae4519c3194c9d2088333a4028965ecb58ca7c/cpp/kernels/xqa/gmma.cuh)
lines121–148 establishes M64, N granularity8, typed shared/register-A variants,
and explicit fence/commit/wait operations. It establishes available instruction
mapping, not a particular executed binary or useful overlap.

### Page size and arithmetic are real incompatibilities

The [defines](https://github.com/NVIDIA/TensorRT-LLM/blob/cfae4519c3194c9d2088333a4028965ecb58ca7c/cpp/kernels/xqa/defines.h)
lines70–100 document fixed uniform speculative length, FP8 cache selection and
contiguous/page layouts. Its page-size comment lists0/16/32/64/128. However,
the actual [host support gate](https://github.com/NVIDIA/TensorRT-LLM/blob/cfae4519c3194c9d2088333a4028965ecb58ca7c/cpp/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImplJIT/kernelUtils.cpp)
lines98–139 admits paged QGMMA sizes **8/16/32/64/128**; HMMA lines185–188
admits16/32/64/128. Both reject1. Thus the supplied page list omitted supported
size8, while its page-1 limitation is confirmed by dispatch code, not comments
alone. No library invocation or page-1 support claim follows from this audit.

The [tensor-map builder](https://github.com/NVIDIA/TensorRT-LLM/blob/cfae4519c3194c9d2088333a4028965ecb58ca7c/cpp/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/tensorMapUtils.cpp)
lines92–104 builds page-shaped boxes with token extent `min(page_size,64)`.
The producer's per-page transfer loop means a naive page1 extension would
require64 page transfers for a64-key tile versus4 at page16. This is source
operation-count inference, not measured latency or proof that such an extension
is legal/efficient. A live row gather is the relevant independent adaptation.
The [cache pointer helpers](https://github.com/NVIDIA/TensorRT-LLM/blob/cfae4519c3194c9d2088333a4028965ecb58ca7c/cpp/kernels/xqa/mhaUtils.cuh)
lines66–91 also distinguish original and separate-pool token/head strides;
matching a page count does not establish layout compatibility.

For FP8 cache, XQA's MathElem remains FP8; this is not BF16 widening. Its cache
ABI also exposes a shared K/V cache scale. The probe requires independently
placed K/V scales, BF16-rounded Q then E4M3 encoding/widening, BF16 K/V and
BF16 P with FP32 softmax/carry. Copying an FP8 XQA math path would change this
experiment's declared precision. Cache/input/output type definitions are
explicit in [mha.h](https://github.com/NVIDIA/TensorRT-LLM/blob/cfae4519c3194c9d2088333a4028965ecb58ca7c/cpp/kernels/xqa/mha.h)
lines25–79; they do not imply our fused normalization/gating contract.

## FA3 already provides grouping, page gathers and staging

At the FA3 pin, [launch dispatch](https://github.com/Dao-AILab/flash-attention/blob/060c9188beec3a8b62b33a3bfa6d5d2d44975fab/hopper/flash_fwd_launch_template.h)
packs query positions and GQA heads into query-M, schedules KV heads in packed
mode, and retains batch/split coordinates. The
[mainloop](https://github.com/Dao-AILab/flash-attention/blob/060c9188beec3a8b62b33a3bfa6d5d2d44975fab/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp)
uses query-M/key-N QK and query-M/output-D PV. Its FP8 path uses FP8 Element
operands, with register/shared PV and row-major V transpose staging. Neither
generic GQA reuse nor V transposition nor asynchronous staging is missing by
default. This remains upstream-source evidence, not resolved selected dispatch.

The [page loader](https://github.com/Dao-AILab/flash-attention/blob/060c9188beec3a8b62b33a3bfa6d5d2d44975fab/hopper/paged_kv.h)
already computes live page index/offset per row and uses cooperative cp.async;
when K/V iterations differ, it preserves prior V pointers. Page1 does not
require pretending pages form one contiguous TMA tile. Short excerpt:
`We use CpAsync for K and V if PagedKV`.

The [epilogue](https://github.com/Dao-AILab/flash-attention/blob/060c9188beec3a8b62b33a3bfa6d5d2d44975fab/hopper/epilogue_fwd.hpp)
and [combine](https://github.com/Dao-AILab/flash-attention/blob/060c9188beec3a8b62b33a3bfa6d5d2d44975fab/hopper/flash_fwd_combine_kernel.h)
own split outputs/LSE and rescale them into final query/head outputs. That ABI
differs from our unnormalized FP32 numerator/max/sum partials and RMS/gate merge.
The [D64 one-byte tile policy](https://github.com/Dao-AILab/flash-attention/blob/060c9188beec3a8b62b33a3bfa6d5d2d44975fab/hopper/tile_size.h)
returns192x160 with flags at lines37–40. It proves neither installed geometry
nor24x/3x executed waste. Historical64x128 symbols and unresolved current
dispatch remain separate provenance findings.

## Exact local difference and admission constraints

Selected `_fused` lines92–110 already assigns `(batch,KV head,split)` and
groups query rows `j*G+g`. Lines137–175 read live page-table rows, gather each
K/V row, perform query-major QK/PV and keep FP32 online max/sum/numerator.
Lines177–204 give the **last-index split** the new-token normalization,
quantization, cache writes and causal attention even when history is empty.
Lines213–243 retain split-major partials and optional elected-finisher merge;
`_merge` lines246–279 handles the selected separate L1 merge. None of this is
new capability supplied by XQA.

The intended producer changes history to K[64,64] x Q.T[64,8] and
V.T[64,64] x P.T[64,8]. Softmax must now reduce across all64 key rows, then
transpose/correspond numerator state back to the unchanged partial ABI. Its
proposed two raw-FP8 gather slots, BF16 shared publication and explicit waits
still require source/native lifetime review. The original N16 new-token tail,
FP32 carry and final merge must remain intact. No completed integrated kernel
or buffering equivalence is asserted here.

Before further GPU admission, require source freeze; original masks/scales/live
metadata/cache ownership; full-history plus empty/tail/address CPU models;
both actual native N8 groups; complete cross-warp reductions; preserved waits
and safe raw/BF16 buffer reuse; zero spills/local and the predeclared full
producer resource gate. Gluon `num_stages=3` alone does not prove automatic
Triton-style pipelining. Native correctness/replay and fresh complete-call
comparisons against selected3da remain required after those gates.

There is no primary-source reason to block an otherwise valid CPU integration
compile. There is also no public-source basis to promote it, claim novelty,
predict speed from XQA, change page layout, lower precision, import a different
split scheduler or bypass the declared gates. SGLang's historical page1 SIMT
and grouped-dot implementations were located but are not needed for these
constraints; no unresolved current-main identity is used as runtime evidence.
