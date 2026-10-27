# Triton 3.8 Gluon/Hopper eligibility: conditional, verify-first

Public **v3.8.0** source can express the required Hopper asynchronous MMA and
masked page-table gathers. This establishes language-level feasibility, not
an installed, working attention backend or a predicted speedup. The saved
image receipts verify Triton3.8.0 but do not verify a Gluon import or its source
files; that eligibility prerequisite remains **unknown**. No GPU, service,
installation, credential action or candidate implementation was performed here.

## Version-bound evidence

The saved architecture probe records H200/sm90, Triton3.8.0, Python3.12.3,
Torch2.14.1+cu130 and CUDA13.0. Its Triton distribution metadata hash is
`d97a89f102848d1b74c4ce40e76150c3bbbbced8912eb5e0f56c203914788bd0`,
RECORD hash`bd578657969c238daafa979493e06aad9167774f174320ee6efa93591cad58e0`,
but VCS revision is null. The CPU inventory result SHA256 is
`f7f114b91ef6fded836f4502e15dbd4e0972a2872ac3e1142c20ad8ad847c382`.
These are in the sibling`attention-architecture` worktree's
`architecture-probe/profiling-run/inventory/result.json` and
`architecture-probe/probe/native-analysis.json`. A version string cannot prove
the installed package exactly matches upstream or includes importable Gluon.

Pinned primary sources were read on2026-10-25:

- [Hopper API](https://github.com/triton-lang/triton/blob/v3.8.0/python/triton/experimental/gluon/language/nvidia/hopper/__init__.py)
  provides explicit asynchronous`warpgroup_mma` and a wait with dependency
  tokens. It exposes a precision argument and`max_num_imprecise_acc`; the
  latter defaults differently for FP8, so a strict prototype must set it
  explicitly rather than inherit an imprecise FP8 default.
- [WGMMA tutorial](https://github.com/triton-lang/triton/blob/v3.8.0/python/tutorials/gluon/05-wgmma.py)
  shows Hopper-only execution, at least four warps/M64, shared B operands,
  register accumulators and a double-buffered schedule that delays waiting
  while other loads proceed. A synchronous call immediately waits; merely
  generating a WGMMA instruction does not establish useful overlap.
- [Masked asynchronous copy API](https://github.com/triton-lang/triton/blob/v3.8.0/python/triton/experimental/gluon/language/nvidia/ampere/async_copy.py)
  accepts a pointer tensor and mask, with shared destination of the same shape.
  Hopper re-exports it. This is the eligible route for live page1 mappings.
- [Hopper TMA API](https://github.com/triton-lang/triton/blob/v3.8.0/python/triton/experimental/gluon/language/nvidia/hopper/tma.py)
  uses shape/stride descriptors and tile coordinates; device-side descriptor
  construction exists, but there is no arbitrary-index gather operation in
  this Hopper API. The
  [v3.8 shared-input test](https://github.com/triton-lang/triton/blob/v3.8.0/python/test/gluon/test_core.py#L940-L1016)
  enables TMA gather/scatter only on Blackwell and calls the Blackwell API.
  Do not import that mechanism as Hopper page1 support.
- [Gluon core](https://github.com/triton-lang/triton/blob/v3.8.0/python/triton/experimental/gluon/language/_core.py)
  exposes E4M3, BF16, FP32 types, casts, shared buffers and a warp-specialization
  API. The last API's presence alone does not verify this installed backend's
  Hopper lowering; it is an additional prerequisite if used.
- [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma)
  documents BF16 WGMMA with FP32 accumulators. Existing local M64/D64 native
  BF16 artifacts already prove the older probe reached those instructions;
  their immediate-wait pipeline was slow. That result rejects the tested
  pipeline, not every future asynchronous schedule.

## Contract matrix

| Muse requirement | Gluon/Hopper assessment | Remaining blocker |
|---|---|---|
| Current image installation | Triton3.8.0 recorded; upstream tag has APIs | No saved Gluon file/import receipt; do not assume installation |
| D64, GQA8, full verify L8 | L×G=64 naturally fills WGMMA M64; D64 is compatible with BF16 K16 steps | New layout, resource and native precision validation required |
| Full decode L1 | Only8 valid rows versus minimum M64 | Padding-only reuse wastes8× row work,4× versus current M16; no automatic decode win |
| FP8 KV/Q quantization | Load E4M3, widen per tile to BF16; retain Q BF16→E4M3→BF16 and one scale placement | Exact source/native cast and original-oracle proof required; no direct imprecise FP8 MMA assumption |
| BF16 PV / FP32 state | API can represent BF16 operands and FP32 accumulator/state | Async carry/softmax ordering must preserve original numerical contract |
| page1/page64 arbitrary mappings | Masked pointer-tensor async copy can read GPU page table and gather rows | Correct masking, alignment, widening and gather cost remain unmeasured |
| Hopper TMA direct sequence gather | Regular affine TMA is not arbitrary page-table gather | Reject direct TMA-gather proposal; no Blackwell API or whole-KV repack |
| Live lengths/page tables/replay | Can load on GPU each invocation and derive masks there | New fixed-address replay/cross-call tests; no host-derived cached descriptors/taskmaps |
| Planning/assignment | Any input-derived plan must execute per call inside captured/timed work | No untimed plan cache or cross-layer reuse; avoid planner in first prototype |
| Generic asynchronous overlap | Explicit tokens, waits, fences and shared buffers exist | Need genuine independent work before waits and verified buffer lifetimes; no runtime overlap claim from assembly alone |

Finite E4M3 values fit BF16 exactly as a representation argument, but changing
operation ordering still needs native validation. The prototype may not bypass
established query quantization, widen raw unquantized Q, change PV precision or
omit historical/new keys. Full-KV repacking is excluded; only bounded per-tile
shared staging/widening is contemplated.

## Smallest useful mechanism falsifier

First recover read-only package-file/import evidence from the **existing**
image, without GPU context or installation. If Gluon/Hopper APIs are absent,
stop this route under the no-install rule. Do not substitute a private backend
entrypoint or a different version.

Then consider one compile-gated **full-verify L8/G8, M64/D64/N64** producer
micro-prototype, with identical page1 gather, FP8 encoding/BF16 widening,
FP32 softmax and BF16 PV in both arms. The causal control immediately waits;
the alternative double-buffers bounded tiles and moves only proven-independent
address/copy or next-QK work between MMA commit and wait. Keep load volume,
geometry, math, ownership and merge the same. No planner, full-cache transform,
formal route change or broad knob sweep is part of this first question.

The required evidence is different from the earlier compile-only WGMMA test:
prove delayed waits with useful independent work, correct dependency tokens,
no shared-buffer overwrite while MMA is in flight, strict conversions, and
no spills/resource collapse. FP32 rowmax/softmax consumes completed QK; an
unfinished accumulator may not be read or rescaled. A possible later pipeline
can issue next-QK before current-PV, wait for the older score group while the
newer PV remains outstanding, and overlap next softmax preparation. That is a
conditional schedule design, not proven code or a claim that it is resource-feasible.

If the compiler still immediately waits, requires lower precision, or extra
buffers/carries destroy resources, reject before end-to-end timing. If it
passes, original-oracle/cache/replay tests precede paired complete-call timings
against **current FA3**, with the same-geometry immediate-wait arm retained
for causal attribution. Beating the previous slow custom kernel is insufficient.

## Upside boundary: this is not a ready four-family winning route

Using the saved official5090 shape/load times, full B16/B32 verify shares of
their loads are0.7597004 and0.8280845. With all other paths fixed,20% cuts in
both model+5.9338% overall; even deleting both entire full-verify paths has a
loose mathematical factor ceiling1.7008×. This is below the parent's reported
leader/current ratio of roughly2.359×. Thus verify-only Gluon cannot by itself
close the reported gap, even under an impossible zero-time ceiling.

All four batched full decode/verify paths have shares0.8291759/0.8402623/
0.7597004/0.8280845; a conditional20% cut in all four models+12.5840% overall.
These are Amdahl sensitivities, not forecasts or HBM measurements. A viable
decode mechanism or another large path family is required for broader upside.
No evidence here validates request packing, a new persistent scheduler or a
library replacement. Actual FA3 already has asynchronous/persistent machinery;
it remains the comparator, not a serial straw man.
