# Completed Pro research and independent source review

Observed complete on 2026-10-25 at 17:59 UTC. This is a research record, not a
new experiment or performance result. No candidate source changed, GPU job
started, official check consumed, or credential changed in this stage.

## Receipt and scope

The existing user-authorized draft was sent once in the user's Mac Chrome.
[Research architectural diagnosis](https://chatgpt.com/c/6ade40a4-9438-83e8-b546-958708e69fd0)
shows the single attached `PRO_RESEARCH_BRIEF.md`, the intended research request,
Pro selected, and **Response complete**. The UI reports “Worked for 6m 3s” and
three research subagents. Web search was enabled; the separate Deep research
mode was not selected. No public share link was created.

The attached file is 16,694 bytes, SHA256
`2b532569e6ae626ae512589f8b4b2829be4b919ba91e70ce8e321a73260fc8e7`.
The previously blocked upload was resolved by the user's specific confirmation.
The later stop was honored while the attachment remained an unsent draft; only
the subsequent explicit reauthorization triggered this one send. No request
is pending or uncertain. The browser tab is preserved as the deliverable.

This file is a faithful condensed capture and critical review, not a verbatim
archive of the entire response. The conversation contains the original memo
and its sources; `pro-request-status.json` records the receipt.

## What Pro recommends

Pro ranks credible progress per implementation risk ahead of maximum modeled
upside. This differs from our pre-research `PLAN.md`, which ranked the matrix
pipeline feasibility probe first. Preserve that earlier analysis as history.

| Rank | Mechanism | Isolated target/change | Conditional whole-score value |
|---|---|---|---|
| 1 | Extend existing fused preparation; retain FA3 | Full decode B16 and full verify B16/B32 | Saving 5–7 microseconds in **each complete call** models +0.84–1.19% |
| 2 | Exact widening after E4M3 quantization, enabling a different Hopper matrix pipeline if the compiler emits one | Compile-only L8/M64 custom producer; QK BF16 operands, unchanged FP32 accumulation policy and BF16 PV | A hypothetical 10–20% reduction in both complete full-verify paths models +2.80–5.94% |
| 3 | Eliminate FA3 split/combine where enough work remains | Sliding verify B16/B32 only, `num_splits=1`; preserve all math and preparation | Assumed 2–4 microseconds saved per target models roughly +0.57–1.15% |

Every value above is a sensitivity scenario, not a forecast, confidence
interval, achieved improvement, or new authorization to run. The third idea
should initially exclude B1 because very little unsplit parallelism remains.

## Independent review: the first experiment is smaller than Pro describes

The current A2-13 source is `../mixed_fa3_fp8_qk.py`, SHA256
`e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e`, source commit
`21fefb946524a25d8a06562dd28b631d49cd5175`.
No candidate code was altered during this review.

The existing `_fa3_full_verify_fused` helper (line 528) already derives its grid
from `batch * q_len`. It is batch-generic. `kernel_fn` (line 562) routes only
full B1 through it; the three proposed targets still use ordinary preparation.
Therefore the smallest falsifiable candidate is **only a dispatch-guard
extension** for `(B, L) = (16, 1), (16, 8), (32, 8)` with zero window and RoPE.
No new producer, indexing rewrite, matrix kernel, precision change or FA3
argument change is needed. Keep full decode B32 on its existing custom route.

The reviewed helper retains FA3 scheduler/page arguments, non-unit descales,
BF16-before-FP8 cache rounding, and writes through physical `cache_loc`.
Existing full-attention dedicated tests mainly cover B1. Prior batched sliding
fusion does not demonstrate correctness or performance on these full targets.
Existing dispatch tests explicitly expect ordinary preparation for them, so
this is not a repeat of an adopted experiment.

The five-microsecond threshold is assumed materiality, not evidence. Prior
small-path wins and about 7.5–8.5 microseconds of preparation profile activities
support testing the idea, but overlapping activities cannot be subtracted
from whole-call latency. This is a low-risk incremental opportunity, not an
architectural explanation of the gap to leaders. A2-11 stays selected.

## Useful corrections and limits

- Overlapping marginal score confidence intervals are **not** a significance
  test for the difference. Use paired log-ratio/score-difference uncertainty
  and independent repeated sessions; many graph replays are not independent
  experimental sessions. Our caution about noisy promotion still applies.
- A2-15 changed generated scheduling and buffering. Its 5.189% paired target
  reduction does not isolate the causal contribution of page-ID lookahead.
  The exact six-load transfer model is only +0.749614% overall. The source
  analysis and native artifacts do not measure hardware stall latency.
- Existing upstream FA3 already contains asynchronous KV loading and paged
  address handling. “Add asynchronous paging” is not a new mechanism without
  showing what the installed path lacks. Upstream FP8 tile geometry is a lead
  to inspect, not evidence of executed padding or wasted work in our binary.
- Exact E4M3-to-BF16 widening preserves represented operands, not reduction
  order or whole-kernel numerical identity. WGMMA instruction presence alone
  does not prove overlap. The unchanged custom L8 producer, cast-only producer,
  and complete incumbent FA3 route are three distinct necessary controls.
- Do not relax `max_num_imprecise_acc=0` merely to obtain FP8 WGMMA. F32 output
  type does not itself guarantee full-precision intermediate accumulation.
  A correctness pass does not override the competition's precision contract.
- Pro could not retrieve some exact pinned repositories/API details. It
  explicitly treated the brief's pinned analysis, tolerances and scoring as
  supplied evidence and used current upstream sources elsewhere. Its rounded
  arithmetic is consistent with our model; `evidence.json` retains the exact
  receipt-based calculations. Neither current upstream nor an unrelated
  issue proves the installed binary's behavior.

## Minimal next validation, not executed

Freeze the complete A2-13 parent and change only the three dispatch targets.
Update dispatch tests and compare preparation outputs/cache bytes with the
ordinary path. Native validation must include all three targets, scheduler
metadata, page sizes 1 and 64, scales 1 and 1/64, scattered pages, live slot 0,
page-boundary append, changing actual lengths, zero/short histories, L8
causality, unchanged cache rows and immutable inputs. Repeat same-address
A→B→A graph replay; separately test overlapping independent invocations with
disjoint writable caches and invocation-local scratch. CPU models are not
native FA3/Triton evidence.

Pro proposes four predefined warmup pairs and twelve balanced measured pairs
per target, fresh equivalent inputs/cache and cold-L2 whole-call graph timing,
then an independent session before promotion. Preserve every sample. Require
correctness first, useful whole-call gains, and no material control regression;
if preparation activities shrink without the complete call improving, reject
the performance hypothesis. Run all 16 prerequisite and 12 scored shapes
before adopting a complete candidate. These gates remain unexecuted here.

Before a broader mainloop rewrite, obtain the actual installed FA3 symbol,
template/grid/binary identity, its own registers/shared memory/spills and
generated loads/matrix/wait instructions. Permitted hardware counters would
help distinguish bandwidth, issue and dependency bottlenecks; none were
collected by this research. Keep instrumented profiling separate from timing.

## Primary-source references supplied by Pro

These are external claims to audit against the pinned runtime, not executed
software or granted permissions. Our local pinned analysis remains in PLAN.md.

- [FA3 mainloop](https://github.com/Dao-AILab/flash-attention/blob/main/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp),
  [paged KV](https://github.com/Dao-AILab/flash-attention/blob/main/hopper/paged_kv.h),
  [tile choices](https://github.com/Dao-AILab/flash-attention/blob/main/hopper/tile_size.h),
  [dispatch](https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_api.cpp).
- [FlashAttention-3 paper](https://arxiv.org/html/2407.08608v2) and
  [FlashInfer paper](https://arxiv.org/html/2501.01005v1).
- [Triton Hopper warpgroup MMA API](https://triton-lang.org/main/gluon/api/generated/triton.experimental.gluon.language.nvidia.hopper.warpgroup_mma.html)
  and [Triton FP8 lowering issue](https://github.com/triton-lang/triton/issues/12392).
- [NVIDIA WGMMA precision specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma)
  and [Nsight Compute measurement guidance](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).
- [Official rules](https://openkernelchallenge.dev/rules) and
  [attention problem](https://openkernelchallenge.dev/problems/muse_attention_decode).
  Standalone performance does not prove serving integration or prize eligibility.

No external implementation was executed or copied into a candidate. The
existing official-baseline attribution remains intact. Independent local
review also made no file changes or GPU calls.

## 工程实验卡

问题：大批量 full attention 的 FA3 主计算之外，仍有分开的预处理开销。
假设：复用已经存在的融合预处理，能减少三条完整调用的延迟。
改动：只扩展 B16 decode、B16/B32 verify 的分流，不改计算精度或 FA3。
预期：若每条少 5–7 微秒，按真实权重模型整体约提高 0.84–1.19%。
现状：源码复核确认可做；本轮未实现、未测速。先补批量缓存及图重放验证，
再用配对完整调用测量接受或否定假设。A2-11 继续保留为已选版本。
