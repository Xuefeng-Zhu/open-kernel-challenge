# Reject sliding B16/B32 num_splits=1

The one-argument candidate passed native identity, correctness and actual
combine-elimination gates, then failed the predeclared whole-call benefit gate.
Keep automatic splits and the validated preparation-fusion control. No official
check was sent, and original A2-11 remains selected.

| Sliding verify target | Raw whole-call median, control→candidate | Median balanced pair mean saving | Paired candidate slowdown | Favorable pairs |
|---|---:|---:|---:|---:|
| B16 | 27.088→35.568 µs | −8.528 µs | +31.4944% | 0/6 |
| B32 | 32.992→35.808 µs | −2.600 µs | +7.8550% | 0/6 |

The unchanged full-B16-verify control's paired candidate change was −0.2237%
(slightly faster), well within the 5% regression gate. Four warmup and twelve
measured two-arm samples per family were retained; each adjacent reverse-order
pair used the same fixture version. Timings measured complete cold-L2 graph
device work, separately from profiling. These are one-session results with
two fixtures and six balanced pairs per family, not an official score or an
independent-session interval. Both targets clearly fail the required ≥2µs
median balanced absolute saving and 6/6 positive pair-ratio gates. Do not spend
an official check on this combined candidate or sweep more split counts.

The single Torch/CUPTI activity session confirms the intended mechanism on both
actual installed specializations: mainloop 1/combine 1/scheduler-prepare 1 becomes
mainloop 1/combine 0/scheduler-prepare 1. Physical kernels decrease 5→4. Raw event
groups additionally contain a CUDA record-function annotation, which is not a
physical kernel. Scheduler preparation persists exactly as the source review
warned. Instrumented mainloop activities grow from 13.855→25.663µs for B16 and
17.695→26.079µs for B32; these are diagnostic intervals, not benchmark samples
or additive latency accounting. The single split flag changes complete-call
performance, but useful CTA parallelism, occupancy and stalls were not measured.
Lost parallelism or an implicit specialization change remains an explanation
to investigate, not a proved hardware cause.

The installed visible `sgl_kernel.flash_attn.flash_attn_with_kvcache` signature
explicitly exposes keyword-capable `num_splits=0`. Both modules use the same
callable. Its 16,160-byte wrapper source was fully retained and hashed:
`610747f2c68495c5f3ba67f1247f0a1b4e4f7df116e2f89cd8e8983f0271f9ac`.
The observed extension `sgl_kernel/flash_ops.abi3.so` is 1,122,619,328 bytes,
fully streamed to SHA256
`cdea0997a51412b8db8bf8da49d896a6c9d12a39bb1b548d840bb8dc428f13ef`.
No relevant observed extension was omitted. This identifies the loaded files
observed by the reviewed worker, not every transitive dependency or the
official judge image. Native environment: H200, Torch 2.14.1+cu130,
Triton 3.8.0, CUDA 13.0, Python 3.12.3. No installation or permission expansion.

All seven required cases passed: two scored sliding cases, two soft page64/
scale1⁄64 boundary cases, two unmodified official-generated window-edge cases,
and one unchanged full-B16-verify control. There were 12 exact preparation groups
(36 Q/K/V byte comparisons), 42 A→B→A official oracle calls, 14 exact return-A
checks, 42 cross-arm cache-byte comparisons, four disjoint-cache concurrent-call
oracle checks, four post-activity replay checks and 96 checked timing replays.
All original tolerances, window/causal masks, scaling, cache ownership and
immutable/untouched-row checks were preserved. Concurrent submission correctness
was checked; actual hardware overlap was not measured.

The isolated branch/worktree is `attention-sliding-unsplit`, source freeze
`5e494062a91a643df674887140125b9333f2fb92`. Source SHA256s:

| Source | SHA256 |
|---|---|
| Validated fusion control | `5c3d1c5b00f0e759f1c499b8f567a0f47c77ce0fa38566f3028a571fccc614a0` |
| Rejected conditional candidate | `586074a4bbff57ff28c53070dfcdfcf881631b9aeffeb016032a7a07b32309cf` |
| Native worker | `953537670113411a0de1445864bbaa52d4b3cfecb276da5a01ca5fdf0ee6d812` |
| Raw native result | `46aed5e3fba5b978d3206de15106ce7bd5961fd63b1bc8e76ac887073fb4f0ab` |

`runtime/analyze_result.py` verifies the published request/source/transport
hashes, exact fixture sets, native gates, event accounting and paired arithmetic.
`runtime/RESULT_REVIEW.md` independently confirms performance rejection.
The preparation manifest describes the historical frozen stage; the separate
result manifest binds the subsequent raw result and sanitized analyses.

App `ap-SoIIp01s2fzbOD9KKGSw9W` stopped at 20:52:50UTC. An independent read at
20:53:15.259613UTC confirmed stopped/tasks0 and all listed apps stopped/tasks0;
the parent received immediate slot-release/outcome commentary before polishing.
The initial controller stopping/tasks1 observation remains unaltered in the
published launch receipt. Its exit1 reflects incomplete immediate shutdown
observation; the native worker completed with exit0 and validation passed,
while the separate performance-admission field is false.

Client wall time 61.398s; worker internal time 48.876s. At the independent cutoff,
rounded workspace credit usage increased $0.06, detailed usage $0.06003806,
actual paid $0. The conservative closed allowance is $0.194892333 within the
reviewed $0.58 free-credit reservation. Delayed workspace deltas are not an exact
per-app invoice. No retry, second Modal job, official check, formal entry,
credential renewal or billing-setting change occurred in this stage.

The previously completed fusion result remains at verified private commit
`05c9cc08948c42fed9e09aae1cd2b41edf966c74`:
[fusion report](https://github.com/Xuefeng-Zhu/open-kernel-challenge/blob/05c9cc08948c42fed9e09aae1cd2b41edf966c74/reviews/attention-v2/fused-preparation-probe/RESULTS.md).
The next separately authorized assessment is composition of that validated
fusion with the already measured A2-15 full-decode-B32 pipeline variant.
It will use fusion as control and automatic splits; no composition candidate
or additional measurement is included in this rejected experiment.
