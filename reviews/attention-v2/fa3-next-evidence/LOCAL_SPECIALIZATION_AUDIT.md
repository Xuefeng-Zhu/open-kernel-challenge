# Saved FA3 specialization evidence and next missing measurement

Local read-only audit at 2026-10-25T22:59:02.937525+00:00. No new GPU,
profiler/NCU session, package install, network request, credential action,
algorithm edit or branch merge was performed. Only this report is written.
The selected working source remains fusion plus lookahead, SHA256
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`.
The alpha-one family has been rejected by the parent's completed native result;
this audit does not reopen it or any earlier rejected architecture/split family.

## Concrete outcome

The raw saved activity trace contains an **untruncated, executed FA3 full-B32-
verify specialization**, including `VarlenDynamicPersistentTileScheduler`, grid,
block, register and shared-memory fields. It supersedes the earlier historical
`scheduling/fa3-audit.json` statement that exact scheduler identity was unknown
from then-available truncated profiles. FA3 already used dynamic persistence.
Grid132 means persistent workers, **not**132 attention work tiles.

The current official profiles for full decode B16 and full verify B16/B32 are
truncated textual reports. They contain **no grid/block/register/shared/split
fields**. The old full-B32-verify trace cannot establish the exact specialization
of these current calls or diagnose their hardware bottleneck. No defensible new
kernel mechanism is established. The highest-value next step is a small,
controlled same-fixture FA3 activity/auxiliary-shape diagnostic, described below,
not a parameter sweep or another custom attention implementation.

## Exact saved trace context and provenance

- Native execution freeze: `e26bb0cf291dfdbfc6e70065b6d121bf315423ca`.
- Existing image: `im-cPTbGpNiX3RarKResPgX7h`.
- Activity started2026-10-25T18:26:49.647278UTC; one FA3 and one rejected BF16-
  custom graph replay in one Torch CPU/CUDA activity session. Four original
  oracle checks before/after passed. This is instrumented activity, not a paired
  benchmark. This audit uses only the unchanged FA3 arm.
- FA3 source then: A2-13 `e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e`;
  full B32/L8, TP8=16Q/2KV heads, D64, ctx80000, agentic/coherent,
  page1, unit scales, seed135701. Actual histories are retained in summary;
  minimum74201, maximum85893;
  page table32×86008. This precedes the later preparation-fusion source change.
- Environment: H200 SM90,132 SMs, Torch2.14.1+cu130, Triton3.8.0, CUDA13.0,
  driver610.57.04; trace CUDA driver API13030/runtime13000/CUPTI130001.
  The current official job reports H200 driver580.159.03 and a different judge
  image. Do not assert binary/driver identity across those environments.

| Saved evidence | SHA256 |
|---|---|
| Raw activity trace,62,221 bytes | `e49b1677de2dfbc9fcca43e63924a026cd7c101321e2bee1131f87ee41355ffc` |
| Activity summary | `a48846ac5a4a79de771822130eee0d46705fa5350c49fafff9508125d977ba15` |
| Profiling launch receipt/image | `6e551833fb414660694eb2ffd2bbff4190290cb55b342603a0b7d8d04f84b55b` |
| Later exact installed-wrapper/extension identity receipt | `46aed5e3fba5b978d3206de15106ce7bd5961fd63b1bc8e76ac887073fb4f0ab` |
| Current official5090 result | `a66e15b71116d6758cd1aae75138f19c0a7db4cb2e89ba27394d6d09df2d61cb` |
| Current official profile JSON | `3a9f9f841665ef0c678001aa62396b44557a375ce20d5e830548d257ff6f1300` |
| Primary fusion official2f5 result | `46f9818a029a68995d6bad8b99c79a72f390ab09e464c01eb9a79b64bb160001` |

The first three files are in the independent `attention-architecture` worktree
under `reviews/attention-v2/architecture-probe/profiling-run/`. The later identity
receipt is in independent `attention-sliding-unsplit`, under
`reviews/attention-v2/sliding-unsplit-probe/runtime/result.json`. No old files
were edited or merged to obtain these facts.

## Full executed FA3 symbol

UTF-8 full symbol SHA256: `33239a4fe9825c98df5c44acd8bc412ee2c441fab7b704ffa6124fb68de96711`.
The exact symbol below exposes the leading mainloop template parameter2,
nominal tile tuple64×128×64, E4M3 input/FP32 accumulator, BF16 epilogue,
SM90 and persistent scheduler type. Do not assign meanings to every boolean or
numeric template argument without matching the installed C++ declaration.

```cpp
void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<64>, cute::C<128>, cute::C<64> >, 64, cutlass::float_e4m3_t, float, cutlass::arch::Sm90, true, false, false, true, true, false, false, false, true, true, true, true, false, cutlass::bfloat16_t, false, 1>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<64>, cute::C<64>, cute::C<128> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 128, true, true, true, true>, flash::VarlenDynamicPersistentTileScheduler<64, 128, 128, 128, true, true, true, true, true, true> > > >(flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<64>, cute::C<128>, cute::C<64> >, 64, cutlass::float_e4m3_t, float, cutlass::arch::Sm90, true, false, false, true, true, false, false, false, true, true, true, true, false, cutlass::bfloat16_t, false, 1>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<64>, cute::C<64>, cute::C<128> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 128, true, true, true, true>, flash::VarlenDynamicPersistentTileScheduler<64, 128, 128, 128, true, true, true, true, true, true> > >::Params)
```

Exact combine symbol:

```cpp
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cute::tuple<cute::C<16>, cute::C<64> >, 4, 256, 1, false, true, cutlass::bfloat16_t, float, cutlass::arch::Sm90> >(flash::FlashAttnFwdCombine<cute::tuple<cute::C<16>, cute::C<64> >, 4, 256, 1, false, true, cutlass::bfloat16_t, float, cutlass::arch::Sm90>::Params)
```

In particular, the integers2 in `prepare_varlen_num_blocks_kernel<2,true>` and4
in `FlashAttnFwdCombine<...,4,256,1,...>` are **not established num_splits**.
Neither the persistent grid nor combine grid gives the actual resolved split
count or per-request dynamic task map. Those runtime values were not saved.

## Recorded launch resources

All rows below belong to graph2/correlation115/stream7 in the saved trace.
Shared bytes and register counts are launch/resource metadata; durations are
instrumented activity and cannot be substituted for whole-call latency.

| Stage | Grid | Block | Registers/thread | Shared bytes | Activity µs |
|---|---:|---:|---:|---:|---:|
| _qk_rms_norm_fwd_kernel | 576×1×1 | 128×1×1 | 23 | 1,024 | 2.304 |
| FP8 conversion | 32×1×1 | 128×1×1 | 32 | 0 | 1.152 |
| FP8 conversion | 64×1×1 | 128×1×1 | 18 | 0 | 2.976 |
| store_kvcache_kernel<128,128,32,true,long> | 128×1×1 | 32×2×1 | 24 | 0 | 1.536 |
| FP8 conversion | 256×1×1 | 128×1×1 | 32 | 0 | 1.312 |
| prepare_varlen_num_blocks_kernel<2,true> | 1×1×1 | 64×1×1 | 20 | 1,056 | 3.712 |
| FA3 forward persistent mainloop | 132×1×1 | 256×1×1 | 255 | 56,320 | 378.752 |
| FA3 combine | 8×1×32 | 256×1×1 | 48 | 17,536 | 3.296 |
| norm_gate<BF16,64,16,4,...> | 1×256×1 | 128×1×1 | 30 | 0 | 2.144 |

The mainloop records grid132×1×1/block256,255 registers/thread and56,320 shared
bytes; its nested estimate reports65,536 allocated registers and57,344 allocated
shared bytes per block. The GPU properties record65,536 registers/SM,
233,472 shared bytes/SM,232,448 opt-in shared bytes/block and49,152 default
shared bytes/block. Under the trace's static allocation model,
`floor(65536/65536)=1` block/SM, while shared-memory capacity alone permits
`floor(233472/57344)=4`. This is a **conditional static resource bound**, not
achieved residency, active warp time, eligible warps or a stall diagnosis.
Hopper warp-specialized `setmaxnreg` redistribution can make the reported maximum
registers/thread misleading as a description of each warp's live allocation.
No generated SASS or per-warp register-allocation trace for this FA3 binary was
saved, so do not convert255×256 into a measured execution bottleneck.

The trace's `blocks per SM` is visibly grid work/SM count: for QK norm,
576/132=4.363636, and for the custom grid704/132=5.333333. It is **not a
measurement of simultaneous CTA residency**. The mainloop value1 and warps/SM8
similarly do not establish achieved occupancy or persistent-worker tail loss.

The mainloop's nested occupancy fields paradoxically say activeBlocks0,
SMEM-limited and estimated achieved occupancy0%, despite successful kernel
execution. The reported56,320 bytes exceed the49,152-byte default block limit
but fit the opt-in limit; this default/opt-in mismatch explains why the estimate
can be invalid. The profiler's exact implementation was not recovered, so treat
that as a consistent explanation, not proof of its internal arithmetic. Zero
estimated occupancy is neither zero real execution nor a permission denial.

No real DRAM bytes/throughput, tensor utilization, active CTAs, eligible-warp
statistics, stall reasons or CTA service/tail timeline was obtained. The two
historic NCU attempts timed out at60s and55s, respectively; neither yielded an
admitted metric row. They remain timeouts, not evidence of permission denial or
a hardware bottleneck. No counter retry is proposed.

## Installed wrapper and extension binding

A later exact identity read in the **same immutable Modal image** found:

- Wrapper`/opt/sglang/lib/python3.12/site-packages/sgl_kernel/flash_attn.py`,16,160 bytes, SHA256
  `610747f2c68495c5f3ba67f1247f0a1b4e4f7df116e2f89cd8e8983f0271f9ac`.
- Loaded extension`/opt/sglang/lib/python3.12/site-packages/sgl_kernel/flash_ops.abi3.so`,1,122,619,328 bytes, SHA256
  `cdea0997a51412b8db8bf8da49d896a6c9d12a39bb1b548d840bb8dc428f13ef`.
- Python signature defaults: `num_splits=0`, `pack_gqa=None`, `sm_margin=0`,
  `ver=3`, `scheduler_metadata=None`. The current selected full FA3 helper does
  not override the first four options. Decode B16 passes actual serving-supplied
  scheduler metadata; verify B16/B32 passes None so the call prepares its schedule.
- The exact C++/CUDA source revision and all transitive GPU binaries were not
  identified by that receipt. The trace itself did not hash its loaded extension;
  the later same-image identity is useful provenance, not proof that the official
  judge has this exact binary or that every runtime template flag is mapped.

The saved wrapper offers a concrete, already-supported observability route.
Lines230–268 unpack `out, softmax_lse, *rest = torch.ops.sgl_kernel.fwd.default(...)`.
Line270 returns `(out, softmax_lse, *rest)` when`return_softmax_lse=True`.
That flag changes only the Python return container, not the arguments sent to
`fwd`. A separately audited invocation with identical prepared Q/cache/metadata
can record auxiliary tensor shapes/dtypes/schema without replacing loaded
functions. The meaning of each `rest` field still needs the actual schema/source;
do not label an arbitrary dimension as split count. The current whole-call
candidate and all timed calls should remain unchanged.

## Current official profiles: expensive stage known, specialization missing

Official5090 full paths currently use the proven fused preparation, unchanged
FA3 calculation and normal output gate. Profile reports show:

| Path | FA3 forward activity µs | Report total µs | What is absent |
|---|---:|---:|---|
| Full decode B16 | 297.3 | 304.2 | Full template, grid/block/resources, resolved splits |
| Full verify B16 | 193.3 | 204.3 | Same absent fields; visible prep template`<1,true>` |
| Full verify B32 | 372.2 | 384.1 | Same absent fields; visible prep template`<2,true>` |

The profile JSON contains only case/event/index/report for each profile; the
kernel names were truncated before export. No hidden resource/launch fields
can be recovered from it. A particularly important inconsistency remains:

| Same unchanged full-decode-B16 path | Benchmark calibrated µs | Benchmark raw µs | Profile forward / total µs |
|---|---:|---:|---:|
| Fusion job2f5 | 194.360665 | 198.256001 | 185.7 /192.1 |
| Composition job5090 | 194.271909 | 198.063999 | 297.3 /304.2 |

Reported case/seed metadata matches benchmark and profile **within** each job,
but the retained evaluator implementation and exact replay tensor state are not
available here. The stable benchmark and ~60% higher forward activity do not
identify whether profiling, fresh-state differences or a runtime dispatch/schedule
change caused the disparity. It is much larger than the ~4µs replay adjustment.
Do not infer a recoverable60% loss or convert activity shares to a bottleneck
fraction. This is a concrete reason to pair the next trace with same-graph,
same-fixture unprofiled measurements before and after collection.

## Whole-score leverage and optimistic bounds

Use calibrated official5090 benchmark rows, **not profile durations**, with
`C_load=46*sliding+16*full`. If only one full path loses fraction`r`, with all
other candidate and reference times fixed, score factor is
`(1-f*r)^(-1/6)`, where`f=16*full/C_load`. For all three paths, multiply their
three factors. These are conditional scenarios, not expected recoverable time.

| Full path | Whole-call µs | Full share of its load | Saved µs at5% /10% | Modeled overall gain at5% /10% |
|---|---:|---:|---:|---:|
| Full decode B16 | 194.271909 | 82.917588% | 9.713595 / 19.427191 | 0.708208% / 1.453088% |
| Full verify B16 | 197.602134 | 75.970038% | 9.880107 / 19.760213 | 0.647508% / 1.325555% |
| Full verify B32 | 380.380746 | 82.808451% | 19.019037 / 38.038075 | 0.707253% / 1.451076% |
| All three paths at the same cut | — | — | — | 2.077176% / 4.289581% |

An overall3% gain across all three would require about7.1248%
whole-path reduction in **each** under this fixed-other-path model. The actual
recoverable loss is unmeasured. Eliminating an activity stage or doubling
occupancy is not assumed to deliver any of those cuts.

## Highest-value next measurement and cheap falsifier

**No-go for a new algorithm from this trace alone.** One future bounded
same-geometry diagnostic would resolve the largest missing facts before code:

1. Use the exact selected source/hash and original oracle in the existing image.
   Fixed cases: full B16/L1, B16/L8 and B32/L8, TP8/D64/page1/unit scales,
   ctx80000/agentic/coherent; one fresh declared fixture per case. Save the actual
   lengths, config, page-table shape, source/image/driver/loaded wrapper+extension
   identity and supplied scheduler-metadata shape/dtype/hash. No cross-layer or
   cross-call task-map cache. Internal scheduling remains in each timed call.
2. Capture complete candidate entrypoints once, preserving input/cache/output
   addresses. Validate original output/new-KV before and after. Restore the same
   saved tensor contents and cold-L2 treatment around unprofiled CUDA-event graph
   measurements **before and after** one activity session containing one annotated
   replay per case. Keep these timings and instrumented stage durations separate;
   no profile-overhead subtraction or performance-variant comparison.
3. Export the untruncated kernel symbols and all launch-resource fields for all
   three cases. Optionally, in a separately labeled untimed supported-wrapper
   call with identical prepared tensors and fwd arguments, record raw returned
   `rest` tensor shapes/dtypes and operator schema. Interpret resolved splits only
   with an exact field mapping. If unavailable, retain “unknown”; do not invent
   a split count or modify dispatch to expose it.
4. Use the existing160s worker/210s watchdog envelope only after fresh serialized
   admission, free-credit/$0 paid verification and independent harness review;
   one app, one activity session, zero retries, no counters/new installs. This
   local report neither reserves funds nor authorizes a launch. Independently
   stop/verify tasks0 and release the slot before interpretation.

The cheap falsifier is **same-source/same-fixture invariance**: if the before/
after unprofiled graphs agree while the activity duration or specialization
changes, the trace is unsuitable for latency attribution; stop that inference.
If full symbols/options and auxiliary metadata do not expose a specific avoidable
work/dataflow/resource cost, do not build another kernel from the static255-reg
number or repeat the rejected split experiments. A later algorithm needs a
measured, source-bound mechanism capable of the required whole-path savings.

No new performance candidate, official check or GPU run is prepared or started
by this audit. Current selection and its immutable A2-11 rollback remain intact.
