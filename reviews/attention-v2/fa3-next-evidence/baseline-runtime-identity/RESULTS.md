# Current selected FA3 identity recovered; diagnostic protocol incomplete

One unchanged-baseline Modal batch ran in app `ap-G4ZFD70w5GGIZfRCYV4ehw`,
accepted 2026-10-27 03:55:34 UTC. The child exited 1; remote-entry elapsed time
was 45.678 s and the internal worker reported 43.306 s. Its
CPU-time-window parser excluded the B32 mainloop. The complete bounded raw
trace and identity report were retrieved. No repeat run was launched.

The exact app was independently verified **stopped, tasks 0**, at 03:57:07 UTC;
all listed apps were stopped with zero tasks. The measurement slot was released.
The controller's earlier 03:56:27 observation still saw `stopping...`, tasks 1;
the later independent receipt reconciles that transition without another run.
The selected 1.2336330588627955× candidate and A2-11 rollback remain unchanged.
No official check, formal entry, credential action or paid resource was used.

## What actually executed

Frozen diagnostic commit: `43c491e19baa9873a138dc5d0de59329de3f3407`.
Selected source SHA256:
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`.
Image `im-cPTbGpNiX3RarKResPgX7h`; H200/SM90 with 132 SMs,
Python 3.12.3, Torch 2.14.1+cu130, Triton 3.8.0, CUDA 13.0.

The loaded public `sgl_kernel.flash_attn.flash_attn_with_kvcache` wrapper is
SHA256 `610747f2c68495c5f3ba67f1247f0a1b4e4f7df116e2f89cd8e8983f0271f9ac`.
Its native operator schema has 37 arguments; scheduler metadata, num_splits and
pack_gqa are arguments 31/32/33. The selected verify calls retain metadata None
and public defaults num_splits 0/pack_gqa None. Those are API inputs, **not** the
resolved native split count or packing decision.

`flash_ops.abi3.so` is 1,122,619,328 bytes, SHA256
`cdea0997a51412b8db8bf8da49d896a6c9d12a39bb1b548d840bb8dc428f13ef`,
GNU build ID `e40ad4a2cb6c0e7d20e7d43f5d5c6647715f3f01`.
The loaded mapping and hashed file have matching inode 2100. This establishes
the current Modal file/loaded mapping identity. It does not establish an upstream
source revision or the official judge's binary. SGLang reports 0.0.0.dev0;
sgl-kernel distribution metadata is absent despite its loaded extension.

Both original generated cases are full verify, L8/B16 or B32, TP8, D64, page1,
ctx 80000, agentic lengths/coherent inputs/unit scales/seed 135741. Complete actual
length vectors, configuration, tensor metadata and page-table hashes are saved.
Both mainloops have the same complete 1380-character symbol: SM90 forward,
`CollectiveMainloopFwdSm90` with tuple `64,128,64`, FP8 E4M3 type names,
BF16 epilogue and `VarlenDynamicPersistentTileScheduler`. Each launches
grid `(132,1,1)`, block `(256,1,1)`, 255 registers/thread, 56320 shared bytes.
Thus the current observed specialization is not supported by merely assuming
the public 192×160 default. Complete symbol text is authoritative; opaque template
booleans are not relabeled as PackGQA or split policy without their definitions.
The logical packed row count is 8 queries×8 GQA heads=64; this alone does not prove
the resolved packing flag or any utilization ratio. QK/PV machine instruction
types, actual split count, achieved occupancy, traffic and stall reasons remain
unmeasured. In particular the trace's zero estimated occupancy is invalid as an
achieved occupancy conclusion.

## Timings and protocol failure

Two initial original-oracle calls and ten warmup/pre-profile replay invariance
checks passed, including output/new-KV byte equality, stable addresses, immutable
inputs and unchanged non-new cache rows. No tolerances were changed.

| Case | Three pre-profile whole-call CUDA-event samples, µs | Median, µs | Mainloop activity, µs |
|---|---|---:|---:|
| Full verify B16 |199.424,199.552,200.320|199.552|187.621|
| Full verify B32 |385.472,383.552,383.104|383.552|365.462|

These are selected-source measurements on one fixed fixture per case, with 256 MiB
eviction outside events and no empty-graph subtraction. They are not an official
score, candidate gain, cross-run comparison or confidence interval.

The one Torch/CUPTI activity session contains five graph kernels per case:
fused preparation, scheduler preparation, FA3 mainloop, combine and norm/gate.
The original worker associated kernels by CPU annotation timestamp containment.
B32's CPU marker starts about 39.960 µs **after** its GPU mainloop timestamp,
so that parser sees zero matching mainloops. It also includes unrelated B16
restore copies under that temporal rule. This is an attribution-method failure,
not a missing kernel, numerical failure, counter-permission denial or evidence
of a hardware bottleneck.

Offline attribution instead follows the annotated CPU `cudaGraphLaunch`
correlation 113/232, graph ID 2/5, and matching GPU annotation External ID 16/32.
This recovers one mainloop and all five stages for each case. Small apparent
same-stream overlaps remain in the raw activity timestamps; do not infer actual
overlap, critical-path savings or precise residency from them. No timestamps or
raw artifacts were edited. The frozen worker remains unchanged and its raw
`incomplete_or_failed` status is preserved.

Because parsing raised before later stages, **no** post-profile oracle,
after-profile sample, auxiliary-return replay, NCU metric query or hardware-counter
session executed. Whole-call profile invariance, resolved auxiliary split shapes
and hardware-counter capability remain unknown. Offline recovery does not turn
the unfinished protocol into a complete PASS.

## Tools, costs and next prerequisite

Fresh inventory found NCU 2025.3.1, NSYS 2026.4.1, CUDA 13.0 cuobjdump/nvdisasm,
and supported Torch CPU/CUDA activities. `RmProfilingAdminOnly:0` was observed.
That setting and executable presence do not prove counter access. The two older
NCU application-replay/graph sessions timed out; they were not permission denials.
This batch did not reach its distinct eager/kernel-replay capability test.

Before admission the billing page showed $27.51 free credit and $0 Spend limit.
The reserved $0.70 was a conservative allowance, not a charge. At independent
cleanup the observed account ephemeral-cost breakdown rose by $0.06505812;
the controller gross-summary account delta was $0.06162934. These are provisional
account-level metering observations, not a final per-app invoice. A 55 s app
lifetime times the configured rate models $0.07178417. Actual billed spending
remained $0. See `COSTS.json` and `independent-cleanup.json`.

Next useful prerequisite is the already reviewed deterministic eager
single-counter capability gate, separated from the failed activity parser, then
one unchanged FA3 target only if that gate succeeds. First fix and regression-test
attribution offline against this saved trace. No new architecture, candidate
promotion, permission change or repeat run is justified by this partial result.
