# Prospective B1 SIMT compile falsifier

This local compile harness tests whether eight valid query heads can remain
unpadded in a strict rank2 SIMT producer while retaining actual K/V reuse.
Source-level eight-head loops and loads outside them are not machine-code
proof. The source-owned `../EXPERIMENT_CARD.md` and `../SOURCE_REVIEW.md`
remain the semantic/dispatch contract; current production3da is unchanged.

## Fixed comparison and payload

Exactly three units compile: `baseline.producer` (`_fused`),
`candidate.producer` (`_fused_b1_simt`) and `shared.merge` (`baseline._merge`).
The entire merge function's source bytes must match candidate `_merge` before
either module is loaded, so a second merge compilation is unnecessary.

Both producers use the same21 runtime arguments, B1/L1/TP8/D64/window2048,
page1/unit scales, RoPEtheta500000, eps1e-5, QKfactor5.4730064863875,
SMscale0.125, QKVstride2304 and PageTablestride80001. Geometry stays external
S16/BN64, grid(1,2,16), four warps/three stages. Baseline internal M16 becomes
candidate M8, while candidate `PARTIAL_M=16` preserves the partial layout and
the identical M16/S16/BS16 merge at grid(1,16). No geometry/precision sweep.

Compile pointers follow actual official types: QKV/output BF16, K/V E4M3,
history/PageTable int32, cache locations int64, scales/partials FP32. Typed
one-element allocations are deliberately undersized and must never execute.
Only explicit `--compile --output PATH --deadline 90` enables warmup and module
loading; default command is CPU source/merge preflight. No remote controller,
upload, GPU reservation, kernel invocation, numerical or timing stage is
performed by this preparation. Future payload is only worker, immutable
baseline/candidate and license/notices; metadata query helpers are self-contained.

## Resource budget declared before compilation

The root accepted a prospective **candidate producer cap192 registers/thread**,
zero spills/local-memory traffic, shared at most64KiB, and actual legal maximum
at least2CTA/SM at128 threads. Control/merge resource observations are retained;
there is no old96-register cap or requirement of register parity with MMA.

The engineering ledger distributes a single active FP32 Kf/Vf[N64,D64] tile
(32 words/thread), one rank2 FP32 product(32), Q/acc/PV/logit/P state(~24),
worst-case full64-bit[N,D] addressing(64) and scalar/control allowance(16):
about168 words/thread, rounded with margin to192. This is **not a proof of
compiler liveness, layout balance or register capacity needed by the32-CTA
grid**. Broadcast replication, unrolling and staging may exceed it and then
reject this exact source; no register-cap, stage, warp, tile or precision rescue.
The CUDA occupancy query gives a legal maximum, not achieved residency or a
concurrency guarantee. Unknown exact resource metadata is inconclusive.

## Mandatory native structural gates

1. Verify actual eight-head Q, online-softmax and accumulator state; no
   padded16-head compute and no[G,N,D] materialization. M16 partial addressing
   alone is intentionally preserved and is not padded arithmetic evidence.
2. Candidate producer must contain no Tensor MMA in TTGIR/PTX/SASS. FP8 Q/K
   encoding and explicit BF16 P/V boundaries stay fixed, with FP32 reduction/
   softmax/partial state. Different reduction ordering is not bitwise identity
   and requires the unchanged original oracle later.
3. Map global K/V load addresses and head reuse in generated code: do not
   accept eight full payload reloads per head merely because source loads
   occur outside `static_range`. Retain matched logical work/load-volume
   evidence and all raw TTIR/TTGIR/LLIR/PTX/CUBIN/SASS.
4. Verify original window masks, empty splits, unique last-split new-KV stores,
   M16 partial indexing and identical merge; then apply the declared resource
   screen. Missing readable ISA/dataflow is inconclusive, not a pass.

Resource-screen success only marks independent structural review pending; no
native invocation or benchmark follows automatically. A structural/resource
failure rejects this source without a tuning sweep. Artifacts are bounded to
18 fixed filenames,4MiB/file,8MiB raw archive and2MiB final report.

Only after all native gates pass may a separately scoped original-tolerance
oracle/cache/replay/isolation stage be reviewed. Its prospective timing bar is
at least15% complete-call reduction against the actual current best3da on each
fresh target fixture, with full controls and all preparation/stores/merge/output
work included. No present speedup, official score or numerical pass is claimed.
