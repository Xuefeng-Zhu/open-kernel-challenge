# One B1 falsifier: unpadded G8 SIMT producer

**Choose execution family as the primary hypothesis. Defer G2 subgrouping.**
Compare one dedicated G8 SIMT producer with actual selected3da's MMA producer,
keeping SPLITS16, the same128-key historical chunks, BN64 softmax blocks,
32 producer CTAs and the unchanged16-CTA parallel merge. The128-key width is
for the scored full window; shorter histories retain the original live-span
formula `ceil(ceil(span/16)/64)*64`. This is credible as
a source/code-generation falsifier now. It is not yet a performance-admitted
candidate: saved measurements do not resolve producer instruction/layout cost
versus memory-level parallelism or startup. No implementation or GPU job is
part of this card; Projection owns the measurement slot.

## Evidence, mechanism and fixed boundary

Control: `../fusion-lookahead-probe/candidate.py`, SHA256
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`.
Target B1/L1, TP8,HQ16/HKV2/GQA8,D64,page1,sliding2048. Current M16 has
eight valid query rows; padded matrix work/state and partial stores remain.
The full window is2047 historical keys plus one new token. GQA reuse, exact
window pruning, Q FP8 encoding, preparation fusion and gate fusion already
exist. The question is whether a valid-row execution layout offsets the loss
of tensor-core throughput and extra scalar reductions.

Current official5090 whole-call adjusted latency is11.477114us. Separate
instrumented producer/merge activities are7.8/1.7us; their sum is not the
whole-call metric. [B1 budget](B1_WINDOW_BUDGET.md) binds those observations,
the source bytes and historical rejected query-hoist/serial-merge experiments.
The [public FlashInfer vector structure](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.0/include/flashinfer/attention/decode.cuh#L58)
supports feature/head/KV lane partitioning as a structural precedent; its
arithmetic and library API are not a ready Muse adapter.

## Smallest proposed producer schedule

Use feature16×head8×KV1=128 threads. Four feature elements per lane cover D64.
Each CTA cooperatively gathers/stages each K/V tile **once**, then reuses it
across heads; independent head groups must not accidentally turn this first
experiment into eight duplicate global KV reads. No full-cache repack.

Preserve each original N64 block: form only logits[8,64] FP32 (2KiB shared),
complete its rowmax/alpha and FP32 probabilities/denominator, then round P to
BF16 into P[8,64] (1KiB shared). Stream staged V through four output features
per lane into a separate FP32 PV accumulator, then apply the original
old-numerator rescale/add. Avoid a materialized `[G,N,D]` outer product or
64 logits in every thread: `[8,64,64]` FP32 alone is128KiB and would require
256 registers/thread at128 threads before other state. Small shared buffers
have explicit publication/reuse barriers, not per-key CTA barriers.

Per-key QK feature reductions/PV accumulation are allowed implementation
steps, but **per-key online softmax is excluded**: it changes the normalizer
at which P is rounded BF16. The current64-token rowmax-before-P-cast boundary
must remain. The current masked BLOCK_L16 new-token tail remains a distinct
tail update. Preserve BF16 norm/RoPE, Q BF16→E4M3→exact widening before scale,
FP8 KV and scales, BF16 P/V with FP32 PV, FP32 QK/state and every visible key.
Changed FP32 reduction order is not bitwise equivalence and needs the original
oracle. No approximated attention, rescale threshold or tolerance relaxation.

Write only valid g0..7 into the **old BM16 partial addresses**:
`((kvh*16+split)*16+g)*(D64)` for numerator, with matching max/sum indices.
Keep the135168-byte allocation and existing merge; it already reads only
valid rows. Producer writes fall132→66KiB, merge reads remain66KiB. Unwritten
padded rows must never be consumed; poison them in local/replay checks.
Last split alone computes, quantizes and writes new KV, exactly as control.

## Deferred G2: explicit cost, no simultaneous second hypothesis

Only after the G8 execution-family question is resolved could the same SIMT
primitive use feature16×head2×KV4=128 threads, S16/grid128. That is a separate
subgroup/parallelism experiment. It duplicates historical KV/page loads4×:
524032→2096128 logical KV bytes. Intra-CTA KV-lane reductions and locally
computed new-KV work also cost time. It is not new GQA reuse.

With compact valid writes, each subgroup writes disjoint
`g=subgroup*2+local_g` into the **same old BM16 addresses**. Allocation remains
135168B; total valid producer writes67584B and merge reads67584B remain the
same as compact G8, rather than growing4×. Every subgroup locally computes
the correctly rounded new K/V it attends; only subgroup0,lastsplit stores
the physical cache entry. Reading another subgroup's global new-KV store
without synchronization is excluded. No G2 implementation/sweep is proposed
as part of the first falsifier.

## Cheap rejection gates and honest opportunity

1. **Before native work:** source ownership/lifetime and AST checks must prove
   fixed G8/S16/window/chunks, cooperative single-copy KV, bounded live shapes,
   original tile-softmax/P rounding, old partial addressing, unique cache
   writer and unchanged merge/other dispatch. Stop if the language/layout
   cannot express this without the outer-product register expansion or a
   per-key softmax rewrite. Rules/tool eligibility is audited separately.
2. **One future compile falsifier:** only if that source passes, inspect actual
   coalesced gathers, SIMD/shuffle reductions, live registers, shared buffers,
   synchronization and partial stores. Require no spills/local-memory traffic
   or accidental padded-M16/duplicated-global-load lowering. Account for64
   half-warp QK reductions per tile,64 streamed PV steps, shared traffic and
   CTA barriers; removing padding does not make these free. No geometry,
   warp/stage or precision rescue sweep. A resource pass alone is not speed.
3. **Before any timing:** original tolerance/oracle, cache bytes/untouched rows,
   nonunit scales, page1/64, history0/partial/window2047–2049, live length/page
   replay and independent scratch/output checks. Both arms include all normal
   preparation, append, merge and gate work inside complete captured calls.
   Existing query hoist, counter initialization or serial-finalizer rescue is
   not added. No cache/task-map work moves outside timing.
4. **Future useful-result gate:** require at least15% whole-call target reduction
   on both fresh fixtures, with an unchanged-route control, before considering
   an official check. That means about9.756us versus the saved11.477us anchor;
   actual paired control times determine the comparison. Source/compile
   success alone never triggers timing or official admission.

Purely illustrative halving of the7.8us **instrumented producer**, if it mapped
one-for-one into whole-call latency, saves3.9us and models only1.035061× overall
score (1.233633→1.276886). That mapping is unmeasured. The15% target-only cut
models1.014445× overall (1.251453); even impossible zero target latency is
bounded by1.142215× overall. No producer stage fraction has been measured
well enough for a latency forecast. This B1 sliding component cannot close
the public leader gap alone and does not generalize to80K full or L8 paths.

The existing B1 activity/resource diagnostic remains useful before a timing
decision; another measurement is not necessary to reject an unimplementable
source/layout cheaply. If the compile costs or native instructions provide
no credible advantage, stop this execution family without a benchmark.
All work in this card is prospective; selected3da and old artifacts stay fixed.
