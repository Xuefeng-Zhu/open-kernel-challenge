# B1 short-window budget: work, parallelism and remaining evidence

Local/source-only audit,2026-10-26UTC. No kernel edit, GPU, official check,
private competitor access or service action. Public scores cannot identify a
competitor's implementation. The analysis below identifies legal structural
possibilities and their costs, not an explanation established for any leader.

## Exact selected route and work

Selected source is `../fusion-lookahead-probe/candidate.py`, SHA256
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`.
For scored B1/L1/sliding2048/page1/TP8, HQ16,HKV2,D64,GQA8, the dispatch reaches
`_e2_kernel_fn`, **not FA3**. Source already has GQA reuse, `j*G+g` mapping,
BF16→E4M3 Q encoding, strict accumulation, exact window pruning, fused
norm/RoPE/new-KV work, and merge-fused output RMSNorm/gate. These are not new
optimization hypotheses. See the separate [B1 route history](B1_ROUTE_AUDIT.md).

`_num_splits` at281–285 gives `min(ceil(264/2),floor(2048/128))=16`;
the source calls32 producer CTAs `(1,2,16)` and16 per-query-head merge CTAs
`(1,16)`. M16 contains eight valid query heads. For scored history80000,
`lo=seq-2048+1`, so historical span is2047 and new-token count is1.
Partition width is128: fifteen128-key histories plus one127-key history.
Each producer executes two64-token historical tiles. Only the last split
normalizes/quantizes/stores/attends the new K/V token; it cannot be deleted
when its historical interval is empty. B1 short-history tests can leave many
empty splits because allocation depends on the window, not live lengths.

### Static parallelism ceiling, not achieved occupancy

There are32 useful historical64-token tiles per KV head,64 in total. Increasing
split count alone cannot create132 independent useful history tiles at BN64.

| Illustrative geometry | Producer CTAs | Nonempty history CTAs | Tail-only CTAs | Historical tiles total |
|---|---:|---:|---:|---:|
| Current G8,S16 |32|32|0|64|
| G8,S32 |64|64|0|64|
| G8,S64 |128|64|2|64|
| G4 subgroups,S32 |128|128|0|128, duplicated across head groups|

Current producer can occupy **at most32 distinct SMs** at one instant,24.24%
of132 SMs, irrespective of achieved CTA residency. This is a grid-size upper
bound, not a measured utilization figure. The merge similarly exposes16 CTAs.
At S64 there are62 wholly empty CTAs plus two tail-only CTAs: the empty ones
still execute Q preparation and partial writes. Reporting128 CTAs as128 useful
history workers would be wrong. The actual source's `most` bound prevents this
configuration. This table is analytical, not a split sweep or recommendation.

## Logical payload and repeated state

Counts use actual FP8 cache, int32 page table, BF16 Q/gate/output and FP32
partials. They are source-level logical bytes, **not measured HBM traffic**;
cache reuse, transactions, masked execution and compiler CSE are unknown.

| Current B1/D1 sliding item | Bytes / work |
|---|---:|
| Historical K+V payload:2 KV×2047×64×2 |524032B|
| New FP8 cache writes:2 KV×1×64×2 |256B|
| Page mappings across two KV groups:2×2047×4 |16376B|
| Unique Q projection input |2048B|
| Q source loads incl. RoPE partner, repeated over16 splits |65536B; one preparation per KV group would read4096B by the same source formula|
| Producer partial writes:32×M16×(D64+2)×4 |135168B =132KiB|
| Merge reads only valid query rows:16 heads×16 splits×66×4 |67584B =66KiB|
| Partial write+read total |202752B =198KiB|
| Gate input / BF16 output |2048B /2048B|

There are256 valid query-row norm/RoPE/FP8 preparations versus16 distinct
query rows. Source dot shapes execute64 historical M16×N64×D64 QK/PV tiles,
plus two M16×N16×D64 new-token tails:16,908,288 shaped FLOPs versus8,388,608
useful attention FLOPs. This is about2.016× source-shape arithmetic, not a claim
of2× hardware time: FP8/BF16 tensor throughput, padding optimizations, trig,
reductions and register pressure differ. The padded tail contributes little.

At the official nominal4.8TB/s,512KiB K+V is0.109us and198KiB partial
traffic0.042us. Those ideal streaming calculations neither explain nor bound
real dependent page gathers or startup at this grid. They show why labeling
the11.48us call "HBM-bound" from byte count alone is unsupported.

### Cost of the illustrative G4/S32 mapping

Splitting each existing G8 group into two G4 groups buys128 one-tile history
CTAs, while retaining reuse across four heads; **GQA reuse already existed**.
It duplicates historical K/V and page loads2×. If the current M16 primitive
and full partial layout remain, historical/tail shaped arithmetic doubles,
partial writes rise132→528KiB, merge reads66→132KiB, and valid Q preparations
double256→512. Total partial traffic becomes660KiB,3.33× current. Each CTA has
only four valid rows. A genuine unpadded primitive/compact state could change
these costs, but must be proved, not assumed. New-KV writes need one unique
subgroup owner; all subgroups must still attend correctly quantized new KV
without an unsynchronized cross-CTA cache-read dependency.

## What is actually measured

Current official check5090 passed16 prerequisites/12 shapes. Profile JSON
SHA256`3a9f9f841665ef0c678001aa62396b44557a375ce20d5e830548d257ff6f1300`
records B1 sliding `_fused`7.8us and `_merge`1.7us, one call each. Its separately
calibrated whole-call time is11.477114us versus20.576001us reference. The
instrumented9.5us sum is not the calibrated whole-call time. No current B1
grid/block/register/shared fields, CTA residency, DRAM/tensor counters or
launch-gap timeline is present in that profile; source-derived CTAs above
must not be presented as traced launch metadata.

Historical A13 ownership report SHA256
`8af336b264806bc39f12875a500d26e9777645660012accf5d71452c08ebbb41`
has B1 sliding activity7.740us producer, then merge starting8.316us and lasting
1.951us. It is truncated to first intervals and used A2-11's older QK source,
not current3da; its0.576us interval gap is **not** a transferable launch-cost
measurement. Cross-graph overlap was not observed there, which does not prove
serial execution or safety in general. No current raw B1 hardware trace was
found in the inspected saved evidence.

Two past experiments constrain the next ideas. [A2-12](../A2-12-RESULT.md)
already hoisted repeated Q work to a timed preparation launch: producer
activity8.1→6.8us, added preparation2.5us, and target whole-call normalized
latency **regressed8.676%**. Preserve that rejection; no duplicate Q-hoist or
prologue-fusion proposal follows from the repetition count. [A2-11](../A2-11-RESULT.md)
replaced a serial separate merge with the current per-head parallel reduction;
B1 normalized latency improved25.868% against A2-10. Generic serial/last-CTA
merge is historical, not a fresh mechanism.

Measured evidence establishes a material producer and a remaining merge
activity. It does **not** distinguish fixed startup/trig/layout costs from
limited memory-level parallelism, useful SM exposure, reductions or spills.
The [already prepared diagnostic](DIAGNOSTIC_CARD.md) includes this exact
custom B1 stratum; any future same-fixture activity/resource observation should
answer that uncertainty without inventing another broad profiler sweep.

## Two conditional structural hypotheses and cheap falsifiers

1. **Unpadded short-query execution family.** A vector/SIMT primitive can map
   feature lanes, valid GQA heads and KV lanes without a mandatory M16 tile;
   public [FlashInfer decode source](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.0/include/flashinfer/attention/decode.cuh#L58)
   supplies this structural precedent. First retain G8,S16,32 CTAs and the same
   merge/ownership to isolate padding/layout cost. Preserve explicit Q E4M3,
   BF16 P/V and FP32 accumulations; public SIMT P arithmetic is not a drop-in.
   Cheap falsifier: allowed-source audit and compiled active-lane/resource/
   reduction proof. Reject if tensor throughput is simply replaced with more
   expensive scalar work, precision boundaries change or padded state remains.
   This is a structure hypothesis, not an installed FlashInfer adapter.
2. **Buy useful CTA exposure with a smaller query subgroup.** G4/S32 is a
   concrete costed illustration, not a sweep. A prospective matched mapping
   test would compare G8 versus G4 at the **same S32 and same admitted primitive**,
   retaining current G8/S16 as the deployment comparator. First prove CPU
   coverage, the byte/partial changes above and unique new-token stores; then
   use actual native dataflow/resources to reject unhelpful padding or loads
   before timing. No current measurement proves the extra CTAs recover time.
   [Public grouped decode source](https://github.com/sgl-project/sglang/blob/v0.5.5/python/sglang/srt/layers/attention/triton_ops/decode_attention.py#L269)
   supports subgroup indexing; its default M16/two-stage structure is already
   similar to ours, so copying the default is no new optimization.

Merge integration is **not a third ready candidate**. Folding16 external
splits into one CTA per KV head leaves two CTAs and sacrifices the parallelism
just identified. Public intra-CTA state merge does not combine arbitrary CTAs.
Preserving32 producers while removing the global merge requires a separately
legal, race-free synchronization/lifetime design and timed initialization;
no spinning barrier, new global cache or old serial finalizer is proposed.
The B16/B32 FA3-unsplit failure does not measure B1 custom behavior, but it
reinforces that launch removal alone is not evidence. See
[public-structure notes](B1_PUBLIC_STRUCTURE_NOTES.md) for source qualifications.

## Reach: a component, not the public-leader explanation

B1 sliding is54.96885% of the decode1 weighted load. With all other selected
times fixed, halving this target models a1.055022× whole-score factor; even
impossible zero latency gives only1.142215×. Making the instrumented1.7us merge
free **under an unproven direct-latency-subtraction assumption** models only
1.014255× overall, and cannot be claimed from the profile. If all six sliding
paths were hypothetically50% faster, current score1.233633 models1.478142;
even deleting all sliding times models1.902072, below the parent's22:57UTC
leader snapshot2.9097. These are coverage scenarios, not predicted gains.

Full B1 history80000 carries about20.48MB K+V, roughly39× this2048-token
window, and currently uses FA3. Batched full paths carry still more data.
Their reuse, pipeline and decomposition need a separate plan; aggressively
duplicating KV to buy short-window CTAs cannot be blindly transferred there.
No private leader algorithm, missing GQA feature, unfair method or universal
hardware bottleneck is inferred from the leaderboard gap.
