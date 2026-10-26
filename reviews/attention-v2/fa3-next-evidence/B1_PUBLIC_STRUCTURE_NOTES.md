# B1 short-window structure: one execution-family hypothesis, two causal stages

Read-only local/public source audit, 2026-10-26 UTC. No implementation, GPU,
service, installation, credential, official check or commit. Public sources
describe legal structural possibilities; they do not identify competitors'
implementations or prove performance. CUDA/source eligibility remains a
separate audit before any implementation.

The credible difference is **unpadded, vector/SIMT work partitioning for tiny
query groups**, optionally followed by splitting the GQA group to expose more
CTAs. Merely adding GQA reuse, correcting row indexing, pruning the window or
removing the merge launch is not a new hypothesis here.

## Exact current route

Control is [fusion-lookahead candidate](../fusion-lookahead-probe/candidate.py),
SHA256 `3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`.
The audited target is B1/L1, sliding2048, TP8, HQ16/HKV2/GQA8, D64, page1.
Source lines84–330 already share K/V across the eight query heads, map
`r=j*G+g`, quantize Q through BF16 to E4M3, prune historical positions to the
exact window, and fuse normalization/RoPE/new-KV preparation in the producer.
Its merge already fuses output RMSNorm and sigmoid gate.

The producer has grid `(1,2,16)` =32 logical CTAs, BM16/BN64. Eight of16 query
rows are valid, but the dot/state dimensions and partial stores remain16.
Its FP32 partial allocation is135,168 bytes; valid rows account for67,584 bytes.
This proves padding and storage capacity, not that half the whole latency is
wasted. With a full window there are2047 historical keys plus the new key;
each split handles at most128 historical keys. The last split remains the
unique new-KV cache writer. Sixteen per-head CTAs perform the separate merge.

[B1 route audit](B1_ROUTE_AUDIT.md), SHA256
`9f8744baefff2f0a72f85607bddd62ed272022b5fd18d7eff13411bf3809b23b`,
records11.477114us calibrated whole-call time. Its rounded instrumented
producer/merge activities are7.8/1.7us; their sum is not a measured critical
path and cannot be subtracted from calibrated timing.

## What the public implementations actually contribute

| Fixed public source | Relevant structure | Consequence for this target |
|---|---|---|
| [FlashInfer v0.7.0 decode.cuh](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.0/include/flashinfer/attention/decode.cuh#L58) | Vector FP32 QK reduction; feature lanes `bdx`, GQA heads `bdy`, KV lanes `bdz`; `sync_state` merges lane states inside a CTA | An execution family without a mandatory16-row matrix tile exists. Its arithmetic boundaries are not a Muse drop-in. |
| [SGLang v0.5.5 grouped decode](https://github.com/sgl-project/sglang/blob/v0.5.5/python/sglang/srt/layers/attention/triton_ops/decode_attention.py#L269) | Head-subgroup indexing is parameterized by `BLOCK_H`; shipped host path uses16; separate per-head stage2 | Source supports the subgroup mapping concept. Its default resembles our padding/two-stage structure, so copying it gives no established advantage. |
| [Original Flash-Decoding explanation](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) | KV splitting adds parallelism, followed by state reduction | Long-context gains do not establish the best decomposition for a2048-token window. |

FlashInfer's [intra-CTA sync](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.0/include/flashinfer/attention/decode.cuh#L149)
does not fuse arbitrary external CTAs. Its [single-decode dispatch](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.0/include/flashinfer/attention/decode.cuh#L660)
still chooses partitioning and a separate merge for longer inputs. The public
[cascade dispatcher](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.0/include/flashinfer/attention/cascade.cuh#L603)
also distributes a many-split merge per output head; that structural feature
is already present in our current merge.

## Stage A: does avoiding the M16 matrix tile help this tiny row group?

**Single causal hypothesis:** a vector/SIMT producer handling only the eight
valid heads can reduce padding/layout/reduction overhead enough to offset
losing tensor-core throughput. Preserve external split16, exact128-key split
boundaries, producer grid32, partial interface and the existing16-CTA merge.
First compare at identical G8 geometry, rather than mixing a subgroup change
into the first timing result. Compact only the unused query-row state; this
does not introduce new GQA reuse.

The concept must retain the original BF16 normalization/RoPE and Q E4M3
rounding, FP8 cache, QK FP32 accumulation, BF16 P/V operands with FP32 PV
accumulation, FP32 softmax/partials, scales and new-KV ownership. Public
FlashInfer's vector P update cannot simply replace the explicit BF16 P
rounding boundary. Changed reduction order is not bitwise equivalence and
must pass the original oracle. No approximate attention or alpha threshold.

**Cheap falsifier:** first audit allowed implementation facilities, then one
source/artifact comparison: actual active head lanes, eliminated padded
state, coalesced gather, shared-buffer lifetime, registers, spills/local
traffic and synchronization. Reject if padding merely moves into another
layout, the required precision boundary is lost, or scalar work/reduction
resources invalidate the rationale. Compilation is not timing evidence.
If admitted later, use a small fresh-fixture correctness/replay check before
one paired complete-call comparison against unchanged3da; retain identical
merge, window and split geometry as the causal control.

## Stage B, conditional on a viable A: buy CTAs with a smaller GQA subgroup

**Different causal hypothesis:** at this tiny batch, additional independent
head groups can outweigh duplicated K/V loads. Using the same admitted SIMT
primitive, compare G8 with exactly one G2 mapping, still split16:128 producer
CTAs instead of32. Index heads as `kvh*8 + subgroup*2 + lane`; do not alter
token indexing, window or external partitioning. Keep output merge per head.
An illustrative128-thread layout is feature16 × head2 × KV4, with each KV
lane carrying its own FP32 state; actual feasibility needs source/codegen
proof. This layout is our proposal, not a published tested Muse kernel.

Costs are explicit: up to4× logical historical K/V reads across subgroups,
more page-table work/CTA setup, and intra-CTA state reduction. It retains
reuse between two heads and never caches input-derived work. Only one
subgroup may store new KV; every subgroup must still attend to the same
properly quantized new key/value. If preserving that ownership requires
cross-CTA consumption or an added preparation launch, account for it inside
the complete call and reconsider the card.

**Cheap distinguishing test:** analytically verify coverage/unique stores,
then compare G8 versus G2 with the same SIMT primitive and split16. Prove
the expected CTA/state/byte changes before execution. Reject without a
split/subgroup sweep if native whole-call benefit fails or extra loads erase
it. Observed CTA count alone is not achieved residency or a bottleneck proof.

Both stages require unchanged original tolerances, scale1 and1/64, page1/64,
shuffled mappings, empty/short history, window edges2047/2048/2049, new-KV
bytes, immutable historical cache/input checks, live graph length/page
mutation and independent output/scratch ownership. Timing must be
uninstrumented, cold-L2, paired and confirmed on a fresh seed; controls must
include an unchanged route. No performance promise is made.

## Why merge integration is not a third ready candidate

Collapsing all16 external splits into intra-CTA KV lanes at full G8 leaves
only2 producer CTAs. The old last-arrival path likewise had only2 finalizers
and serialized split merging; the measured move to the current parallel
merge improved the route. Neither is a justified return experiment.
Cross-CTA fusion preserving all32 producers needs an independently legal,
race-free synchronization design; public intra-CTA `sync_state` is not that
design. It must not silently add spinning barriers, task-map caching or
unmeasured counter initialization. Fusing more output heads in a merge also
trades16 CTAs for fewer; RMSNorm/gate are already fused.

Coverage is deliberately narrow: these cards address only the scored B1/L1
sliding path initially, not full attention, L8 verify or all B16/B32 paths.
With all other selected times fixed, even an impossible zero target time
would multiply the whole score by only1.142215; halving it models1.055022.
These follow from weights46/16 and six-load geometric scoring, using the
current B1 sliding/full times11.477114/27.031303us. They are conditional
coverage bounds, not achievable speed forecasts. This can be one component
of the broad gap plan; it cannot explain or close the public leader gap
alone. No claim about private leader algorithms follows.

Ordinary FlashInfer host planning and XQA page1 rejection remain the
[prior eligibility blockers](FLASHINFER_ELIGIBILITY_NOTES.md); these notes
propose original kernel structures, not a workaround adapter or an installed
API claim. No implementation or launch is authorized by this audit itself.
