# Architecture plan: separate primitive viability from production gain

**No genuinely missing FA3 page1 mechanism has been proved.** The executed saved
full-B32-verify symbol already uses a staged SM90 mainloop and
`VarlenDynamicPersistentTileScheduler`. Public FA3 has non-TMA paged handling;
the installed wrapper already accepts the workload. Dynamic persistence,
staging or WGMMA presence is therefore not a new optimization by itself. The
exact current per-path specialization/runtime split mapping and installed
C++ revision remain incomplete. We have no admitted bandwidth, tensor-use,
residency or stall counters establishing the bottleneck.

The working candidate remains fusion plus lookahead, SHA256
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`.
A2-11 remains its rollback. A separately scoped local matched Gluon primitive
prototype is now permitted to test implementation viability. This is not a
production dispatch change or proof of a missing FA3 mechanism. This plan starts
no service, GPU experiment or official check. It does not reopen alpha-one,
blind split removal, dynamic-reorder or the rejected immediate-wait BF16 path.

## Required scale and complete coverage

Rounded public leader2.9097 / our nonranking1.233633 score =2.358643,
equivalent to57.60% uniform weighted-load reduction. Improving only the three
large FA3 full paths cannot close that gap with other paths frozen: even their
impossible zero-time model limit is2.81676. All six full paths at nominal TASK
floors model2.35851, still short. These are conditional arithmetic, not true
hardware ceilings or attainable performance claims.

The strategy must cover **all eight B16/B32 paths and all four B1 paths**.
Halving the eight large-path times plus cutting four B1 times70% models2.92527,
without crossing TASK floors. With large paths halved, B1 needs69.52% to match
the rounded point. This is a broad engineering target, not a forecast, a ready
algorithm or a reliable winning margin. See [GAP_MODEL.md](GAP_MODEL.md).

| Coverage | Separate question to investigate | First rejection condition |
|---|---|---|
| Four large full paths: B16/B32 × decode/verify | Can an eligible page-gather/data-reuse/overlap schedule improve an independent implementation and eventually beat the actual working path? | Repeating a mechanism already present is insufficient for novelty or production gain; no net benefit is established |
| Four large sliding paths: B16/B32 × decode/verify | Can a low-latency window mapping reduce demonstrated per-call preparation/launch/gather/merge cost while retaining useful parallelism? | Work is already fused or proposed saved overhead is small relative to required family gain |
| Four B1 paths: sliding/full × decode/verify | Can a dedicated low-latency complete-call mapping substantially reduce fixed phases and small-query inefficiency? | Only a tiny epilogue saving or an uncosted padded tile is offered |

These are hypotheses, not three manufactured candidates. Audit each actual
route first: existing preparation fusion and custom routes must not be proposed
again as missing work. Full L8/G8 naturally has64 query rows; L1/G8 has only8.
An M64 verify result cannot be transferred to M8 decode by padding. Decode needs
its own useful-work, masking, gather, reuse, merge and resource model; combining
unrelated requests also needs an explicit lawful mapping and cost proof.

## Ordered falsifiers and gates

1. **First executable falsifier completed: CPU image identity, no GPU.**
   `image-inventory/result.json`, SHA256
   `91c05d95ba02b797ff12d46c7a57318cdfba0b9741f0d23a430de371dd734d78`,
   confirms Triton3.8.0 Gluon/Hopper `warpgroup_mma`/wait and pointer-based
   `async_load` source/signatures are present. This removes source-availability
   uncertainty, not native precision/lowering or performance uncertainty.
   Installed FlashInfer0.7.0.post1 XQA rejects page1 via whitelist16/32/64/128;
   stop that direct adapter. Page64 relabeling/full KV repack is not a cure.
   FA3 wrapper hash matches prior610747… evidence, but none of the five fixed
   FA3 C++ source directories exists. SGLang's module origin was omitted by
   the local path guard, not denied by the service. Exact FA3 revision remains
   unknown. The CPU app was stopped/tasks0; no GPU context or kernel ran.

2. **Define the strict primitive contract before the local microprototype.** Match exact installed
   source/API and public contract to H200/SM90,D64,page1/GQA8,E4M3 KV and causal
   L8. Preserve Q normalization→BF16→E4M3 before any widening, scales once,
   BF16 PV, FP32 softmax/partials and strict accumulation. Require GPU-live
   lengths/page mappings on every replay, private state and all new per-call
   assign/descriptor/gather/merge costs inside the call. No host plan cache,
   cross-layer task map, unsupported private API or precision substitution.
   These prerequisites permit a same-geometry independent implementation even
   when exact FA3 C++ source is unavailable. It can falsify whether our strict
   page-gather/convert/MMA schedule compiles and is viable. Matching known FA3
   features does not block that microprobe; it blocks any claim of novelty or
   inferred production advantage. A production candidate still needs measured
   net benefit, complete-call costs and an honest coverage path.

3. **Isolate independent primitive viability and useful overlap.** With an
   eligible installed source/API, predeclare one
   same-tile, same-precision causal pair, initially natural-M64 verify when
   justified. Both arms have identical page1 gathers, conversions, math,
   load volume, geometry, ownership and merge; only the proved schedule differs.
   One waits immediately; the other performs genuinely independent work before
   waiting. Compile artifacts must prove correct dependency tokens, buffer
   lifetimes, wait placement and accumulation precision, with resource/spill
   gates declared against that control before compilation. A WGMMA opcode or
   predicated instruction alone is insufficient. This serial arm is a mechanism
   control, **not a description of FA3**. The existing FA3 pipeline remains the
   separate production comparator. Local matched-kernel implementation is in
   progress; compilation/execution still requires its own reviewed frozen
   harness and existing bounded resource admission. No GPU run follows from
   this document or from an implementation passing CPU/source checks.

4. **Then validate correctness and actual complete-call benefit.** Original
   oracles/tolerances, fresh inputs, empty/tail/new-KV ownership, page1/page64,
   scales, admitted head configurations, fixed-address metadata replay and
   independent output/scratch reuse precede timing. Report overlap only if it
   is observed. Paired uninstrumented whole calls must beat the current working
   path on declared fresh fixtures with unchanged controls; a mechanism-only
   win over a slow custom or immediate-wait arm is insufficient. Keep profiler
   durations separate from timings. The saved194µs decode benchmark versus
   297µs profile-forward discrepancy is unresolved, not recoverable60% loss.

5. **Recompute family reach before any official check.** Retain all twelve
   paths and six weighted loads, including B1 and sliding regressions. Insert
   measured whole-call cuts into the fixed-reference model and distinguish
   achieved evidence from conditional upside. A verify-only success or small
   one-path gain cannot become the central gap-closing strategy. No unchanged
   source check, dedupe workaround, threshold sweep or formal entry follows.

The CPU read resolves availability, not a missing mechanism. A strict matched
primitive prototype can independently test implementation viability while FA3
source remains unknown. Keep that microprobe separate from production-candidate
admission: no novelty, speedup or gap-closing claim follows until the complete
working-path comparison and coverage evidence support it. If viability fails,
record the exact blocker; do not relax precision or manufacture a production
candidate merely because asynchronous primitives are present.

Basis: [GAP_MODEL.md](GAP_MODEL.md),
[LOCAL_SPECIALIZATION_AUDIT.md](LOCAL_SPECIALIZATION_AUDIT.md),
[ELIGIBILITY_AUDIT.md](ELIGIBILITY_AUDIT.md) and its pinned Gluon/FlashInfer notes.
Static allocation estimates are not achieved residency; historical counter
timeouts are not permission denials or bottleneck evidence. No counter retry
is part of this plan.
