# B1 route history and remaining architecture evidence

Local source/result audit, 2026-10-25 UTC. No new service, GPU execution,
profiler, official check, permission, source change or commit. D1 means decode
q_len1; V1 means verify q_len8. Both scored B1 paths use TP8 (16Q/2KV heads),
D64, page1 and nominal ctx80000. Historical source hashes were recomputed from
the saved files; an independent local history search agreed with the results.

## Finding

The B1 route split follows measured history, rather than an untested assumption
that FA3 is always preferable. Old custom full D1/V1 passed official correctness
but were slower; replacing them with FA3 and then fusing preparation improved
their target measurements. Custom sliding D1 subsequently benefited from a
parallel per-head merge and remains faster than the official SGLang reference.
Sliding V1 had already moved from custom to FA3, then fused preparation.

All three current FA3 B1 routes already fuse the obvious Q/K normalization,
optional RoPE, FP8 conversion and live KV append. Custom sliding D1 already
includes preparation in its attention producer and norm/gate in its merge.
There is no newly identified missing prologue fusion. The remaining separate
norm/gate activity is about1.1–1.2us, insufficient evidence for the70% B1
whole-call reduction in the broad gap-closing scenario.

## Current exact source and complete calls

Working source: [`fusion-lookahead-probe/candidate.py`](../fusion-lookahead-probe/candidate.py),
SHA256 `3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`.
Official check `5090b83f1d63407b` passed16 prerequisites and12 timed shapes;
score1.2336330588627955. Its result SHA256 is
`a66e15b71116d6758cd1aae75138f19c0a7db4cb2e89ba27394d6d09df2d61cb`.

| Scored B1 path | Current dispatch | Whole-call adjusted us | Same-run reference us | Same-run S/C |
|---|---|---:|---:|---:|
| D1 sliding | `_e2_kernel_fn`: fused producer + parallel merge | 11.477114 | 20.576001 | 1.792785 |
| D1 full | `_fa3_full_verify_fused` | 27.031303 | 31.184000 | 1.153626 |
| V1 sliding | `_fa3_sliding_verify_fused` | 16.988207 | 23.008000 | 1.354351 |
| V1 full | `_fa3_full_verify_fused` | 29.568836 | 35.552002 | 1.202347 |

The first full guard at source lines585–589 catches B1 D1/V1; the q_len8
sliding guard catches V1 (590–592); D1 sliding reaches the fallback (602).
The later overlapping legacy guards do not change these scored routes.

FA3 calls (456–582) have one timed preparation kernel producing FP8 Q and
appending new FP8 K/V, the unchanged FA3 attention call, then output RMSNorm
and sigmoid gate. Full paths use no RoPE table. Sliding uses the existing
input-independent cosine/sine table helper. BF16 normalization/RoPE and scaled
KV rounding boundaries are explicit in the fused producer. Cache slot0 is
live; store masks use the actual `cache_loc` and capacity. QK factor is folded
once into softmax scale. The source leaves FA3 split/scheduler options at their
baseline defaults and passes live lengths/page tables.

The official contract supplies FA3 scheduler metadata for full decode only
(`official/TASK.md:31`, `official/reference.py:151–171`). Thus current full D1
has no per-call schedule-preparation event, while both V1 calls retain one.
Moving V1 task-map work outside the timed call or caching live maps is not an
available fusion. The source-owned Q temporary and result allocations belong
to each invocation/capture; the saved activity table is not host-allocation
timing, and does not measure launch/host overhead independently.

Custom D1 sliding (84–330) uses SPLITS16 from the existing occupancy heuristic,
BM16/BN64 and producer grid(1,2,16):32 logical CTAs. It allocates135,168 bytes
of FP32 output/max/sum partials, then launches16 per-query-head merge CTAs
(BS16). The producer repeats Q norm/RoPE per split, gathers paged historical
FP8 KV, computes softmax/PV, and writes partials. The last split alone prepares,
quantizes and stores the new KV and attends to it; it remains necessary even
with no historical work. Merge combines FP32 partials and applies output
RMSNorm/gate. There is no standalone prologue, gate or counter-zero kernel on
this L1 path. Removing split/merge also removes useful producer parallelism;
a launch count alone does not justify unsplitting.

## Actual route-change evidence

C is the saved adjusted candidate time. The normalized cut is
`1 - (C_new/S_new)/(C_old/S_old)` using each job's own reference. Every row below
passed the saved16 prerequisites and12 benchmarks, with the same evaluator
between its parent/new pair. **These are separate jobs, not same-session paired
custom-versus-FA3 arms.** Same-run C/S compares with the official reference.

| Change | Old job -> new job | Target C old -> new us | Reference-normalized cut |
|---|---|---:|---:|
| V1 sliding: old custom -> ordinary FA3 | `e0b3cd9474084069` -> `4f626ab7437a42be` | 43.976461 -> 22.542432 | 47.3036% |
| V1 full: old custom -> ordinary FA3 | `4f626ab7437a42be` -> `6e761bdd1d9747ad` | 76.347735 -> 36.173308 | 53.6052% |
| D1 full: old custom -> ordinary FA3 | `bdd11e743365439a` -> `66b44d7586184708` | 41.446812 -> 31.669972 | 22.3375% |
| V1 sliding: ordinary -> fused FA3 | `66b44d7586184708` -> `50465a4c440f4b75` | 23.389947 -> 16.822679 | 28.1779% |
| V1 full: ordinary -> fused FA3 | `05b52e7834064b14` -> `ba9e02c1d6584cc4` | 35.4007 -> 30.484769 | 15.7734% |
| D1 full: ordinary -> fused FA3 | `ba9e02c1d6584cc4` -> `cf979895a3f24685` | 31.874407 -> 27.428827 | 12.5358% |
| D1 sliding: serial -> per-head parallel merge | `08235841c4044048` -> `cfa8a8670e394f80` | 16.056079 -> 11.820900 | 25.8681% |

Evidence: [`E6_SLIDING_FA3.md`](../E6_SLIDING_FA3.md),
[`A2-01-RESULT.md`](../A2-01-RESULT.md),
[`A2-04-RESULT.md`](../A2-04-RESULT.md),
[`A2-05-RESULT.md`](../A2-05-RESULT.md),
[`A2-07-RESULT.md`](../A2-07-RESULT.md),
[`A2-08-RESULT.md`](../A2-08-RESULT.md),
[`A2-11-RESULT.md`](../A2-11-RESULT.md).
Current D1 sliding's1.792785x reference ratio is not a comparison against a
newly forced fused-FA3 D1 sliding route. Ordinary official correctness is also
distinct from targeted graph mutation, scale/page/DP and concurrency coverage.

| Historical source | SHA256 |
|---|---|
| E5 `mixed_fa3.py` | `708d3f06f46bfb24a7a43963022eb51426f3a68bb2155c77dea8e9b6ae255423` |
| E6 `mixed_fa3_sliding.py` | `7a3bc5862d117f448879982158554e3a4fd9a80d92c8e6de4c8b91f93acd7755` |
| A2-01 `mixed_fa3_verify1.py` | `2c7bd6a0456c18555d0579b046e741086e527e2290dc3a11b642f8fb37eea357` |
| A2-03 `mixed_fa3_decode16.py` | `8e1db32fe3d2c803efecbe62bbff5dd752e133cc093743ef28143352a3d28905` |
| A2-04 `mixed_fa3_decode_small.py` | `5ef5c9e93b56ca382aea363d0022d558a8163f5df0ef8ae5673ba1a854bbf97b` |
| A2-05 `mixed_fa3_fused_prepare.py` | `a7930cda0389cf55f60759ea3f9247f0562bf3412fad00c81dcbfc48520f93de` |
| A2-06 `mixed_fa3_fused_verify.py` | `c40127a320442953630276206b0dfd0ebadd550cc58bdbfe124d800b31b13b18` |
| A2-07 `mixed_fa3_fused_full_verify1.py` | `f20126cb295419276c00eb1cb1927c8920a6a96f0c7d3e62f362b10d2176f95d` |
| A2-08 `mixed_fa3_fused_full_small.py` | `34b0c6bc67357aac9b3306e32010f5254bd4df2ae14b07b827b8686902fa60e9` |
| A2-10 `mixed_fa3_decode_merge.py` | `94cfb257da13daddec1e8eeb5e7c0b9abe768f54029ecfe2bca8733f8088b93f` |
| A2-11 rollback `mixed_fa3_decode_parallel_merge.py` | `c9a0847502a0b2fe2be55f546a2b3ea7eefa931c851e328e680302ebd828baf0` |
| A2-13 `mixed_fa3_fp8_qk.py` | `e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e` |

The old full-B1 custom result is not an evaluation of the current fallback.
E6 used `BF16(normQ*qk_factor)`; A2-13/current fallback uses
`E4M3(BF16(normQ))` with factor in the logit scale, strict FP32 QK accumulation,
BF16 PV and FP32 softmax/partials. Current official dispatch retains full B1
FA3, so5090 does not exercise a forced modern custom full-B1 route. Saved native
fusion/lookahead/common-PV harnesses target B16/B32; A13 explicitly names its
tiny B1/L8 full case `retained_fa3`. No saved same-session forced-current-FP8-
fallback versus fused-FA3 full-B1 comparison was found in the inspected local
workers/probes/harnesses. This is a scoped evidence gap, not a claim that such
a test has never existed elsewhere or that the current fallback would win.

## Remaining phase budget and smallest useful gap

The current official profile JSON SHA256 is
`3a9f9f841665ef0c678001aa62396b44557a375ce20d5e830548d257ff6f1300`.
Its B1 rows are rounded **instrumented activities**, separately from the
whole-call calibrated timings above:

| Path | Forward/producer us | Combine/merge us | Schedule us | Fused preparation us | Norm/gate us | Activity sum / GPU busy us |
|---|---:|---:|---:|---:|---:|---|
| D1 sliding | 7.8 | 1.7 | inside producer | inside producer | inside merge | 9.5 / not reported |
| D1 full | 14.7 | 6.8 | supplied input | 1.2 | 1.1 | 23.7 / 22.3 |
| V1 sliding | 7.8 | 4.0 | 1.8 | 1.3 | 1.2 | 16.2 / 14.7 |
| V1 full | 14.9 | 7.4 | 2.7 | 1.2 | 1.2 | 27.4 / 25.9 |

The profile explicitly says overlapping kernel duration includes predecessor
wait. These rows cannot establish a critical path or be subtracted from the
calibrated whole-call medians. No hardware bottleneck/counter or achieved CTA
residency is proven. The symbols are truncated; resolved B1 FA3 split count,
full specialization, auxiliary shapes and actual task map are not saved.

Under the deliberately optimistic **activity-accounting assumption** that the
producer/forward stays fixed and everything else becomes free, removable
activity is approximately17.8% (custom D1 sliding),38.3% (D1 full),51.8%
(V1 sliding),45.5% (V1 full), using reported shares. These are conditional
budgets, not attainable latency savings, hardware ceilings or forecasts.
Only eliminating the outer FA3 preparation plus gate accounts for roughly
2.3/2.5/2.4us activity; a combine epilogue that absorbs only gate has about
1.1–1.2us of observed gate activity to target. Neither establishes70% benefit.

For scale, a70% complete-call reduction would require current B1 times to
reach3.443/8.109/5.096/8.871us in table order. This is the engineering scenario
in [`GAP_MODEL.md`](GAP_MODEL.md), not an inferred capability. There is no
defensible new B1 architecture candidate from the saved data alone.

The smallest high-value architectural evidence gap is the **actual B1 FA3
decomposition**: recover each executed specialization and resolved split/
partial shapes, then distinguish forward work, combine work and dependency
wait on the same live fixture. Full D1 already has supplied metadata, so
planner fusion cannot explain its remaining gap. Verify metadata must remain
per call. Exact installed FA3 C++ source was not found by the bounded inventory;
the saved wrapper does not expose a proven gate-fused epilogue. Source/API
eligibility is required before proposing that integration. No B1 missing
dynamic persistence or staging mechanism is established.

A second, narrower missing comparison is matched current-FP8 custom versus
current fused FA3 on full B1, with the unchanged complete call and original
oracle/replay/cache requirements. It would resolve a route-selection gap,
not validate a novel mechanism; old custom losses and today's phase budgets
provide no70% upside prediction. Neither gap authorizes a launch in this audit.
No blind split sweep, duplicate prologue fusion or unmodified official check
is proposed. A2-11 remains preserved as rollback; no selection was changed.
