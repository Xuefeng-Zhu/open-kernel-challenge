# Attention strategy audit and A2-14 local prototype

2026-10-25. This is a local research result, not a new GPU result. No check,
upload, push, Modal launch, credential action or formal entry was attempted.

The competitive gap requires changing more of the attention computation than
the four QK paths reached by A2-13. The next bounded experiment is a denominator
dataflow change on full decode B32. It is a feasibility test for a core mechanism,
not a claim that one small patch can close the gap to the leaders.

## Preserved evidence and scoring

| Version | Source / official job | Measured score | Status |
|---|---|---:|---|
| A2-11 | `mixed_fa3_decode_parallel_merge.py` / `cfa8a8670e394f80` | 1.187678756× | Selected attention version |
| A2-13 | `mixed_fa3_fp8_qk.py` / `c2fb7b23433f4888` | 1.210990803× | Promising; 95% interval overlaps A2-11 |
| P039 | `projection_router_m1_fp32_gemv.py` / `d5636ac8d1404abd` | 1.173238999× | Separate selected projection version |

A2-11 and A2-13 each passed 16 prerequisites and 12 timed shapes. A2-13's
interval is [1.192825941, 1.229155666]; A2-11's is
[1.169863575, 1.210934250]. The observed cross-job score change is +1.9628%,
not a paired A/B confirmation. Source and receipt hashes, evaluator identity,
all raw medians and profiles are in [weighted_analysis.json](weighted_analysis.json).
The original official receipts are `../A2-11-official-result.json` and
`../A2-13-official-result.json`.

For each of six loads, `C = 46 × sliding_candidate + 16 × full_candidate`,
and similarly `S` for the same-receipt reference. The score is
`exp(mean(log(S/C)))`; it is not the average of 12 shape ratios. The following
calibrated candidate times come from official paired ratios. Raw graph medians
and replay-overhead estimates stay separate in [weighted_shapes.csv](weighted_shapes.csv).
These two historical jobs recorded 60 timing samples per shape except full
verify B32, which recorded 56. These are receipt sample counts, not new tests.

| Shape | A2-11 candidate µs | A2-13 candidate µs | A2-13 SGLang µs | SGLang / A2-13 |
|---|---:|---:|---:|---:|
| decode 1/sliding | 11.821 | 10.912 | 20.352 | 1.8651× |
| decode 1/full | 27.297 | 27.183 | 31.264 | 1.1501× |
| decode 16/sliding | 15.157 | 14.254 | 26.672 | 1.8713× |
| decode 16/full | 200.575 | 200.209 | 199.936 | 0.9986× |
| decode 32/sliding | 21.036 | 20.271 | 30.992 | 1.5289× |
| decode 32/full | 362.877 | 316.412 | 384.832 | 1.2162× |
| verify 1/sliding | 16.466 | 17.141 | 22.656 | 1.3217× |
| verify 1/full | 29.886 | 29.802 | 35.952 | 1.2064× |
| verify 16/sliding | 22.562 | 22.577 | 29.008 | 1.2849× |
| verify 16/full | 205.992 | 204.454 | 204.208 | 0.9988× |
| verify 32/sliding | 27.454 | 27.913 | 34.336 | 1.2301× |
| verify 32/full | 389.601 | 389.167 | 388.992 | 0.9995× |

For example, A2-13 decode B32 has
`46 × 20.270870572 + 16 × 316.411809862 = 5995.049004 µs`.
Its reference load is `7582.944054 µs`, giving `1.264867735×`.

| Load | A2-13 weighted µs | Reference weighted µs | Ratio |
|---|---:|---:|---:|
| decode 1 | 936.883 | 1436.416 | 1.533187× |
| decode 16 | 3859.007 | 4425.888 | 1.146898× |
| decode 32 | 5995.049 | 7582.944 | 1.264868× |
| verify 1 | 1265.322 | 1617.408 | 1.278258× |
| verify 16 | 4309.796 | 4601.696 | 1.067729× |
| verify 32 | 7510.684 | 7803.328 | 1.038964× |

Full attention accounts for 75.9–84.4% of each batched A2-13 load. Full decode
B16 and full verify B16/B32 remain almost exactly at their SGLang reference
times. They are the largest unexploited weighted paths, not measured regressions
that can be fixed simply by routing them back to the same library.

## Why FP8 QK only moved the aggregate about 2%

A2-13 changes QK operands on sliding decode B1/B16/B32 and full decode B32.
It leaves eight FA3 routes unchanged. The normalized target-path cuts are
5.584%, 3.533%, 2.243%, and 12.671%, respectively. Full decode B32 contributes
73.65% of the four changed paths' allocated normalized log-score gain.
Changing only those four normalized ratios models +2.6403% overall; drift in
controls and reference timings explains why the observed total is different.
This is a mathematical decomposition, not proof that code caused every change.

The source still performs page-table gathering, online maxima and denominator
updates, BF16 probability/value multiplication, accumulator rescaling, cache
writeback and output normalization. It has one KV head per CTA, eight useful
query rows for L1 in a minimum 16-row tile, head dimension 64 and historical
key tiles of 64. Full B32 uses 8,192-token chunks and a separate merge. QK
encoding changes do not remove the rest of that work. Nor does a source dtype
prove which native FP8 MMA instruction the compiler selected.

The A2-13 full B32 profile reports `_fused` 311.1 µs and `_merge` 2.6 µs.
That points investigation toward the producer, but profile activity sums are
not the calibrated 316.412 µs whole-call time. Current-candidate registers,
spills, shared memory, generated instructions and bandwidth/stall counters
were not captured. It would be unjustified to call this definitively memory-
bound or register-bound. Historical A03/A05 resource metadata belongs to other
kernels and cannot fill that evidence gap.

Numerical margin also matters: A2-13 output relative errors were 3.935–9.903%,
within the official adaptive bounds but higher than the BF16 parent. Its own
same-address graph/concurrency/untouched-cache-row stress has not run. The
earlier A13 diagnostic applies only to the exact A2-11 source.

## Competitive distance and modeled value

The public attention leader observed at 07:17–07:21 UTC was pjmullin3,
2.909672404×, job `c524687784b14b1b`. The projection leader was colinlikescode,
1.609861742×. These are provisional public results, not our measurements.
Under identical normalization, reaching the attention leader from A2-13
would require a uniform 2.40272× acceleration, or a further 58.38% latency cut.
Cross-run normalization and unknown per-shape results limit this comparison.
[Attention board](https://openkernelchallenge.dev/api/v1/leaderboard/muse_attention_decode),
[projection board](https://openkernelchallenge.dev/api/v1/leaderboard/muse_bf16_proj).

The pinned public `attention_balanced_hybrid.py` has a historical 1.493864714×
check receipt under the same evaluator object as A2-13. That is 23.36% higher
score, equivalent to an 18.94% latency cut under equal normalization. The
archived receipt omits per-shape times, so it cannot identify the winning
mechanism. The source combines different split dispatch, FP8 queries, partials
and parallel merge with large-full-verify FA3 routes; those differences are
confounded. We inspected it but did not copy or execute it. It is neither the
current leader nor our result. [Pinned manifest](https://github.com/MadanChaollaPark/open-kernel-experiments/blob/40154c27d4b694c7d641c7959b00d7ff24a7c2e4/outputs/best_manifest.json),
[receipt](https://github.com/MadanChaollaPark/open-kernel-experiments/blob/40154c27d4b694c7d641c7959b00d7ff24a7c2e4/outputs/attention-balanced-hybrid-check-result.json).

The following are arithmetic scenarios holding all other A2-13 times and its
reference fixed. They are not achieved speedups or predictions of this patch.

| Hypothetical change | Modeled total score |
|---|---:|
| Full decode B32 latency −10% | 1.228929× |
| Full decode B32 latency −20% | 1.248910× |
| Every full-attention latency −20% | 1.404226× |
| Every full-attention latency −50% | 1.858591× |
| Full decode B16 and full verify B16/B32 latency −50% | 1.567643× |
| Every full path replaced by the published harness floor | 2.35626× |

The last row uses optimistic simplified bandwidth/compute estimates, not
attainable latencies or a hardware guarantee. Even that full-only scenario
does not reach the observed leader. Under this simplified floor model, sliding paths also need improvement.
The 368 scenarios in the saved analysis are calculations, not GPU experiments.

## Ranked mechanisms, excluding repeated parameter changes

| Rank | Mechanism | Potential and evidence | Risk / information gained |
|---|---|---|---|
| 1: immediate feasibility test | Defer the denominator's horizontal reduction across historical tiles | Removes one recurring reduction from the producer. A 20% full B32 cut would add 3.13% overall; extension beyond that is conditional. | Low implementation scope, meaningful compiler/register risk. A native result tests whether saved reductions outweigh longer-lived FP32 state. |
| 2: larger architectural research | Explicit staggered QK/PV pipeline with paged-address lookahead | Overlap next QK/load work with previous PV, instead of serially repeating the producer sequence. Potentially reaches costly batched paths. No measured benefit yet. | High implementation risk: buffer lifetime, waits, accumulator rescaling and register occupancy. Requires generated-code/resource evidence before a broad rewrite. |
| 3: alternate decode engine | SIMT/vector producer for L1 | Eight useful GQA rows currently occupy a minimum 16-row tensor-core tile. A vector path could trade that padding for useful work in small decode loads. | May lose tensor-core throughput and KV reuse, and add shuffles. Keep FP8 Q semantics, BF16 P/V and existing partition/merge; test whole-call latency. |

FA3 keeps thread-local denominator contributions and combines them later;
its Hopper implementation also explicitly overlaps attention stages. These
are algorithmic references, not evidence that our Triton layout will benefit.
Its paged-cache implementation does address sharing and asynchronous copies;
arbitrary page gathers should not be assumed to map to one TMA transaction.
See pinned [softmax](https://github.com/Dao-AILab/flash-attention/blob/060c9188beec3a8b62b33a3bfa6d5d2d44975fab/hopper/softmax.h#L115-L139),
[pipeline](https://github.com/Dao-AILab/flash-attention/blob/060c9188beec3a8b62b33a3bfa6d5d2d44975fab/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp#L1083-L1135),
and [paged cache](https://github.com/Dao-AILab/flash-attention/blob/060c9188beec3a8b62b33a3bfa6d5d2d44975fab/hopper/paged_kv.h#L15-L66).
FlashInfer's [decode implementation](https://github.com/flashinfer-ai/flashinfer/blob/a88349f9f43df74d31d1d52ad5aa20c28824a790/include/flashinfer/attention/decode.cuh#L59-L138)
is a reference for vector/thread decomposition, not code run in this project.

Earlier negative evidence changes the plan: A2-10 serial merge regressed;
A2-12 query hoisting lost 8.676% on its target despite a shorter producer;
E3's loop `num_stages=3` moved the aggregate only 0.2502%, within overlapping
intervals. The proposed pipeline above requires actual dataflow overlap,
not repeating that source flag. The Triton 3.8
[CombineDotAddPattern](https://github.com/triton-lang/triton/blob/v3.8.0/lib/Dialect/Triton/Transforms/Combine.cpp#L237-L278)
already recognizes relevant dot-plus-accumulator forms, so a cosmetic
`tl.dot(..., accumulator)` rewrite is not worth a check without generated-code
evidence. The actual worker compiler version is not in these receipts.

Simple dual-head, wide-tile, materialized-QK/PV, query-hoist and unchanged
actual-span-balancing variants are excluded as requested. Missing local
receipts for some variants are not evidence of measured failure. Projection's
TMA experiment was also confounded by geometry/algorithm changes; its ordinary-
copy control never ran on GPU. We do not repeat it or attribute the loss to TMA.

## A2-14: precise local implementation and gate

Candidate: [`../mixed_fa3_deferred_denom.py`](../mixed_fa3_deferred_denom.py).
Parent: A2-13 source commit `21fefb946524a25d8a06562dd28b631d49cd5175`, SHA256
`e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e`.
Local source commit: `2092957b1c8312b1fc90b43f64c3ed5eed8f4658`.
Candidate SHA256:
`bba54e6e5cd779074ad0c372ca60556048085251edd6317463312e4a74cee43d`.
The prototype deliberately uses the promising FP8 parent to isolate this change;
it does not replace selected A2-11.

Only `B=32, L=1, window=0, no RoPE` enables `DEFER_DENOM`. Instead of reducing
`p` into a scalar on each historical tile, preserve FP32 mass per column:
`u = alpha*u + p`. Reduce once before the unchanged new-token branch.
The real-arithmetic identity is `sum(alpha*u+p) = alpha*sum(u)+sum(p)`.
FP32 addition order changes. Empty historical chunks retain neutral partials.
Do not remove the denominator by asserting RMSNorm cancels it: epsilon leaves
an exact dependence on `epsilon*(denominator/v_scale)^2`.

The `[16,64]` live FP32 tile contains 1,024 elements, about eight per thread
under uniform distribution across 128 threads, before compiler layout effects.
This is a storage estimate, not measured register allocation or an upper bound.
Extra scalar multiplications and register pressure may outweigh the reduction
saving. FA3's lane-local representation is more specialized than this prototype.

QK/PV precision, maxima, coverage, partition, cache ownership, new-token writes,
merge, geometry, warps/stages and FA3 routing are unchanged. AST reconstruction
checks the complete candidate against its parent after reversing only the
declared edits. No global input-derived cache or new inter-call state is added.

Local validation: **7 source/CPU-model tests + 11 analysis tests pass** on
macOS arm64, Python 3.12.13, Torch 2.14.1 CPU. Tests cover neutral empty chunks,
partial tiles, changing maxima, lengths around 8,192, FP32/FP64 denominator
agreement, retained RMS epsilon and fresh CPU A/B/A contents. Syntax parsing
does not compile Triton, and CPU A/B/A is not CUDA graph replay.

None of the 16 official prerequisites dispatches to this new B32 condition.
The full decode B32 timed case will exercise it, but target-specific page64,
non-unit scale, nearby length, untouched-row and concurrent-graph checks remain
unverified. An official pass must not be described as all those tests passing.

After separate authorization, the smallest next external experiment is one
candidate-only $0 off-board check at `openkernelchallenge.dev`. The exact
payload and command are in [approval_request.json](approval_request.json).
No new push, Modal run, credential or ranked entry is requested with it.
First recheck platform quota/auth and reconcile existing jobs; an uncertain
request is never duplicated. Adoption requires all correctness gates, a
meaningful normalized target gain (predeclared 20% goal), no unexplained >5%
regression, and a supported overall improvement. Small changes within noise
remain inconclusive. A failure is saved, not followed by arbitrary knob sweeps.
If compiler resources are unavailable from the receipt, report that gap rather
than substituting another kernel's metadata.

## Competition objective beyond one kernel

Read-only inspection in the user's Mac Chrome confirmed that all $100,000 in
cash prizes are in Serving; kernel-track top 50 receives Serving queue
priority. The public Serving board had no entries and was not yet open.
Serving measures throughput at stock SGLang interactivity across loads
1/4/16/32. Attention-only work can help a component and queue position; it
does not establish a prize-ready end-to-end result. The overall board does not
expose a numeric aggregation formula sufficient to average our problem scores.
[Prizes](https://openkernelchallenge.dev/#prizes),
[machine-readable overview](https://openkernelchallenge.dev/llms.txt),
[overall board](https://openkernelchallenge.dev/api/v1/leaderboard).

Serving integration uses a unified diff against SGLang
`a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f`, within the allowed source paths.
Our kernels still need correct TP8/EP8 collectives, rank-specific weights,
cache/backend metadata, graph lifetimes and model-stage integration. P039's
FP32 router is one concrete adapter opportunity; end-to-end benefit is unknown.
No new problem or Serving implementation is begun here.
[Serving rules](https://openkernelchallenge.dev/rules#serving),
[pinned model](https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/srt/models/muse_spark.py#L774-L845).

There is a deadline conflict: the site overview says kernel close November 4,
Serving open November 5, and Serving close November 19, 2026; current terms
v4 section 3 instead ends the contest November 18 at 11:59 p.m. Pacific, and
section 5 gives the rules precedence. Plan conservatively and obtain organizer
clarification. Eligibility and final rerun review remain separate from kernel
correctness or a license; no prize is promised. [Official terms](https://openkernelchallenge.dev/terms.pdf).

## Reproduction, provenance and next action

From the project root:

```sh
.venv/bin/python -B reviews/attention-v2/strategy/test_deferred_denom.py
.venv/bin/python -B reviews/attention-v2/strategy/weighted_analysis.py --self-test
.venv/bin/python -B reviews/attention-v2/strategy/weighted_analysis.py
```

The analysis regenerates deterministic JSON/CSV from saved receipts. Re-running
it does not send code or consume GPU quota. [preflight.json](preflight.json)
records actual local execution, source hashes and independent review.

A2-14 is an original narrow algebra edit to our existing source. No competitor,
FA3 or FlashInfer source was copied or executed. Existing SGLang adaptation
notices and Apache-2.0 text remain intact. The upstream ideas were read under
[FA3 BSD-3-Clause](https://github.com/Dao-AILab/flash-attention/blob/060c9188beec3a8b62b33a3bfa6d5d2d44975fab/LICENSE)
and [FlashInfer Apache-2.0](https://github.com/flashinfer-ai/flashinfer/blob/a88349f9f43df74d31d1d52ad5aa20c28824a790/LICENSE).
The pinned public comparator's notices license its original additions under
Apache-2.0 while preserving upstream rights; see [public_comparison.json](public_comparison.json).
A source license alone does not establish competition eligibility.

**Recommended next step:** authorize the one A2-14 native correctness/timing
test, with the stated target and rejection criteria. If it fails to deliver,
use generated-code evidence to decide whether the larger explicit pipeline is
worth implementing. Do not widen the search by counting cheap guesses as
progress. The cash-prize path eventually requires a separately scoped Serving
integration decision; current component scores cannot substitute for it.
