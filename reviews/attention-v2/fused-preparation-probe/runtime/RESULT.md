# Native preparation fusion passed; useful whole-call gain in one session

The frozen candidate passed strict native preparation equivalence and every
unchanged official output/cache check in this bounded H200 diagnostic. All
three changed paths improved in the paired whole-call measurements. This
supports one complete official check; it is not an official score, an
independent-session confidence interval or adoption of a new selected version.

| Target | Median reverse-order control/candidate ratio | Paired latency reduction | Positive reverse-order pairs |
|---|---:|---:|---:|
| Full B16 decode | 1.039325312 | 3.7837% | 6/6 |
| Full B16 verify | 1.042815412 | 4.1058% | 6/6 |
| Full B32 verify | 1.024166375 | 2.3596% | 6/6 |

The ratio is the median of six adjacent reverse-order pair geometric means.
Each pair uses the same input version, with versions alternating between pairs.
Whole-call CUDA graph device work includes preparation, cache stores, unchanged
FA3 scheduling/compute/combine and output gate; host allocations, fixture
creation, validation and L2 eviction are outside the event interval. There was
no profiler or hardware counter collection.

Pooled raw medians are 264.224→255.824, 211.200→202.704 and
397.152→388.448 microseconds respectively. The decode fixtures have materially
different lengths/scheduler outcomes: per-version raw medians are
322.608→312.048 and 207.648→199.024 microseconds. Its pooled median is therefore
not representative of either fixture. The paired ratios are the primary
comparison; all samples, versions and actual length vectors remain in
`result.json`. Do not compare these absolute times directly with earlier jobs.

The six required cases cover the three changed routes with scored-family
page1/scale1 and boundary page64/scale1⁄64 variants, using seeds135701/135702.
Boundary histories include 0/1/63/64/65/127/128/129, scattered physical pages,
live slot zero and negative unused page-table entries. Decode exercises actual
serving-generated scheduler metadata, refreshed with each fixture. Verify
metadata remains None. No preparation or scheduling data derived from one call
is cached for another call.

All 12 preparation comparisons passed: FP8 query and entire post-store K/V
cache bytes were exact against ordinary preparation (36 byte comparisons),
with finite queries, immutable input and untouched-row checks. Both complete
entrypoints then passed A→B→A at fixed addresses in all six cases: 36 unchanged
official oracle calls, including 12 exact return-A checks. Three disjoint-cache
independent graph pairs submitted on separate streams added six passing oracle
calls. This establishes concurrent submission correctness, not measured actual
hardware overlap. All 96 warmup/measured replays matched their own previously
oracle-checked output/new-KV bytes and preserved inputs/untouched rows.

Environment: NVIDIA H200 sm90, Torch 2.14.1+cu130, CUDA 13.0, Triton 3.8.0,
Python 3.12.3. No installation, credential change, tolerance change, matrix or
FA3 split/scheduler/precision change was performed. The source freeze is
`0280972637e600e5f82a45f35778aed3f72886b1`. Candidate SHA256 is
`5c3d1c5b00f0e759f1c499b8f567a0f47c77ce0fa38566f3028a571fccc614a0`;
the exact raw result SHA256 is
`576d8e09e41b344d6c262c5bbdeb830e5800c6e3a1e19ed6f1aa8cb266e06868`.
`analyze_result.py` verifies published request/result/source/support hashes,
the exact case/version sets, native gates and paired arithmetic. The independent
`RESULT_REVIEW.md` reaches the same result.

The app `ap-I6PNHOGgL6uRL1AjjG2yoV` stopped at 20:17:43UTC. An independent
read at 20:18:08.870706UTC confirmed stopped/tasks0, and all listed apps stopped
with zero tasks. The controller's earlier stopping/tasks1 observation remains
unaltered in `launch-receipt.json`; it returned exit1 for incomplete immediate
closure although the worker itself completed successfully with exit0. Terminal
closure is separately retained in `reconciliation.json`.

Worker internal time was 70.489s; client wall time 89.073s. The initial billing
read had not yet advanced; at the independent cutoff, rounded workspace usage
increased $0.09 and detailed workspace usage$0.08614060. These delayed workspace
deltas are not an exact per-app invoice. Actual paid spend remains$0. The
conservative closed allowance is $0.267981667, within the reviewed $0.58 free-credit
reservation. No retry or second Modal job occurred. The serialized slot was
released immediately after terminal closure; later charges belong to later
work and must not be attributed to this diagnostic.

The complete 16-prerequisite/12-scored official coverage and independent-session
repeatability were outside this Modal diagnostic. A2-11 remains selected. The
parent separately granted one nonranking official check of these exact bytes;
its receipt and result are recorded under `../official-check/`.
