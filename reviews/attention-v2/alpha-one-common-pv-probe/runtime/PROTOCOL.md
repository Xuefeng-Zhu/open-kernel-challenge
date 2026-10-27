# One bounded alpha-one frequency, correctness and paired stage

The compile-only phase passed its exact source/resource/pipeline gates. This
protocol is the next independent stage, prepared locally while Projection P049
owns the serialized measurement slot. No app, upload or credit reservation
follows from this file. The parent conditionally permits one bounded stage
after complete local harness review and slot coordination; no official check
can follow unless performance and fresh-input confirmation support it.

## Question and controls

Does exact all-valid-row alpha-one suppression occur on actual scored-family
inputs, preserve the original output/cache/replay contract, and improve the
complete uninstrumented call enough to exceed predicate/shuffle overhead?
The SASS predicated FMUL slots can still incur instruction issue costs. Counts
cannot answer the timing question, and compile success is not numerical or
performance evidence.

The control remains working composition SHA256
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`;
the sole uninstrumented candidate is
`14a8fa2e3d7b64fa299fe69c8da6569501c069425a62e07cc5f11b1e9b15f3e5`.
Both retain automatic FA3 policy, original precision/tolerances, lookahead,
split/warp/stage geometry, partial allocation and last-split new-KV ownership.
No failed-source repair, threshold sweep or register cap.

Only full-B32-decode changes. On the last official score its load share was
0.8402623; a2% whole-path reduction conditionally models only+0.282867%
aggregate with all other paths fixed. This modest question is not presented
as an award-level advance or a precise prediction. The raw same-job control
and unchanged full-B16-verify family are mandatory.

## Six fixed cases, two fresh fixtures

Use seeds135731/135732, never the previous selected fixture seeds.

| Case | Shape | Input family |
|---|---|---|
| Scored target | TP8, B32/L1/full, ctx80000, page1, scale1 | Agentic/coherent generator, unchanged recipe |
| First history maximum | TP8, B32/L1/full, ctx8193, page64, scale1/64 | Boundary vector, forced first history tile |
| Middle history maximum | Same | Forced middle history tile |
| Last history maximum | Same | Forced last history tile |
| Generic DP multi-chunk | DP128Q/16KV, B32/L1/full, ctx8193, page64, scale1/64 | Boundary vector, normal fresh soft inputs |
| Unchanged control | TP8, B16/L8/full, ctx80000, page1, scale1 | Agentic/coherent generator, unchanged recipe |

Boundary histories are`[0,1,63,64,65,8191,8192,8193]×4` and the reverse.
Generate fresh page mappings/values, preserve live physical page0 and negative
unused suffixes, and rebuild actual supplied scheduler metadata outside capture.
Use page-size-aware valid extents. Both sources receive identical fixture
versions and geometry. Live tensor contents, including lengths/page tables/
positions/metadata, are restored at fixed graph allocations across A→B→A.

The three forced cases are explicit adversarial fixtures: every Q head has only
dimension0 nonzero, new K channels are zero, and historical K is zero except
dimension0=representable E4M3 value32 at each request/KV head's leader physical
row. V and gate values remain fresh. Leader tile is0,`(n_tiles-1)//2`, or
`n_tiles-1`, with position`64*tile`; empty histories have no leader. The valid
normalization/quantization/scaling and positive leader yield a clear history
maximum, while new K has zero score. Log actual positions; do not describe
these remapped fixtures as untouched scored distributions. Non-forced target
and DP cases retain ordinary nonzero cache-store observability.

## Original correctness and independent graph reuse

Both complete uninstrumented entrypoints must pass the unchanged original
oracle on A→B→A for all six cases:36 oracle calls, twelve exact return-A and
36 K/V whole-cache cross-arm byte comparisons. Eighteen paired output/new-KV
byte comparisons must also match. Compare bytes through uint8 views, including
signed-zero representation; floating `equal` alone is not a byte proof.
Finite outputs, fixed addresses, immutable inputs and untouched cache rows
remain mandatory. No precision or tolerance change.

On the middle-maximum case add an independent candidate graph with disjoint
cache/output ranges. Establish its own A expectation with one oracle call;
submit primary A and independent B on separate streams, then apply two oracle
checks. Total original-oracle count is39. After submission, replay primary B
and independent A, checking active and inactive retained state after each:
four isolation checks after two extra replays. Each graph owns both oracle
expectations; no transfer of expected output between graphs. Record pool IDs
when supported, otherwise state the observation limit. No actual overlap or
same-graph overlapping replay certification is claimed.

## Separate frequency diagnostic

`diagnostic.py` is a separately frozen, audited source transformation of exact
candidate14a. It adds only three scalar int32 history counters per CTA:
`[visited, fast, slow]`, written once to fresh designated output
`alpha_stats[batch*HKV*SPLITS,3]`. It reuses the exact computed predicate;
empty-history CTAs write zeros, and the new-token tail is not counted. No
atomics, input-derived cache, changed floating recurrence or timed counter code.

Run ten diagnostic calls: cases1–5, both fresh versions. For each also replay
the uninstrumented candidate once with that same version, checking its own
previous oracle expectation. Diagnostic output/new-KV and entire post-store
K/V bytes must match the uninstrumented state; all other inputs and untouched
rows remain exact. Only the designated`alpha_stats` output is excluded from
diagnostic immutable-input views. No such exclusion or extra field exists in
the timed sources or original candidate.

For CTA split`s`, the expected visited history-tile count is
`ceil(max(0,min(seq,(s+1)*8192)-s*8192)/64)`, repeated across KV heads.
Require per-CTA`fast+slow=visited`, exact shape/count bounds, zeros on empty
splits and the forced first/middle/last record-max transition counts. Report
raw counts/fractions per case/version without translating them into cycles.
Both ordinary scored fixtures must expose some fast opportunity; no arbitrary
frequency percentage threshold is introduced. If neither a faithful diagnostic
nor scored opportunity is established, stop before performance and retain the
specific completed negative/partial coverage. Never time the diagnostic.

## Uninstrumented paired complete-call acceptance

Only after all correctness/diagnostic gates, time the exact target and unchanged
control entrypoints. Each family uses four warmup plus twelve measured two-arm
samples, with adjacent reversed orders and the same fixture within each pair.
Six balanced pairs provide three per fresh fixture. Evict256MiB L2 outside
the CUDA-event interval; input/cache restore and fixture/reference work are
also outside it. Whole graph work includes preparation, stores, compute/merge
and output gate. Use raw paired graph times; no empty-graph subtraction is
needed and no absolute comparison with calibrated official times is valid.

All64 warmup/measured arm replays must reproduce their own oracle-checked
output/new-KV bytes and immutable/untouched state. Thirty-two paired cross-arm
output/new-KV byte checks cover those samples. The source card's predeclared
gate remains: target median balanced latency reduction≥2% **in each fixture**,
all six target pairs favorable, and unchanged-control median slowdown≤2% in
each fixture. Preserve every sample, order and fixture. A completed correctness
run with failed benefit is a valid negative result, not a launch error.

Passing one session does not promote the candidate or send an official check.
A fresh-input confirmation and review must precede that decision. No posthoc
threshold change, score claim from point timing, instrumented comparison or
unchanged-source quota use.

## Bounds and closure

One reviewed existing-image H200/2 CPU/8GiB-host invocation, max one container,
zero retries;160s worker/170s child/185s function/20s startup/210s watchdog.
Reserve at most$0.58 of verified free credits, actual paid cap/current paid$0,
live rates/app checks and exact committed source/support/diagnostic hashes.
Existing protected credentials via supported config only; no installation,
renewal, permission, billing, visibility or formal-entry change.

Stop on source, original-oracle, byte/cache/address/isolation, diagnostic,
access, timeout or unexpected-app failure. Save raw partial results, never
retry an uncertain action. Stop and independently verify stopped/tasks0,
release the serialized slot promptly, preserve initial shutdown observations
and record a cutoff-qualified cost before later measurements.
