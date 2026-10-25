# Composition: validated preparation fusion plus the measured A2-15 variant

This is local preparation, not a new measurement. Use the validated fusion
source as the control: commit `05c9cc08948c42fed9e09aae1cd2b41edf966c74`, source
SHA256 `5c3d1c5b00f0e759f1c499b8f567a0f47c77ce0fa38566f3028a571fccc614a0`,
official nonranking check `2f5f1fb54b48486d` with aggregate 1.225555436×.
The rejected sliding-unsplit source is not the control. Keep `num_splits`
automatic and original A2-11 selected. No GPU, Modal job, official request,
installation, credential/billing change or external Pro communication is
authorized by this document; a separate parent serialized-slot grant is needed.

## Evidence and the limited question

The prior A2-15 paired run compared A2-13 source
`e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e` against
`mixed_fa3_page_lookahead.py`, SHA256
`fe09ee9736b3039b7c8edcd02968dc247c4457fe503629930fb41b7ea3c7fb3c`.
Its exact raw result is
`92b5ba197689c3e17ec3c86cefdefdb1edd4bfe0222171b0bd63bd67f000e867`.
The six reverse-order pair ratios reconstruct to median 1.054731138,
or 5.189108% whole-call target latency reduction. All 12 measured comparisons
favored A2-15. The run covered the full-decode-B32 scored family and a scaled,
paged boundary case, with fixed-address A→B→A and immutable/untouched-cache
checks. It did not include unchanged-path timing controls, independent calls,
complete official coverage or an independent session.

The old 10% target was not met, and remains failed. The compiler evidence showed
both page-address scheduling and K/V buffer changes, so neither the old timing
nor this proposed composition isolates a causal benefit from page IDs alone.
The measured object is the exact A2-15 producer/pipeline variant, not an
arbitrary new prefetch edit. Its old artificial $2 stage ceiling is historical
and does not apply now; paid authority remains $0.

The actual composed source is now SHA256
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`.
Source ancestry/isolation is documented separately by the source audit; this
score model binds both the new complete candidate and its exact control copy.
The new question is whether that measured variant remains correct and retains
a useful complete-call gain when composed with the independently validated
preparation fusion. Source ancestry and isolation are prerequisites, not
assumptions. The source audit must establish that the fusion routes and the
target producer do not overlap in execution or math, and that the measured
A2-15 choices are preserved. If the variant already exists, cannot be applied
to only its intended path, or changes unsupported semantics, stop rather than
manufacturing another candidate.

## Exact current score leverage

The current official fusion full-decode-B32 time is 317.205125874µs and its
sliding partner is 19.915056634µs. The load cost is
`C = 46×19.915056634 + 16×317.205125874 = 5991.374619172 µs`.
Full decode B32 contributes 84.709809294% of that load. The six loads are
equally weighted geometrically. Holding every other path fixed, transfer a
relative target latency cut `r` by:

`factor = (1 − 0.847098092940 × r)^(-1/6)`.

| Assumed target cut | Conditional overall gain |
|---|---:|
| 2% | +0.2852% |
| 3% | +0.4299% |
| 4% | +0.5762% |
| Prior measured 5.189108% | +0.7520% |
| 10% | +1.4862% |

About 3.4803%, 6.8416%, 10.0883% target cuts would model 0.5%, 1.0%, 1.5%
aggregate gain respectively. These are sensitivity calculations using a
different official run and calibration; they are not forecasts, confidence
intervals or achieved composition scores. The prior gain would be a modest
increment, not a major architectural result. A complete official check would
have correctness/integration value, but its minimum aggregate interval can
still be wider than this modeled gain.

`model.py` binds the exact control source, historical A2-13/A2-15 sources,
prior raw paired receipt/analysis and current official receipt. It reconstructs
the current six-load score and old pair arithmetic, rather than trusting a
rounded memo or subtracting raw times from different jobs.

## Source and local gates

The source audit owns the exact transplant. Require a standalone complete
candidate whose only execution change is the intended full B32/L1,
window0/RoPE0 custom route, retaining supported head layouts; its scored family
is TP8/D64. Keep every fusion/FA3 adapter and other dispatch
path unchanged. Do not reuse the rejected split override. Preserve the exact
measured pipeline settings, source precision boundaries, E4M3 storage/query
quantization, BF16 PV, FP32 softmax/partials and `max_num_imprecise_acc=0`.
Do not change split sizing, warp/stage settings, grid, masks or merge policy
unless they are themselves part of the exact previously measured variant;
record that distinction explicitly rather than attributing everything to
address lookahead.

Prove the pre-existing last-split ownership of new K/V and the lifetime of
partial/scratch buffers. Empty historical splits can still own new-token work;
do not remove them. No input-derived plan, cache, output or data-dependent host
branch may be introduced. CPU models and AST/call-contract tests should bind
the exact sources, isolate the target config among nearby batch/length/window/
head/page choices and exercise fresh input restoration, physical slot0,
quantization/scales and boundary masks. They are not native correctness or
timing evidence.

## Bounded native replication protocol, after a parent slot grant

Use the complete fusion control and composition entrypoints. Do not patch a
loaded candidate or import the oracle into it. Verify committed payload hashes,
original reference/check hashes, environment/library identity, and the
resource/split/ownership audit before timing. Keep original tolerances.

Five fixture families address this focused question:

1. Changed target: full B32/L1, TP8/D64, context 80000, page 1, scale 1,
   coherent/agentic lengths. The declared new seeds are135711/135712 for this
   replication/generality question. This is not an exact rerun of the old two
   fixtures, and no held-out official evidence is claimed.
2. Target boundary: full B32/L1, page64, scale 1/64, soft distribution,
   histories 0/1/63/64/65/8191/8192/8193 repeated to 32 and reversed. Rebuild all
   positions, live new-cache locations, cache lengths and serving-generated
   scheduler metadata; retain scattered physical pages and −1 unused suffixes.
   Make the zero-history request write physical slot0. These cases cover empty
   history, BN64/page tails, 8192-token chunk boundaries and last-split new-token
   ownership. Preserve the same shape/capacity for A and B.
3. Unchanged control: full B16/L8, TP8/D64, context 80000, page 1, scale 1,
   coherent/agentic lengths, through the validated fusion+FA3 route. Verify
   metadata remains None; target decode metadata is valid and changes with
   each fixture even if the custom route legitimately ignores it.
4. Unchanged custom fallback: full B4/L1, TP8/D64, page64, scale 1/64,
   context 8193 capacity and histories 0/63/64/8193, then reversed. Rebuild
   consistent live metadata and make the zero-history new row physically 0.
   This case must retain `PAGE_LOOKAHEAD=False` and the original custom path.
   It checks that the newly embedded producer does not alter nearby non-target
   execution; it is correctness/replay coverage, not a timed claimed gain.
5. Target-route head-layout smoke: full B32/L1 with supported DP heads,
   page64, scale1/64, context65 capacity and histories0/63/64/65 repeated eight
   times and reversed. This cheap case exercises the same generic source guard
   at the larger head layout. It has no timing or performance claim.

Capture both complete entrypoints independently for all five cases and replay
A→B→A at fixed input/output addresses:30 unchanged official oracle calls and
10 exact returning-A checks.
Each returning A must exactly reproduce its own first-A output/new-KV bytes.
Check immutable tensors and every untouched cache row, and compare post-store
K/V bytes across arms exactly after each replay label. Attention output bytes
need not be equal across implementations; original oracle correctness is the
requirement. Unexpected cache-byte differences stop timing for investigation,
not tolerance relaxation.

For the target boundary family, separately capture a second candidate graph
with disjoint writable caches and private default graph memory, then submit A
and B on separate streams. Two additional oracle calls check target scratch/
partial/merge ownership. This tests independent concurrent submission;
do not claim measured actual hardware overlap. No output/input cache may cross
calls or layers. Preserve original fallback counter semantics; do not claim
fallback concurrency coverage from this target-only independent-call test.
All native gates must pass before performance measurement.

For the scored target and unchanged control, retain four warmup two-arm samples
then 12 measured samples, grouped into six adjacent reverse-order pairs sharing
one input version per pair. Alternate versions by pair. Restore all inputs and
caches, evict 256MiB from L2 outside the event interval, and time all captured
`kernel_fn` device work: preparation, historical/new K/V processing, merge and
gate. Replay reuses graph memory; host allocation, fixture generation,
restoration, eviction, validation and any diagnostic collection are outside
the event interval. Check all 64 warmup/measured replays against their own
previously oracle-validated output/new-KV bytes, addresses and immutable rows.
Keep raw order/version/sample data and unclipped empty-graph diagnostics.
No profiler counters or repeated tuning knobs are needed for this test.

## Predeclared composition-replication decision

The proposed new gate is at least 4% median paired whole-call latency reduction
on full decode B32, with all six reverse-order pair ratios favoring composition
and no unexplained >5% median paired regression in the unchanged control.
Compute reduction as `1 − 1/median(pair_geomean(control/candidate))`, not from
pooled medians over unlike fixtures. Four percent is below the old worst pair's
approximately 4.78% cut and models about +0.576% overall on the current official
costs. This is a preregistered replication/materiality criterion, not a
significance threshold or a revision of the old failed 10% architectural goal.
Parent review can set a different gate before launch, but it must be explicit
and frozen before inspecting new measurements.

If source isolation, correctness, ownership, replay, control stability or this
replication gate fails, reject the composition without an official check.
Do not relax precision or sweep parameters to rescue it. If all gates pass,
report ready for one complete official `check`; first stop the Modal app and
independently verify stopped/tasks0, then wait for the parent's official slot.
One session and the prior session are useful replication evidence, not proof
of aggregate score significance. No automatic official or ranked submission.

Plan within the existing one-H200/CPU2/8GiB, worker 160s/child 170s/function 185s/
startup 20s/watchdog 210s envelope and a conservatively reserved $0.58 in verified
free credits, actual paid $0. No new credentials, installation, payment or limit
changes. Read current rates and guardrails before admission; preserve prefix
results and reconcile any timeout/uncertain request instead of retrying.
No GPU job is started by these files.
