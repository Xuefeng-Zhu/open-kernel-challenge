# Experiment card: exact alpha-one numerator rescale bypass

**Evidence.** Current selected fusion-plus-lookahead source has explicit
historical old-numerator rescaling. Its exact A2-15 producer's saved native PTX
contains eight `mul.f32` carry instructions before PV MMA per thread, rather
than an already-eliminated identity. The current official full-B32-decode
profile reports297.2µs producer plus2.7µs merge; this is instrumented activity,
not the calibrated304.031893µs whole-call latency and not a rescale attribution.
Actual alpha-one frequency and rescale bottleneck share are unknown.

**Single causal hypothesis.** When all valid query rows have computed alpha
exactly1, a uniform branch can skip historical numerator multiplies often
enough to exceed its predicate/control-flow cost without perturbing PV carry,
buffering or resources. This first variant does not skip denominator rescaling,
change maxima/exp/P, defer maxima or omit any key/value contribution.

**Exact isolated source.** Control`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`;
prepared candidate`e66f502812f10f070170efcc9a41a7ef566576799d7461d27cd860a10c94f764`.
Only the historical numerator assignment is wrapped under existing target
constexpr`PAGE_LOOKAHEAD` and a padded-row-safe exact alpha predicate. Slow
expressions, denominator and new-KV tail are unchanged. See`SOURCE_AUDIT.md`.

**Optimistic latency and weighted-score bound.** Only one of six equally
weighted loads changes. In official job5090, full-B32-decode is304.031893µs;
its weight16 contributes4864.510291µs of the decode32 load's5789.276070µs,
fraction`f=0.8402622767`. Holding all other paths fixed, a fractional whole-path
cut`r` gives score factor`(1-f*r)^(-1/6)`.

| Conditional target cut | Saved target latency | Modeled overall gain |
|---:|---:|---:|
| 2% | 6.080638µs | 0.282867% |
| 5% | 15.201595µs | 0.717918% |
| 10% optimistic scenario | 30.403189µs | 1.473544% |
| 20% | 60.806379µs | 3.113931% |

These are sensitivity scenarios, not predictions. Even eliminating the entire
304.031893µs path has a loose mathematical ceiling of35.7580% aggregate gain;
this deliberately unrealistic bound says nothing about rescale cost. A3%
overall improvement would require a19.3411%/58.8030µs target cut. No source or
profile evidence currently establishes that much recoverable rescale time.
The true recoverable-latency upper bound is the unmeasured rescale critical-path
share, capped by producer time, not the sum of source FLOPs divided by peak
throughput. Do not report a precise hardware bottleneck or strong score upside
from these operation counts.

**New costs and resource risks.** Each historical tile adds a valid-row
comparison/reduction and uniform branch. Code size/phi moves, duplicated PV,
layout conversions, register pressure, barriers, spills or lost lookahead
buffering can dominate eight saved per-thread multiplies. This is distinct
from A2-14's live16×64 denominator array. Geometry, splittings, cache I/O and
private partial sizes must remain identical. Compile/probe instrumentation
requires a separately reviewed bounded existing-image free-credit job with
paid cap$0; no budget is reserved or GPU launched by this card.

**Same-geometry control.** Compile and run the exact current working control
alongside this one-source-site candidate with the same producer/merge geometry,
flags, mappings, data and graph setup. Use an unchanged full-B16-verify whole-call
family to expose session/control drift. Keep instrumented and timed modules,
fixtures, artifacts and reports separate; never compare instrumented times.

**Cheap falsifiers, in order.**

1. Six offline source/predicate/IEEE-model tests pass without GPU imports.
2. Inspect actual TTGIR/PTX/SASS and resource metadata. Prove the alpha-one path
   skips old-numerator multiplies, the false path retains them, PV uses the old
   carry with unchanged MMA order, and buffering/shared/register/spill resources
   do not regress materially. If the compiler removes the branch, unconditionally
   rescales, breaks the carry order or introduces spills, reject before timing.
3. A separate bounded instrumented producer counts visited/fast/slow historical
   tiles per CTA and validity mask, using fresh inputs. It never supplies benchmark
   times. Confirm both outcomes on designed cases and measure actual scored-family
   frequency; do not assume IID record-max statistics apply to coherent inputs.
4. Only if mechanism/frequency are viable, run strict original-oracle/cache/replay
   checks, then uninstrumented paired complete-call timing.

**Predeclared correctness and performance gates to freeze before launch.**
No relaxed tolerance, approximate threshold, input-dependent cache, algorithm
retiming or selection of favorable fixtures. Require unchanged original oracle
and finite output/new-KV; exact cross-arm cache bytes; A→B→A/live metadata and
untouched-row checks; target TP8 and generic DP; empty history, chunk/page tails,
unit/1⁄64 scales, padding and maximum-score first/middle/last-tile cases. Require
the same merge/new-KV ownership and independent-call isolation coverage as the
working control. Any failure stops the performance branch.

For a first measured benefit gate, use fresh seeds135731/135732 (not previous
selected fixtures), cold-L2 eviction outside the complete-call event interval,
two arms in reversed order, three balanced pairs per fixture/six total. Require
at least2% target latency reduction in **each** fixture, all six favorable pairs,
and unchanged-control median slowdown≤2% in each fixture. All warmups/replays
must match their own oracle-checked state. One session remains limited evidence;
a fresh-input confirmation must precede any promotion or complete official
check. These gates are recorded before results, not tuned after seeing them.

**Decision.** Reject a source-only/compile-only success as performance evidence.
Reject if branch cost or buffering/spill changes dominate; do not sweep alpha
thresholds. A viable paired improvement is a modest component, not an award-level
claim. A complete official check would have new-branch correctness value, but
its minimum score uncertainty cannot resolve a small modeled aggregate gain.
An approximate deferred-max threshold variant is a separate numerical experiment
and is explicitly outside this card. Execution requires the parent-coordinated serialized slot; this card launches nothing.
