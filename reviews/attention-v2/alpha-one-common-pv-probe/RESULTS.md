# Common-PV native validation passed; performance gate failed

The exact common-PV candidate passed all39 original-oracle checks and every
predeclared cache, byte, replay, independent-graph and diagnostic-fidelity gate.
Its scored-family alpha-one fraction was about79.29%. Nevertheless, both fresh
fixtures were slightly slower, with zero of six favorable balanced pairs. The
predeclared≥2%-per-fixture improvement gate failed. Reject this candidate and
stop the exact alpha-one route; do not promote it, repeat a confirmation or
consume an official check.

| Whole-call family | Raw control→candidate medians | Fixture0 paired slowdown | Fixture1 paired slowdown | Favorable pairs |
|---|---:|---:|---:|---:|
| Full B32 decode target | 296.208→297.440µs | +0.54976% | +0.47631% | 0/6 |
| Unchanged full B16 verify control | 203.888→203.920µs | −0.07786% | +0.21450% | 3/6 |

Negative slowdown means faster. Paired per-fixture statistics are primary;
pooled raw medians describe this session. Each fresh fixture has three balanced
reverse-order pairs. All six target pairs favored the control, while control
drift stayed well inside2%. This is one session, not a population confidence
interval or an official aggregate. It establishes failure of the declared
benefit gate without proving a precise universal slowdown or hardware cause.

The two source forms demonstrate different failure modes: the earlier PV-inside-
both-branches variant was rejected before execution because registers increased
96→128 and V buffering disappeared. This common-PV form restored96 registers,
the original K/V pipeline and carry order, then passed runtime correctness but
failed end-to-end timing. Compile success and frequent predicates were useful
admission evidence, not sufficient performance evidence.

## Frequency is separate from timing

| Diagnostic case | Fixture | Visited history tiles | Fast | Slow | Fast fraction |
|---|---:|---:|---:|---:|---:|
| Scored full B32 decode | 0 | 79,528 | 63,056 | 16,472 | 79.2878% |
| Scored full B32 decode | 1 | 79,604 | 63,120 | 16,484 | 79.2925% |
| Forced first maximum | Both, each | 3,120 | 3,056 | 64 | 97.9487% |
| Forced middle maximum | Both, each | 3,120 | 3,032 | 88 | 97.1795% |
| Forced last maximum | Both, each | 3,120 | 3,032 | 88 | 97.1795% |
| DP multi-chunk boundary | 0 | 24,960 | 19,603 | 5,357 | 78.5377% |
| DP multi-chunk boundary | 1 | 24,960 | 19,815 | 5,145 | 79.3870% |

Ten separate instrumented calls matched uninstrumented candidate output/new-KV
and complete K/V cache bytes. Every CTA replaced its negative sentinel, including
empty-history zeros; per-CTA visited totals and forced-record-max transitions
matched independent CPU models, with fast+slow=visited. Only three private
int32 history counters were added. No instrumented time supplied any performance
sample, and new-token tail work was not counted as historical work.

The compiled source uses predicated FMUL: alpha-one suppresses arithmetic/write,
but instruction slots are not proven eliminated. The added predicate and three
butterfly shuffles can cost work. The timing result is consistent with that
overhead offsetting the saving, but no execution counter or stall measurement
isolated the cause. Do not translate79.29% into saved cycles or latency.

## Correctness and measurement scope

Six fixed cases use fresh seeds135731/135732: scored full-B32-decode TP8
page1/unit-scale; three page64/scale1/64 ctx8193 boundary cases with first,
middle and last history maxima; DP128Q/16KV multi-chunk boundary; and unchanged
full-B16-verify control. The maximum-position fixtures deliberately modify Q,
new K and historical K outside capture while leaving V/gates fresh. Their
recorded positive leaders and zero new K make them adversarial correctness
fixtures, not untouched scored input distributions.

Both complete uninstrumented entrypoints passed36 A→B→A original-oracle calls,
twelve exact return-A,36 whole-cache K/V byte comparisons and18 cross-arm
output/new-KV byte comparisons. An independent candidate graph added its own-A
setup plus two separate-stream submission oracle checks, totaling39. Two more
mutating isolated replays gave four active/inactive retained-state checks.
Observable outputs/caches and graph pools were disjoint with stable addresses;
actual hardware overlap and internal scratch addresses remain unmeasured.

After correctness, ten diagnostic calls and ten exact-own-reference candidate
replays passed uint8 byte fidelity, immutable-input, untouched-row, sentinel and
counter checks. Only the dedicated diagnostic stats output was excluded from
its immutable-input view. The original oracle, tolerances, timed candidate and
reference were unchanged. Scored fast opportunity existed in both fixtures,
so the predeclared timing branch was entered.

The uninstrumented target and unchanged control each used four warmup and twelve
measured two-arm samples, six balanced reversed-order pairs with matching input
versions within each pair. All64 arm replays and32 cross-arm output/new-KV byte
comparisons passed. A256MiB L2 eviction and all input/cache restore/validation
were outside CUDA-event intervals. Whole-call graphs include preparation,
cache stores, compute/merge and gate; no replay-cost adjustment, profiler,
hardware counters or instrumented module entered timing. Absolute times must
not be subtracted from separately calibrated official results.

## Source, closure and cost

Execution freeze: `032c2ffaf494c4f847e0b9f4bc97b7dd23e627fc`.

| Artifact | SHA256 |
|---|---|
| Working control | `3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8` |
| Rejected common-PV candidate | `14a8fa2e3d7b64fa299fe69c8da6569501c069425a62e07cc5f11b1e9b15f3e5` |
| Independent diagnostic | `ee5488bd8c6b8ee1e4bb9b6fccc1184215949136008dd15df11859b876f76c60` |
| Worker | `c3e9193d8c7dbb8cd68c06ea363b2f77cc8b4d64657be6a20ad832ac3e1739bb` |
| Controller | `ab4b88f430ab2934f4cd0e75ce5bfa1b16fd1f6c1ca3801631a3f0b77ef7f0e3` |
| Raw report | `24c7ea033298ea155d31ef161e887ed086a3820860cb077919d8276990e3d0c9` |

Thirty-four offline tests and two independent harness reviews passed before
the exact frozen-head preflight and single admitted app. The existing image,
original graph helpers/reference/checks and protected credential configuration
were hash-bound. No precision, geometry, tolerance, installation, credential,
billing-setting or visibility change occurred. No formal entry was made.

App`ap-wE6ZSXzC1dnP253vX0kM5l` stopped at22:50:11UTC. Independent
22:51:11.150348UTC reconciliation confirmed stopped/tasks0 and all listed apps
stopped/tasks0; immediate outcome/slot release preceded report polishing. The
initial stopping/tasks1 receipt is preserved. Worker exit0/status complete with
failed benefit is a valid completed experiment; the initial controller exit1
instead reflected incomplete immediate shutdown verification.

Worker time was51.2960s and client63.1893s. At22:53:22.201691UTC, a no-new-
app/all-stopped cutoff observed rounded credit usage+$0.07 and detailed
usage+$0.07047792, actual paid$0. Conservative closed allowance$0.200113 fits
the reviewed$0.58 reserve. Delayed workspace deltas are not exact app invoices;
no later jobs were assigned to this cutoff. Zero retries and160/170/185/20/210-
second worker/child/function/startup/watchdog limits were retained.

The validated composition remains the working source at official1.233633059×,
with A2-11 rollback and fusion control unchanged. No official quota was used.
The main research direction now shifts from small one-path savings to a
quantitative public-leaderboard gap plan and multi-path mainloop/data-movement
redesign. Existing passing compile evidence is retained as a reusable mechanism
result; no further alpha thresholds or lowering variants are launched.
