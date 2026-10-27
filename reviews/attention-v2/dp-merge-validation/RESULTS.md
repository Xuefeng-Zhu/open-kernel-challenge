# DP multi-chunk merge correctness passed

The unchanged working composition passed the two predeclared native DP cases,
including two-split and three-split reductions. All15 original-oracle calls,
four exact A→B→A return checks, twelve whole-cache K/V byte comparisons and
four independent-graph isolation checks passed. Candidate algorithm, dispatch,
precision and original tolerances were unchanged. This is additional correctness
evidence, not a new latency result or official score.

| DP full B32/L1 case | Shape-bound tokens | Splits / merge padding | Source-derived FP32 partial bytes | Result |
|---|---:|---|---:|---|
| page64, ctx8193, scale1/64, soft | 8256 | 2 / BS2 | 4,325,376 | Pass |
| page1, ctx16385, scale1, coherent-generated then boundary-remapped | 16386 | 3 / BS4 | 6,488,064 | Pass |

Both use128 query/16 KV heads, D64 and fresh seeds135721/135722. The second
case covers a non-power-of-two split reduction and the unused fourth merge
lane. Vectors include empty history, page/tile/8192-token chunk boundaries and
last-partial tails; their reversals and fresh page mappings replace live tensor
contents at fixed addresses in the same captured graphs. Live physical page0,
negative unused page-table suffixes, last-split new-KV ownership, immutable
inputs and untouched cache rows all passed the unchanged checks. The coherent
generator's explicit boundary remapping is an adversarial fixture, not a claim
to reproduce the untouched official scored distribution.

The fifteen oracle calls consist of twelve two-arm A→B→A checks, one own-A
setup for a second independent candidate graph, and two checks after separate-
stream graph submission. Each graph owns its own previously oracle-checked
expectations; none are transferred between graphs. After submission, two
additional isolated replays change the primary graph to B and the independent
graph to A. Four checks confirm the active and inactive graph output/new-KV
bytes and immutable/untouched state. Both graph output/cache address ranges
were disjoint and retained stable addresses. Installed graph-pool handles were
observed as distinct `[2,0]` and `[3,0]`.

Source audit establishes fresh per-call FP32 partial allocation and the decode
counter-free separate-merge route. Internal scratch addresses are not returned
and were not directly inspected. Independent streams and graph pools do not
prove simultaneous hardware execution; no activity profiler or counters were
introduced. Same-graph overlapping replays and universal serving concurrency
remain outside this bounded validation.

## Exact source and execution

Preparation freeze: `8d38550d061af14df3257afe0d438dbf0fc82f5f`.

| Artifact | SHA256 |
|---|---|
| Unchanged working candidate | `3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8` |
| Unchanged fusion control | `5c3d1c5b00f0e759f1c499b8f567a0f47c77ce0fa38566f3028a571fccc614a0` |
| Worker | `4507a19cdaea450415b6d7a40315e9e0021741919f5a33842fed290e0f2af59d` |
| Controller | `0299612f835e68b26ece31f5d5d031c959953988ce4d998d95941ddd9d778f17` |
| Raw native result | `3640d3df3c6d31deff277c98d04e1bdc4d685a45d1d5ee35333932301b7cae47` |

Seven worker and eight controller offline tests passed before the independent
source/protocol/harness review and real frozen-head preflight. Review caught an
uninitialized expectation in the second graph before compute. The predeclared
protocol added its own oracle setup, making15 total rather than weakening the
isolation check. The frozen worker was the sole executed payload.

One existing-image H200 app`ap-gXUKhAM11JBfvrA3h5rip9` was admitted, with
zero retries and160/170/185/20/210-second worker/child/function/startup/watchdog
bounds. Worker time was23.4248s, wrapper24.9274s and client34.2999s. Native
environment: H200 sm90, Torch2.14.1+cu130, CUDA13.0, Triton3.8.0,
Python3.12.3, existing image`im-cPTbGpNiX3RarKResPgX7h`. No new installation,
credential, permission, precision, billing setting or official request occurred.

## Closure, cost and next decision

The app stopped at21:51:32UTC; independent21:52:08.731188UTC reconciliation
confirmed stopped/tasks0 and all listed apps stopped/tasks0. The parent received
immediate slot release before report polishing. The original controller receipt
observed stopping/tasks1 and remains unchanged; controller exit1 signaled that
initial incomplete closure, while worker exit0/status passed are separately
preserved and independently reconciled.

At21:53:27.435772UTC, a further accounting read verified no new apps and all
apps stopped. Rounded workspace usage increased$0.03, detailed usage$0.03730389,
actual paid$0. Conservative closed allowance is$0.1244133334, below the reviewed
$0.58 free-credit reserve. These delayed workspace deltas are not an exact
per-app invoice; the retained cutoff excludes any later measurement. Initial
unchanged billing was not reported as free GPU execution.

The DP multi-chunk/merge gap identified in the preceding report is closed for
these two declared cases. The working source remains composition, with A2-11
and fusion preserved as fallbacks/controls; prior official score1.233633059× is
unchanged. No official quota was consumed or duplicate-source check submitted.

The next algorithm question is a strict conditional numerator rescale when
every valid query row has `alpha==1`. Its source audit and experiment card are
being prepared separately. It must first prove generated skip behavior and
resource preservation, then measure rescale frequency outside performance
timing and pass fresh-input original correctness/replay checks. This report
does not start or reserve another measurement. Grouped-request M64/WGMMA is
deferred by its arithmetic/resource model in`NEXT_ALGORITHM_ANALYSIS.md`.
