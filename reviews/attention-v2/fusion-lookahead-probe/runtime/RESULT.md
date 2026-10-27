# Native composition replication passed

The composed source passed all native correctness, replay, cache and independent
submission checks, then met its new predeclared replication gate. This supports
one complete nonranking official check of the exact source. It is not an
official aggregate or a selected-version promotion.

| Whole-call family | Raw median control→composition | Median reverse-order ratio | Paired latency change | Favorable pairs |
|---|---:|---:|---:|---:|
| Full B32 decode target | 313.552→297.392 µs | 1.050670052 | 4.8226% reduction | 6/6 |
| Unchanged full B16 verify fusion control | 203.840→204.272 µs | 0.998469987 | 0.1532% slowdown | 2/6 |

All twelve target observations favored composition. Six adjacent reverse-order
geometric pair ratios give target reductions of 4.5092–5.5895%. Each pair uses the
same fixture version; versions alternate by pair. The control's largest pair
slowdown is about 0.638%, below its 5% gate. Raw pooled medians are descriptive;
paired ratios are primary. Four warmup and twelve measured two-arm samples per
family, all orders/versions and before/after empty-graph calibration remain
retained. Separate calibration is unclipped and is not official judge timing.
No profiler, hardware counters or device-overlap measurement entered the run.

The source audit verifies that only exact A2-15 `_fused` and `_e2_kernel_fn`
definitions replace the corresponding fusion definitions. Their guard enables
lookahead only for full B32/L1 without RoPE; all FA3 helpers/arguments, fusion
dispatch and automatic split policy remain unchanged. Original precision,
last-split new-KV ownership, private partials and merge are preserved.

Five cases with fresh seeds 135711/135712 passed: the scored B32 target, its
scaled page64/tile/chunk/empty-history boundary case, an unchanged B4 custom
fallback with lookahead false, a small B32 DP-head single-chunk smoke with
lookahead true, and the unchanged full-B16-verify control. There were 30 A→B→A
original-oracle checks, ten exact return-A checks, 30 cross-arm whole-cache byte
comparisons and two additional original-oracle checks after independent target-
boundary graphs with disjoint writable caches were submitted on separate
streams. All 64 warmup/measured replays exactly reproduced their own previously
oracle-checked output/new-KV bytes and passed immutable input, untouched-row and
fixed-address checks. Actual hardware overlap was not asserted.

This is one new session with two new random fixtures. The historical A2-15 run
used 135701/135702, measured a 5.189% target cut against A2-13, and missed its old
10% goal. The new ≥4% criterion was declared for composition/replication before
this run and does not change that historical conclusion. Both results are
consistent with a modest useful component, but they are not a confidence
interval, proof of the global score difference or evidence that page-address
scheduling alone caused the gain. The donor also changed generated buffering;
this run did not inspect new native assembly or hardware stalls.

Source freeze `c9062363c665e37535d3370c1b6bfb54e04fc7c8` uses control SHA256
`5c3d1c5b00f0e759f1c499b8f567a0f47c77ce0fa38566f3028a571fccc614a0`,
composition SHA256
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`,
worker `03c1261d8cb18b9c2ea266556463d087ee83d2da8cc121e1be80ff04fdc3cb15`.
Exact raw result SHA256:
`5e6447008a1f55baff65939aa2af75521235891fad976434ac378b4dbc55395d`.
Environment: H200 sm90, Torch 2.14.1+cu130, CUDA 13.0, Triton 3.8.0,
Python 3.12.3, fixed existing image `im-cPTbGpNiX3RarKResPgX7h`.

App `ap-EY8FNKYqqFMGgZAatYZXOQ` stopped at 21:15:59UTC. An independent read
at 21:16:21.481125UTC confirmed stopped/tasks0 and all listed apps stopped/tasks0;
the parent received immediate slot release before report polishing. The earlier
controller stopping/tasks1 observation stays unaltered in the published launch
receipt; worker exit0 and successful benefit admission are separate from the
controller's initial incomplete shutdown observation.

Worker internal time 54.297s; client wall 66.630s. Rounded workspace usage increased
$0.05 and detailed workspace usage $0.05481666 at the closed cutoff, actual paid $0.
The conservative closed allowance is $0.207944, within the reviewed $0.58 free-
credit reservation. These delayed workspace deltas are not an exact per-app
invoice; no later charges were assigned to this run. No retry or extra Modal
job occurred. A2-11 remains selected.

`analyze_result.py` verifies published request/source/transport hashes, exact
five-case sets, 32 oracle checks, replay/cache gates and all paired arithmetic.
Independent `RESULT_REVIEW.md` confirms the same result. The separately
authorized official check is recorded under `../official-check/` and must be
evaluated independently from this local paired result.
