# B32/L8 FA3 counters collected; graph transfer remains inconclusive

One unchanged selected-baseline batch completed in app
`ap-FrbiCpeL0XEh1biFdE7zTF`, from frozen commit
`37b21ebc5ec4a3c51a6fdb41ad41b9d54f15880f`. Collection and correctness passed;
the predeclared exact graph/eager symbol-identity gate did **not** pass. Keep
those outcomes separate. No extra GPU attempt or post-result gate relaxation
belongs to this result.

The case is full verify B32/L8, TP8 (16 Q / 2 KV heads), D64, page1,
nominal context 80000, agentic lengths, coherent inputs, unit scales and
seed 135741. Selected source SHA256 is
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`.
Its actual dispatch is `kernel_fn` → `_fa3_full_verify_fused`, retaining the
fused preparation, public FA3 and norm/gate. External Q/K/V are FP8 E4M3 and
output is BF16. Scheduling metadata stays None; split 0 and pack None are API
inputs, not resolved native policy. The L1 M64N8 producer was not included.

## Whole-call primary measurement and correctness

These are unprofiled whole-`kernel_fn` CUDA-graph samples on one fixed fixture,
with input restoration and 256 MiB L2 eviction outside the event interval. No
empty-graph cost was subtracted. They are not official scores or code gains.

| Sample phase | Three raw samples, µs | Median, µs |
|---|---|---:|
| Before Torch activity |383.136,385.152,380.480|383.136|
| After Torch activity, before NCU |381.280,381.216,381.344|381.280|
| After NCU terminated |383.968,380.064,381.056|381.056|

All 14 main-process graph checks and all three eager-child checks passed.
There were five actual original-oracle calls: three in the graph process and
two in the NCU child. Output/new-KV byte invariance, immutable inputs, non-new
cache ownership and checked graph addresses passed. The post-NCU oracle and
three graph samples were completed before interpreting counter output.

## One valid eager counter report

Installed NCU 2025.3.1 version/help and all four fixed metric queries completed
before one profiling invocation. It used the previously tested kernel-replay,
application-only, skip1/count1 configuration; no CUDA graph, NVTX, application
replay, clock override or permission change was added to the NCU child.
Exactly the middle of three eager selected calls was targeted. The NCU command
exited 0 in 19.310275 s; the worker completed in 61.937420 s.

Independent parsing of the preserved raw CSV found one actual kernel row,
275 columns, correct percent units and no percentage-range warnings:

| Exact requested metric | Value |
|---|---:|
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` |11.47%|
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` |36.77%|
| `smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct` |43.40%|
| `sm__warps_active.avg.pct_of_peak_sustained_active` |12.49%|

The three ordered child markers (`pre`, `measured`, `post`) all passed; their
output/new-KV digests agree with the graph result. No timeout or permission
denial was observed. NCU also emits automatic launch/device fields; only the
four metrics above were explicitly requested.

## Identity gate: preserve the failure

Grid `(132,1,1)`, block `(256,1,1)`, complete input fingerprints, output/new-KV
bytes, wrapper and loaded-file mapping identities match. The Torch activity
contains the 1380-character mainloop name; NCU reports a 425-character simplified
name. The exact name comparison after whitespace normalization is **false**, so
`necessary_identity_checks_passed=false` and the raw conclusion remains
`inconclusive_eager_graph_proxy_mismatch`.

The installed help captured in this batch explicitly shows
`--rename-kernels arg (=1)`: NCU simplifies demangled names by default. The
observed output drops namespaces, prints boolean arguments as 1/0 and uses a
`Params` alias. This supplies a concrete naming explanation to test; it is not
permission to equate the strings after seeing the result. Neither the raw gate
nor frozen code was changed. Opaque split/capture-state equivalence remains
unproven even if a later full-name comparison succeeds.

## Conditional hypothesis and next measurement

For the **profiled eager launch**, dependency latency with limited latency
hiding is a candidate explanation worth testing: long-scoreboard stalls are
43.40%, active warps 12.49%, tensor-pipe activity 11.47% and aggregate DRAM
throughput 36.77%. This does not establish HBM bandwidth saturation, identify a
page-table/gather instruction, prove register pressure caused the low active-warp
value, or measure useful tensor FLOP utilization. The percentages have different
denominators and are not fractions to add together. The failed identity gate
prevents transferring this diagnosis to the timed graph as a confirmed bottleneck.

The next smallest discriminating measurement is to retain the same source,
fixture, correctness, whole-call and metric protocol while disabling the
documented display simplification with `--rename-kernels 0`, after validating
that option against the installed help. Retrieve full symbols and retain the
grid/block and scheduling-state checks. This tests the naming explanation;
do not normalize away namespaces or booleans post hoc. No candidate rewrite is
supported yet, and no new GPU measurement was performed for this report.

## Closure and costs

The app was independently verified **stopped, tasks 0** at
**2026-10-27 05:00:58 UTC**, with all listed apps stopped and zero tasks.
Actual billed spending remained **$0**. The observed account ephemeral-cost
delta was $0.05742655 at that cutoff; it may include delayed metering and is not
a final per-app invoice. Reserved free credit was at most $0.70. The existing
selected/ranked implementation remains unchanged; no official check, ranked
submission, new credential or security change was made.

Raw source bindings, full symbols, original samples, trace and NCU stdout/stderr
are preserved in `result.json` and the separately indexed artifacts. This is
Modal-image evidence, not proof of the official judge binary.
