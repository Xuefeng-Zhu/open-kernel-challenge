# Sliding verify B16/B32: test split-combine elimination, not launch-count intuition

This is local preparation only. The baseline is the newly validated fusion
candidate at private commit `05c9cc08948c42fed9e09aae1cd2b41edf966c74`, source
SHA256 `5c3d1c5b00f0e759f1c499b8f567a0f47c77ce0fa38566f3028a571fccc614a0`.
Original A2-11 remains selected. The sole proposed algorithm-facing change is
the supported FA3 `num_splits=1` argument for sliding B16/B32, L8, window2048,
RoPE500000. B1, decode, full attention, the validated fused preparation, output
gate, precision, paging, scaling and causal/window arguments must stay unchanged.

The installed Python interface, exact package/source identity and the
force-one specialization remain admission gates owned by the source/runtime
audit. Pinned upstream source is explanatory evidence, not proof of the next
Modal image's installed behavior. No GPU, Modal app, official check, profiler,
installation, credential change or external Pro message was initiated here.
Projection owns the serialized GPU slot.

## Existing executed evidence and the causal question

Official check `2f5f1fb54b48486d` passed16 tests and12 scored shapes for the
fusion baseline. Its actual sliding-verify profiles contain all of:

| Target | Mainloop activity | Combine activity | Varlen scheduler preparation | Sum of activities | GPU busy union |
|---|---:|---:|---:|---:|---:|
| B16 | 10.5µs | 4.2µs | 1.8µs | 19.7µs | 18.3µs |
| B32 | 14.1µs | 4.4µs | 1.8µs | 24.2µs | 22.7µs |

These are rounded0.1µs activity observations, separate from calibrated
complete-call timing. The profiles explicitly report overlapping kernels/PDL;
combine activity includes waiting and cannot be subtracted from whole-call
latency. The trace does not establish the default split count, CTA grid,
residency, or the unsplit mainloop's cost. The fact that scheduler preparation
is present today does not imply it will disappear with one split.

The hypothesis is that removing cross-split combine saves more complete-call
time than the unsplit path loses in parallelism. Under the conditional geometry
of one packed M≥64 query tile per batch/KV-head for L8×GQA8, Hkv2, the useful
query work count would be32/64 tasks on132 SMs. These counts are not measured
CTA grids, active-SM counts or occupancy, and depend on the actual installed
packing/tile specialization. A persistent launch can have more CTAs than useful
tasks. Splits can be useful precisely because the unsplit query work is small.
No mainloop penalty or recoverable latency is inferred from these counts.

## Conditional score value, using the current official baseline

The official calibrated complete-call times are22.892704769µs sliding and
197.827062966µs full for verify16, and27.459687322µs sliding and
382.479524745µs full for verify32. With weights46 sliding/16 full, their load
costs are4218.297426832µs and7382.818012738µs. Six loads have equal weight in
the geometric mean. Freeze every other path and replace each target complete
call time by its time minus an assumed saving `s`:

`factor = [(C16/(C16−46s)) × (C32/(C32−46s))]^(1/6)`.

| Assumed saving in each complete target call | Conditional overall gain |
|---|---:|
| 1µs | +0.2873% |
| 2µs | +0.5782% |
| 3µs | +0.8726% |
| 4µs | +1.1708% |

Equal savings of1.7324/3.4288/5.0899µs would be needed to model0.5%/1.0%/1.5%
overall gain. These are mathematical sensitivity scenarios, not forecasts,
confidence intervals, activity subtraction or achieved results. In particular,
5.0899µs is not a claim that such savings are recoverable. The same load weights
make B16 savings more valuable than equal B32 savings. `model.py` binds the
baseline source, official receipt and raw profile artifact hashes, reconstructs
the existing1.225555436× aggregate, and writes all arithmetic to `model.json`.

## Local source gates before compute

The complete candidate must differ from the fusion baseline only in the FA3
split choice for the two intended config routes. Compare all other function
ASTs and call expressions exactly; dispatch sentinels should cover batches
1/2/8/15/16/17/31/32/33, L1/7/8/9, full/sliding and mismatched window/RoPE
configs. Preserve the explicit causal flag, `(window−1,0)` window bounds,
softmax scale, live page/cache tensors, descales and None verify metadata.
No dtype conversion, query padding, cache mutation rule, split-count sweep or
new scheduler should accompany this change.

Hash-bound local host tests can validate config selection, parameter forwarding
and unchanged preparation/cache formulas. CPU attention stand-ins may confirm
the selected calls obey the original oracle and fixed-address A→B→A contract,
but they cannot prove an installed FA3 flag, native mask, instruction path or
performance. Reuse the exact official reference/check hashes; do not change
tolerances or import the oracle into candidate code.

## Minimal native protocol, only after a parent slot grant

1. Before expensive work, record the installed module/wrapper path, package
   versions and readable source or binary hashes; inspect the actual signature
   and supported one-split argument. Stop if unsupported or semantics cannot
   be established. This is not an invitation to install another FA3 or bypass
   a permission failure.
2. Use complete baseline/candidate entrypoints. For each batch16/32 L8, test
   the scored-family page1/scale1/ctx80000/agentic/soft case with two independent
   seeds and live GPU lengths/page mappings. Add a page64/scale1⁄64 soft boundary
   case with histories0/1/63/64/65/2047/2048/2049 and a distinct reversed vector;
   ensure new physical slot0 is live, page append crosses boundaries, unused
   table entries are−1 and all corresponding positions/cache lengths/locations
   are rebuilt. Verify metadata staysNone.
3. Separately add a page1/unit-scale `dist=edges`, fixed-ctx2048 fixture for each
   batch. Generate its edge keys from the actual official generator, rather
   than changing lengths after edge placement and claiming the keys remain at
   the window boundaries. Two seeds change QKV/page mappings even when this
   dedicated fixed-length edge case has the same length vector. The soft
   boundary/scored cases, rather than this fixed case, establish changed-length
   replay. Keep all eight verify-token causal rows under the unchanged oracle.
4. For these six cases, capture each complete entrypoint privately and replay
   A→B→A at fixed tensor/output addresses:36 unchanged oracle calls. Add one
   unchanged full-B16-verify scored-family control, using its validated fusion
   route and two fresh inputs, for six more A→B→A oracle calls. Each A2
   must return exactly its own first-A output/new-KV bytes. New cache bytes
   should match across arms because preparation is unchanged; output bytes
   need not match across arms because the reduction partition changes. Check
   all immutable inputs and untouched cache rows. Submit two candidate graphs
   with disjoint writable caches on separate streams for each boundary target,
   adding4 oracle calls; claim concurrent submission correctness, not measured
   hardware overlap. Any native oracle/cache/replay failure stops timing.
5. Use one bounded Torch/CUPTI activity diagnostic of the two scored graphs
   per arm. Retain the actual kernel names, graph-node/grid information when
   supported, stage counts and raw trace. The control must actually call
   combine and the candidate must omit it. Record varlen scheduler preparation
   independently: it may remain and is not counted as a saving. Stop this
   launch-removal hypothesis if combine is absent in the control or remains in
   the candidate. Do not force a different packing/scheduler path to obtain the
   expected names. No NCU counters or counter retries are part of this test.
6. Stop the profiler, then do four balanced warmup two-arm samples and twelve
   measured two-arm samples per scored target and the unchanged full-B16-verify
   control, grouped into six reverse-order
   pairs using the same input version per pair. Measure cold-L2 whole-call
   CUDA graph device work, with256MiB eviction and input/cache restoration
   outside the event interval. Check every replay against its own previously
   oracle-validated output/new-KV bytes. Retain all samples and raw/empty-graph
   diagnostic values; profiler durations must not become benchmark timings.

This is a proposed minimal protocol, to be reconciled with the reviewed worker
before an immutable source freeze. The trace must be bounded and its capture
overhead kept separate from performance claims. If the whole workload cannot
fit the reviewed timeout/reserve, tighten case scope transparently or stop;
do not silently extend a running job.

## Predeclared benefit and failure gates

The combined two-target candidate warrants one official check only if installed
semantics and actual combine elimination are established, all native gates
pass, and both complete targets save at least2µs by the median of their six
balanced absolute differences. For each reverse-order pair, that difference is
the arithmetic mean of its two `(control_us−candidate_us)` observations on the
same fixture; do not subtract pooled medians across unlike input versions.
Require all six geometric pair ratios above1 for each target. This models at
least about+0.578% overall under the fixed-cost baseline; it is a materiality
criterion, not a forecast or significance threshold. Require no unexplained
>5% regression in the unchanged full-B16-verify control. Preserve all
rejected/unfavorable samples. A single session still needs cautious uncertainty
language, and any promotion needs complete official coverage and review.

If any precision/cache/causal oracle fails, the API is unsupported, combine does
not disappear, or useful complete-call benefit fails, reject this combined
candidate without spending an official check. Scheduler preparation persisting
alone is not failure. If only one target improves, a restricted future candidate
would need a separate, explicit evidence question and local gates; do not
automatically consume remaining quota on ablations. No trial will be justified
solely by having three checks remaining before00:00UTC.

For planning, the prior210s watchdog/one-H200/CPU2/8GiB envelope has conservative
double-live-rate cost0.56817USD plus its included0.02USD slack, covered by a
0.58USD free-credit reserve at the last observed rates. Exact worker/child/
function bounds and trace transport need review for this new workload. Fresh
free-credit and paid0 guardrails, existing supported credentials, a parent
serialized-slot grant and independent stopped/tasks0 closure remain mandatory.
No source of authority here permits paid spending, new credentials, installing
tools, ranked submissions, visibility changes or another job without admission.

Local arithmetic checks:

```sh
python3 -B reviews/attention-v2/sliding-unsplit-probe/test_model.py
python3 -B reviews/attention-v2/sliding-unsplit-probe/model.py
```
