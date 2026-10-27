# P068 — baseline collector and bounded native diagnostic

## Native result: completed, 2026-10-27 03:35 UTC

**258 original-oracle comparisons passed**: 240 in the uninstrumented timing
stage and 18 in separate activity/eager checks. All nine traces were retrieved.
P046 remains selected at official **1.1841399409×**; no candidate changed and
no official check or new performance score was produced.

- Frozen launcher commit: `1ffbcddf6bcad200531854b31d804eeae189e74c`.
- App: `ap-Hw1HKh6pn7adn4Tc5E0lXE`; native collector wall time 12.876 s
  (timing process 6.995 s, activity process 5.848 s). These include setup and
  checking, not GPU-only active time. Local client lifecycle was 22.984 s.
- Independent observation at **03:35:45 UTC**: app stopped, tasks 0, all apps
  stopped. The measurement slot was released. No uncertain invocation remains.
- H200, SM90, 132 SMs; Python 3.12.3, Torch 2.14.1, Triton 3.8.0, CUDA 13.0.
  Trace driver/runtime/CUPTI versions: 13030 / 13000 / 130001. Package metadata
  also reports CuTe DSL 4.8.0, TVM FFI 0.1.11, Quack 0.6.5 and einops 0.8.2;
  none of those optional libraries was imported or benchmarked here.
- Base image ID `im-cPTbGpNiX3RarKResPgX7h`; immutable image digest unavailable.
  Seven loaded-library hashes are saved. They do not identify the exact vendor
  kernel binary or prove that this test image equals the official judge image.

| Case | Raw graph medians A / B (µs) | Paired B/A latency movement | Separately instrumented phase medians (µs) |
|---|---:|---:|---:|
| QKV M128 | 21.632 / 21.536 | −0.121% | GEMM 14.058; reduction 3.456 |
| Pooler M128 | 28.288 / 27.984 | −1.979% | GEMM 19.522; reduction 3.424 |
| Pooler M8 | 22.096 / 22.320 | +0.324% | Single kernel 17.215 |

A and B are **identical P046 graphs**, so these movements are repeatability
evidence, not improvements. Each case retained all 24 measured alternating-order
rounds (12 reversed-order pairs), after six predefined warmup rounds. Pooler M128
is just below the 2% median-drift flag and 11/12 pairs favor B. Pooler M8's pairwise
latency movement ranges from −2.91% to +5.61%. A single session cannot reliably
establish a small 1–2% candidate benefit. No replay-cost calibration was applied.
Eager CUDA-event and synchronized host times are separately retained and are not
judge-equivalent performance measurements.

The visible QKV M128 and pooler M128 paths contain a vendor GEMM followed by a
separate FP32 split-K reduction. The main kernels report 168 registers/thread,
384 threads/block and shared memory of 221,492 and 213,252 bytes respectively;
their grids are `[8,12,1]` and `[2,64,1]`. Pooler M8 has one vendor kernel with
grid `[4,16,1]`, 384 threads/block, 168 registers/thread and 164,308 shared bytes.
The symbols match prior saved evidence; M8 also reproduces P053's launch fields.
This confirms dispatch sequences and static resource metadata, not achieved
occupancy, a tile-to-CTA map, internal persistence or actual overlap.

No material positive gap appears between main and reduction in these traces.
Tiny negative gaps down to −0.032 µs are retained as timestamp uncertainty, not
interpreted as overlap. Reduction represents about 19.73% and 14.92% of the sums
of instrumented phase medians, but those percentages are **not removable latency
or whole-score forecasts**: changing finalization can change the mainloop.

`ncu`, `nsys` and `nvcc` are present, and the read-only driver parameter reports
`RmProfilingAdminOnly=0`. This does not establish working counter collection.
No standalone profiler/counter invocation or permission change was attempted;
the two earlier NCU timeout recipes were not repeated. Torch/CUPTI activity
collection succeeded. Exact vendor cubin/tactic, per-CTA tile/K assignment,
buffer/barrier ownership and hardware traffic/stall counters remain unavailable.
P067's Stream-K decision therefore remains unresolved; launch-grid arithmetic
alone is insufficient justification for an implementation.

Coverage includes same-address A → B(outliers) → A, sparse power-of-two and zero
inputs before and after timing; original weights, prepared state and inputs
were checked for mutation. It covers three shapes and one weight seed, sequential
graph replay only. Capture warmups/executions were not individually oracle-checked;
capture input/state integrity was checked. Concurrent calls, all official shapes
and a new optimization were not tested.

Before launch, the browser and CLI confirmed $27.53 free credits, $0 spend cap,
$0 paid usage and no active apps. Post-run billing still showed $2.47 cumulative
usage and $0 paid, with no newly reported delta. This is delayed billing, **not a
claim of zero run cost**. Retain the $0.579916500143 free-credit reserve until
accounting catches up. At 03:40:29 UTC, a further read reported a $0.02
workspace usage increase and still $0 paid. That rounded, delayed workspace
delta is not a settled per-app invoice. No credential contents were read,
renewed or published.

Evidence: [assessment](../results/projection-p068-assessment.json),
[timing](../results/remote-P068-baseline/timing.json),
[activity](../results/remote-P068-baseline/activity.json),
[lifecycle](projection-p068-modal.json). All raw samples remain in the structured
receipts. Original full traces/logs remain ignored and owner-only locally;
published traces retain only kernel/copy/memset intervals and selected launch
fields. Raw and published hashes are paired in the assessment. No vendor binary
is redistributed.

Exact launch (already completed; **do not rerun this ledger**):

```bash
python -B projection_p068_modal.py launch \
  --frozen-head 1ffbcddf6bcad200531854b31d804eeae189e74c \
  --control-root "$CONTROL_ROOT"
```

The first local invocation used an abbreviated commit and was rejected by the
40-character freeze gate before any service call/reservation. The full hash
invocation above created the single app. All 51 CPU launcher/collector/helper
tests passed. Offline result validation and publication used
`projection_p068_result.py`; no second native call was needed.

## Resumed acquisition, 2026-10-27

Existing runtime access was restored separately. The unchanged collector's 12
allowlisted files total 138,546 bytes; selected P046 remains frozen at SHA-256
`001ea12e8764564d507de51e433769f5bd9a9916bfc4c014f0f87522c278545e`.
The resumed batch is a baseline diagnostic, not an optimization or official run.
Its three cases are QKV M128, pooler M128 and pooler M8. The planned checks remain
240 original-oracle timing comparisons and 18 activity/eager comparisons.

The external launcher adds one ephemeral H200, fixed 2 CPU / 8 GiB limits,
300-second function and 45-second startup limits, no retries, and a 330-second
app watchdog. A 390-second reserve with 10% margin plus $0.02 covers the planned
lifecycle at the verified resource rate: $0.579916500143 of free credit.
This is a conservative reservation, not a settled charge. The collector retains
its own 90-second internal / 120-second hard limit for each of two stages.
App shutdown and tasks 0 must be verified independently, even if a child or
transport step fails. No counter query, permission change, credential creation,
package installation or candidate edit is part of this batch.

The existing 38 collector/helper/model CPU tests passed again before launch.
The native result and verified cleanup are recorded above.

## Original preparation checkpoint (historical)

Status at 2026-10-27 00:00 UTC: **CPU-tested diagnostic tooling, no native run**.
P046 remains selected at official 1.1841399409× (job `41ba778763a5433b`).
No candidate, numerical tolerance, judge input or score changed.

## Deliverable and commands

`projection_p068_collector.py` implements the bounded P067 baseline acquisition:
QKV M128 and pooler M128, with pooler M8 as the single-kernel control. This is an
A/A repeatability/activity diagnostic, not a new optimization comparison. The
hypothesis and Stream-K rejection rules remain in
[`P067_BASELINE_PROFILING_PROCEDURE.md`](../docs/P067_BASELINE_PROFILING_PROCEDURE.md).

The collector checks 11 frozen source/reference/license files and hashes itself,
giving an explicit 12-file payload. P046 SHA256 remains
`001ea12e8764564d507de51e433769f5bd9a9916bfc4c014f0f87522c278545e`.
The dry run lists every path/hash. No cloud client, credential access, package
installation, automatic upload or NCU invocation is included.

From this checkout, Python 3.12, offline and without Torch import:

```bash
python -B projection_p068_collector.py plan --output-dir /tmp/p068-baseline
python -B -m unittest tests.test_projection_p068_collector tests.test_projection_p046_confirmation tests.test_projection_p053_analysis tests.test_projection_gap_model -v
```

Inside a separately authorized Linux H200 ephemeral container with the allowlisted
payload and existing dependencies, the exact native entry point is:

```bash
python -B projection_p068_collector.py run --output-dir /tmp/p068-baseline
```

The output directory must be new; its parent must exist. Optional
`--image-digest sha256:<64 lowercase hex digits>` records caller-provided
provenance, not an independently verified identity. Neither command provisions
anything. Never upload this collector as an OKC scoring candidate.

## Implemented protocol, not H200 results

| Stage | Planned execution | Oracle checks if complete |
|---|---|---:|
| Timing | Two identical P046 graphs, three cases; A → B(outliers) → A-return, sparse and zero before/after; six warmup rounds and 24 alternating-order A/A rounds (12 reversed-order pairs of rounds) per case | 240 |
| Activity | Fresh process, same source/environment; three instrumented graph replays and three separate eager full calls per case | 18 |

Original FP64 tolerances remain. Fresh X contents retain the captured address.
Original/prepared weights, input/config/metadata and A-return outputs are checked.
Capture-time mutation is checked before fresh data can hide it. Capture includes
18 eager warmups + six capture calls in timing and nine warmups + three capture
calls in activity; those calls are not individually oracle-checked. Separate
outputs do not establish independent hidden vendor workspace or concurrency safety.

Each measured replay follows 256 MiB L2 eviction. Both timed members finish
before either oracle. All samples/tails remain; six warmups are excluded by
design. Gross CUDA-event whole-`kernel_fn` graph times retain fixed replay cost.
A/A paired drift above 2% is flagged, not reported as a gain. Instrumented
intervals and eager CUDA-event/synchronized host-wall times are separate
descriptive evidence. Three cases cannot produce a whole-problem score.

Activity requires a complete timing receipt before Torch import: source hashes,
environment/DSO identity or explicit absence, all cases/samples/seeds/orders,
oracle counts and recomputed A/A summaries. Runtime identity must then match.
Every trace binds case, source hash, repeat and raw trace hash; missing launch
metadata stays null.

## Identity, counters and cleanup limits

Records include Python executable hash, package versions, CUDA runtime, GPU/SM
identity and precision flags. Loaded allowlisted CUDA/Torch DSOs get basename,
byte count and SHA256 where readable; full maps/paths are not exported. A DSO
hash is **not** an exact vendor kernel/cubin/tactic binding. Vendor instructions,
CTA work assignment and driver version remain absent unless supported trace
fields provide the latter.

Counter requirements remain explicit: per-SM work/activity, PC-correlated
synchronization waits, DRAM/L2 traffic and warp/issue activity. No counters are
collected. Prior NCU timeouts are not retried; a supported capability change is
required. Grid/132 and static resources are not achieved bandwidth, occupancy
or stalls. This collector cannot alone settle the complete Stream-K question.

Two sequential workers each have a 90-second internal deadline and 120-second
supervisor bound, plus short termination/reaping grace. Failure/timeout stops
later stages. The controller signals the process group and reaps its direct
worker; it does not independently verify descendant absence or cloud shutdown.
`cloud_app_stop_verified` remains false. The external owner must verify credits
and net paid $0, reserve startup/run/cleanup cost, enforce app lifetime, serialize
the GPU, stop the app and verify stopped/tasks0. CPU tests/process bounds are not
a cost quote.

Outputs are owner-only (directory 0700, logs/JSON 0600). Raw `.local` logs/traces
may contain native paths and are never automatically published. Review/sanitize
future results before private push. Partial receipts/errors are retained;
timeouts are not correctness passes.

## Actual verification and review

CPython 3.12.13 on Darwin: **38 CPU tests passed in 1.286 seconds** on 27 October.
Twelve cover this collector; 26 cover reused P046 helpers and P053/score analysis.
Tests include an input-dependent stale-output oracle, capture/input mutation,
malformed receipts, event/oracle ordering, payload hashes, dry-run import
isolation, real CPU process-group termination/worker reaping and prevention of
activity after failure. AST parsing and the frozen 12-file dry run also pass.
No native module or GPU code executed.

Independent review found capture-time mutation and incomplete receipt-identity
gaps; both are repaired. The new capture test first failed because its fake Torch
omitted `uint8`; the fixture was corrected and the suite passed. An earlier
test expected `/tmp` rather than macOS's `/private/tmp`; its expectation now uses
`Path.resolve()`. Neither is a GPU failure/result. Review found no remaining
blocking issue.

## Stop state and exact prerequisite

P068 started no Modal app, official check, credential action or paid resource.
No new job or uncertain admission exists. Old stopped-app receipts remain
historical; no cloud/billing query occurred here. The known Modal credential
expired at 2026-10-26 03:39:46 UTC. Valid authorized runtime access, an external
app watchdog and fresh billing/active-app preflight are required. Credentials
were not renewed. Projection pauses here with P046 preserved.
