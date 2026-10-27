# P068 — baseline collector assembled; native access still required

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
