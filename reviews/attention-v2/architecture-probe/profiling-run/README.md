# Bounded hardware profiling: activity complete, counters timed out

The fixed B32/L8 full-verify profiling task is closed. One CPU inventory and two
serialized H200 apps ran, with exactly one dual-arm activity session and two
FA3 counter attempts total. No more profiling jobs are permitted in this stage.
The three stage apps are stopped/tasks0; the measurement slot is released.
No algorithm, selected candidate, precision, formal route or billing setting changed.

## Resource closure and accounting first

Independent CLI verification at2026-10-25T18:44:16.949229UTC confirms:

| Stage | Exact app ID | Final state |
|---|---|---|
| CPU inventory | `ap-1W3buQKnds7sDfi6eVnJIo` | stopped, tasks0 |
| Activity + first NCU attempt | `ap-aLYZMcYLfYZoshhSzwyZ7y` | stopped, tasks0 |
| Final one-metric access check | `ap-mEXdTixH5oBF9hXapbXGxj` | stopped, tasks0 |

The last app stopped at18:40:40UTC; the independent closure receipt is
`RESOURCE_CLOSURE.json`. Initial immediate reads that saw stopping/tasks1 are
retained in exact published launch records, with later reconciliations separate.
There is no uncertain request, active reservation or pending profiling resource.
Projection may use the serialized measurement slot; no new profiling job follows.

Observed workspace usage rose$0.26 rounded, or$0.25824844 in detailed ephemeral
usage, across this stage. Actual paid cost remains$0. Conservative closed
runtime allowances total$0.581925, below the approved$0.67 reservation. Source
snapshots end18:41:49UTC; later projection charges must not be attributed here.
The bill revised during observation, so no exact per-app invoice or refund is
inferred. See `COSTS.json` and each stage's `run-receipt.json`/`reconciliation.json`.

Existing free-credit/$0spend-limit state was read live before launches. The first
new billing-tab action was automatically rejected under the earlier browser
restriction. A new direct human authorization allowed only this read-only Modal
billing operation; a single retry succeeded. The rejection and resolution are
preserved in `control/browser-block.json`. ChatGPT/Pro tabs and settings were
untouched. The artificial$2stage ceiling remains removed; actual paid authority
stays$0. No credentials were copied/renewed, no install or permission expansion.

## Installed tools: measured inventory, not assumptions

The unchanged image is `im-cPTbGpNiX3RarKResPgX7h`. The CPU-only worker completed
in0.997s and did not request a GPU or CUDA context. It found:

- Nsight Compute2025.3.1.0, build36398880, launcher `/usr/local/cuda-13.0/bin/ncu`.
- Nsight Systems2026.4.1.191, existing `/usr/local/bin/nsys`.
- CUPTI `libcupti.so.2025.3.1`, SHA256
  `e2f9ed861fe27c492b8bb52b5e3220ef5120f3edcda36312e96b7fd8a186be3e`.

No installation was needed. Tool presence, a successful metric-name query and
successful ordinary CUPTI activity do not establish hardware-counter permission.
NCU queried the installed H200 metric suffixes successfully and selected eight
supported requested names. Neither collection completed. No permission denial
was observed; access/permission therefore remains unknown.

## Fixed source and case

Both arms use full verify B32/L8, D64, Hq16/Hkv2, ctx80000, agentic lengths,
page1, scale1, coherent inputs, seed135701. Actual lengths are saved in the
raw summary and match the final access target. The source remains immutable:

| File | SHA256 |
|---|---|
| Exact A2-13 candidate / unchanged full-verify FA3 | `e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e` |
| BF16 experimental helper | `59fa2bf6aa1e97e55a23b821b56bdd5573b3fdf2edb0613913c2e93dc7acde82` |
| Activity worker | `07c6c36b06995df46b49b90208b0454eba1aa3efb17787c7a2fbdf67b1efa326` |
| First counter helper | `cdf06463a840d75ce8cd3e7ef497fa60c0481b63ff7de6300d4061b23ea12408` |
| Immutable NCU target | `910e0b3613681edf318d43df25356d8c9ed286b416c2757063d4c0d431142c3f` |
| Final counter-only parent | `7c2b3cfd6b53151cc69d0d17b7673d6f481be5ecbe891d3c851440d3df88f9f7` |

CPU payload freeze:`829ea288166a87dcc56b99d2f324e1f47979c153`.
Activity/counter freeze:`e26bb0cf291dfdbfc6e70065b6d121bf315423ca`.
Final access freeze:`052665dd2496f12952d684ee6fd5a4b0e1075228`.
Complete source/receipt/artifact hashes are in the final `manifest.json`.

## Activity result: stage placement only

One CPU+CUDA Torch profiler session captured exactly one fenced graph replay
per arm, after compilation/capture/warmup. Four unchanged official-oracle checks
before/after passed, including immutable inputs and untouched KV rows. Both
post-profile output/new-KV byte strings exactly match their own pre-profile
results. The complete62221-byte Chrome trace is retained unmodified at
`gpu/artifacts/activity-trace.json`.

| Instrumented observation | FA3 | BF16 custom |
|---|---:|---:|
| Recorded GPU activity union |393.536µs|589.089µs|
| Recorded GPU activity span |396.288µs|589.697µs|
| Mainloop / integrated producer+merge |378.752µs|588.417µs|

FA3 stage interval unions: QK norm2.304µs, three FP8 conversion events5.440µs,
cache store1.536µs, length-schedule preparation3.712µs, mainloop378.752µs,
combine3.296µs, output norm/gate2.144µs. Intervals show apparent overlaps,
including same-stream events; these recorded intervals do not independently
prove concurrent execution or scheduling behavior. They must not be summed
into latency. Recorded activity outside the mainloop interval totals14.784µs; this is not a recoverable
fusion saving or a new score model.

Custom has one integrated producer/merge kernel and a0.672µs FillFunctor<int>
event. Its destination is not independently identified by the trace. This
activity cannot separate internal QK, PV, softmax and ticket merge time.
CPU graph-launch API intervals and annotated spans are retained, but include
instrumentation and synchronization; they are not normal host-launch latency.

These are single-session instrumented layout observations, not performance
measurements or confidence intervals. The prior unprofiled paired benchmark
remains immutable and separate. High confidence: main compute dominates the
recorded device activity. Unknown: whether its cause is bandwidth, gather
latency, tensor issue, occupancy or a stall category. No new kernel tuning or
adoption follows from these traces. See `analysis/RESULT.md` and its independent
raw-trace reconstruction/negative admission checks.

## Exact counter limits: two timeouts, not permission denial

Attempt1 used installed NCU application replay with strict/grid matching,
node-level graph profiling, exact FA3 mainloop filter and an NVTX range, leaving
cache/clock settings unmodified. Eight names came from the actual metric query:
duration, DRAM read/write bytes, tensor activity, active warps/CTAs, long
scoreboard and barrier. The FA3 session hit its60s timeout. No numeric metric,
NCU report, target-check file or pass count was returned. The original helper
did not retain timeout stdout/stderr; this evidence limitation is explicit.
BF16's counter session was not started.

The remaining authorized slot was used for an access diagnostic: same FA3
case and unchanged target, only already-supported `dram__bytes_read.sum`, with
a standalone stdlib parent and durable bounded logs. Conservative closed costs
plus the$0.28 remaining reserve were$0.6692016667, within$0.67. This added no
activity session or algorithm change. Two diagnostic factors changed (metric
set and parent context); outcomes cannot isolate the first timeout's cause.

Attempt2 hit its55s timeout, exit124. Its retained stdout is exactly272bytes,
stderr0bytes, neither truncated. NCU warns about unmodified cache/clock settings
and reports connection to process29. The target's two oracle checks and exact
post-profile bytes passed, proving at least that target execution completed;
they do not prove all profiler passes or collection completed. No CSV metric,
pass count or `.ncu-rep` was returned, and no permission-denial signature appeared.
The complete logs and validation JSON are preserved under `counter-access/artifacts`.

Therefore DRAM bytes/throughput, tensor utilization, warp stalls, aggregate
residency and sampled tail remain unmeasured. Exact CTA-to-SM/start/end tracing
was not demonstrated or inferred. Attaching to a process is not counter-access
proof; a timeout is not a hardware bottleneck or an access denial. The specific
remaining blocker is NCU2025.3.1 collection failing to complete within the bounded
60s/55s application-replay sessions. Root cause is unknown. There will be no
third attempt in this stage. See `analysis/COUNTER_ACCESS_SUPPLEMENT.md`.

## Validation, next step and delivery

CPU inventory guards6, GPU worker/controller/counter guards22, final access
guards5, and independent artifact/negative-admission analyses passed. Every
frozen payload byte and published transport hash was checked. Secret-pattern
scans include decoded retained logs. Selected A2-11 and original source stay
unchanged; no official check, public/formal submission or visibility change.

Next step is read-only investigation of NCU2025.3.1 CUDA-graph/NVTX matching,
application-replay and completion behavior using these retained logs and
installed-version documentation. This is an observability question, not a
kernel-parameter sweep. Any future hardware collection requires a new bounded
plan; do not infer privilege failure or request broader permissions from these
results. No further profiling resource is running or scheduled.

Reviewed sanitized sources, raw trace, validation/timeout logs and receipts are
prepared for the same private branch. Delivery is established by the verified
remote SHA receipt, not this paragraph.
