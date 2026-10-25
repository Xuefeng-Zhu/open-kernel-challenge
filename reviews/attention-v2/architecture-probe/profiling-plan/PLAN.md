# Minimal profiling feasibility and measurement plan

Preparation complete; no new GPU/container job, installation, credential,
algorithm edit, billing change or private-repository push occurred this turn.
Current compute cost is $0. Existing activity tracing is demonstrated; Nsight
installation and counter permission remain unknown. The first future action is
capability inventory, not another optimization benchmark.

## Current admission and exact blocker

The existing image `im-cPTbGpNiX3RarKResPgX7h` previously ran Torch/Kineto CUDA
activity profiling successfully. It can provide a bounded CPU/CUDA stage trace
using the existing code path. `cuobjdump` also worked. Existing records do not
establish `ncu` or `nsys` versions/installation or NVIDIA performance-counter
permission. The Mac has CPU-only Torch and no Nsight executable in the searched
scopes; that does not inventory the remote image.

Pending action: admit a hardware-counter or temporal-residency diagnostic on
that H200 image. State: **not attempted/submitted, not denied; capability and
permission unknown**. The present local/preparation-only scope prevents the
remote inventory and actual permission probe. No `ERR_NVGPUCTRPERM` occurred.
No lack of data is interpreted as a memory, tensor, occupancy or stall bottleneck.
An ordinary activity trace remains feasible independently of the counter gate.

## Frozen workload: one case, two unchanged arms

Reuse existing exact sources and protocol; implement only a future profiling
wrapper around them. Do not alter algorithms or benchmark the rejected path
again merely to reconfirm its latency.

| Item | Fixed value |
|---|---|
| Case | Full verify B32/L8, D64, Hq16/Hkv2, ctx80000, agentic lengths, page1, scale1, coherent distribution |
| Input | Official generator seed135701; save actual GPU lengths/page-table shape and input metadata |
| FA3 arm | Exact A2-13 `candidate.kernel_fn`; current full-verify FA3 dispatch unchanged |
| Custom arm | Existing `candidatecustomhelper.bf16_custom_fn`; strict BF16-widened QK, private per-call tickets/integrated merge |
| Graphs | One independently captured graph per arm; fixed addresses; no shared allocation pool or concurrent replays |
| Correctness | Restore input/cache; validate output, new KV, immutable tensors and untouched rows before and after each profiler session using unchanged oracle/tolerances |
| Timing claim | None from profiler sessions; retain prior unprofiled paired result separately |

B16 or seed135702 is conditional follow-up only if B32 produces a specific,
falsifiable unresolved question. No split/warp/stage/pack-GQA sweep is included.
No taskmap cache, cross-layer reuse, precision change or formal entry is allowed.

Source SHA256:

- Candidate: `e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e`.
- Custom helper: `59fa2bf6aa1e97e55a23b821b56bdd5573b3fdf2edb0613913c2e93dc7acde82`.
- Existing paired worker: `3818221cb069ed1d605c37871183aebc5cf711bdcf22fd28fcc6768b3dc120c7`.
- Input/oracle: `386f7f67a1d788501e8fb9005d340423d098a0946e23810ba10878228c11a37b`.
- Checks: `9a046c3c38db28b325dcd1557baecdf6bc9ef941d2175f9a900ba4c230bb8055`.
- Prior raw result: `b55b87101004742d43d33272a632250d094a2345f1a3e37fce64589f76d931e9`.

## Gate 1: existing-tool inventory, without installation

A future bounded CPU-only inspection of the same immutable image can inventory
files/versions without reserving a GPU. It cannot prove H200 counter permission.
Allow30s, fixed2CPU/8GiB, no retry or persistent storage; expected planning
reserve <=$0.02 using last verified rates. Do not launch it in this preparation turn.

Record only an allowlisted JSON: executable paths/binary hashes/version/help,
Torch/CUDA/Triton/CUPTI library identity, and supported sections. Search PATH and
bounded known CUDA/NVIDIA install directories, not arbitrary home files. Do not
capture environment dumps, tokens or authentication diagnostics.

```sh
command -v ncu nsys
ncu --version
ncu --help
ncu --list-sections
nsys --version
nsys profile --help
```

On a subsequently allowed H200 instance, query device-specific metric support:

```sh
ncu --query-metrics --query-metrics-mode all --devices 0
```

If supported by that installed version, also query the PM-sampling collection
and record its available metrics/intervals. Record NVIDIA driver profiling
restriction and effective capability metadata if readable; absent files or
container root identity do not establish access. No sudo, capability additions,
driver changes, token expansion or alternate access route is proposed.

Missing Nsight: record exact path/version checks and stop its counter phase.
Use already-installed Torch activity tracing only. Unsupported metric: record
and omit that specific measurement, rather than invent an equivalent value.

## Gate 2: ordinary stage timeline, existing demonstrated path

Future worker warmup/capture and input restore happen before the profile range.
Enable CPU and CUDA activities with shapes/stacks/memory tracing disabled.
Profile one complete graph replay per arm with arm/seed/graph/stream identifiers,
export complete Chrome trace and a compact event table, and validate afterward.

Preserve kernel/API start/end, CPU-to-GPU correlation and overlap. Attribute
FA3 norm/query conversion, cache store, `prepare_varlen_num_blocks`, mainloop,
combine and norm/gate separately. For custom, distinguish ticket initialization
from the integrated producer/merge kernel; the latter cannot be split into
internal stage times by an ordinary kernel trace. Report wall span and interval
unions; do not sum overlapping kernel intervals into total latency. This measures
instrumented stage placement, not the original benchmark's host overhead.

If already installed, Nsys is an optional substitute, not a second required run:

```sh
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --cuda-graph-trace=node --capture-range=cudaProfilerApi \
  --capture-range-end=stop -o stage-trace \
  python profile_worker.py --case b32 --arm fa3
```

`profile_worker.py` is a proposed wrapper, not an implemented/approved upload.
This template requires installed-help validation and matching profiler start/stop
API calls. If Nsys is absent, do not install it; use the demonstrated Torch path.
No device-wide GPU sampling is needed for this stage.

## Gate 3: minimal counter comparison, only after access succeeds

Discover installed section/metric definitions first. A successful kernel-activity
trace or metrics-name query does not prove collection permission. The first real
counter collection is the admission test; a denial stops that collection for
both arms. No retry with broadened privileges.

Use one selected mainloop invocation per arm, under a unique `PROFILE_ONE`
NVTX push/pop range enclosing only one graph replay. Warmup, restore, eviction,
validation and compilation stay outside the range. Prefer deterministic
application replay so every pass reconstructs input, counters and graph state;
keep strict kernel matching. Internal kernel replay is not an automatic fallback
for mutable ticket/cache state. Every profiler session must retain pass count,
cache/clock/replay settings and pass-specific results.

Select at most eight individually queried metrics, avoiding `--set full` and
broad wildcard expansion. The minimal semantic bundle is:

| Need | Metric/report evidence | What it does not prove |
|---|---|---|
| Launch/resource limits | Grid/block, registers, static/dynamic shared memory; LaunchStats/Occupancy if installed | Actual per-CTA scheduling from resource ceilings |
| Achieved residency | Queried active-warps/active-CTAs aggregates and elapsed/active denominators | Time-resolved residency or exact CTA-to-SM assignment |
| Memory traffic | Measured DRAM read/write bytes plus supported throughput utilization | Memory-bound causality from utilization alone |
| Tensor activity | Supported Hopper tensor-pipe active/issued metric; record exact definition | WGMMA instruction efficiency or a compute bottleneck by itself |
| Stall reasons | Small queried warp-state set covering long-scoreboard, barrier/wait and execution-pipe throttling | A stall percentage as a fraction of total kernel wall time; exact tail timing |

Choose names from that actual H200 query, not guessed cross-version aliases.
If eight metrics cannot fit one or a few passes within the hard envelope, retain
a partial result and stop; do not grow the measurement set or time budget.

Illustrative version-gated mainloop command (the proposed wrapper is not yet built):

```sh
ncu --replay-mode application --app-replay-match grid --app-replay-mode strict \
  --nvtx --nvtx-include PROFILE_ONE/ --graph-profiling node \
  --kernel-name-base demangled --kernel-name 'regex:.*FlashAttnFwdSm90.*' \
  --launch-count 1 --cache-control none --clock-control none \
  --metrics RESOLVED_SUPPORTED_METRIC_CSV -o fa3-one \
  python profile_worker.py --case b32 --arm fa3
```

Use the same command/settings for custom with filter `regex:.*_fused.*` and its
arm/output name. A unique NVTX range is necessary: launch-count alone could
capture a warmup or preparation kernel. Do not use forced kernel termination
that skips validation. Cache-control none preserves the explicit cold-L2
precondition and inter-node cache flow, but multipass cache state can still
vary; document this limitation. Clock-control none avoids requesting GPU clock
changes. Record version-specific boost defaults; stop if the tool requires any
unauthorized device-state change. These are diagnostic counter reports, never replacement benchmark times.
[NVIDIA CLI behavior](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html).

## Tail question: sampled evidence versus exact CTA trace

Only if installed NCU PM Sampling supports H200 and the permission test succeeds,
use a tiny single-pass PM metric subset on this same case: SM-active/active-warps,
tensor-active and DRAM activity as available. Record actual interval, context
filtering, sample timestamps, overflow/dropped data and collection passes. The
online documentation's minimum interval is not evidence of the installed tool's
minimum. Define the reported tail as the sampled terminal region where active
work falls below its own plateau; report interval-resolution uncertainty. Do not
combine different passes into one simultaneous curve or infer individual CTAs.

Ordinary CUPTI kernel intervals do not contain each CTA's SM/start/end record.
PC/warp-state stall sampling is not a temporal CTA schedule. Nsys's default
100us device-metric bins would provide only roughly2–6 samples over these
0.2–0.6ms calls and cannot resolve a short tail. Device-wide counters also need
attribution/isolation checks. If temporal PM data is unavailable or insufficient,
report **tail unmeasured**, not zero tail or a scheduling bottleneck. Exact CTA
tracing remains outside the admitted capability; no kernel instrumentation or
binary patch is proposed here.

## Future resource envelope, cost and stop rules

This turn spends $0 and creates no resource. A future GPU diagnostic, if admitted,
uses one reused H200 image, max1container,2CPU,8GiB, no retries, network blocked,
credential access restricted and an explicit source/artifact allowlist. Suggested
bounds:165s worker alarm,175s child kill,190s function timeout,20s startup and
210s app watchdog begun before app creation; each NCU subprocess gets at most60s.
At most one activity session and two NCU mainloop sessions total; PM sampling
must fit the same envelope or remain unmeasured. No B16 follow-up in that job.

Last verified hourly rates: H200$4.54 +2CPU×$0.0473 +8GiB×$0.008 =
$0.0013051667/s. Reserve up to235s including cleanup, double the rate model
because earlier workspace billing exceeded a simple wall-time model, then add
$0.02: approximately$0.63343, rounded up to **$0.65 free-credit planning reserve**.
This is a conservative estimate, not a per-app billing guarantee. Refresh rates,
free balance and the actual$0paid cap before any later launch. The artificial
$2stage ceiling stays removed; no payment/limit/credential change is authorized.

Stop on missing tool, permission denial, source/hash mismatch, failed oracle,
wrong graph/kernel match, nondeterministic replay, unresolvable contamination,
report >8MiB, counter overflow/loss that invalidates interpretation, or deadline.
Preserve exact target/command/version/exit/error and all partial evidence. Do not
turn a missing metric or timed-out profiler into a workload conclusion. Stop
same app, independently verify stopped/tasks0, reconcile delayed usage and
actual paid$0, then release reservation. No retry or second job is automatic.

Deliver raw activity trace, NCU reports if admitted, inventory/help/metric
manifest, exact source/input hashes, correctness and terminal/cost receipts,
and an evidence table separating measured facts, unavailable observations and
hypotheses. Keep the existing unprofiled benchmark immutable. No algorithm or
candidate-selection decision follows from profiling overhead alone.
