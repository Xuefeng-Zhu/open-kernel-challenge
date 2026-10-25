# Installed profiling tools: evidence and missing capabilities

The existing H200 image has demonstrated PyTorch/Kineto CUDA activity capture.
There is no saved evidence establishing whether Nsight Compute (`ncu`) or
Nsight Systems (`nsys`) is installed, or whether hardware-counter access is
permitted. No hardware-counter request or access denial was observed. Missing
counter evidence must not be presented as a memory, tensor or occupancy bottleneck.
This is a local preparation audit only: no network, GPU, browser, installation,
credential action, algorithm change or incremental compute cost.

## Demonstrated on the reused image

Image `im-cPTbGpNiX3RarKResPgX7h` was used by both current diagnostics. The native
receipt records H200, capability9.0, driver610.57.04, Torch2.14.1+cu130,
CUDA13.0 and Triton3.8.0. Exact Triton revision is unavailable; its package and
compiler-source hashes are retained in that receipt. The historical base-plan
OCI digest is recorded in `tools-audit.json`; it does not inventory every
current image layer.

| Facility | Actual evidence | Limit |
|---|---|---|
| Torch CUDA activity | `paired/result.json` contains eight complete CUDA kernel names from a separate successful B32/L8 FA3 profiler replay. | Names alone contain no stage durations or hardware metrics. |
| CPU/CUDA activity trace | Earlier A13 ownership diagnostic exported Chrome traces, extracted kernel intervals, stream/graph IDs, and observed cross-graph overlap in one reduced-context decode fixture. | Saved compact receipt retained only two intervals; full raw trace was deleted. It cannot reconstruct all intervals or individual CTA residency. |
| Static compiler/driver metadata | Current native artifacts report registers, shared memory, raw spill field, TTIR/TTGIR/PTX/SASS. | Static resource ceilings are not achieved occupancy, active CTAs/SM, memory traffic or tensor utilization. |
| `cuobjdump` | Four native records report `sass_tool_exit_code=0` and verified SASS/cubin hashes. | Successful path was not recorded. This establishes disassembly availability, not profiling permissions. |

The current paired worker uses `torch.profiler.profile` only after timing, with
CUDA activity enabled. The older ownership worker uses CPU and CUDA activities,
with `record_shapes=False`, `profile_memory=False`, `with_stack=False`, then
exports a Chrome trace. This is a demonstrated existing mechanism for a future
bounded stage timeline. Its instrumented durations must remain separate from
unprofiled paired benchmark results. To inspect launch/preparation, compute and
combine, preserve all events and correlate graph/call/stream IDs; do not keep
only names or sum overlapping activities as total latency.

## What remains unknown

No saved remote `which ncu/nsys`, `--version`, metric query, Nsight report,
CUPTI Profiling API result, NVIDIA driver counter restriction value or actual
counter-denial error was found in the inspected local records. A historical
projection resource probe explicitly says **“No Nsight/CUPTI hardware metrics
requested; ordinary PyTorch CUDA activity only”**. Its reported optimistic
resident-CTA ceilings were arithmetic estimates, not measured occupancy.

Current Modal function flags `block_network=True` and
`restrict_modal_access=True` describe the experiment sandbox. They do not
establish NVIDIA counter privilege or perf-event policy. No permission expansion,
admin/capability change or counter workaround is part of this audit.

The Mac has Torch2.14.1 with `torch.version.cuda=None` and CPU-only supported
profiler activity. PATH lookups and bounded searches in `/usr/local`, `/opt`
and `/Applications` found no ncu/nsys/CUPTI/Nsight matches. These are scoped
local negatives; they do not prove remote tools absent. The Mac cannot provide
H200 hardware counters for the remote experiment through its local CPU build.

## Feasibility gate for the parent plan

Existing tools can execute an ordinary CPU/CUDA activity-stage trace in a future
GPU run. They have not demonstrated the requested hardware metrics. Before
claiming availability, a separately allowed minimal remote capability check must
record existing ncu/nsys paths and versions, available metric names, and a real
single-kernel counter-access result. Do not install tools, broaden privileges,
renew credentials, change driver settings or retry through another access route
on a denial. A failed tool/counter request is a capability blocker, not a
bottleneck measurement.

Exact per-CTA residency/tail remains unproven. Standard kernel intervals cover
whole kernels, not CTA start/end or SM assignment. Aggregate achieved occupancy
or active-CTA counters, if supported, still need a temporal/per-SM method to
identify a tail. Any report must label aggregate residency versus exact CTA
tracing separately. No kernel instrumentation or algorithm change is proposed
in this preparation task.

The exact action currently blocked is **claiming/executing a hardware-counter
or exact-CTA-residency diagnostic on the existing H200 image**. Its state is
**capability/permission unknown, not denied or submitted**. Preparation is
complete; the missing remote inspection is intentionally not performed because
this turn forbids GPU launch.

Complete paths, source hashes, search scopes and machine-readable findings are
in `tools-audit.json`. Primary current result SHA256:
`b55b87101004742d43d33272a632250d094a2345f1a3e37fce64589f76d931e9`.
Current native summary SHA256:
`4aef9c653af05a09276b7385e14bf2a3d5889fefdb66885f319715178e1ff6d7`.
