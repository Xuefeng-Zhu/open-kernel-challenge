# Official-documentation profiling feasibility

Preparation only; no GPU, install, credential, browser UI or algorithm action was
performed. These are supported method candidates, **not evidence that Nsight is
installed or counters are permitted**. Exact local evidence hashes are in
`docs-evidence.json`. The existing PyTorch CUDA trace proves one activity path
worked, not hardware-counter access.

## What each method can establish

| Question | Minimal candidate | Interpretation limit |
|---|---|---|
| Launch/prep, mainloop, combine and gaps | PyTorch CPU+CUDA activity export, or Nsight Systems CUDA/NVTX node timeline | Activity timestamps and graph-node attribution; profiler overhead is not a benchmark. |
| Resident-work/resource pressure | Nsight Compute launch/occupancy sections | Theoretical resource ceiling and achieved average; neither is a time-resolved per-CTA schedule. |
| Memory, tensor and scheduling limits | NCU selected throughput, memory, scheduler and warp-state sections | Use queried metrics and definitions; high/low utilization alone does not prove causality. |
| Within-kernel tail | NCU PM Sampling, if supported and permitted | Sampled device activity, with timestamps; not every CTA's SM/begin/end record. |
| Exact CTA residency/service times | No established existing-tool path | Requires additional supported tracing/instrumentation evidence; do not relabel a utilization curve as a CTA trace. |

NCU describes occupancy and periodic PM sampling separately. PM support for
GA10x and later has a documented minimum interval of1µs, sometimes higher for
particular configurations. Warp-PC/state sampling has instruction correlation
but no time resolution. Multipass sampling can misalign. Replay may restore
memory, serialize work, affect caches and clocks; captured metric duration must
not replace the prior uninstrumented paired timings. Query actual collection
support and retain pass count, sampling interval, dropped/merged samples and
context filtering. [NCU profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).

CUPTI13.0's normal kernel activity record supplies kernel timestamps, grid/block
sizes, resource metadata, stream and correlation IDs. It does not supply an
individual block-to-SM execution interval. This is the concrete limitation of
that record, not a claim that every possible CUPTI extension lacks other data.
[CUPTI kernel record](https://docs.nvidia.com/cupti/13.0.0/api/structCUpti__ActivityKernel9.html).

CUPTI PC sampling can correlate sampled stalls with SASS, but its documented
samples lack temporal resolution. It cannot, by itself, locate an exact final
CTA tail. Avoid adding a custom injection library in this minimal plan.
[CUPTI13.0 PC sampling](https://docs.nvidia.com/cupti/13.0.0/main/main.html#cupti-pc-sampling-api).

Nsight Systems node-level graph tracing can reveal stage boundaries, with
higher overhead than graph-level tracing. GPU metrics are device-wide and need
separate permissions. Its default10kHz is100µs/bin: only about2–6 bins across
our0.2–0.6ms calls, an inference too coarse for a short tail claim. A higher
supported rate still needs loss/overflow checks and cannot assign each sample
to one process or CTA. [Nsight Systems guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html).

## Version-discovery and capture templates

These commands are a future runbook, not commands executed this turn. First
record executable path, binary hash, version and local help. Do not guess flags
from current online documentation when the installed version differs.

```sh
command -v ncu nsys
ncu --version
ncu --help
ncu --list-sections
ncu --query-metrics --query-metrics-mode all --devices 0
nsys --version
nsys profile --help
```

NCU section/metric names must appear in local discovery; do not use the full
set. Filter the exact mainloop and one matched invocation. Push/pop NVTX filters
use a trailing slash. Node and whole-graph profiles answer different questions.
The explicit query flag is redundant in current documentation, where query mode
implies querying, but improves runbook clarity. [NCU CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html).

```sh
# Conditional on these options and named sections being present in local help:
ncu --nvtx --nvtx-include PROFILE_ONE/ --graph-profiling node \
  --kernel-name-base demangled --kernel-name 'regex:.*_fused.*' --launch-count 1 \
  --replay-mode application --app-replay-mode strict --app-replay-match grid \
  --cache-control none --clock-control none \
  --section LaunchStats --section Occupancy -o ncu-one python profile_worker.py --arm bf16_custom --case b32

# Activity-only timeline, no device-wide GPU sampling requested:
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --cuda-graph-trace=node --capture-range=cudaProfilerApi \
  --capture-range-end=stop -o timeline python profile_worker.py --case b32
```

The proposed worker is not implemented or run. Reuse frozen B32/L8,
80K-agentic/page1/scale1/coherent inputs, seed135701. Initialize fresh inputs,
caches and ticket state on every process run; warm up and capture before the
profiling range, restore and evict consistently, profile one replay per arm,
then validate outside the range. Use unchanged FA3 and the rejected BF16 helper.
NCU's FA3 filter must match its recorded full mainloop name; `_fused` selects
the custom arm only. Stop if the intended kernel is not matched.

Application replay reruns the entire program per pass. Require deterministic
filtered launches; strict matching must succeed in exact order, using name,
grid/block and order matching. It avoids relying on individual-kernel replay
to restore mutable tickets/cache state, but repeats initialization and can
raise cost. Stop on mismatch or the bounded deadline. [NCU replay guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#replay).

`cache-control none` leaves cache invalidation to the fixed harness and can
produce history-dependent metrics; `clock-control none` requests no GPC/memory
frequency changes, not fixed physical clocks. Record all defaults, including
any version-specific boost control. Check help before adding these options;
no automatic fallback or privilege change. [NCU control options](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#command-line-options).
Multiple-pass counters cannot form a synthetic same-instant trace. PM samples
remain aggregate activity, never exact CTA-to-SM intervals. Keep B16/seed135702
for a later specific question; no repeated knob sweep or performance claim.

If Nsys is absent, the already-demonstrated `torch.profiler` CPU/CUDA activity
path can export a bounded Chrome trace. Use explicit activities, no shape,
stack or memory recording unless necessary. Its CUDA tracing depends on CUPTI.
[PyTorch2.14 profiler](https://docs.pytorch.org/docs/2.14/profiler.html).
Modal documents the profiler wrapping the same function environment; existing
artifacts can be returned with the current bounded transport, avoiding a new
persistent Volume. [Modal profiling example](https://modal.com/docs/examples/torch_profiling).

## Permission and evidence gate

`ERR_NVGPUCTRPERM` means denied GPU counter or hardware-event access. Record the
exact tool, target, version, exit/error and the requested metric, then stop that
collection. Do not use sudo, add capabilities, change driver parameters or
otherwise widen access. An unavailable counter is an observability blocker,
not evidence that memory, compute or occupancy is the workload bottleneck.
[NVIDIA permission explanation](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters).
Modal's host kernel-driver components are not customer-controlled; a container
root identity cannot establish permission to change them. No permission fix is
part of this plan. [Modal CUDA stack](https://modal.com/docs/guide/cuda).
