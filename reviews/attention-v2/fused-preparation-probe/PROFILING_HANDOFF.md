# Existing local profiling guidance for the parent

The completed attention diagnostic is available in the sibling worktree:

`/Users/frank/Documents/Codex/2026-10-25/task-2/attention-architecture/reviews/attention-v2/architecture-probe/profiling-run/`

`gpu/worker.py` demonstrates the already-supported Torch/CUPTI activity route
and bounded raw Chrome-trace transport. `analysis/analyze.py` computes interval
unions and signed gaps. `RESOURCE_CLOSURE.json` records all three stage apps
stopped/tasks0; the measurement slot was released. The final private commit is
`ad46ee907952d0bec8b3fc98e429417328776420` on `attention-architecture-probe`.

Use one explicit profiler session around graph replays after warmup and
correctness. Keep a separate clean session for benchmark timing. Profiler CPU
spans include collection/synchronization and are not ordinary launch overhead.
Graph replays reuse captured allocations. Report raw activity intervals and
overlaps honestly; summing overlapping stage durations is not complete latency
and subtracting a stage duration is not a predicted fusion saving. Do not infer
exact per-CTA residency/tail from ordinary CUDA activity records.

The existing image `im-cPTbGpNiX3RarKResPgX7h` contained Torch 2.14.1+cu130,
Triton 3.8.0, CUPTI 13 and installed NCU 2025.3.1/Nsys 2026.4.1 during the prior
run; a new run must record its own environment. No new installation is needed
for the successful activity path. The two NCU collections timed out at 60 and
55 seconds. The second retained connection and target-correctness evidence but
no report, metric or permission error. Counter access and the hardware
bottleneck remain unknown; do not retry counters in this stage or describe a
timeout as permission denial or bandwidth proof.

The first activity run used the standard whole-call BF16 output/new-KV oracle,
fixed-address input restores and 256 MiB L2 eviction for *separate* paired
timings. Projection should bind its own source/fixtures and chosen cold/warm
protocol rather than reuse attention-specific timings or claim profiled timing
as benchmark performance. This handoff is local only; no data was sent to Pro.
