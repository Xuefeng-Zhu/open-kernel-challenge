# Single activity session, fixed B32 case

Locally prepared, not launched. This worker never imports Modal, handles
credentials, installs software or changes an algorithm. After completed activity
validation it may call the separately reviewed existing-NCU counter module. The parent's fresh paid-cap/free-credit and execution gates remain
required. Counter installation and access are checked separately; no permission
is inferred from activity tracing.

`worker.py` expects its allowlisted support files in the same remote directory:
exact `candidate.py`, `candidatecustomhelper.py`, `modal_worker.py`, plus
`official/reference.py` and `official/checks.py`. It checks their fixed hashes
before importing them. The additional reviewed `counters.py`, `counter_target.py`
and exact CPU `inventory.json` are hash-recorded before imports; the controller
matches those hashes against its frozen payload. The candidate and helper are unchanged from the rejected
BF16 comparison. There is no module/global mutation, new routing or formal check.

One official generated input is fixed at B32/L8, full attention, ctx80000,
TP8 Hq16/Hkv2/D64, agentic lengths, page1, scale1, coherent distribution,
seed135701. The worker records actual GPU lengths, config and tensor metadata.
Each arm has a private cloned input/cache and independently captured graph.
Warmup/capture occur before the profiler session. All four validation calls use
the unchanged official oracle/tolerances, stable addresses, immutable inputs and
untouched KV rows. Both pre-profile checks must pass before profiling. Both
post-profile checks must pass and match their own pre-profile output/new-KV
bytes exactly before a completed result is admitted.

There is exactly one `torch.profiler.profile` context with CPU and CUDA
activities, `record_shapes=False`, `profile_memory=False`, `with_stack=False`.
It captures one complete graph replay per arm, labeled with distinct
`record_function` and NVTX ranges. Input/cache restores and a256MiB L2-eviction
read happen in separately labeled ranges; a synchronize completes them before
the arm range. The arm range includes launch and completion synchronize.
Both post-profile oracle checks run after the profiler context closes, avoiding
large oracle activity in the trace. No CUDA-event benchmark samples are taken.

The complete, unmodified Chrome trace is written as `activity-trace.json`.
It is bounded at7936KiB so the256KiB summary plus trace fit an8MiB raw transport.
An oversized trace is deleted with an explicit failed status, never truncated
or interpreted as complete. Optional raw counter reports share the same8MiB
actual artifact budget and are admitted only if they fit. Caller compression/
chunk limits remain additional bounds. `/tmp/profiling-run` is the default output directory and must not
already exist; the caller must not precreate it. Summary updates are atomic.

Compact analysis fences each arm by its CPU annotation and requires exactly
one FA3 `FlashAttnFwdSm90` or custom `_fused` kernel. Missing/ambiguous attribution or missing expected FA3 norm/conversion/store/
schedule/mainloop/combine/output stage coverage fails closed while preserving
the raw trace. A mainloop-only trace cannot claim complete stage capture. Event records keep complete names,
start/duration, PID/TID and raw activity args (including available correlation,
stream and graph/node IDs). Unavailable IDs are not invented. GPU union and
span are reported separately from summed activity durations; overlapping events
must not be summed into wall latency.

FA3 stages are classified by observed kernel-name tokens: QK norm, FP8
conversion, KV store, length-schedule preparation, mainloop, combine and output
norm/gate. Name-only FP8 conversion cannot identify query versus cache
conversion. The custom `_fused` remains one integrated producer/merge activity;
no internal stage durations or per-CTA service times are fabricated. Memsets
are conservatively labeled without independently proving ticket attribution.

If optional counter logs cause summary overflow after activity has passed,
verbose counter details are reduced first, retaining status/count and a hash of
the original counter summary. Completed activity remains valid and its raw trace
is retained. A verbose event table may be omitted with an explicit marker, while
activity stages/unions remain. Activity-only overflow still persists an explicit
failed summary and cannot claim completed capture.

CPU arm spans contain synchronize/profiler overhead. All activity durations
are instrumented observations, not replacement measurements for the immutable
unprofiled paired timings. The activity trace alone cannot establish exact CTA residency,
DRAM throughput, tensor utilization, stall reasons or a bottleneck. Optional
counter evidence is reported separately.

After the activity trace parses and all four oracle checks pass, counters may
run only with at least25s remaining, inside the same165s overall alarm. At most
two NCU mainloop sessions are allowed, one per arm. Installed tool/metric/access
gates, deterministic application replay and exact targets live in the counter
module. Denial, timeout, unsupported metrics or unexpected errors leave the valid
activity result intact and record partial/unavailable counter status. No PM or
exact CTA tracing is performed. No retry/permission expansion is allowed.

A165s worker alarm/gates bound this worker. The caller must separately enforce
175s child kill,190s function timeout and210s app watchdog, no retry and one
container; transport the allowlisted files with hashes; stop and independently
verify stopped/tasks0; and reconcile billing without claiming a failed immediate
teardown read is terminal. Local protocol/parser tests use no Torch/GPU/network.
