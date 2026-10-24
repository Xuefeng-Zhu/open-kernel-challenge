# Hopper diagnosis of unchanged projection kernels

Completed 2026-10-24T20:16:32.989508+00:00. No algorithm was changed or newly benchmarked. Best remains
full composition c34c497b, official check48b1357016134e52,1.1486499523x SGLang,
95%CI[1.1314202031,1.1658797016]. No new official check/ranked entry/push.

Successful diagnosticM34: app **ap-GwiDybwrxgrysaFh3d8DRM**,10.304sworker/18sapp,
source bundle commit0bb14a9. Existing three kernel sources unchanged. Exact
compiled metadata, full PTX/TTGIR with hashes,10ordinary CUDA timelines and20fresh
CUDA-graph oracle checks: results/remote-modal-M34-resource-probe-fixed.json.
Alloraclechecks pass. No hardware performance counter was requested.

M33failed before diagnostic execution: app ap-vd8ZPjum8f0SpBMhBTTYDE,7sapp,
exit2 due omitted resource-v1 CLI whitelist. It was definitely terminal/stopped,
not an uncertain request or access denial. Failure retained in its own commit and
JSON. Routing/zero-repetition regression tests were added before new M34 app.
Both resources stopped; no blind retry or discarded failure.

## What the evidence does and does not establish

| Existing native pooler | Registers/thread | Driver n_spills (raw) | Shared/CTA | Optimistic register/shared CTA ceilings perSM |
|---|---:|---:|---:|---|
| BM32/BN128 (M30) |95|0|40960B|5 /5|
| BM32/BN32 (M32) |45|0|16384B|11 /14|

Same values at m128and256. These ceilings ignore register allocation granularity,
SM scheduling and other limits; they are **not achieved occupancy**. Device reports
132SM/65536registers perSM/233472Bshared perSM. Neither specialization reports
spill, and neither PTX contains local-memory loads/stores. Thus a spill-based
explanation is unsupported for these inspected kernels. This does not disclose
resource usage inside the opaque cuBLAS library kernels.

Both generated PTX variants use BF16 mma.sync and cp.async; neither contains
wgmma.mma_async or cp.async.bulk. The earlier kernels already have asynchronous
copy instructions: it would be wrong to call them completely unpipelined. The
specific Hopper WGMMA/TMA instruction paths are absent in this generated PTX.
PTX occurrence counts are static, not dynamic instruction counts or a SASS audit.

| Case | Existing best CUDA activities (us) | BN128 single kernel (us) | BN32 single kernel (us) |
|---|---|---:|---:|
| pooler m128 |21.506GEMM +3.424reduce|62.139|61.474|
| pooler m256 |29.209GEMM|70.632|121.772|
| QKV m16 |11.616GEMM +2.112reduce|—|—|
| QKV m128 |13.696GEMM +3.488reduce|—|—|
| unpooler m128 |19.903GEMM|—|—|
| unpooler m256 |24.421GEMM|—|—|

These are one instrumented ordinary CUDA activity per existing graph, not a new
balanced benchmark/score. Full earlier unprofiled balanced samples remain in
M26/28/30/32 evidence. The native pooler's excess time is present inside its GPU
kernel despite fewer launches; reducing launch count alone cannot explain away
its measured regression. QKV and pooler m128 have explicit reduction kernels;
unpooler largeM and pooler m256 already have one GEMM.

**Do not use profiled host cudaGraphLaunch times** (0.58–1.92ms) as steady launch
cost. No unprofiled host-launch control was measured, so its instrumentation overhead
cannot be quantified here. No host/GPU duration was
subtracted from graph measurements, no guessed overhead applied, no profiler
sample used to claim an official gain. CUDA activity times also carry profiling
and single-sample limitations.

The source-level operand request model exposes a tradeoff:

| m | BM/BN | CTAs | Logical X requests | Logical W requests |
|---|---|---:|---:|---:|
|128|32/128|128|64MiB|256MiB|
|128|32/32|512|256MiB|256MiB|
|256|32/128|256|128MiB|512MiB|
|256|32/32|1024|512MiB|512MiB|

BN32 raises CTA count and reduces compiler resources, but quadruples logical X
requests without reducing logical W requests. That is a plausible cost of the
change, **not measured HBM traffic or proof that memory bandwidth caused the
slowdown**. Cache reuse can satisfy repeated requests. Actual DRAM bytes/cache hit
rates, stall breakdown and achieved occupancy remain unknown. No restricted
counter access was attempted, denied or bypassed. A kernel name does not reveal
competitors' implementation or the library's internal resource allocation.

## Two concrete hypotheses, assessed before any algorithm benchmark

### H1 — BF16 Hopper pipeline plus greater M-tile weight reuse, pooler largeM

Proposed first target: pooler m256, then m128 only if justified. Starting geometry
BM64/BN64/BK64 with an explicit supported Hopper TMA/WGMMA BF16 path, preserving
FP32accumulation/original weights and other dispatch. This is a future prototype,
not implemented or compiler-validated here. For causal comparison, hold the new
BM64/BN64 geometry fixed in a conventional-copy control versus the async-pipeline
variant; do not attribute a combined geometry/instruction change to one cause.
Separate compatibility/smoke from
algorithm timing; verify the pinned3.8 APIs and emitted instructions first.

The case for testing: current generated PTX lacks those paths; both slow native
variants rerequest full weights over multiple M tiles. BM64/BN64 halves logical W
requests and doubles logical X requests versus BM32/BN128, reducing combined
logical requests by20% for these two shapes. Grid remains128CTAs at m128/256at
m256; simply increasing CTA count is not its hypothesis. Hopper's async engines
provide a plausible way to overlap movement and MMA, not an established win.

Benefit boundary: if an eventual candidate reduced both official pooler m128/256
latencies by10% while all other timings stayed frozen, the scoring formula yields
only ~1.01047 aggregate multiplier. That is a conditional scenario, not a forecast.
The failed native m256 kernel currently needs more than2x improvement merely to
match the library; beating an existing optimized library is a high-risk objective.

Risks: larger M tile/resource changes, too few CTAs at m128, descriptor alignment/
tails, synchronization and compiler pipeline compatibility. Nearby widths need
correct masking or a legitimate general fallback. BF16must stay BF16; tutorial
FP16/FP8 examples do not authorize converting contest BF16weights/activations.
Full39CUDA gates, fresh/probe/nearby cases and balanced cold-L2 CUDA-graph timing
against immutable best are required before promotion or any official check.

### H2 — single-output-owner full-K QKV at m128, remove split-K reduction

Keep m1/8/16/32/256 and every other family unchanged initially. Proposed BF16/FP32
accumulator full-K Hopper GEMM with starting BM64/BN32, subject to compiler gates;
for scored m128/N2304 this gives144CTAs. Each CTA owns its output tile; no global
partials/reduction buffer. Input/output values never precomputed or cached.

The case for testing: actual QKV m128 activity contains a3.488us reduction after
13.696usmain; the library layout screen did not remove the aggregate limitation.
The objective is to remove that reduction while keeping enough useful parallelism.
Replacing it with a slower full-K main loses, as the pooler counterexample shows.
No assumption that eliminating a3.488us kernel saves exactly3.488us overall.

Benefit boundary: even10%lower latency at that one shape gives only1.00366
aggregate multiplier under frozen official timings. Thus prioritize H1; H2 alone
may be too narrow to warrant an official check. Risks: full-K dependency/latency,
loss of split-K parallelism, BF16rounding/FP32accumulation order and cache traffic.
No benefit measured, no source implementation or GPU algorithm benchmark yet.

Recommendation: do not sweep more basic tile constants or use official quota now.
A first architecture-aware compatibility prototype is plausible from this evidence,
but its expected speedup is unresolved. Select H1first, keep an explicit failure
criterion versus the existing library, and only continue to H2 if it adds value.
Actual hardware counters would sharpen the diagnosis but are not prerequisites to
try a controlled prototype; this task did not request restricted access.

## Evidence, budget and commands

Actual H200 SM90,132SM,Torch2.14.1+cu130/Triton3.8.0; pinned public image remains
different from private judge overlay. CUDA timeline availability succeeded. Local
108tests pass (including CLI regression and parser/traffic interpretation tests);
these tests do not execute GPU kernels. Reproduce local checks:

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m py_compile projection_resource_probe.py modal_gpu_worker.py modal_diagnostics.py
```

Exact historical launch/allowlist/finalize commands: experiments/033-resource-probe.md
and034-resource-probe-fixed.md. Do not rerun accepted IDs.20allowlistedfiles, no
credentials/repo upload, one ephemeral H200,180sfunction/60sstartup/150sinternal,
no detached deployment/retries. Both diagnostic apps terminal; all10apps stopped.
Conservative cumulative stageguard1.034494554of2USD; final billing snapshot0.75USD
fully credited,0paid, observed gross highwater0.87077815. Counters asynchronous/
revised, hence use the larger conservative guard for future admission. Net paid
cap0reverified in user's Mac browser; no new credentials/cap/payment/terms/push.
Official second-cycle2/10used,8remaining, no active/uncertain request.

Primary references, read2026-10-24:
- [Triton3.8 compiled metadata and asm](https://github.com/triton-lang/triton/blob/v3.8.0/python/triton/compiler/compiler.py)
- [Official warmup/resource inspection example](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
- [Hopper resources and TMA mechanisms](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
- [Triton persistent/TMA examples](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html)
- [PyTorch2.14 ordinary CPU/CUDA activities](https://docs.pytorch.org/docs/2.14/profiler.html)

No outstanding denied operation. M33's exact failed action was worker protocol
parsing inside appap-vd8ZPjum8f0SpBMhBTTYDE; resolved by separately committed/tested
M34, not a permission bypass. Missing actualDRAM/occupancy is a measurement limit;
new architecture-aware algorithms remain unimplemented/unmeasured.
