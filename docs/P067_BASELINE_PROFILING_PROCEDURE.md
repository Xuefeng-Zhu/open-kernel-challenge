# P067 — minimal baseline profiling procedure

Status: **preregistered procedure, not a completed native run**. P068 has now
assembled and CPU-tested the baseline-only collector from the helpers below;
see [its commands, coverage and limits](../experiments/068-baseline-native-collector.md).
No native execution occurred. Expired Modal access must be resolved before
live capture, without automatic credential renewal. Counter-dependent conclusions
require functioning counters; activity/code-only capture can still be useful if
counters remain unavailable. This is not an instruction to rerun the failed NCU recipe.

## Question and fixed scope

Does actual P046 leave a materially reducible work-assignment/finalization cost
on **QKV M128** and **pooler M128** that a work-centric scheduler could address,
or does the presumed missing mechanism already exist in the vendor path?
Pooler M8 is the known single-kernel control. Do not infer the answer from grid
size, kernel names, public CUTLASS code, or lower bounds alone.

Freeze P046 source SHA
`001ea12e8764564d507de51e433769f5bd9a9916bfc4c014f0f87522c278545e`,
original reference/checks hashes from `projection_p046_confirmation.ORACLES`,
and prepared weights. Use exactly the official dimensions/distributions and
existing weight seed. Repeat the result with independently seeded weights only
if the first observation supports a decision; weight preparation is untimed but
all input-dependent computation and per-call synchronization remain timed.

First reproduce the old evidence, without GPU/network:

```bash
python -B projection_p067_audit.py --output results/projection-p067-audit.json
python -B projection_p053_analysis.py --output /tmp/p053-reaudit.json
python -B -m unittest tests.test_projection_p053_analysis tests.test_projection_gap_model
```

The first command verifies selected source/official receipt, six published
trace hashes and raw bindings, and reconstructs all 30 profiles and the score.
The second analyzes owned custom PTX, **not vendor P046 instructions**.

## Live acquisition sequence, once access exists

1. **Access and identity gate, no GPU yet.** Existing valid profile only; read
   credits/net-paid-$0 cap and active apps. No changes to credentials, billing,
   counter permissions or drivers. Record exact image digest, GPU/SM count,
   Python/Torch/CUDA/driver/CUPTI/profiler versions and native library build IDs.
   Record CUDA graph, TF32 and BF16-reduction flags without changing them.
   Inventory pinned Quack prerequisites using package metadata without import,
   install, JIT or tuning. An arbitrary current Quack revision must not silently
   replace judge-installed packages. A separate test image is not judge proof.

2. **Correctness and uninstrumented repeatability.** Reuse `benchmark.capture`,
   `projection_p046_confirmation`'s frozen-state/input/oracle helpers and
   `ColdGraphTimer`. Capture two independently owned graphs of identical P046
   (A/A noise control), separate outputs, the same X address. For each of
   the three fixed shapes run A→B(outliers)→A, all-sparse power-of-two and zero
   inputs before and after. Require exact input/config/weight immutability, original
   FP64 oracle tolerances and exact A-return output reproduction. Output aliasing
   is rejected; hidden vendor workspace independence remains unverified. This
   diagnoses sequential replay, not concurrency.
   Use six predefined warmup rounds, then 24 reversed-order rounds; a 256 MiB L2
   eviction precedes **every** measured replay. Retain all event samples, including
   tails; no guessed fixed-cost subtraction or outlier deletion. Copy fresh X
   before a pair, validate both outputs after the pair. All oracle work stays out
   of event timing. A/A drift above 2% or unstable tails makes timing inconclusive.

3. **Separate activity capture.** In a fresh process with identical frozen code,
   environment and flags, repeat three single graph replays per case. Follow the
   P053 Torch/CUPTI capture/export pattern, record_shapes/stack/memory profiling
   disabled. Check the output after each trace. Save complete local traces plus
   sanitized publications and their hashes. Extract exact symbols, launch grid,
   block/cluster metadata where exposed, registers, shared bytes, streams,
   timestamps and reducer gaps. Missing fields remain missing. Timeline gaps and
   profiler duration are descriptive; they cannot replace uninstrumented times.

4. **Executable and work-assignment evidence.** Obtain the actual loaded module
   identity/code only through a supported profiler/disassembler interface. Hash
   the library and loaded executable, not merely the Python wrapper. If a vendor
   JIT binary is unavailable, mark that fact; do not infer it from another cuBLAS
   tactic, intercept private evaluator state or pretend static vendor source is
   the deployed kernel. Analyze visible TMA/copy, WGMMA commit/wait, barrier and
   output/fixup sequences, and whether output tiles/K ranges are statically or
   dynamically assigned. Static instructions show possible ordering, not measured
   achieved overlap. Do not publish proprietary vendor binaries without checking
   applicable redistribution rights; publish hashes/allowed summaries instead.

5. **Counter gate only after a material capability change.** Two earlier bounded
   NCU sessions timed out, even with one DRAM metric. A renewed ordinary token
   alone does not establish that counters work. Require a documented provider/tool
   fix or a successful sanctioned bounded capability check before collection;
   never repeat the unchanged timeout or request elevated counter privileges.
   Otherwise finish with activity/code evidence and the remaining unknowns.

## Minimum counter questions, if supported

Query the installed tool's `--list-sections` / `--query-metrics` first; record the
returned names. Do not assume names or suffixes from another NCU version.

| Question | Required observation | Insufficient substitute |
|---|---|---|
| Is work uneven or ending in a long tail? | Per-SM active-cycle/CTA-work instances or a supported per-CTA timeline, actual partition/cluster assignment, full kernel duration | Grid/132, theoretical occupancy, aggregate SM utilization |
| Is synchronization material? | Barrier/membar wait sampling tied to PCs and the actual instruction sequence; separate finalization intervals | Percentage of one stall category treated as additive wall time |
| Does a changed partition duplicate traffic? | Actual DRAM and L2 byte/transaction counters at identical cold-cache inputs | Unique tensor byte count or logical load count |
| Is the proposed mainloop already present? | Loaded SASS and buffer/wait ownership, plus eligible/active-warp and issue evidence where supported | WGMMA name, public source, or a same-named kernel from another run |

Keep collection bounded to those questions and the two targets; no full-metric
sweep. Save replay/cache/clock settings and number of profiling passes. Node-level
graph profiling can isolate/change interactions; whole-graph profiling cannot
attribute individual nodes. Never compare instrumented latency with acceptance
timing. If a valid report includes machine code, documented offline inspection is:

```bash
ncu --import CAPTURE.ncu-rep --page source --print-source sass
```

This command only reads an existing report. No such P046 report currently exists.
References: [NVIDIA CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html),
[profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).

## Predeclared Stream-K decision rule

- **Reject a mislabeled test before execution:** pinned scheduler resolution must
  contain genuine work-centric units; `splits>1` in the inspected CUTLASS API
  routes to Split-K, and even splits=1 can resolve to DP/Split-K. Record the actual
  mode, number of work units, ranges, partial boundaries and deterministic fixup.
- **Reject this rationale** if actual P046 already implements equivalent work
  assignment and no different reducible cost is identified. Unknown is not absent.
- **Reject porting onto P052/P053's fixed mainloop:** its optimistic partial-only
  result already loses to complete P046, and the old equal-cost 132-worker model
  offers only 1.56% work-depth reduction. This rejects that implementation, not
  every possible scheduler/mainloop combination.
- **Require value before a prototype:** the six visibly split pooler/QKV/unpooler paths need about
  20.63% uniform latency reduction to model +5% whole score. A QKV128-only 20% cut
  models +0.780%. Router's four vendor-split paths are outside this six-path model.
  Scheduling may justify a small diagnostic for information, but
  cannot be presented as a gap-closing strategy without defensible coverage.
- If a later prototype is warranted, compare a fixed scheduler and work-centric
  scheduler with the same BF16/FP32 mainloop, layouts, tile geometry and epilogue,
  plus **actual P046**. Count all workspace/reset/fixup costs and test nearby/tail
  shapes. Reject if it only beats the weaker custom control, fails the original
  oracle/replay checks, or loses its predicted gain after full-call costs. Use
  fresh confirmation after choosing a configuration; no post-hoc gate change.

## Operational limits and stop condition

No launch in P067. A later baseline-only acquisition should first set a fresh
reserve from actual rates/credits, one H200, no retries/detach/deploy, internal
90-second and external 120-second process bounds per stage, and a bounded app
watchdog covering startup and cleanup. Stop after the minimum answer or a stage
failure; never consume a further stage automatically after auth/counter failure.
Stop the app and independently verify stopped/tasks0; unknown stop or transport
outcomes must be reconciled before another launch. These bounds are not a cost
quote or evidence of native execution. P068 implements and CPU-tests process
supervision; cloud provisioning/shutdown still needs an external app watchdog.

Current exact blockers: expired Modal credential, no documented official-runner
native-artifact export, and no working counter capture established. The completed
saved-evidence audit supplies the baseline facts while those prerequisites remain
unresolved. No candidate or score changed.
