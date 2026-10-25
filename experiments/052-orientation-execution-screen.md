# P052 — execute the matched orientation mechanism

Status: native correctness passed; performance gate failed. P046 remains
selected. The one Modal app is stopped with zero tasks. No official check or
promotion resulted from this experiment.

实验卡：P051 已证明调换矩阵计算方向能改变真实 GPU 指令和等待顺序。
这次检验它是否既正确又有实用速度。先验证边界形状与同地址 A→B→A
图重放，再同时对比原自定义内核和当前 P046。只战胜较慢的自定义对照
不算成功；本轮不改分块、split-K、精度、归约或 P046。

## Evidence and single hypothesis

Frozen P051 source: `projection_p051_orientation.py`, SHA256
`445ab7d44e36cdceafdf1ecdbe19d7fafa247d8033ab580ab38f74b98f3533db`.
P051 native compilation emitted BF16 mma.sync for the token-first control and
BF16 WGMMA for the weight-first variant. The latter keeps one committed group
potentially outstanding while issuing the next copies. Registers changed from
42 to 36 per thread; shared memory doubled from 10,240 to 20,480 bytes; raw
driver local-memory metadata was zero for both. These are static observations,
not proof of achieved overlap, occupancy or speed.

Hypothesis: preserving the logical tile, arithmetic, split partition and final
reducer while changing orientation can improve whole-call graph latency enough
to justify developing this dataflow further. Layout conversion and doubled
shared memory are explicit possible costs. P046 may already have a superior
vendor route, so an improvement over the matched control alone is insufficient.

## Frozen controls and scope

P051's fixed BM16/BN64/BK64, split-K 8, four warps, two stages, BF16 operands,
FP32 accumulation/partials/reduction and BF16 output remain unchanged. The
external partial tensor remains `[split, token, column]`. No candidate algorithm
is edited in this experiment.

The current working control is the original P046
`candidates/projection_pooler_column_batch2_hybrid.py`, SHA256
`001ea12e8764564d507de51e433769f5bd9a9916bfc4c014f0f87522c278545e`.
Its real prepared state and dispatch are retained: M8 uses the output-column
batch-two BMM, while nearby M7 uses the transposed-weight matrix multiply.
Neither route is forced by modifying the candidate or library globals.

The first stage must pass the original oracle on all five P051 cases: canonical
pooler M8, nearby M7, M15/N65/K131 tails, M1/N63/K1, and M16/N67/K521 split tail.
Its 70 comparisons cover normal/outlier/sparse/zero inputs and captured graph
A→B→A content changes at the same address for both orientations. Timing cannot
start after a compile-only or incomplete receipt.

The execution screen times canonical pooler M8 and nearby M7 against both the
matched orientation control and P046. It uses fresh input values, initialized
CUDA events, 256 MiB L2 eviction before each replay, six predefined warmup
rounds and 24 measured rounds (12 reversed-order pairs) per comparison. The
comparison order also alternates across rounds. Oracle work occurs after both
members of each pair. All samples and tails are retained. Input and prepared
state mutation checks, additional pre/post A→B→A/sparse/zero checks, and exact
source hashes are recorded. No post-hoc sample deletion or fixed-cost subtraction.

## Predeclared gates and modeled value

Correctness is mandatory before any timing interpretation. Both M cases must
show at least a 10% median paired latency reduction against the matched control,
with at least 10 of 12 paired wins each. If the ratio is control/candidate, a
10% latency reduction means ratio >= 1/0.9, not merely 1.1. In addition, the
candidate may be no more than 5% slower than actual P046 on either case.

Passing these gates justifies a broader architecture study; it does not promote
this narrow prototype or establish an official score. P046's pooler M8 occupies
33.378% of one of six equally weighted loads. Even eliminating that path entirely
would only model about +7.0% overall; a 20% target reduction models about +1.16%.
These are conditional score sensitivities, not predictions. The research value
is whether a precision-preserving mechanism can later extend across higher-weight
pooler/QKV/unpooler/O paths. That extension is not implemented or validated here.

## Operational boundary

One ephemeral H200 container on the existing image, existing authentication and
free credits only, paid cap $0. Two bounded child stages, no automatic retry,
no persistent deployment, no official check. The launcher validates exact committed
allowlisted bytes, current billing/credit guards and the serialized GPU slot.
It must stop the app and verify zero tasks after success, failure or timeout.
Raw reports and PTX are source-bound and returned in bounded chunks. Credentials
and unrelated repository material are excluded.

Results, exact commands, checks, app ID, observed billing and conservative reserve
are recorded below. Missing GPU evidence remains explicitly missing.

## Local preflight

Independent source review found no blocker after receipt schema reconciliation.
The timing worker uses isolated function globals and retains the frozen P051
function bodies and constexpr annotations. It counts actual oracle records and
requires 320 screen checks, separately from the prior 70 correctness checks.

Mac Python 3.12.13, standard-library tests only:

```bash
python -B -m unittest \
  tests.test_projection_p052_orientation_screen tests.test_projection_p052_modal \
  tests.test_projection_p051_orientation tests.test_projection_p051_modal
```

Result: **59 passed in 0.317 seconds**. An earlier local test used the wrong
summary key `min` instead of the existing `minimum`; the fixture was corrected.
No kernel or frozen helper changed to address it. Four launcher self-test guards
also passed. These tests do not execute Triton or a GPU.

Frozen new worker SHA256:
`042c395d20ffa6b191629aa79d9b68b9766cb92ea94df56d7763e741d46cdd59`.
Launcher SHA256:
`40b8c8aca0e6e1e673da44bf278fff8326791c6009839918c10e0c3f4a5e0d70`.
The allowlist contains 14 files, 156,788 bytes; no credential/configuration files.
The source-bound launcher enforces 65-second internal/80-second hard limits per
stage, 180 seconds per function, 45 seconds startup and a 225-second app watchdog.
The 255-second resource reserve plus margin is **$0.38609925** at the observed
rate floor, raised if live rates are higher. Current browser evidence shows
$27.86 credits, paid $0 and spend cap $0. Read-only CLI verification at
23:28:17 UTC found existing authentication valid and every app stopped.

## Native result and decision

Source commit: `39fd8cf2538f63ea7511fc2da8f4c88a2473cdc0`.
App: `ap-tarMKbDOkq7ba0yqmHM3UH`, accepted **2026-10-25 23:32:19 UTC**.
The app was created at 23:32:18 and stopped at 23:32:42. Independent read-only
reconciliation at **23:33:07 UTC** verified this app and all other apps stopped,
with zero tasks. GPU slot released immediately after verification.

Reproducible coordinator command, with the existing private control workspace
provided separately rather than embedding a user-specific path:

```bash
python -B projection_p052_modal.py launch \
  --frozen-head 39fd8cf2538f63ea7511fc2da8f4c88a2473cdc0 \
  --control-root "$OKC_CONTROL_ROOT"
```

This exact launcher refuses an existing reservation; the command documents the
completed run, not a request to duplicate it. The remote commands were:

```bash
python -B projection_p051_orientation.py --output correctness.json \
  --deadline-seconds 65 --outer-timeout-seconds 80
python -B projection_p052_orientation_screen.py --output timing.json \
  --correctness-receipt correctness.json \
  --deadline-seconds 65 --outer-timeout-seconds 80
```

**390 original-oracle checks passed**: 70 five-case correctness comparisons,
then 320 screen checks (192 measured replays, 48 predefined warmup replays and
80 pre/post correctness replays). Six capture calls and 18 untimed capture
warmups are separate and not claimed as individually oracle-checked. Original
and all prepared weights remained immutable; same-address sequential A→B→A,
outliers, sparse and zero inputs passed. No concurrency test or full official
9-prerequisite/30-shape suite was run.

Native environment: NVIDIA H200 SM90, Torch 2.14.1+cu130, CUDA runtime 13.0,
Triton 3.8.0, Python 3.12.3. TF32 was false; the existing vendor BF16 reduced
precision reduction setting remained true, unchanged. Our kernels use BF16
operands and FP32 partials/reduction. Both stages ran in the same one-container
invocation. Correctness stage wall time 7.523 seconds, screen stage 5.285 seconds;
total worker 12.809 seconds and client 24.417 seconds. These are wall durations,
not GPU utilization or kernel timing totals.

Each row has 24 measured rounds / 12 reversed-order pairs. The ratio is
opponent/candidate. Latency cut is `1 - 1/ratio`; negative means slower.

| M | Opponent | Opponent raw median µs | Candidate raw median µs | Paired ratio | Candidate latency cut | Winning pairs |
|---:|---|---:|---:|---:|---:|---:|
| 8 | matched token-first | 26.832 | 26.176 | 1.022964 | +2.2448% | 9/12 |
| 8 | actual P046 BMM | 21.728 | 26.304 | 0.824282 | **−21.3177%** | 0/12 |
| 7 | matched token-first | 26.816 | 26.368 | 1.017889 | +1.7574% | 9/12 |
| 7 | actual P046 MM | 25.088 | 26.304 | 0.953484 | −4.8785% | 1/12 |

Both shapes miss the matched-control ≥10% / ≥10-of-12 gates. M8 also clearly
fails the P046 ≤5% regression gate. M7 only passes that regression tolerance,
not the improvement gate. The one P046 M7 sample of **51.680 µs** remains in
the raw data (median 25.088, p90 26.147); no outlier was removed. Candidate
M7 tails reach 29.312 µs against the matched control and 28.000 against P046.
Median raw times and paired ratios use different estimators and must not be
substituted for one another. No fixed-cost calibration, formal confidence
interval or official score is inferred.

**Reject this narrow variant as an optimization candidate.** P046's official
1.1841399409× remains the working result. Do not spend an official check on
P052, and do not extrapolate its small custom-control gain to other families.

## What the experiment establishes, and what it does not

The compiler mechanism is real and its arithmetic passes this bounded GPU
coverage, but the whole-call performance hypothesis fails. This rules out
“orientation alone is sufficient” for these shapes. It does not show whether
copy latency, matrix issue, reduction/launch cost, address arithmetic, shared
memory or layout conversion dominates; no hardware counters or phase trace
were collected in P052.

Prerequisite-stage resources for M8 remain control 42 registers / 10,240 shared
bytes versus weight-first 36 / 20,480; M7 is 42 / 10,240 versus 34 / 20,480.
The reducer is 20 registers / 2,048 bytes. All raw driver local-memory values
are zero. Both canonical shapes emit the expected BF16 mma.sync versus WGMMA.
The M8 instruction text is unchanged from P051; raw PTX hashes differ because
the embedded remote source path and corresponding debug data changed. Saved
resources/PTX belong to the executed prerequisite stage; the timing process
uses the same frozen function bodies and launch settings, but does not export
its own second binary dump. No achieved occupancy, spill-traffic or hardware
overlap claim follows from these static records.

The next useful causal question is whether a competitive full-call design can
reduce the additional partial/reducer path while retaining enough parallelism
and genuine operand overlap. Simply dropping split-K already failed for QKV
M128 in M36, and the M8 vendor BMM is already faster here, so repeating either
configuration is not justified. Before another GPU experiment, use the saved
phase traces and exact API/source capabilities to choose one distinguishable
mechanism with a cross-family benefit model. FlashInfer 0.7.0.post1 presence in
the separate read-only image inventory narrows the source audit but does not
prove a legal precompiled tactic API in the official judge. No new candidate
or compute was started as part of this result interpretation.

## Evidence, cost and shutdown

Raw, byte-preserved reports and 15 PTX artifacts: `results/remote-P052-screen/`
(20 files, 546,528 bytes). `results/router-p052-assessment.json` records all
artifact hashes, recomputed gates, per-shape distributions and resource fields.
The durable lifecycle record is `experiments/projection-p052-screen-modal.json`.
No credentials, private config or unrelated files were transferred.

The launcher initially exited 1 because its immediate post-stop query still
reported one terminating task. Both worker stages had exited 0 and all artifacts
were retrieved. A subsequent **read-only** query verified stopped/tasks0;
no run or stop retry was needed. This was shutdown reconciliation, not a kernel
failure or an uncertain duplicate admission.

Paid usage remained **$0**. The immediate workspace billing delta was zero,
then the later gross snapshot slightly decreased; these are delayed workspace
snapshots, not evidence of a free run or a settled per-app invoice. The full
**$0.38609925** conservative reservation is retained until accounting settles.
No official check, ranked entry, credential change or persistent resource was
created. Current app state is terminal, all tasks zero.
