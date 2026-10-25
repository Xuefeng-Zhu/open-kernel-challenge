# P044 — unchanged P039 activity and cache-sensitivity diagnostic

Completed on H200. Exact app `ap-pF3Ammg5fT5638UgFJTrox` was accepted at
2026-10-25 21:05:40 UTC and stopped at21:06:23 UTC with zero tasks. No new official
check or score was produced. P039 remains selected at1.173238998969285×.

实验卡：保持 P039 不变，检查五条主要路径的主计算、归约和冷／热缓存表现。
五条路径全部通过正确性与重复图回放。发现部分路径有独立归约，但已有分片
也加速了主计算；不能把“删除归约”直接当成收益，更不能从热缓存对照推断
HBM 瓶颈。

## Observed result

Source freeze/controller commit `b80c5941cc2856ab086e9e17c7bcadf9b1481fbc`
(worker introduced in `b0b3007`). The exact P039 SHA below was checked in both
processes. Environment: NVIDIA H200 SM90,132SMs, Torch2.14.1+cu130,
CUDA13.0, Triton3.8.0, Python3.12.3. Timing and activity subprocesses took
21.012s and14.302s; combined worker35.314s, client43.418s, app lifetime45s.
No NCU attempt, new install, credential change or algorithm change occurred.

All five representatives passed the original FP64 oracle in both modes:
60 same-address ABA checks,60 retained but excluded warmup measurements,
480 measured observations and10 post-activity checks —610 successful oracle
checks in total. There are24 fresh-input rounds and12 reversed-order pairs
per shape/cache condition. All10 nonempty traces and all15 original artifacts
were hash-verified on receipt. Derived prepared-state tensors were not separately
snapshot-compared; original weights and inputs were, and outputs used independently
frozen weights. This does not establish all39 official cases or concurrent-call
correctness anew.

Ratio direction below is **baseline time / P039 time**; greater than1 favors P039.
Times are gross CUDA-graph microseconds, not calibrated official times. Pair p10/p90
are empirical percentiles, not confidence intervals. The ratio need not equal the
ratio of individual medians.

| Path | P039 cold median µs | Baseline cold median µs | Paired ratio median | Pair p10–p90 | P039 warm/cold median |
|---|---:|---:|---:|---:|---:|
| pooler m8 |24.320|25.584|1.05205|1.03048–1.06613|1.1030|
| pooler m128 |28.128|29.728|1.05239|1.03492–1.07335|1.0886|
| pooler m256 |33.344|35.264|1.07268|0.99137–1.10262|1.0246|
| QKV m128 |21.888|22.112|1.01354|0.98384–1.04015|1.0801|
| unpooler m16 |22.832|22.752|1.00500|0.98492–1.01692|1.1353|

P039 won12/12 cold pairs at pooler m8/m128,10/12 at m256/QKV128 and9/12
at unpooler16. These repetitions establish a limited same-run comparison,
not a new full-workload selection or proof that small near-parity gaps are real.
Warm repeats were slower by about8–14% on four paths in both runners. This
control changes queue depth and cache history; host event/launch gaps and device
state remain confounders. It therefore provides no evidence for an L2/HBM
bottleneck. Preserve the slow/outlying samples; no post-hoc trimming was done.

| Path | P039 main/reduction µs | Baseline main/reduction µs | P039 reducer share |
|---|---:|---:|---:|
| pooler m8 |17.919 /2.272|18.497 /2.080|11.25%|
| pooler m128 |19.106 /3.360|23.839 /0|14.96%|
| pooler m256 |29.296 /0|28.605 /0|0|
| QKV m128 |13.603 /3.296|13.792 /3.264|19.50%|
| unpooler m16 |18.110 /0|17.824 /0|0|

These are single separate instrumented samples classified by visible symbol,
not acceptance timing. In particular, the m256 activity ordering differs from
the paired timing result; profiler overhead and single-sample variation were
not estimated and must not be converted into a speedup claim. Candidate m128's
main is faster in the trace despite its added reducer, so deleting split-K can
lose useful parallelism (as previous failed single-pass experiments demonstrated).

All main kernels report384threads/block and168registers/thread. Candidate
shared-memory launch attributes are164292,213252,213268,221492,180692 bytes
for the table order; grids contain128,128,128,96,128CTAs. Reducers report32
registers/thread and zero shared memory. Trace device properties report65536
registers and233472shared-memory bytes per SM. These constrain static residency;
they do not measure achieved occupancy, spills, actual traffic or stalls.
Trace blocks/warps-per-SM fields equal launch work divided by132SMs and are not
occupancy counters. QKV and unpooler have identical visible symbols/grid/block/
register/shared metadata for both runners; all five P039 main symbols match
historical official profiles. **No exact vendor PTX/SASS or spill evidence was
captured.** A loaded binary/symbol mapping is still needed for disassembly.

Preparation is separate, ordered and not paired: timing mode P03947.316ms,
baseline0.440ms; activity mode P03931.615ms, baseline18.864ms. These include
initialization/order effects and are not per-call latency comparisons.

## Cost, shutdown and evidence

Latest preflight Mac browser: free credits$28.22, net spend cap$0, paid$0.
Existing auth verified; no changes. Actual live-rate reserve was$0.709128,
below the approved$0.71. Stop was explicitly requested once at21:06:20;
initial list showed stopping/tasks1, then a read-only follow-up verified stopped/
tasks0. No replacement app or second invocation was launched. GPU slot released.

At21:07:32UTC workspace gross usage was$1.85074494, a$0.07047953 increase
from the preflight snapshot, with paid usage$0. This is an observed workspace
snapshot delta, potentially delayed, not a settled per-app invoice. A separate
50-second exposure model plus margin estimates$0.09178417. The$0.709128 reserve
is preserved in the ledger; delayed accounting is not interpreted as zero cost.

`experiments/projection-p044-modal.json` retains admission, exact payload hashes,
stop observations, rate/reserve and billing snapshots. `results/remote-P044-diagnostic/`
contains every timing sample and sanitized traces; its publication manifest maps
original hashes to published hashes. Only incidental trace host/name/ID metadata
was removed; numerical/resource values are unchanged. Original15files remain
in ignored local artifacts. Re-running the offline assessor against the published
bundle passes, as does assessment of the original bytes.

At21:16:04UTC official quota is Projection4/20 used,16left; Attention17/20
used,3left; both ranked0/10. This diagnostic consumed no official check.
The next reset is2026-10-26 00:00UTC.

## Ordered next local queue — no further GPU admission

1. **Pooler output-column-sharded library BMM.** Preserve BF16 weights, inputs,
   full K and total arithmetic, but pack static output-column halves as a batch
   in prepare. Expanded activation views and interleaved disjoint output views
   can avoid explicit input/concatenation copies. Target only full4096×8192
   pooler with2<=m<=128, preserving m256 and all other paths. The hypothesis is
   that a different batched library decomposition may reduce the measured
   split/reduction cost without the slow custom GEMM routes. Batching does not
   automatically add CTAs, remove split-K or improve traffic. First gates are
   API/address/precision proof and CPU fresh/ABA/edge checks. A native profile
   must establish the actual tactic/copies before any mechanism claim. Reject
   promotion on correctness failure, target regressions>5%, or less than10%
   normalized target improvement and1.5% frozen-target overall benefit. Preserve
   all control shifts and uncertainty. No shard/tile sweep is proposed.
2. **Timing-control preflight.** Review preinitialized/reused CUDA events and
   queue-depth controls before interpreting warm/cold differences. This is a
   measurement reliability change, not an optimized candidate or score claim.

Deleting only the two pooler reducers above, while freezing historical P039
adjusted times, models+1.4521% overall. This deliberately mixes activity with
adjusted timing only as a leverage illustration, not achievable savings. Uniform
20% pooler reductions model+6.9829% over all six loads, or+4.3446% within the
proposed m2–128 scope. These are conditional models; no new gains were achieved.

## Historical pre-admission rationale and protocol

## Frozen parent and motivation

Source: `candidates/projection_router_m1_fp32_gemv.py`, source commit
`cdf5bc2eb1d8c066f9a5cbf1e8f40d6605e6f8da`, SHA256
`0c0c41119ef925b03a33ccab230bc64176cc22c242e3fea77ce01e317376defc`.
Official selected receipt `d5636ac8d1404abd` passed nine prerequisites and thirty
scored cases, with score 1.173238998969285. Preserve this source and selection.

The saved six equally weighted load totals place about 32.62% of latency in
pooler, 22.54% in QKV and 20.93% in unpooler. Cutting all pooler m>1 times by 10%
while freezing everything else models +2.664% overall. This is an estimate of
leverage, not a promised or achieved improvement. Each Projection family has
weight 62; the Attention 46/16 formula does not apply.

## Bounded causal review before this diagnostic

P043 failed decisively despite exact weight compression and full official
correctness. At m8, its reference-normalized latency is approximately 2.249 times
historical raw R06, and R06 is about 1.149 times P039. The matched P043 raw-weight
control has never run on GPU. Historical comparison cannot isolate decoder,
geometry, generated scheduling or register effects. Neither P043 nor selected
P039 currently has saved exact generated instructions or hardware counters.
P043's partial kernel dominated activity; that is not proof of bandwidth or spills.

P039's router m16/m32/m128/m256 has a real separate final half-add kernel after a
wide hi/lo GEMM and split-K reduction. Its instrumented time is 1.5/1.5/1.6/1.6us.
An illustrative subtraction from official adjusted timings models only +1.37%
overall even if that add disappears entirely, or +0.68% if half disappears.
Mixing instrumented activity with adjusted medians is a sensitivity calculation,
not an achievable upper bound. Reproduce exact arithmetic with:

```sh
python3 -B projection_p044_analysis.py
```

A proposed two-GEMM `addmm(out=acc)` route is not queued: old R01's two GEMMs and
their reductions, excluding its final add, already total 11.6/11.8/12.6/13.8us,
versus P039's complete 7.4/7.6/8.6/9.8us activity. A fresh beta=1 library tactic
could differ, but the saved evidence does not justify spending a check merely
because the API exists. [`torch.addmm` 2.14](https://docs.pytorch.org/docs/2.14/generated/torch.addmm.html)
supports the relevant CUDA FP32 output overload; this is API compatibility, not
performance evidence. The [CUDA implementation](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/cuda/Blas.cpp)
can copy `self` into a separate result, so a functional call would introduce work.
The [cuBLASLt epilogue contract](https://docs.nvidia.com/cuda/cublas/index.html#cublasltepilogue-t)
does not expose a documented paired-column fold that directly preserves our one
wide GEMM. Input K packing would require timed activation duplication. No new
candidate has been invented to fill the quota.

The exact saved P039 activity differentiates the cases already: pooler m8 is
18.1us main +1.6us reduction; m128 is21.9+3.0us; m256 is one25.2us kernel.
QKV m128 is12.8+3.1us; unpooler m16 is one18.0us kernel. These are historical
instrumented samples. Therefore eliminating a separate reducer cannot benefit
all five paths, and the failed single-pass QKV result must not be ignored.
The new diagnostic tests whether these stage relationships and cache sensitivity
reproduce on the same source before designing a replacement.

## Measurement protocol and limits

`projection_p041_probe.py` retains the original unmodified oracle. Each runner
captures its own graph and tests same-address normal A, outlier B, then A again;
the returning A must match exactly. Original inputs and weights are checked
against frozen copies. Every timing repetition uses fresh content and the
official FP64 oracle with original tolerances. This is sequential replay, not
concurrent-stream testing or all thirty scored shapes. Derived prepared copies
are not independently snapshot-compared; P039 contains no writes to them and the
output oracle uses frozen original weights. An empty CUDA trace is missing
activity evidence even if numerical correctness passed.

Six warmups are excluded by a predefined rule. Twenty-four rounds make twelve
reverse-order pairs. Cold and immediate-replay warm controls have balanced
ordering. All samples are retained. Timing is gross CUDA-event whole-call time:
there is no guessed four-microsecond subtraction or official-score claim. Warm
cache results are explanatory controls, not acceptance evidence. Preparation is
host wall time bracketed by synchronization and excluded from kernel timings.

Activity runs separately after timing and records one instrumented trace per
runner/representative. Raw trace and hashes are preserved. Register, shared-memory,
grid and block fields are reported only if Kineto actually emits them. Any
occupancy-related trace estimate is not a measured occupancy counter. These five
paths call library GEMMs; Torch activity alone does not provide their PTX/SASS or
spills. Missing instruction evidence remains missing. Both prior NCU attempts
timed out without metrics; this experiment does not retry them.

## Execution prerequisites and accounting

Existing Modal workspace authentication was read-only verified at 20:08 UTC.
The Mac browser showed $28.40 credits, $0 paid charges, $0 net spend limit and $30
usage limit. No cap, payment, credential, or subscription changes were made.
The account CLI reported no live apps. These observations must be refreshed at
admission if stale. The launcher uses the existing image only, an explicit
source/contract/license allowlist, one H200 container and no NCU or installs.
The prospective fixed reserve is $0.709018880 at the existing rate floor
($0.00130496/s for H200 +2CPU +8GiB), including 480 seconds, 10% margin and $0.02.
Live rates can only increase the reserve. Each worker process has150s internal
and180s external limits; the function390s, startup45s, app watchdog420s and stop
allowance15s bound the run. This reserve is a conservative guard, not an invoice
or a promise about cloud enforcement latency. Retrieval
uses bounded chunks, and the exact app must be verified stopped with zero tasks.

Official quota observed 2026-10-25 20:05:12 UTC: Projection checks 4/20 used,
16 remaining; Attention checks 16/20 used, 4 remaining; ranked runs both 0/10.
At 20:11:38 UTC, supported `okc show muse_bf16_proj --json` freshly confirmed
reset at 00:00 UTC (next 2026-10-26). No official check was sent in this phase.

## Decision after evidence

Rank a new mechanism only if the trace and paired distributions identify a
specific reducible stage or a reproducible shape/cache difference. Prefer an
intervention that preserves P039's successful math and improves multiple loads.
Generated resources can constrain a hypothesis, but no claim of HBM saturation,
spills or pipeline stalls is permitted without the corresponding evidence.
If the diagnostic times out or cannot expose a needed fact, preserve that failure
and state the exact missing evidence rather than substituting another kernel's
metadata or performing an arbitrary tile sweep.

## Reproducible commands

```sh
python3 -B -m unittest discover -s tests -p 'test_projection_p041*.py' -v
python3 -B -m unittest discover -s tests -p 'test_projection_p044*.py' -v
python3 -B projection_p044_modal.py self-test
# Requires fresh verified local preflight, existing auth and explicit coordinator slot:
python -B projection_p044_modal.py launch --frozen-head <reviewed-commit> --control-root <existing-control-workspace>
python3 -B projection_p044_assess.py --artifacts-dir local-notes/projection-p044-artifacts --output results/router-p044-assessment.json
```

The launch command is prospective until an accepted app ID is recorded. The
13-file payload contains only the frozen candidate, diagnostic helpers, selected
official contract/reference/baseline and applicable licenses. Auth/config/repo
history are excluded. Local tests validate boundaries and transport, not H200
execution. The result commit must replace this prospective status with observed
coverage, partial failures, exact app ID, hashes, stop state and cost accounting.

Local preflight at2026-10-25 20:17 UTC: 18 plan/worker tests and16 launcher tests passed, plus3 local launcher self-checks. No Modal SDK/network call is made by the self-test. Static diff check passed. GPU compilation, correctness, timing and traces remain pending.

The result assessor adds10 passing offline tests for source binding, partial runs, missing traces, malformed samples and hash/path failures. This brings P044-related local coverage to44 tests; no GPU evidence is implied. Worker/controller payload is frozen by commit b0b3007439e39adf8eb447cd6d53e0743c83d713.
