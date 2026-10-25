# P044 — unchanged P039 activity and cache-sensitivity diagnostic

Status: local preparation. No P044 GPU app or official check has been admitted.
The measurement slot was released to the ready Attention probe at 20:11 UTC;
Projection must receive a new coordinator grant before launching.

Experiment card: preserve the selected computation and investigate its dominant
paths before changing another kernel. Compare P039 with the official SGLang
baseline on pooler m8/m128/m256, QKV m128, and unpooler m16. Capture preparation
time separately, gross cold/warm CUDA-graph timings, and ordinary activity traces.
This is a diagnostic, not a new optimization or an official score.

实验卡：先不改 P039。观察五条主要路径的主计算、归约和缓存敏感性，找出
值得修改的重复工作；不把时间线当成 HBM 流量或占用率测量。

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
