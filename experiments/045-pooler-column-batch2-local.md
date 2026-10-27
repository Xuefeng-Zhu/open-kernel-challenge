# P045 — two independent output-column batches using library BMM

Status: official check completed; all correctness passed, performance gates failed. Reject promotion and preserve selected P039. Local preflight/admission sections below are historical checkpoints.
P039 remains selected. P044 source, receipts and result commit `aa730b8` are intact.

实验卡：P044 实测 pooler m8/m128 有独立归约开销。把输出列分成两个互不重叠
的批次，保留完整 K 和库矩阵乘法，直接写入同一输出的左右两半。尝试改变库
的并行分解方式，避免已经失败的自写单遍 GEMM。是否去掉归约、是否更快，
都必须由实际 GPU 内核和整次调用计时决定。

## Parent, isolation and mathematical contract

Parent `candidates/projection_router_m1_fp32_gemv.py`, source commit
`cdf5bc2eb1d8c066f9a5cbf1e8f40d6605e6f8da`, SHA256
`0c0c41119ef925b03a33ccab230bc64176cc22c242e3fea77ce01e317376defc`.
Official selected job `d5636ac8d1404abd`,1.173238998969285×,9+30 correctness passed.

Candidate `candidates/projection_pooler_column_batch2.py` targets only pooler
N4096/K8192, BF16 compatible contiguous inputs,2<=M<=128. The scored targets
are m8/16/32/128; m1,m256 and every other family keep the exact P039 dispatch.
Other sizes/dtypes/strides and prefix output widths use P039. This is a single
fixed two-shard decomposition, not a sweep.

Static packing stores `W.reshape(2,2048,8192).transpose(1,2).contiguous()` in
`prepare()`. It preserves every BF16 bit and adds64MiB while retaining P039's
transpose for fallback. No activation enters prepare. The timed call expands
X to[2,M,8192] with batch stride0, allocates one contiguous[M,4096] output, and
uses its[2,M,2048] view with strides[2048,4096,1] as `torch.bmm(...,out=...)`.
Each address is `batch*2048 + row*4096 + column`; the two halves are disjoint
and cover every output exactly once. There is no extra concatenation in source,
no duplicated arithmetic, no cross-shard sum, and no persistent input-derived
scratch. Each invocation owns its output allocation.

BF16 inputs/weights/output are preserved. The pinned PyTorch v2.14.1
[CUDA Blas.cpp](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/cuda/Blas.cpp)
requests FP32 compute for standard BF16 bgemm (lines581–602). Its output-stride
branch at637–640 borrows this output and skips the clone/copyback branches;
input preparation at59–85 accepts the expanded-X and packed-weight within-matrix
strides. Lines712–720 forward the batch/leading strides. This is source evidence
about ATen, not evidence of cuBLAS's selected tactic, generated instructions,
absence of internal scratch or CUDA correctness. No backend/precision flags change.

## Why this is different, and why it may fail

P044 measured candidate main/reducer17.919/2.272us at m8 and19.106/3.360us at
m128; a reducer consumes11.25%/14.96% of those instrumented samples. However,
m128's main is faster than the baseline's single main kernel, so eliminating
split-K alone can lose useful parallelism. The prior full-K custom GEMMs failed
badly. This candidate preserves vendor BF16 multiplication and full K while
changing independent-output grouping; it does not repeat the transpose-only,
custom tile, compression or single-pass Triton mechanisms.

Two shards do not automatically increase CTAs. The batched library may retain
split-K, choose a slower algorithm, produce hidden internal work, or expose less
parallelism. Extra static weight storage is real. Warm/cold P044 controls did
not establish a memory bottleneck, and this proposal makes no such claim.

## Predeclared value and acceptance

Holding all original official P039 timings fixed except the four targets,
uniform target latency cuts5%/10%/20% model+1.0432%/+2.1142%/+4.3446% total
score. These are sensitivity scenarios, not predicted or achieved gains.
An illustrative deletion of only P044's two measured reducers models+1.4521%;
mixing instrumented times with calibrated medians is not an achievable bound.

Before any native admission require local source/packing/address/edge/ABA tests
and independent review. Native correctness must use original tolerances and all
official prerequisite/scored cases. The profile must identify the actually
executed BMM tactic and any reducer/copy; no claim of reducer elimination without
that evidence. Report all30 timings and unchanged controls.

Performance gate: geometric mean reference-normalized latency improvement on
the four targets >=10%, no individual target regression>5%, and frozen-parent
target-only modeled score gain>=1.5%. Investigate all control shifts>5% and
interval/noise limits. Define target improvement as
`1 - geomean(parent_shape_speedup / candidate_shape_speedup)`; this compares
reference-normalized latencies, not the geometric mean of percentages.
A higher aggregate point alone cannot promote. Failed
gates preserve P039. If official source deduplication returns historical results,
do not edit comments or add a nonce to force another measurement.

## Local evidence and outstanding work

An initial CPU API/address model passed81 checks, including15 full-width
A→outlier-B→A checks at m7/8/16/32/128. Its maximum relativeL2 was7.381e-5,
worst-row2.461e-4, with exact sparse probes and unchanged inputs/original/packed
weights. This predates the final candidate test suite and is not CUDA evidence.
The final candidate coverage and source identity are recorded below. CUDA compilation, graph replay, fresh correctness,
library tactic, resources and performance remain unverified.

Code is our narrow modification of the preserved project parent, with embedded
Apache-2.0 attribution intact. Pinned PyTorch source was consulted for contract
behavior; no third-party whole-source program or competitor kernel was executed.


## Final local checkpoint — 2026-10-25 21:26 UTC

Candidate SHA256: `c751d3ec5bff008e9f893f3cc6113184e3a78ae255578e6b398c9d22122d3f58`.
Tests SHA256: `aba51473df9202d9e87535b9e9d2e7d472836a2420c73ae1a69dbe6357d06d7f`.

```bash
python -B -m unittest discover -s tests -p test_projection_pooler_column_batch2.py -v
python -B static_check.py --submission candidates/projection_pooler_column_batch2.py
python3 -B projection_p045_model.py
```

Run these commands with the existing Python 3.12.13 / PyTorch 2.14.1 environment
activated (Darwin arm64, CUDA unavailable). The worktree has no `.venv`
subdirectory; a first coordinator command using that relative path failed before
testing and was corrected to the existing project environment. NumPy is absent;
Torch warns at import, but these Torch-only tests pass without it.

The bounded final suite passed **13/13 tests**, exit 0, unittest elapsed 5.750 s
(wrapper elapsed 6.721 s). A coordinator rerun using the existing project Python
environment also passed 13/13 in 5.674 s; static checks passed again. Static checks, official snapshot hashes, and
`git diff --check` passed. This is Mac CPU execution, not CUDA or graph execution.
Coverage consists of:

- Full K=8192/N=4096 candidate execution at M=2/7 on normal and outlier inputs;
  full-width M=2 sparse/zero and same-address A→B→A; original weights, packed
  state, and inputs remain unchanged.
- Every BF16 bit pattern is preserved by packing, including NaN/Inf payloads.
  This tests storage only; arithmetic correctness is checked on finite inputs.
- M=8/16/32/127/128 helper arithmetic with K=131 and real N=4096 output layout,
  two fresh weight seeds. These are reduced-K models, not full-K validation
  of the four scored paths. Full-K larger-M CUDA execution remains unverified.
- Output address ownership, guard boundaries, device/dtype/stride rejection,
  100 nearby/other-family comparisons, and AST identity of unchanged parent
  functions. Large unchanged fallbacks use dispatch mocks plus AST comparison.

Two earlier CPU suites were intentionally interrupted for excessive full-size
fallback work, each after five completed tests. Neither was a passing suite;
the second interruption elapsed 33.008 s and exited 130. The final suite reduces
unchanged fallback work and clearly labels reduced-K helper coverage. A sandbox
process-list diagnostic was denied and not retried; it is not a GPU or upload
failure. No test process remains active.

Independent source/address review found no definite contract or bounds defect.
It confirmed all output offsets are unique and exhaustive and every preexisting
kernel is unchanged. Remaining native risks are CUDA BMM layout acceptance,
zero-stride input handling, library tactic/internal workspace, graph capture,
fresh input/replay correctness, and runtime. No reduction removal, extra CTA
parallelism, memory-traffic reduction, or speedup has been established.

The reproducible model is `results/router-p045-score-model.json`. Its impossible
zero-target-latency algebraic ceiling is +28.0157% overall; this is not a physical
bound or useful expectation. The practical decision scenarios remain +2.1142%
for a 10% target cut and +4.3446% for a 20% cut. Do not choose this candidate on
an aggregate point estimate alone.

P044 result/source commits are already privately pushed through `aa730b8`.
P045 is local only pending the coordinator's measurement-slot release. The
previous quota snapshot (21:16 UTC, not a reservation) was 16 Projection checks
remaining, with a 00:00 UTC reset. Recheck authentication, account jobs, quota,
source hash, and serialization before any admission. No new check or Modal app
was started by this preflight, and P039 remains selected.

## Ordered follow-up queue

1. Evaluate this fixed two-batch output decomposition once against the original
   oracle and all 30 shape timings; inspect the actual vendor kernel/reducer
   profile. Reject on the predeclared target/control gates above.
2. If this decomposition is a clear loser, do not sweep shard counts. Preserve
   the negative evidence and inspect the executed tactic before selecting
   another mechanism.
3. Before any new claim based on P044 warm/cold behavior, repair the diagnostic
   control: preinitialize and reuse event pairs and compare matched queue depth
   on the same frozen source. This is a timing-method check, not a new kernel
   optimization or a reason to consume an official candidate check.


## Admission and assessment readiness

Source commit: `a3aba7ba7d70ef84bacd957ced35c0f0c9939c0b`. The local runner plan
is `P2-08-pooler-column-batch2`; it has no request/reservation/job ID. After a fresh
coordinator slot release and the wrapper's quota/rules/source checks, the exact
planned command is:

```bash
python -B cycle2_check.py send P2-08-pooler-column-batch2 candidates/projection_pooler_column_batch2.py
```

This command has **not** been executed. A terminal saved receipt can be assessed
without network or GPU activity using:

```bash
python3 -B assess_p045.py results/remote-P2-08-pooler-column-batch2.json --output results/router-p045-official-assessment.json
python3 -B -m unittest discover -s tests -p test_projection_p045_assess.py -v
```

The assessor's 12 focused tests pass. They exercise synthetic known gains,
reference normalization, control movements, failed correctness, malformed rows,
source mismatch and the real P039 times used as an explicitly synthetic
no-change control. They are analysis-tool tests, not P045 performance evidence.
The tool requires all 30 valid unique rows for a numerical comparison and fails
closed on incomplete receipts. Failed compilation/correctness receipts must
still be retained and reported even when no complete timing comparison exists.
Profile execution remains a manual, unverified gate and the tool never changes
the selected parent. No actual P045 receipt exists at this checkpoint.


## Official result — terminal, 2026-10-25 21:35:44 UTC

Exactly one off-board check was accepted at 21:34:36 UTC, job
`b2c1e6c724144e87`, source commit `a3aba7b`, SHA256
`c751d3ec5bff008e9f893f3cc6113184e3a78ae255578e6b398c9d22122d3f58`.
The H200 run started at 21:34:50 and finished at 21:35:44 (about 54 seconds).
It passed all **9 prerequisites and 30 scored correctness cases**, with no flags.
All 30 timings and 60 retained observations/ratios per shape are preserved in
`results/remote-P2-08-pooler-column-batch2.json` and its CSV. The selected P039
source remained unchanged, and no second request was made.

**Decision: reject promotion.** Official score **1.179863621×**, interval
[1.162165667, 1.197561575], versus P039 **1.173238999×**. The point-score change
is only +0.5646%, with overlapping intervals; no reliable aggregate win follows.

| Pooler M | P039 calibrated µs | P045 calibrated µs | Reference-normalized latency cut |
|---:|---:|---:|---:|
| 8 | 20.162 | 17.747 | +8.925% |
| 16 | 19.932 | 18.337 | +6.042% |
| 32 | 20.237 | 18.294 | +7.972% |
| 128 | 22.550 | 24.093 | −5.542% |

The four-target normalized geometric-mean cut is **4.5185%**, below the 10%
gate. M128 exceeds the permitted 5% regression. Holding controls fixed, the
actual observed target changes model only **+0.9638%** total score, below 1.5%.
All three predeclared performance gates fail. All 26 unchanged controls are
reported: only pooler M256 moved beyond 5%, improving 6.155% after normalization.
That path's code is unchanged, so do not attribute its movement to batching.
The original +2.1142% scenario assumed a uniform 10% target cut; it was not met.

### What the real profiles establish

All four target profiles show exactly one listed main kernel, with **no
separate split-K reduction or copy kernel**. P039 used a main and an independent
reducer on all four targets. The P045 names and instrumented durations are:

| M | Actual main kernel | Instrumented µs |
|---:|---|---:|
| 8 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_NNT` | 17.7 |
| 16 | `nvjet_sm90_tst_64x8_64x16_1x2_h_bz_NNT` | 18.5 |
| 32 | `nvjet_sm90_tst_64x16_64x16_1x2_h_bz_NNT` | 18.8 |
| 128 | `nvjet_sm90_tst_64x64_64x13_2x1_v_bz_NNT` | 24.1 |

This supports the intended library-dispatch change and removal of a separately
launched reducer. It does **not** establish less HBM traffic, higher occupancy,
absent internal workspace, or the causal hardware reason for M128's regression.
No P045 PTX/SASS/register/spill metadata or hardware counters were collected.
P044's resource numbers cannot be substituted for these new library kernels.

Official fresh-input CUDA graph checks passed as defined by the published
harness; dedicated exact-source A→B→A graphs, overlapping independent streams,
all-bit CUDA arithmetic and arbitrary-stride native stress were not separately
run. CPU coverage remains separately labeled. The official evaluator hashes and
image digest match P039; cross-run device/runtime drift remains possible.

### Follow-through and queue

A restricted composition using only the three smaller-M wins models +1.2370%
over P039 (1.187752203×), holding everything else fixed. This has **not** been
executed and is below the original 1.5% gate. Do not spend an immediate rescue
check or lower the gate after seeing the result.

The next higher-information local candidate is the same independent-output
batching mechanism applied separately to **QKV M8/16/32/128**, retaining P039
for every other family. P039 profiles show split-K reducers on those four QKV
paths (approximately 1.6/1.7/1.9/3.1 µs in separate activity samples). A uniform
10% QKV target cut models +1.5144% overall; a 20% cut models +3.0878%. These are
sensitivity cases, not a claim that the pooler gains transfer. QKV width 2304
and original TNT/TNN tactics differ from pooler; a local layout/precision review
and original oracle tests must precede any new native request. No QKV candidate
was submitted by P045, and no shard-count sweep is justified by this result.

A second queue item remains the cold/warm timing-method control from P044;
ordinary cache labels did not isolate a measured memory bottleneck. It does not
need an official candidate check. A broad unpooler rewrite is lower priority:
its selected m>8 profiles already show one main kernel, so this reducer-removal
hypothesis lacks the same direct evidence there.

The shared slot was marked terminal and released at the 21:36:40 UTC observation,
before analysis/push. Actual platform quota then: checks **5/20 used, 15 left**;
ranked **0/10 used**; reset **00:00 UTC**. No Modal app, paid resource, new credential
or formal entry was used. P044 remains stopped/tasks zero. One accepted request,
zero uncertain requests, zero retries. Sanitized results are eligible for the
reviewed private result commit; remote verification is reported separately.
