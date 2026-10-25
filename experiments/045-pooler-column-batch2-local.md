# P045 — two independent output-column batches using library BMM

Status: local preflight passed; ready for serialized GPU validation. No native run or official check admitted.
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
