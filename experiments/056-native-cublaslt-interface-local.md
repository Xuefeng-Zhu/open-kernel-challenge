# P056 — owned cuBLASLt interface; native compilation failed

Parent and decision control: unchanged P046,
`candidates/projection_pooler_column_batch2_hybrid.py`, SHA256
`001ea12e8764564d507de51e433769f5bd9a9916bfc4c014f0f87522c278545e`.
P055 corrected the blanket compilation prohibition. This experiment prepares a
small owned diagnostic interface through the documented `load_inline` route;
it is not a replacement submission or a measured optimization.

The sections below describe the predeclared preparation. The terminal result is
recorded at the end; a Modal worker was admitted, but no native GEMM executed.

## Experiment card

**Problem:** library heuristics behind ordinary PyTorch calls may hide useful
algorithm choices across QKV/unpooler. P049 showed that a route change affects
both families differently, so an aggregate wrapper change is not an adequate
causal experiment. P052/P053 custom paths lost to the real P046 control.

**Hypothesis:** explicit, precision-preserving plan choice can expose a genuinely
different execution plan at the same layouts, without an activation copy or a
new mathematical approximation. The probe may falsify interface feasibility
without measuring performance at all.

**Isolated interface:** BF16 row-major A `[M,K]`, original row-major W `[N,K]`
described as column-major B `[K,N]`, row-major BF16 output; opN/opN, FP32
compute and alpha/beta=1/0. No transpose, bias, global math setting or hidden
contiguous copy. The complete returned algorithm object is retained, rather
than rebuilding a potentially different configuration from an algorithm ID.

**Initial precision restriction:** a local no-split reduction preference, with
returned split/reduction attributes and `AlgoCheck` checked again. This excludes
potentially useful FP32 split-K plans deliberately; failure of this subset cannot
reject cuBLASLt or split-K in general. Output-type BF16 partial reductions are
not silently accepted because the main compute mode says FP32.

**Costs/risks:** wrapper and descriptor choices, workspace allocation, heuristic
availability, compiler setup time, strict alignment and incomplete algorithm
coverage. A single eligible algorithm provides no plan-selection comparison.
Heuristic waves are estimates, not measured occupancy. The owned interface uses
the current PyTorch CUDA stream and metadata-only plan construction. Plan/handle,
output and independent workspace owners must outlive graph replay; handle
destruction can synchronize and must stay outside capture/timed calls.

**Value:** two-family M>1 cuts of 10%/20% would model +4.07%/+8.54% whole score
from P046, not a prediction from this uncompiled source. Matching the current
leader by only the three main families would require a 35.72% uniform cut. A
small interface probe is justified by information gain and possible extension
to other BF16 families, not a claim that this first route closes that gap.

## Source and local evidence

- Module: `native_cublaslt_probe.py`, SHA256
  `a40916423e9eb7da4b93258e7e986b55777a56ca93268e9c2e3105f437b409cc`.
- Embedded native source SHA256
  `0e427a632fe12c410b2f0389deadf9c3757b919999287fea5705a892aad69fc8`.
- Tests: `tests/test_native_cublaslt_probe.py`, SHA256
  `f9e0229a462b77a6adbcbc3b4a37b1d78cd4236731ae47aa6697f1d44b5176a4`.

```bash
python -B -m unittest tests.test_native_cublaslt_probe -v
```

Eighteen tests passed on local Python 3.12.13. They cover metadata/address
models, all 65,536 BF16 bit-pattern addresses, official shape dimensions and
nearby odd dimensions, no import-time compilation, explicit plan selection,
alignment/alias/workspace failures, and static API/precision/stream guards.
The router dimensions are only a layout check: this interface rejects FP32
router weights. Integer addressing does not test BF16 arithmetic or CUDA.
An initial test string falsely identified `is_contiguous()` as a hidden copy;
the test was corrected to look for `.contiguous(`. Production source was unchanged.

No Torch import, native build/library load, GPU, remote payload or official
check occurred in these tests. The future builder requires an explicit call;
it is never invoked by local tests. Independent primary-source review verified
the API design; compilation against the actual headers remains necessary.

## Smallest next native gate, then a causal timing screen

1. In the approved existing image, compile only this source through
   `build_native_probe(compile_requested=True)` with CUDA and cuBLASLt linking.
   Retain source/build logs/version evidence; do not install missing software,
   guess constants, or bypass errors. This call has not run.
2. Before capture, enumerate at most 32 heuristics for QKV
   `(M,N,K)=(128,2304,8192)` and unpooler `(16,8192,4096)` with a fixed 64 MiB
   workspace limit. Record accepted/rejected metadata and preserve whole plans.
   No matrix multiplication or synthetic timing occurs during enumeration.
3. Validate selected plans on fresh inputs against the original FP64 oracle,
   column probes, zeros, cancellation/outliers and poisoned output. Check
   immutable W, current-stream capture, same-address A→B→A and independent
   scratch. Unsupported alignment/stride inputs must fail or take an explicit
   future fallback, never silently copy. Include nearby dimensions before a
   complete candidate claims contract coverage.
4. Only after those gates, compare at most two preselected eligible plans per case (four total)
   through identical native descriptors/workspace/precision. First eligible is
   the native control; P046 is a separate real-performance anchor. Use predefined
   six warmups and twelve reversed cold-L2 graph pairs, retaining all samples.
   Initial broad-mechanism gate: both representative shapes improve at least
   10% against P046 with at least 10/12 pair wins each and all correctness checks
   passing. Otherwise do not extrapolate broad gains or spend an official check.
   Changes versus P046 alone still combine API/layout-route and algorithm effects;
   the native comparison is what tests plan selection.

This is a prepared protocol, not executed coverage. Compiler compatibility,
actual numerical bounds, graph/replay/concurrency and native performance are
unverified. No official score will be inferred from these two shapes. Attention
currently owns the coordinated GPU slot; no Projection app has been admitted.

Primary interface evidence: [CUDA 13 cuBLASLt reduction definitions](https://docs.nvidia.com/cuda/archive/13.0.0/cublas/index.html#cublasltreductionscheme-t),
[PyTorch 2.14.1 local reduction-mask use](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/cuda/CUDABlas.cpp#L302-L314),
[PyTorch current-stream API](https://github.com/pytorch/pytorch/blob/v2.14.1/c10/cuda/CUDAStream.h#L218-L233).
No third-party implementation was copied; these sources inform API usage.

实验卡：不再只换 PyTorch 包装函数，先把库内的具体算法选择显式暴露出来。
先验证接口、原精度和 graph 行为，再以相同布局比较算法，并保留 P046 作真实
基线。18 项本地检查通过；尚无编译成功或 GPU 加速结论。

## Reviewed execution freeze, 2026-10-26

The explicit next diagnostic uses projection_p056_native_screen.py and
projection_p056_modal.py. Fifty local native/worker/controller tests passed
in 0.088 s, plus four local payload guards. Independent frozen-source review
found no blocker. The first combined test invocation preceded creation of the
worker test file and failed with missing-module; the completed files then passed.

Worker SHA256 17baa53ba3c3c54201f570cb2a08d9415f8cd0ef4db625625ac7399962843a2f.
Controller SHA256 ea8b99f4b6b7df4027e4d0d93498f67b358e6833403f3151530c7b5704f0ff22.
One 13-file allowlist includes native interface, worker, P046, official oracle,
required harness helpers and notices; it excludes credentials and the controller.
Only report.json, summary.json and a bounded screen.log return. Fixed-worker
supervision is separate from the candidate; native compilation is exclusively
load_inline. Internal 240 s, child hard limit 270 s, function 300 s, startup 45 s, watchdog 345 s.
Conservative reserve $0.55838125 free credits, paid cap $0; one H200, 2 CPU, 8 GiB, max 1,
nondetached, no retry. Fresh account preflight at 00:25:03 UTC verified existing
authentication, all 9 apps stopped with tasks 0 and sufficient free credits.

Two plans per case maximum, first eligible control and a different algorithm ID
preferred for the second; otherwise a distinct recorded configuration is explicitly
labeled. Nearby native shapes are skipped by this bounded plan, not implicitly
passed. Independent graphs with private buffers run for native0; native1 concurrency
and actual simultaneous kernel overlap are not proven. No official check.

```bash
python -B -m unittest tests.test_projection_p056_native_screen tests.test_projection_p056_modal tests.test_native_cublaslt_probe
python -B projection_p056_modal.py self-test --control-root /path/to/control-workspace
python -B projection_p056_modal.py launch --frozen-head <reviewed-source-commit> --control-root /path/to/control-workspace
```


## Terminal result — 2026-10-26 00:33:36 UTC

Frozen execution commit: `113e05e2085f08497c58e1f67af0454b8a4a85eb`.
App: `ap-EaioKmzLxCQFWKWwyp9cmP`, accepted at 00:32:24.515351 UTC.
The worker reached `native_build`; NVCC exited 134 with an internal assertion
in `lexical.c:22316`, `find_allocated_name_reference`, at
`__location->~_Tp()`. The generated host `main.o` compilation step had completed,
but the native CUDA translation unit did not compile. This is a technical
compiler failure, not an access denial, precision failure or algorithm result.
The sanitized diagnostic does not identify the originating header, so a specific
standard-library/header incompatibility remains a hypothesis.

Actual image evidence: Torch 2.14.1+cu130, Torch CUDA 13.0, C++11 ABI enabled,
CUDA header version 13000 and cuBLAS API header version 13.1.1.3. Header hashes
are preserved in the raw receipt and assessment. The build used C++20 through
PyTorch. No package, toolchain, credential or permission was changed.

- Native compilation completed: **no**.
- Native execution and oracle checks: **none / 0**.
- Plan enumeration, graph/replay, independent workspaces and timings: **not reached**.
- Official check or score: **none**. P046 remains selected at 1.1841399409×.
- Worker wall time: 22.283691 s; this is not an allocated-GPU billing duration.

Explicit stop was requested at 00:32:51.319705 UTC. The immediate status was
transitional (`stopping`, one task); the controller correctly required follow-up.
An independent read at 00:33:36.348385 UTC confirmed this exact app stopped with
zero tasks and all workspace apps stopped with zero tasks. The measurement slot
was released. No request was duplicated or retried.

Paid billing remained $0. The observed workspace gross delta was $0, but billing
can lag and is not a per-app invoice. Retain the conservative **$0.55838125 free-credit
reserve** rather than claiming a free execution or a settled cost.

Exact launch, using the project's existing Python environment:

```bash
python -B projection_p056_modal.py launch \
  --frozen-head 113e05e2085f08497c58e1f67af0454b8a4a85eb \
  --control-root /path/to/existing/control-workspace
```

The control-root placeholder contains ignored local authentication; it is not
uploaded. Receipt: `experiments/projection-p056-native-modal.json`. Raw bounded
worker artifacts: `results/remote-P056-native/{report.json,summary.json,screen.log}`.
Assessment and exact artifact hashes: `results/router-p056-assessment.json`.
These copies are byte-identical to the returned artifacts; error paths were
already sanitized by the worker.

Next local repair P057 retains this failed source and its exact native binding
bytes. The binding uses host CUDA/cuBLASLt APIs and has no device kernel, so the
supported `load_inline(cpp_sources=..., cuda_sources=None, with_cuda=True)` route
may avoid unnecessary NVCC parsing while preserving CUDA headers/linking. This
is a new build-route hypothesis, not a demonstrated fix; all native validation
remains required. No GPU retry occurs until a fresh coordinated slot.

实验结果：P056 在 NVCC 编译阶段失败，尚未运行矩阵计算，不能据此否定
cuBLASLt 算法路线。原始失败版本和回执已保留；P057 仅改变编译单元归属，
不改计算和精度，先完成本地审查，再验证真实编译。
