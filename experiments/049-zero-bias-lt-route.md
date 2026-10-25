# P049 — public zero-bias GEMM epilogue route

Status: local preparation, no native run or official check. Parent is selected
P046, source `fa56f2b`, SHA256
`001ea12e8764564d507de51e433769f5bd9a9916bfc4c014f0f87522c278545e`.
P046 remains selected throughout this experiment.

实验卡：QKV/unpooler 的矩阵形状和 BF16 权重都不变，只通过公开 `addmm`
接口加一个恒为零的偏置，让库尝试另一种融合 epilogue 路线。假设是算法
选择改善，而不是“加零本身更快”。先用两个代表形状、负对照和独立时间线
检验路线是否真的变化；如果没有稳定收益，就停止，不扫描参数。

## Evidence and single causal hypothesis

P044 shows QKV M128's vendor main plus reducer and unpooler M16's vendor main;
these observations do not prove memory/occupancy bottlenecks. P045 shows why
removing a reducer alone is insufficient: its larger-M route regressed.

[PyTorch v2.14.1 CUDA dispatch](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/cuda/Blas.cpp)
uses beta-zero addmm internally for mm. Thus replacing mm by beta-zero addmm
is a negative control, not a novel optimization. A contiguous one-dimensional
bias, beta=1, BF16 matrices and contiguous output qualify for the dedicated
fused-bias Lt entry. The actual installed build may differ; this does not prove
a different GPU algorithm or a speedup. Batch-one bmm is not pursued.

[The corresponding BLAS implementation](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/cuda/CUDABlas.cpp)
constructs an epilogue descriptor and heuristic request, preserves BF16 operands
and existing reduction policy, and can fall back to the unfused path on failure.
No environment variable, global math mode, custom extension or private binding
will be changed. Source inspection is separate from exact generated instructions.
No upstream source was copied; the candidate uses the existing public Torch API.

Exact isolated change: only canonical contiguous BF16 QKV/unpooler M>1 uses
`torch.addmm(zero_bias[:n], x, W[:n].T, beta=1, alpha=1, out=out)`.
The immutable zero vector is prepared from weight shape/device/dtype, not inputs.
Output is invocation-local; all other paths retain P046. The intended mathematical
result remains `x @ W.T`. Rounding, signed zero, sparse probes and outliers are
checked against the original oracle, not bitwise equality to a different GEMM.

## Value, added costs and cheapest falsifier

Ten scored shapes can be affected across five loads. From the immutable P046
receipt, hypothetical uniform target cuts of 5%/10%/20% model whole-score gains
of +1.993%/+4.075%/+8.535%, holding all other times fixed. These are sensitivity
scenarios, not predictions. For each representative, a 10% local target saving
is recorded in `results/router-p049-model.json`; the formal floor sensitivity is
+18.676%, not an achievable forecast for this mechanism.

New costs/risks: zero-bias reads and extra epilogue work; different workspace or
algorithm/reduction choices; unsupported lazy fallback with a copy; graph/output
lifetime or setup differences. No bias creation is moved out of the timed call
unless it is constant, input-independent state allowed by prepare's contract.

The first native screen covers QKV M128 and unpooler M16. Both runners use the
same input addresses/weights. Original P046 is the performance control; beta-zero
addmm with the same output layout is a separate negative control. Cold-L2 CUDA
graph replays use predefined warmup, balanced order, initialized events and all
samples. Separate Torch/CUPTI activity identifies kernels/copies and is excluded
from timing. It is not a hardware-counter or occupancy measurement.

Predeclared native gate: original correctness passes, target raw paired geometric
latency cut at least 5%, neither representative regresses more than 2%. Reject an
unsupported copy/fallback or identical implementation without robust benefit.
A changed kernel name alone is insufficient. Passing two representatives only
justifies full ten-target/control testing, not score or all-shape claims.
Same-address A→outlier B→A, sparse/zero and mutation checks are required; concurrent
calls and all shapes are explicitly outside the first screen.

No GPU slot is currently reserved for P049. Existing credentials and fresh
credit/paid-cap checks, coordinator serialization, bounded one-app launch and
verified shutdown are prerequisites for the screen. No official request has
been made, and no formal entry is intended.

## Local source and checks

Candidate `candidates/projection_qkv_unpooler_zero_bias.py`, SHA256
`4c3b0ed4181cbdc70f8557fd7445b6937c1ae0b236724c87178faf9d02116ad7`.
Thirteen CPU/static tests passed in 1.786 s on the coordinator rerun; static
submission checks passed. No GPU code or new library was executed on this Mac.
Full-K coverage: M2 QKV and sliced unpooler across normal/outlier/sparse/zero,
same-address CPU A/B/A, plus full-width unpooler M9 outliers. Larger M numerical
models use K131; all 30 scored and nine prerequisite dispatches are metadata
tests. These are not full-shape CUDA correctness passes.

Extra immutable state is 20,992 bytes of BF16 zeros. All other P046 executable
AST is unchanged after removing the intended state/helpers/branch. Source
provenance and inherited notices are retained; no upstream implementation copied.

```bash
python -B -m unittest tests.test_projection_zero_bias -v
python -B static_check.py --submission candidates/projection_qkv_unpooler_zero_bias.py
python -B projection_p049_model.py
```

The local Torch installation emitted an optional NumPy initialization warning;
all tests passed, and no dependency was installed. Native screen preparation
continues separately; no external job or score exists for this source.

Independent source review found no blocker and reran all 13 CPU tests in
2.262 s. Original/static weights, invocation-local output, M1 and noncanonical
fallbacks are preserved. CUDA behavior remains unverified.
