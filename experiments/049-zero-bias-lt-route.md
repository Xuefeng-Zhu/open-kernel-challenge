# P049 — public zero-bias GEMM epilogue route

Status: completed native screen; rejected on declared performance gates. No official check. Parent is selected
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

## Frozen two-case native screen

Candidate source commit `9374d9c8e3a7dc4ec57ab897b184bd5f1285fe53`. The final
worker is `projection_p049_screen.py`, SHA256
`1e6a04fa2a0b0e2105810e1ecd9179f20a18b7bdfb79cb1e868abbd0e8b4d415`.
The coordinator reran 34 candidate, worker and launcher CPU/mock/static tests:
all passed in 1.966 s. Independent worker review found no execution blocker.
These checks do not execute CUDA.

The screen uses QKV M128 and unpooler M16, two separate paired series per case,
six excluded warmup rounds and 24 measured rounds (12 reversed-order pairs) per
series. Full completion performs 326 oracle comparisons, including pre/post
same-address A/B/A, outliers, signed sparse powers of two, zero and immutable
weight checks. All timings precede six separate instrumented activity replays.
Raw samples are retained; no fixed replay-cost subtraction or official score.

The beta-zero control uses its freshly allocated 2-D output as ignored self,
matching mm's native route. The candidate uses the 1-D immutable zero bias. This
is a route-equivalence control, not a beta-only isolation. Negative-control
performance and activity are informational and cannot promote the candidate.
Main parent/candidate activity must be available for the declared screen.

One existing-image ephemeral H200 invocation, 13 allowlisted files only. Internal
deadline 120 s, child hard limit 180 s, function timeout 195 s, startup timeout
45 s, watchdog 240 s, and a conservative 270-second resource reserve plus 10%
and $0.02 margin (about $0.408 at the last verified rate). Fresh credits, net
paid cap $0, existing authentication, no active apps and a coordinator slot are
required before launch. No automatic retries, deployment or retained capacity.
The launcher requires a committed file freeze and refuses an existing ledger.
No P049 app or official request has been created at this preparation checkpoint.

```bash
python -B -m unittest tests.test_projection_zero_bias tests.test_projection_p049_screen tests.test_projection_p049_modal -v
python -B projection_p049_modal.py self-test --control-root /path/to/control-workspace
# Only after fresh budget/auth/slot preflight, using the exact harness commit:
python -B projection_p049_modal.py launch --frozen-head <40-character-commit> --control-root /path/to/control-workspace
```

## Actual H200 outcome — rejected

App `ap-qGKwQTPaVfoieaRJxN3Etw` was accepted at 2026-10-25 22:37:02 UTC,
completed, and independently reconciled stopped with zero tasks at 22:37:36 UTC.
All workspace apps were stopped/tasks0. The initial stop observation was still
`stopping...`; the launcher therefore returned needs-reconciliation. One
read-only status query resolved that known teardown state. No repeated launch
or stop occurred. Measurement slot released before report preparation.

Frozen harness commit `afa9db212c9b7c608f87c227c624cff18ea3e6e3`; candidate source
commit `9374d9c8e3a7dc4ec57ab897b184bd5f1285fe53`. H200 SM90, Torch
2.14.1+cu130, CUDA13.0, Triton3.8.0, Python3.12.3. TF32 remained false; existing
BF16 reduced-precision reduction policy remained true. No global setting changed.
The image is a native diagnostic environment, not the exact official judge overlay.

All **326 original-oracle comparisons passed**: 192 measured, 48 excluded warmup,
80 explicit pre/post probes and six instrumented replays. This includes sequential
same-address A → outlier B → A, signed sparse power-of-two rows, zero and state
immutability. No NaN/Inf output or tolerance failure occurred. Signed-zero bitwise
identity is not established or required by the numerical oracle; no claim of
bitwise equivalence to a different vendor GEMM is made. Other scored shapes,
concurrent calls and official validation remain untested for P049.

| Path | P046 raw median µs | P049 raw median µs | Paired raw latency cut | Winning pairs |
|---|---:|---:|---:|---:|
| QKV M128 | 21.728 | 21.040 | 3.7804% | 10/12 |
| Unpooler M16 | 22.528 | 23.040 | -3.5852% | 2/12 |

The two-case geometric latency cut is **0.1655%**. Both substantive timing gates
fail: below the predefined 5% improvement and an unpooler regression above 2%.
Reject P049, retain selected P046, and do not spend an official check. The QKV
point improvement is not justification to extrapolate to ten shapes or silently
change the declared threshold. Each paired statistic uses adjacent reversed-order
pairs, not the ratio of table medians. No outlier removal or replay-cost subtraction.

The route hypothesis was exercised, not merely assumed:

- QKV parent and beta-zero control execute `nvjet_sm90_tst_96x64_64x10_4x2_h_bz_splitK_TNN`
  plus the cublasLt split-K reducer. P049 executes one
  `nvjet_sm90_tst_48x64_64x15_4x2_h_bz_bias_TNN` kernel. Instrumented stage times
  were parent14.527+3.327µs versus candidate16.831µs.
- Unpooler parent/control execute `nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT`;
  P049 changes to the corresponding `_bias_TNT` symbol. Instrumented times were
  17.952 versus19.358µs. The bias path adds work without removing a listed stage.
- Beta-zero retains the parent symbol sequence in each case; paired timing
  movements +1.0987%/-0.2475% are informational control observations, not wins.
- No visible copy or recognized unfused-fallback warning appeared. That does not
  prove no hidden library work. Symbols are not exact tactic IDs or machine code.

Activity was collected only after all paired timing; instrumented durations are
not the acceptance timing. No instruction/resource/counter measurements occurred
in this screen. In particular, the QKV route change also changes kernel geometry,
so this is an API-selection mechanism test, not isolated reducer overhead.

Worker execution7.081s, supervised child7.807s, client16.390s, recorded app
lifetime16s. Reserve$0.407634500099 retained. Delayed workspace gross usage
delta$0.01641954, actual paid$0; this is not a settled per-app invoice. No new
credentials, installation, formal entry or official check. All raw numerical
artifacts are preserved unchanged with hashes under `results/remote-P049-screen/`;
local process logs remain untracked. Independent audit reproduced counts, hashes,
paired statistics and both failed gates.

The next investigation is local and distinct: inspect whether existing launch-grid
evidence supports any principled larger-pooler partition. Missing grid/resource
data must remain unknown; no blind shard sweep, and no new GPU slot is implied.
