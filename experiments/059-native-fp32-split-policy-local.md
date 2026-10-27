# P059 — admit FP32 split-K with descriptors held fixed

Status: completed on H200; correctness passed, performance gate rejected.
P059 source freeze is `6856cc2cb72d398c88c78b63a375bb3d2e8628e0`.
Parent P057 compiled and passed 444
oracle comparisons, but its four sampled no-split tactics failed the performance
gate. P046 remains selected at 1.1841399409×. P058's independently reviewed
column-layout prototype is parked; this experiment does not incorporate it.

## Experiment card

**Evidence.** Historical P049 activity is bound to the exact P046 source hash
`001ea12e8764564d507de51e433769f5bd9a9916bfc4c014f0f87522c278545e`.
Its QKV M128 path has a split-K main kernel and float-typed cuBLASLt reducer.
P057 excluded all split-K and timed only two of seven eligible QKV tactics.
The remaining ID66 is a known eligible but untimed control. No P057 instructions
or activity were collected, so neither SIMT selection nor a memory/occupancy
bottleneck is established. Historical symbols do not prove an exact reduction enum.

**Hypothesis.** The conservative no-split preference may exclude useful vendor
parallel decomposition. Admitting only FP32 workspace reduction can expose a
useful actual split plan at the same descriptor orientation and data layout.
This is not a rule relaxation: no-split was our first interface gate, not an
official requirement. Native plan metadata must prove the tested mechanism.

**Exact change.** Hold row A, column B, row output, opN/opN and physical X/W/output
buffers fixed. Keep BF16 operands/output, FP32 compute/alpha/beta and current
PyTorch stream. Change only the local reduction preference to
`NONE | COMPUTE_TYPE` and admit returned split plans only when split_k>1 and
scheme is exactly COMPUTE_TYPE. INPLACE and OUTPUT_TYPE remain excluded.
Full returned algorithm and AlgoCheck/workspace validation stay in place. No
manual split count, tile/stage parameter, global setting or input-value tuning.

[NVIDIA's definitions](https://docs.nvidia.com/cuda/archive/13.0.0/cublas/index.html#cublasltreductionscheme-t)
state that COMPUTE_TYPE stores intermediate results in compute-type workspace;
with this binding's FP32 compute, those partials are FP32. INPLACE/OUTPUT_TYPE
use output precision and are excluded. [PyTorch's reduction mask](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/cuda/CUDABlas.cpp#L302-L314)
uses the corresponding precision-safe restriction. Native headers provide the
actual enum values; Python must not guess integer encodings.

**Added costs and risks.** Additional FP32 workspace traffic and a reducer launch
may outweigh parallelism. Plan availability, alignment, workspace size, graph
capture and numerical error remain native test questions. No-split results do
not count as testing this hypothesis. If no eligible actual FP32 split exists,
record a capability result and stop without manufacturing a fallback or sweeping.

**Controls.** One QKV M128 case: replay exact prior row ID41/tile21/stages35;
include eligible untimed row ID66/tile20/stages35; candidate is the first
metadata-eligible actual FP32 split plan; P046 is the separate real anchor.
The ID66 control addresses weak initial sampling. Its result is not attributed
to changing the split policy. All selected descriptors/configurations are logged
before timing; missing controls are reported explicitly.

**Cheap falsifiers.** Local source isolation, scheme/name/number mismatch and
absent-split tests precede any build. Native inventory must contain a valid
actual split before correctness/timing. Original FP64/column-probe tolerances,
new inputs, zero/cancellation/outliers, poisoned outputs and same-address A→B→A
must pass. Private scratch and separate-stream graphs are required; actual kernel
overlap is not inferred. Compare six excluded warmups plus twelve reversed
cold-L2 pairs, retain every sample, then obtain a bounded activity summary
outside acceptance timing. Do not repeat NCU counters.

**Acceptance.** At least 10% paired latency reduction against P046 and ten of
twelve winning pairs, with all correctness passing, would justify broader shape
coverage only. It is insufficient for promotion or an official score. A plan
that only beats the slow ID41 control is not an optimization success. Reject or
retain as diagnostic evidence if the P046 gate fails; never extrapolate a point
from this one shape to all five families.

**Score leverage.** P046 official QKV M128 is 18.193201 us. Cutting only that
shape by 10%/20%/50% models +0.385%/+0.780%/+2.036% whole score. This test is useful
as a mechanism gate that can extend across higher-weight families, not a winning
one-shape tweak. Conditional QKV+unpooler M>1 cuts of 10%/20% model +4.07%/+8.54%;
those wider gains are unmeasured. Raw Modal graph times retain fixed overhead and
must not be substituted for calibrated official timing.

## Execution boundary

Native building remains exclusively the documented host `load_inline` path
validated by P057. No credentials, installs, downloads, arbitrary subprocesses
or public/formal submission. Exact source freeze, allowlisted payload, current
existing authentication, verified free credits / paid cap $0, bounded supervisor
and independent stopped/tasks-zero confirmation are required. These were the
predeclared prerequisites; the execution result is recorded below.

实验卡：P046 的已保存时间线显示 split-K，而首轮接口实验主动禁用了它。
P059 只放开 FP32 工作区归并，保持布局和精度；若没有真实 split-K 算法，就
明确记录缺失。先验证机制，再看是否值得扩大形状覆盖，不靠慢对照制造加速。


## Local interface verification

```bash
python -B -m unittest tests.test_native_cublaslt_fp32_split_probe -v
```

Thirteen CPU/source/metadata/mock tests passed; together with the worker's
nineteen tests, the coordinator rerun passed 32 tests in 0.019 s. The native body is
hash-derived from the frozen original; only reduction preference/filter and
its diagnostic metadata change. Layout, pointers, FP32 compute, BF16 storage,
stream, full algorithm object and runtime workspace guards are identical.
Tests reject unknown/mismatched schemes, unsafe output/in-place reductions,
missing actual splits and changed precision. A small FP32 arithmetic model
covers uneven and empty partitions; a cancellation witness distinguishes FP32
from BF16 partial storage. These are models, not CUDA correctness or native
availability evidence. The builder is mocked; no Torch/native execution occurs.

Source `native_cublaslt_fp32_split_probe.py`, SHA256
`bb2a54cd7cc14bde5ec3a170a6bf8172032abbcade890ee3d6a0262001e614af`.
Derived native-body SHA256
`e3493f7e6d56c18bea112587b620a8bda0bda93cf20bb0d8ade3994ac2865c0c`.

The initial source is preserved in commit
`3c74fa908e8b315c2d69b1c7b8e3b016e8e75123`. A subsequent integration review
found a potential duplicate global pybind `Plan` registration when loading both
extensions in one interpreter. Only P059 now registers `Plan` with
`py::module_local()`; the frozen P057 control and shared ownership remain
unchanged. A regression test reverses that one binding edit and checks the prior
native body exactly. This is a preflight risk fix, not an observed GPU failure.
[Pybind documents global/local coexistence](https://pybind11.readthedocs.io/en/stable/advanced/classes.html#module-local-class-bindings).

The worker `projection_p059_native_screen.py` is frozen at SHA256
`f3ea106e4a3f713c2559d2c102a2aab93b4e375369e16aa1f58e018fb34a5e0f`.
It selects configurations before execution, runs four named paired series,
retains six excluded warmups and 24 samples per series, and records 298 original
oracle comparisons on a completed screen. A→B→A and separate-stream independent
graphs use private mutable buffers. Actual kernel overlap is not established.
Four activity captures occur after all acceptance timing and are not performance
samples. Nearby M127 and all other shapes remain explicitly untested.

```bash
python -B -m unittest tests.test_native_cublaslt_fp32_split_probe tests.test_projection_p059_native_screen -v
```

## Frozen execution harness

Controller `projection_p059_modal.py`, SHA256
`564d26f1c1a6cb8cd48c6ba5f01146fb117a66d4c2e80668c42c634cd3fc55c1`.
The coordinator's complete local rerun passed **54 tests in 0.073 s**, followed
by four local controller guards. No SDK import, native build or GPU action was
performed by these commands:

```bash
python -B -m unittest tests.test_native_cublaslt_fp32_split_probe tests.test_projection_p059_native_screen tests.test_projection_p059_modal -v
python -B projection_p059_modal.py self-test --control-root /path/to/control-workspace
```

The payload contains exactly 17 reviewed files: the two native interfaces,
their frozen provenance, P046, worker, local harness, official oracle/spec and
licenses. It excludes credentials, the controller, P058 and unrelated code.
The controller preserves a capability-only absence separately from compilation,
correctness or successful execution. Safe raw failure receipts are retained.

One H200, 2 CPU and 8 GiB RAM, zero idle containers, one invocation and no retry.
Internal gate 240 s, child hard timeout 270 s, function timeout 300 s, startup
45 s and app watchdog 345 s. The conservative reserve includes 375 s at the
higher of the documented floor and current rates, plus 10% and $0.02:
**$0.5583812501375 at the floor**. Fresh actual free-credit/paid-$0 verification,
existing valid auth and a released serialized GPU slot are mandatory before
launch. An exclusive ledger prevents duplicate or uncertain relaunches. Explicit
stop and independent stopped/tasks-zero reconciliation remain required.

Planned command after those prerequisites, using this report's committed source:

```bash
python -B projection_p059_modal.py launch --frozen-head <reviewed-commit> --control-root /path/to/control-workspace
```

At source freeze there was no P059 admission or cost. The following section
records the subsequent run without altering that frozen payload.

## H200 result — 2026-10-26

App `ap-vbrU0sPpoRjkmN7e7OgK4R` was accepted at **01:09:00.804116 UTC**.
Both host-built extensions loaded successfully: 23.997 s for the old interface
and 22.569 s for the split interface, 46.567 s total. Python 3.12.3,
Torch 2.14.1+cu130, CUDA 13.0, cuBLASLt 130101, NVIDIA H200 (SM90).
No precision globals were changed. The native descriptors explicitly use FP32
compute and scalar types with BF16 input/output and compute-type reduction.

**298 original-oracle comparisons passed**: 48 adversarial/ABA comparisons,
6 independent-graph comparisons, 240 warmup/measured comparisons and four
post-timing activity replays. Fresh content, sparse column probes, zeros,
cancellation/outliers, poisoned output and immutable state passed. Separate
streams had private input/output/workspace buffers; hardware overlap was not
instrumented. Nearby M127 and all other shapes were not executed.

The expanded preference returned eight eligible FP32 split plans. The chosen
metadata-first plan was ID66, tile27, stages35, split2, COMPUTE_TYPE (native enum2),
custom2, workspace **2,359,312 bytes**. The no-split controls were the exact prior
ID41/tile21 and previously unmeasured ID66/tile20 configurations. No plan was
selected by timing. The other seven returned split plans remain unmeasured.

Times below are gross whole-call graph medians in microseconds; **R = paired
parent/candidate**, so R>1 favors the candidate. Each series has six excluded
warmups and 24 measured samples grouped into 12 order-reversed pairs. No samples
were deleted, and no fixed-overhead calibration was applied.

| Series | Parent median | Candidate median | Paired R | Candidate paired latency change | Wins |
|---|---:|---:|---:|---:|---:|
| P046 / row41 | 21.824 | 197.200 | 0.111235591 | +798.993% | 0/12 |
| P046 / row66 | 21.728 | 45.360 | 0.476046888 | +110.063% | 0/12 |
| P046 / FP32 split | 21.840 | 21.856 | 0.996773480 | +0.324% | 6/12 |
| row66 / FP32 split | 45.328 | 21.808 | 2.074843903 | −51.804% | 12/12 |

The split candidate's 51.616 us tail is retained. In the separate row66/split
series, their raw medians changed by −0.071%/−0.220%, respectively; this does not
show a large systematic drift. The result fails the predeclared 10% reduction
and 10/12-win gate against the actual P046 anchor. **Preserve P046; no official
check or promotion.** Correctness is established only for this stated coverage.

Post-timing activity reported the same symbol pair for P046 and the native split:
`nvjet_sm90_tst_96x64_64x10_4x2_h_bz_splitK_TNN` plus the float-typed cuBLASLt
split-K reducer. Their individual instrumented durations were 14.625+3.392 us
and 13.735+3.353 us. These single instrumented replays are not acceptance timing.
The no-split ID66 used `nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNN` (39.906 us),
and row41 used a distinct `sm90_xmma_gemm_bf16bf16_bf16f32...` kernel (192.182 us).
No copy event was reported. No instructions, hardware counters, achieved
occupancy, HBM throughput or stall measurements were obtained.

The useful causal result is that the no-split restriction excluded the route
needed to recover P046-level performance. The 51.8% reduction versus row66
also changes tile20→27 and custom0→2, not just split count; it cannot be assigned
to splitting alone. Equal kernel symbols do not prove identical complete
algorithm objects or launch geometry. The row layout is therefore not sufficient
to explain P057's extreme loss: at that same layout, a precision-safe permitted
plan now reaches P046. P058 stays parked. This does not prove the broader vendor
ceiling, but it removes the premise that merely exposing this native route is a
competitive gain. A next experiment needs a distinct algorithm/dataflow
mechanism with score leverage; a sweep of the remaining seven configurations or
an official check of this flat result is not justified by this evidence alone.

## Shutdown, cost and reproducibility

The supervisor completed in 54.623 s; worker internal wall time was 53.670 s,
including compilation and validation. These are not billed GPU durations.
Stop was requested at 01:10:00.019570 UTC. The first observation was transitional
(`stopping...`, tasks1), so it was not claimed stopped. Independent read-only
confirmation at **01:10:24.310883 UTC** showed this app stopped/tasks0 and all
workspace apps stopped/tasks0. The shared GPU slot was released.

Preflight free-credit lower bound was $27.60740071, paid cap and billed cost $0.
Observed later workspace gross delta was **$0.05740071**; billing can lag and
this is not a settled app invoice. Retain the **$0.5583812501375** conservative
reserve. No new credentials, official check, formal entry or paid resource.
The controller's exit1 reflected the immediate transitional shutdown state;
subsequent reconciliation verified completion and did not relaunch the app.

Raw bytes are preserved in `results/remote-P059-native/`; report SHA256 is
`dcd514c6c4ae2a905b87fe904e582e26ecf55c5778c1057db91eb2c1aa217227`.
The receipt and actual command are bound to the source freeze in
`experiments/projection-p059-native-modal.json`.

```bash
python -B projection_p059_modal.py launch --frozen-head 6856cc2cb72d398c88c78b63a375bb3d2e8628e0 --control-root /path/to/control-workspace
# Historical execution command, not a retry instruction. Existing ledger blocks reruns.
python -B projection_p059_assess.py
```

The read-only assessment rebuilds the JSON decision and 120-row CSV containing
all warmup and measured samples, verifies the receipt, and makes no external
call. Independent review reproduced every paired result and found no sensitive
paths or credential values in the three raw artifacts.

实验结论：FP32 split-K 的能力和正确性已确认，但只追平了 P046。
不能把相对慢对照的 2.07× 当作项目进步。保留 P046，保存失败证据，
不花官方额度验证这次没有收益的版本。
