# P059 — admit FP32 split-K with descriptors held fixed

Status: local preparation, no admission. Parent P057 compiled and passed 444
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
and independent stopped/tasks-zero confirmation are required. No P059 job,
request, cost or GPU result exists yet; wait for coordinated measurement slot.

实验卡：P046 的已保存时间线显示 split-K，而首轮接口实验主动禁用了它。
P059 只放开 FP32 工作区归并，保持布局和精度；若没有真实 split-K 算法，就
明确记录缺失。先验证机制，再看是否值得扩大形状覆盖，不靠慢对照制造加速。


## Local interface verification

```bash
python -B -m unittest tests.test_native_cublaslt_fp32_split_probe -v
```

Twelve CPU/source/metadata/mock tests passed in 0.002 s. The native body is
hash-derived from the frozen original; only reduction preference/filter and
its diagnostic metadata change. Layout, pointers, FP32 compute, BF16 storage,
stream, full algorithm object and runtime workspace guards are identical.
Tests reject unknown/mismatched schemes, unsafe output/in-place reductions,
missing actual splits and changed precision. A small FP32 arithmetic model
covers uneven and empty partitions; a cancellation witness distinguishes FP32
from BF16 partial storage. These are models, not CUDA correctness or native
availability evidence. The builder is mocked; no Torch/native execution occurs.

Source `native_cublaslt_fp32_split_probe.py`, SHA256
`2f369143631f28a98695875a077ed2dc51795e87b2d5b26e103b08fe04b5cafd`.
Derived native-body SHA256
`996ad67db33a2e22b10c525a234ddb65d6996ef7113eca3aa64e02521267e91f`.
The bounded worker/controller remains in local preparation; no GPU admission.
