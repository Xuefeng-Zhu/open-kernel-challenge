# 035 — QKV m128 single full-K output-owner GEMM

Source candidates/projection_qkv_m128_singlepass.py SHA25684e86a5e6fff111220cc6ddc77d3b8885a00f7383b068f360699956de2d39132.
Base officialbest c34c497b/b9b0fd9,job48b1357016134e52,1.14865x. Only QKV m128
uses native full-K BF16 GEMM BM64/BN32/BK64,GM8,warps4,stages3. Grid144CTAs at
N2304, allK8192terms/FP32accumulation and BF16output. No partialbuffer/reducer,
explicitTMA/pipeline rewrite, newframework or changedweightlayout. Derived loop
from tested existing kernel, not its failing pooler dispatch. Other source paths
and prepare are identical to best. NoncontiguousCUDAx falls back to originalGEMM.

TASK permits shape/config dispatch. Canonical m128/QKV N2304/K8192; all other
m1/8/16/32/256 and prerequisite m512 retain bestpaths. Nprefix/KTails masked in
native code; nearby m127/129use unchanged library. No branch on activation values,
outputcache/precomputation/harnessaccess or precision reduction. Re-read official
TASK/AGENTS: full39checks, 1%L2/5xrow bound, sparsecolumnprobes0.1%, BF16weights
stay BF16, router unchanged. CPUmathematicalbranch is not Tritonexecution.

M34ordinary timeline has QKV m12813.696usmain+3.488usreduce. New main may lose
split-K parallelism or bandwidth; eliminating reduce does not guarantee savings.
Conditional10%shape improvement yields only~1.00366 aggregate multiplier with
other officialtimesfrozen, not a forecast. Must show robust whole-workload benefit
in pairedModal timing before spending officialcheck. Preserve any failure.

```sh
.venv/bin/python check.py --submission candidates/projection_qkv_m128_singlepass.py --device cpu --suite all --output results/router-cycle2-qkv-m128-singlepass-cpu.json
PROJECTION_CANDIDATE=projection_qkv_m128_singlepass.py .venv/bin/python -m unittest discover -s tests -p test_projection_cycle2.py -v
.venv/bin/python -m unittest discover -s tests -p test_qkv_m128_dispatch.py -v
.venv/bin/python static_check.py --submission candidates/projection_qkv_m128_singlepass.py
.venv/bin/python -m py_compile candidates/projection_qkv_m128_singlepass.py
```

Actual39CPU+3focused+1nearby/stridedCPU check pass, static/syntax pass. Full
localregression112/112pass before sourcecommit. Environment MacM2/Python3.12.13/Torch2.14.1/noCUDA.
No GPU compilation/correctness/performance evidence at source preparation.
