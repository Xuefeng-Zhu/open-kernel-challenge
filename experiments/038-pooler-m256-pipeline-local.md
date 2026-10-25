# P038: pooler m256 TMA/copy local prototype, pending coordinator review

This is a NEW local source experiment, preserving P037 and its frozen admission
model without edits. No GPU or official request is admitted by this record.

Parent: candidates/projection_full_composition.py, source commit
b9b0fd9fb0cf326b5836167ffc29d1a0d3386fa7, SHA256
c34c497b8219b7c8e37ee7da754db7444db351a7eb27f04473796fdadc1a57f7.
Best official job48b1357016134e52 remains1.1486499523461586x.

## Hypothesis and exact controlled change

An explicit descriptor/TMA load path with compiler-managed three-stage software
pipelining may improve pooler m256 relative to library best. Actual DRAM traffic,
stalls and achieved occupancy are unknown. This is not a measured bottleneck.

Two self-contained files differ ONLY in USE_TMA=True/False. Both use
BM64/BN64/BK64, 4warps, launch num_stages3 and tl.range(num_stages=3), same 2D grid
and masked BF16 store. BF16 x/W enter tl.dot with FP32 accumulation. TMA takes
host descriptor arguments; control takes original pointers. No manual barrier,
warp specialization, autotuning, global allocator mutation, or input-value branch.
The compiler owns asynchronous scheduling; actual TMA/WGMMA emission is UNVERIFIED.

Only pooler m256, positive N/K multiples of64, dense BF16 operands on the same
device, matching K and sufficient weight rows,16-byte aligned bases enter the
prototype. GPU must have compute capability major9. Other devices, noncontiguous
layouts, N/K tails and nearby m255/257 retain the parent's library route. CPU
compatible cases use a mathematical FP32 accumulation model, never Triton.
Weights/activations are not mutated; no activations or outputs are cached.
prepare(), every existing helper/kernel, and kernel_fn after removal of precisely
the new dispatch branch are AST-identical to parent. Tests enforce this.

Comparison against best is essential: changing geometry and descriptor loading
together cannot establish a causal TMA gain. The conventional-copy control is
ready for a separately justified measurement only; no such run is authorized here.

## Exact supported primary API, read 2026-10-25

Pinned v3.8.0, NOT assumptions based on current main:

- https://github.com/triton-lang/triton/blob/v3.8.0/python/triton/tools/tensor_descriptor.py
  TensorDescriptor.from_tensor(tensor,block_shape) creates host metadata; constructor
  checks16-byte base/leading-stride alignment, contiguous final dimension, positive
  shapes and power-of-two block sizes. No allocator or global-state mutation.
- https://github.com/triton-lang/triton/blob/v3.8.0/python/tutorials/09-persistent-matmul.py
  matmul_tma constructs host descriptors, passes them as JIT arguments and uses
  descriptor.load plus tl.dot. Adapted API/load-dot pattern only; MIT attribution
  in candidate headers, THIRD_PARTY_NOTICES.md and licenses/Triton-MIT.txt.
- https://github.com/triton-lang/triton/blob/v3.8.0/python/triton/language/core.py
  tl.range supports num_stages; Hopper uses software pipelining, while automatic
  warp specialization is documented for Blackwell. We do not request it.
- https://github.com/triton-lang/triton/blob/v3.8.0/python/test/unit/language/test_tensor_descriptor.py
  includes BF16 descriptor matmul at64/64/64; upstream test source was read only,
  never executed or vendored. This does not validate this new kernel.
- https://github.com/triton-lang/triton/blob/v3.8.0/third_party/nvidia/backend/driver.py
  make_tensordesc_arg encodes the host descriptor and returns tensor-map arguments.
- https://github.com/triton-lang/triton/blob/v3.8.0/third_party/nvidia/backend/driver.c
  descriptor is copied by value into kernel-argument storage (around line1095).
- https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__GRAPH.html
  CUDA graph kernel nodes copy parameter arrays and argument values. Therefore
  host Python descriptor lifetime need not span replays. Underlying x/W/out device
  allocations must remain valid as usual. This is a source/API lifetime argument,
  not an actual CUDA graph replay result; capture/fresh replays remain required.

The local static import allowlist adds only exactly
from triton.tools.tensor_descriptor import TensorDescriptor with no alias or
extra imported symbols. All previous source/sync/manifest checks remain active.
No other new import is permitted. Official validator compatibility is untested.

## Local commands and actual outcomes

Use existing primary-workspace .venv/bin/python (Python3.12.13/Torch2.14.1,
MacM2/noCUDA/noTriton), working directory /tmp/okc-projection-pipeline-preflight:

```sh
PY=<primary-workspace>/.venv/bin/python
$PY -m unittest discover -s tests -p test_pooler_pipeline_preflight.py -v
$PY check.py --submission candidates/projection_pooler_m256_tma.py --device cpu --suite all --output results/router-p038-pooler-m256-tma-cpu.json
$PY check.py --submission candidates/projection_pooler_m256_copy.py --device cpu --suite all --output results/router-p038-pooler-m256-copy-cpu.json
$PY static_check.py --submission candidates/projection_pooler_m256_tma.py
$PY static_check.py --submission candidates/projection_pooler_m256_copy.py
$PY -m py_compile candidates/projection_pooler_m256_tma.py candidates/projection_pooler_m256_copy.py tests/test_pooler_pipeline_preflight.py static_check.py
$PY -m unittest discover -s tests -v
```

Actual: both39-case CPUoracles pass;5focused tests pass;117full tests pass;
static/syntax and git diff --check pass. Fresh/probe/zero-row model, metadata/tail/
strided/alignment fallback and source-invariant evidence are CPU-only. Full suite
log results/p038-preflight-tests.local.txt is ignored local output; durable summary
and exact source hashes: results/router-p038-preflight-cpu.json. Known missing
NumPy warning remains; no install attempted. No CUDA/Triton execution occurred.

## Remaining native gates and stop criteria

Coordinator review/immutable source commit precede any remote upload. No GPU job
was launched here. Attention exclusively owns GPU scheduling. Eight remaining
projection checks are not admissions to this subagent. No credential/install/
OKC/Modal/push/checkout switch/commit occurred.

If separately admitted: retain full raw terminal/profile outputs even on failure;
check native compilation/instruction path/resources and all9+30oracle cases,
fresh weights/inputs, column probes/outliers, m255/256/257, aligned N prefixes,
N4093 and K tails, noncontiguous/unaligned inputs, and device fallback. Capture
kernel_fn, discard local Python descriptor references, then replay fresh inputs
repeatedly while retaining x/state/out allocations to verify graph lifetime.
Use cold-L2 paired CUDA-graph timings against immutable full-composition best;
never infer performance from this CPU model or eager timing.

Stop on native failure/regression; no automatic retry/tile sweep or control run.
Only measure same-geometry copy control if a material TMA-candidate gain or a
concrete diagnosis warrants it. Do not attribute gains to TMA without that control.
Frozen P037 scenario:10%faster m256 gives0.5373%aggregate;27.0105%latency reduction
is needed for1.5%aggregate;17.4us floor would yield2.4362%. These are scenarios,
not predictions. No gain or official admission is claimed.

Both standalone candidates embed the complete MIT notice and existing Apache2
license as comments, so the single-file upload retains applicable license texts.
Final CPU reports were rerun after this comment-only notice change; their recorded
source SHA256 equals the hashes below. No executable algorithm changed.

Source hashes (uncommitted pending parent review):

- candidates/projection_pooler_m256_tma.py: 443128591305c4a8f79f2e8669d59b3cc402a5243b9920273676583b825f518d
- candidates/projection_pooler_m256_copy.py: 1b6fe24df633fb8e01e69cfa0cc3d78992af6c8deae033f6b08093c668d457cb
