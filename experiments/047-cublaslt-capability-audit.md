# P047 — capability gate for explicit cuBLASLt tactic selection

Status: read-only research and local probe preparation. No FlashInfer import,
JIT, installation, external probe, GPU run or official check has occurred here.
This is not an implemented performance candidate. Selected Projection remains
P039; P045 is rejected and P046 is officially correct and promising, awaiting a direct same-device control confirmation.

实验卡：已测库内核不等于已经达到库的性能上限。下一条高信息量路线是检查
已有绑定能否显式选择 cuBLASLt 算法，而不是继续改 tile。先证明固定环境里
接口、无需编译的加载路径、精度和 workspace 安全性，再决定是否值得测速。

## Established evidence and missing evidence

The latest P045 official receipt pins judge image digest
`sha256:9a7f287b80d772c6f636bbcb57422c619fe326e2aabc53803b85007bca93f68e`.
No saved official package inventory establishes the FlashInfer version, runner
symbols or compatible AOT artifact. The P044 Modal image
`im-cPTbGpNiX3RarKResPgX7h` derives from a different public base; its successful
Torch 2.14.1+cu130 / Triton 3.8.0 run is not proof of the judge's FlashInfer API.
Saved framework availability statements explicitly do not test API versions.

The installed CLI's help exposes no environment/package-inventory command.
A successful kernel check and an evaluator digest do not expose package contents.
Do not spend an official timing check solely to assume an unknown binding exists.

Primary-source research, consulted 2026-10-25, establishes these **upstream** facts:

- The [documented BF16 API](https://docs.flashinfer.ai/generated/flashinfer.gemm.mm_bf16.html)
  does not publicly expose tactic selection or caller-supplied workspace.
- The current [Python implementation](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/gemm/gemm_base.py)
  has `get_mm_bf16_cublaslt_module().cublaslt_bf16_gemm_runner()`, with inputs
  `[a, b, bias, pdl, out, workspace]`, `get_valid_tactics(inputs, profile)` and
  `forward(inputs, tactic=i)`. This internal API is not a stable public contract.
  Its cache key omits device and workspace capacity; keep those fixed per runner.
- The [native wrapper](https://github.com/flashinfer-ai/flashinfer/blob/main/csrc/mm_bf16_cublaslt.cu)
  expects contiguous BF16 A, column-major B and contiguous output with hardcoded
  leading dimensions. QKV/unpooler can use original contiguous W with W.T;
  row-major `pooler_t` would be the wrong layout and must not trigger hidden copies.
- The [BF16 implementation](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/gemm/mm_bf16_cublaslt.cuh)
  explicitly uses BF16 operands and `CUBLAS_COMPUTE_32F`, with FP32 alpha/beta.
  Its heuristic preference constrains workspace bytes, not reduction scheme.
  [CUDA 13.0 definitions](https://docs.nvidia.com/cuda/archive/13.0.0/cublas/index.html#cublasltdatatypes-reference)
  distinguish compute-type from output-type partial reductions, and the default
  mask permits multiple schemes. Compute32F alone therefore does not prove every
  intermediate reduction stays FP32. This is an unverified property, not proof
  that a particular tactic violates the challenge.
- The public wrapper has shared cached scratch (40 MiB upstream); the internal
  runner accepts supplied workspace. Any candidate should own mutable scratch
  per invocation/graph rather than share it through prepared state. Metadata-only
  algorithm descriptors can be shared; activation-derived data cannot persist.
- The [Python getter](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/gemm/gemm_base.py)
  invokes `build_and_load()` on the specification returned by the
  [generator](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/jit/gemm/core.py).
  The [JIT lifecycle](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/jit/core.py)
  may load compatible AOT directly; the [compiler path](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/jit/cpp_ext.py)
  can invoke Ninja subprocesses even for a cached JIT build. No direct download
  was found in this particular generator, which is not a global no-download guarantee.

These links reference mutable upstream `main`, not the installed judge code.
No third-party whole-source software was executed. Actual copied source would
require its [Apache-2.0 license and notices](https://github.com/flashinfer-ai/flashinfer/blob/main/LICENSE).
Using a supported existing API is preferable to copying or compiling a wrapper.

## Eligibility gate before implementation

Saved official TASK permits normal matrix computation, lossless static weight
packing and fresh-input graph execution, and prohibits candidate-started
processes/network and harness inspection. Saved framework notes list FlashInfer,
but do not establish an exception for every lazy-build path. Live rule content
was not independently readable during this audit. Do not loosen the local AST
checker, use ctypes/load_inline, install packages or trigger compilation to make
an unverified binding appear available.

Required evidence before a candidate:

1. Installed distribution versions and exact source hashes from the relevant
   image, including runner signatures and layouts.
2. A compatible AOT loading path that is proven not to spawn a compiler or
   download; a cache filename alone is not enough.
3. Accessible algorithm/reduction metadata or another supported way to preserve
   the original precision contract. No assumption from the compute32F label.
4. A graph-compatible invocation-local workspace plan with fixed capacity and
   no shared mutable scratch. No timing-dependent metadata mutation or synthetic
   benchmark in untimed prepare.
5. A matched same-layout/default-tactic control. Without it, a win combines
   library-wrapper, layout, algorithm and workspace changes and cannot be called
   a tactic-selection effect.

## Minimal next evidence action

Prepare a bounded standard-library metadata probe, without importing FlashInfer,
Torch or CUDA: `importlib.metadata` distributions, METADATA/RECORD entries,
allowlisted source-file hashes and signatures/build markers. Do not use nested
`find_spec`, load shared objects, call a module factory, enumerate secrets or
walk arbitrary cache directories. Record missing files explicitly. An initial
CPU-only inventory in the existing Modal image would address that image only;
judge-overlay compatibility would still need separate supported evidence.
No such external inventory has yet run. A native test is premature until the
availability, lazy-build and reduction-precision questions are resolved.

If those gates pass, the first native experiment should use one QKV M128 case
with a matched default-tactic control, original FP64 oracle, fresh inputs,
same-address graph replay and independent scratch. Tactics are metadata
hypotheses selected before timing; test any timed tuning inside sanctioned
measurement only. A single representative can reject a design but cannot prove
whole-problem correctness or scoring. Preserve all samples and then confirm a
chosen complete implementation on fresh official inputs.

## Whole-score value and limits

The motivation is greater than another isolated B32 adjustment: QKV and unpooler
contribute material time across five loads. Conditional uniform 5%/10%/20% cuts
across their M>1 paths model roughly +1.94%/+3.96%/+8.28% overall in the independently recomputed
P039 receipt model saved as `results/router-p047-score-model.json`. These are
sensitivity scenarios, not predicted benefits. Cost risks include workspace allocation/lifetime, extra wrappers or
copies, changed library setup, precision choices and unsupported lazy compilation.

P045 is the cautionary evidence: eliminating a separately launched reducer did
not guarantee a win at larger M. No new hardware-bottleneck claim is made here.

## Capability gate disposition (2026-10-25)

Unresolved; not evidence that FlashInfer is absent. The receipt's exact image is
addressed through a cluster-internal registry, not an accessible public image
locator. Read-only CLI help and documented API route inspection exposed no package
or AOT inventory endpoint. The advertised public serving base has a different
digest. No image pull, CPU application or package execution was attempted, and
no access-denial outcome is being inferred. A sanctioned exact-image locator or
organizer-provided fixed package/AOT manifest is the missing prerequisite.

Prepared probe: `flashinfer_capability_inventory.py`. Fourteen fixture tests pass:
`python -B -m unittest discover -s tests -p test_flashinfer_capability_inventory.py`.
This is local test coverage only, not runtime inventory. Reads are limited to
64 file attempts, 512 KiB per file, 4 MiB total and 10,000 RECORD rows; relative
paths/symlinks are checked. Only selected package metadata/source and specifically
named mm_bf16_cublaslt binary artifacts are inspected. Binary files are only
statted/hashed, never loaded or decoded. No package import, JIT, compiler,
subprocess, network or environment/cache scanning occurs. The script SHA256 is
`5e928c4b168f5342e461cb5d2032fb558777e1c0f05db4b9b3b2a1562d2e199f`.
The probe is retained for a supported inspection opportunity; it does not block
P046 confirmation and no tactic candidate is implemented on unverified APIs.
