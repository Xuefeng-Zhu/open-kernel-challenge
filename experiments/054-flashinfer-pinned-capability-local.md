# P054 — pinned FlashInfer tactic capability, local/source evidence only

> Correction, 2026-10-26: the official global rules explicitly allow native CUDA
> through PyTorch load_inline/nvcc. The earlier universal precompiled-only/no-compiler
> gate below is superseded by P055. Loader/source observations remain historical
> facts; actual API availability, precision, graph behavior and time limits still
> need verification. No arbitrary process/network exception is implied.
> See [P055](055-rules-correction-and-leader-gap.md).

Status: source audit and bounded metadata-probe update complete; installed probe
not executed. No FlashInfer package import, factory call, build, binary load,
GPU or official check. This is not an implemented submission. P046 is unchanged.

The separate Attention CPU inventory observed `flashinfer-python=0.7.0.post1`
and `flashinfer-jit-cache=0.7.0.post1+cu130` in the shared test image at
2026-10-25 23:26 UTC. That inventory did not inspect GEMM loader artifacts, and
the test image is not proof of the private official judge's contents.

The [official release](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.7.0.post1)
resolves to commit `946200de1ae94fc93fdd0926f0a13afd1fa7f0f1`. This replaces
P047's mutable-main assumptions with pinned source facts; P047's historical
report and original probe hash remain preserved.

## What the pinned source permits us to conclude

The private getter `get_mm_bf16_cublaslt_module()` exposes a runner with
`get_valid_tactics(inputs, profile)` and `forward(inputs, tactic=i,
do_preparation=False)`. Inputs include output and caller workspace. A tactic
index selects from returned heuristics; it is not a stable global algorithm ID.
The public wrapper instead uses shared workspace. For overlapping graphs, a
future design would need invocation-local scratch and immutable metadata tied
to device, layout and workspace capacity. These are source properties, not
tested graph safety. [Pinned Python implementation](https://raw.githubusercontent.com/flashinfer-ai/flashinfer/946200de1ae94fc93fdd0926f0a13afd1fa7f0f1/flashinfer/gemm/gemm_base.py)

The native wrapper uses hardcoded leading dimensions for row-major A and
original contiguous W, with contiguous output. It obtains the current device
stream through TVM-FFI rather than using the stream-zero wrapper examined in
P047. Stride checks must be explicit if bypassing public validation.
[Native binding](https://raw.githubusercontent.com/flashinfer-ai/flashinfer/946200de1ae94fc93fdd0926f0a13afd1fa7f0f1/csrc/mm_bf16_cublaslt.cu),
[stream helper](https://raw.githubusercontent.com/flashinfer-ai/flashinfer/946200de1ae94fc93fdd0926f0a13afd1fa7f0f1/csrc/tvm_ffi_utils.h).

BF16 operands/output and FP32 compute/scaling are explicit. Heuristic preferences
limit workspace without filtering the reduction scheme. Compute type alone
does not establish every split reduction's precision; the absence of filtering
also does not prove a violation. This remains a separate eligibility question.
[BF16 implementation](https://raw.githubusercontent.com/flashinfer-ai/flashinfer/946200de1ae94fc93fdd0926f0a13afd1fa7f0f1/include/flashinfer/gemm/mm_bf16_cublaslt.cuh),
[CUDA 13 cuBLASLt definitions](https://docs.nvidia.com/cuda/archive/13.0.0/cublas/index.html#cublasltmatmulpreferenceattributes-t).

The getter's build/load path can use a compatible prebuilt artifact. An ordinary
JIT cache is not equivalent: it can still enter Ninja dependency scanning, and
missing or incompatible artifacts can trigger compilation. The disable-JIT
setting fails closed when artifacts are absent; it does not create compatibility.
[Pinned loader](https://raw.githubusercontent.com/flashinfer-ai/flashinfer/946200de1ae94fc93fdd0926f0a13afd1fa7f0f1/flashinfer/jit/core.py).

Architecture-specific providers have metadata including version, module set
and CUDA targets. The resolver checks compatibility. A package name or `.so`
filename alone does not prove the selected artifact will load without building.
Provider factories are executable code; our probe only reads declarations and
bounded source/manifest bytes, never invokes them.
[Pinned AOT resolution](https://raw.githubusercontent.com/flashinfer-ai/flashinfer/946200de1ae94fc93fdd0926f0a13afd1fa7f0f1/flashinfer/jit/env.py).

No third-party implementation was copied or executed. The source is Apache-2.0;
existing notices remain. License permission is not competition eligibility.
Saved official TASK prohibits candidate-started processes/network; no exception
for an arbitrary internal loader is inferred, and no local checker is loosened.

## Completed local prerequisite

`flashinfer_capability_inventory.py` now inspects `jit/env.py`, fixed SM90/SM90a
provider metadata and at most eight strictly named provider dependencies declared
by the shim. It reads only RECORD-listed entry points, source and selected JSON
fields. Factories, binaries, arbitrary cache trees and environment values are
never executed or followed. All original limits remain: 64 file attempts,
512 KiB per file, 4 MiB total and 10,000 RECORD rows. Unsupported/dynamic manifests
remain unknown. Version and SM90 declarations are metadata evidence, not ABI,
JIT-free callability, precision or official-environment proof.

Probe SHA256:
`7de966c2828ff275728bc412458c2e750bcdce61ea2ab55d746415fa2d89e6af`.
Twenty fixture tests passed on the local Python runtime. One privacy fixture
initially caught manifest-path text in a source excerpt; the probe now emits
only allowlisted manifest fields and suppresses factory excerpts. No external
inventory was performed while developing or fixing the probe.

```bash
python -B -m unittest discover -s tests -p test_flashinfer_capability_inventory.py -v
```

Next prerequisite is a bounded standard-library read in the identified image,
then a sanctioned judge-compatible artifact/precision check if that read is
positive. Do not jump to a tactic performance sweep. No factory/import/build,
credential change or runtime operation is part of this local preparation.
