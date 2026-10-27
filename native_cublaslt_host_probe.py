"""P057 host-translation-unit repair for the frozen P056 native interface.

The binding contains only host C++ calls into CUDA/cuBLASLt, no device kernels.
Reuse its exact bytes and contract helpers. Only load_inline's translation-unit
route changes: host C++ compilation with CUDA headers/link libraries retained.
This is still a GPU interface, not CPU GEMM or a changed numerical algorithm.

Import is inert: no Torch import, native build, library load or GPU access.
CPU mocks validate build arguments only. Native compile/link and all numerical,
capture and timing properties remain unverified for this repaired build route.
"""

from native_cublaslt_probe import (
    CUDA_SOURCE,
    MAX_HEURISTICS,
    MAX_WORKSPACE_BYTES,
    TensorMetadata,
    dimensions,
    select_plan,
    source_sha256,
    validate_contract,
)


FROZEN_BINDING_SHA256 = "0e427a632fe12c410b2f0389deadf9c3757b919999287fea5705a892aad69fc8"


def build_native_probe(*, compile_requested=False):
    """Explicit future build; CPU tests replace the loader with a mock."""
    if compile_requested is not True:
        raise RuntimeError("native build requires an explicit separately supervised action")
    if source_sha256() != FROZEN_BINDING_SHA256:
        raise ValueError("Frozen P056 binding source mismatch")
    from torch.utils.cpp_extension import load_inline

    return load_inline(
        name="okc_lt_host_probe_" + source_sha256()[:16],
        cpp_sources=CUDA_SOURCE,
        cuda_sources=None,
        functions=None,
        with_cuda=True,
        extra_cflags=["-O3"],
        extra_ldflags=["-lcublasLt"],
        verbose=False,
    )
