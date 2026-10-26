"""P058: column-output descriptor view of the unchanged P056 GEMM buffers.

Physical row-major Y[M,N] is column-major Y.T[N,M]. Describe the same operation
as Y.T = opT(colW[K,N]) @ opN(colX[K,M]), with leading dimensions K/K/N.
Only descriptors and operand pointer order change; no transpose kernel, copies,
repacking, precision setting, algorithm-policy or workspace changes are added.

The native binding is derived with hash- and count-checked replacements from the
frozen original. Build through the supported P057 host-only load_inline route.
Import performs only Python string construction: no Torch/native import, build,
filesystem read or GPU access. CPU address models do not establish native tactic
availability, numerical tolerances, graph behavior or performance.
"""

import hashlib

from native_cublaslt_probe import (
    CUDA_SOURCE as FROZEN_SOURCE,
    MAX_HEURISTICS,
    MAX_WORKSPACE_BYTES,
    TensorMetadata,
    dimensions,
    select_plan,
    validate_contract,
)


FROZEN_BINDING_SHA256 = "0e427a632fe12c410b2f0389deadf9c3757b919999287fea5705a892aad69fc8"

DESCRIPTOR_REPLACEMENTS = (
    ("    cublasOperation_t no_transpose = CUBLAS_OP_N;",
     "    cublasOperation_t transpose_a = CUBLAS_OP_T;\n    cublasOperation_t no_transpose = CUBLAS_OP_N;"),
    ('          &no_transpose, sizeof(no_transpose)), "TRANSA");',
     '          &transpose_a, sizeof(transpose_a)), "TRANSA");'),
    ('cublasLtMatrixLayoutCreate(&a, CUDA_R_16BF, m, k, k)',
     'cublasLtMatrixLayoutCreate(&a, CUDA_R_16BF, k, n, k)'),
    ('cublasLtMatrixLayoutCreate(&b, CUDA_R_16BF, k, n, k)',
     'cublasLtMatrixLayoutCreate(&b, CUDA_R_16BF, k, m, k)'),
    ('cublasLtMatrixLayoutCreate(&out, CUDA_R_16BF, m, n, n)',
     'cublasLtMatrixLayoutCreate(&out, CUDA_R_16BF, n, m, n)'),
    ('cublasLtOrder_t row = CUBLASLT_ORDER_ROW, column = CUBLASLT_ORDER_COL;',
     'cublasLtOrder_t column = CUBLASLT_ORDER_COL;'),
    ('cublasLtMatrixLayoutSetAttribute(a, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row))',
     'cublasLtMatrixLayoutSetAttribute(a, CUBLASLT_MATRIX_LAYOUT_ORDER, &column, sizeof(column))'),
    ('cublasLtMatrixLayoutSetAttribute(out, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row))',
     'cublasLtMatrixLayoutSetAttribute(out, CUBLASLT_MATRIX_LAYOUT_ORDER, &column, sizeof(column))'),
    ('          x.data_ptr(), context->a, w.data_ptr(), context->b, &beta,',
     '          w.data_ptr(), context->a, x.data_ptr(), context->b, &beta,'),
)


def derive_source(source):
    if hashlib.sha256(source.encode()).hexdigest() != FROZEN_BINDING_SHA256:
        raise ValueError("Frozen P056 binding source mismatch")
    for before, after in DESCRIPTOR_REPLACEMENTS:
        if source.count(before) != 1:
            raise ValueError("Descriptor replacement must identify exactly one source location")
        source = source.replace(before, after, 1)
    return source


CUDA_SOURCE = derive_source(FROZEN_SOURCE)


def source_sha256():
    return hashlib.sha256(CUDA_SOURCE.encode()).hexdigest()


def descriptor_contract(m, n, k):
    """CPU-only declaration of physical views; no tensor materialization."""
    dimensions(m, n, k)  # Preserve original positive signed-32-bit shape checks.
    return {
        "a": dict(buffer="w", shape=(k, n), order="column", ld=k, operation="T"),
        "b": dict(buffer="x", shape=(k, m), order="column", ld=k, operation="N"),
        "out": dict(buffer="out", shape=(n, m), order="column", ld=n),
        "logical_output": (m, n),
    }


def build_native_probe(*, compile_requested=False):
    """Future host-only build, never performed by the local CPU tests."""
    if compile_requested is not True:
        raise RuntimeError("native build requires an explicit separately supervised action")
    if CUDA_SOURCE != derive_source(FROZEN_SOURCE):
        raise ValueError("Derived column descriptor source mismatch")
    from torch.utils.cpp_extension import load_inline

    return load_inline(
        name="okc_lt_column_probe_" + source_sha256()[:16],
        cpp_sources=CUDA_SOURCE,
        cuda_sources=None,
        functions=None,
        with_cuda=True,
        extra_cflags=["-O3"],
        extra_ldflags=["-lcublasLt"],
        verbose=False,
    )
