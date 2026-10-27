"""P059: permit only compute-type FP32 split-K beside the P057 no-split path.

The original BF16 inputs/output, FP32 compute/scalars, row-output descriptors,
current stream, guards and caller-owned workspace remain unchanged. This module
does not configure split counts or geometry: it only admits safe configurations
returned by the library's heuristic query. The dedicated selector requires an
actual split; no-split results are never a successful split-K experiment.

NVIDIA CUDA13 cuBLASLt reduction documentation distinguishes COMPUTE_TYPE
workspace from INPLACE/OUTPUT_TYPE reductions in the output dtype. The native
code compares named header enums and exports their actual compiled values.
Python never assumes their integer encodings. Documentation:
https://docs.nvidia.com/cuda/archive/13.0.0/cublas/index.html#cublasltreductionscheme-t

The P059 Plan binding is module-local so it can coexist with the frozen P057
global Plan binding in the same interpreter. This changes Python type
registration only, not native computation or object ownership:
https://pybind11.readthedocs.io/en/stable/advanced/classes.html#module-local-class-bindings

Import is Python string construction only. Tests are source/metadata/arithmetic
models and mocked host builds; no native availability, accuracy, graph safety or
performance claim follows from them.
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
NONE = "CUBLASLT_REDUCTION_SCHEME_NONE"
COMPUTE_TYPE = "CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE"
INPLACE = "CUBLASLT_REDUCTION_SCHEME_INPLACE"
OUTPUT_TYPE = "CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE"

_NATIVE_SCHEME_NAMES = r'''static const char* reduction_name(uint32_t scheme) {
  switch (scheme) {
    case CUBLASLT_REDUCTION_SCHEME_NONE: return "CUBLASLT_REDUCTION_SCHEME_NONE";
    case CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE: return "CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE";
    case CUBLASLT_REDUCTION_SCHEME_INPLACE: return "CUBLASLT_REDUCTION_SCHEME_INPLACE";
    case CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE: return "CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE";
    default: return "UNKNOWN";
  }
}

struct Plan {'''

_NATIVE_SCHEME_INVENTORY = r'''  py::dict schemes;
  schemes["CUBLASLT_REDUCTION_SCHEME_NONE"] = uint32_t(CUBLASLT_REDUCTION_SCHEME_NONE);
  schemes["CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE"] = uint32_t(CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE);
  schemes["CUBLASLT_REDUCTION_SCHEME_INPLACE"] = uint32_t(CUBLASLT_REDUCTION_SCHEME_INPLACE);
  schemes["CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE"] = uint32_t(CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE);
  inventory["reduction_scheme_constants"] = schemes;
  inventory["policy"] = "metadata-only; FP32 compute; NONE or compute-type split-K; no benchmark";'''

REPLACEMENTS = (
    ("    uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_NONE;",
     "    uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_NONE | CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;"),
    ('sizeof(reduction)), "no-split preference");',
     'sizeof(reduction)), "FP32-safe reduction preference");'),
    ("struct Plan {", _NATIVE_SCHEME_NAMES),
    ('    d["split_k"] = attributes.split_k; d["reduction_scheme"] = attributes.reduction;',
     '    d["split_k"] = attributes.split_k; d["reduction_scheme"] = attributes.reduction;\n'
     '    d["reduction_scheme_name"] = reduction_name(attributes.reduction);'),
    ('    else if (meta.split_k > 1 || meta.reduction != CUBLASLT_REDUCTION_SCHEME_NONE)\n'
     '      reason = "outside_no_split_precision_policy";',
     '    else if (!((meta.split_k <= 1 && meta.reduction == CUBLASLT_REDUCTION_SCHEME_NONE)\n'
     '               || (meta.split_k > 1 && meta.reduction == CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE)))\n'
     '      reason = "outside_fp32_reduction_policy";'),
    ('  inventory["policy"] = "metadata-only; FP32 compute; no split-K; no benchmark";',
     _NATIVE_SCHEME_INVENTORY),
    ('  py::class_<Plan, std::shared_ptr<Plan>>(module, "Plan")',
     '  py::class_<Plan, std::shared_ptr<Plan>>(module, "Plan", py::module_local())'),
)


def derive_source(source):
    if hashlib.sha256(source.encode()).hexdigest() != FROZEN_BINDING_SHA256:
        raise ValueError("Frozen P056 binding source mismatch")
    for before, after in REPLACEMENTS:
        if source.count(before) != 1:
            raise ValueError("FP32 split replacement must identify exactly one source location")
        source = source.replace(before, after, 1)
    return source


CUDA_SOURCE = derive_source(FROZEN_SOURCE)


def source_sha256():
    return hashlib.sha256(CUDA_SOURCE.encode()).hexdigest()


class NoEligibleFP32Split(RuntimeError):
    """An empty actual-split set is an absence result, never no-split fallback."""


def select_actual_fp32_split(inventory, heuristic_position=None):
    """Return an actual safe split, selected by position without benchmarking.

Enum integers must come from the same native inventory as its plans. Validate
all exposed eligible plans, so an inconsistent/unsafe record cannot be silently
ignored while claiming the native filter worked.
"""
    names = (NONE, COMPUTE_TYPE, INPLACE, OUTPUT_TYPE)
    constants = inventory.get("reduction_scheme_constants")
    if not isinstance(constants, dict) or set(constants) != set(names):
        raise ValueError("Actual compiled reduction enum map is required")
    if (any(type(constants[name]) is not int or constants[name] < 0 for name in names)
            or len(set(constants.values())) != len(names)):
        raise ValueError("Invalid compiled reduction enum map")
    if heuristic_position is not None and (type(heuristic_position) is not int or heuristic_position < 0):
        raise ValueError("Heuristic position must be a nonnegative integer")
    candidates = []
    positions = set()
    for plan in inventory["plans"]:
        meta = plan.metadata()
        split, scheme, name = meta.get("split_k"), meta.get("reduction_scheme"), meta.get("reduction_scheme_name")
        position = meta.get("heuristic_position")
        if (type(split) is not int or split < 0 or type(scheme) is not int
                or name not in constants or scheme != constants[name]
                or type(position) is not int or position < 0 or position in positions):
            raise ValueError("Inconsistent native reduction metadata")
        positions.add(position)
        if (meta.get("compute") != "CUBLAS_COMPUTE_32F" or meta.get("scale") != "CUDA_R_32F"
                or meta.get("input_output") != "CUDA_R_16BF"):
            raise ValueError("Native precision metadata changed")
        if not ((split <= 1 and name == NONE) or (split > 1 and name == COMPUTE_TYPE)):
            raise ValueError("Native eligible inventory contains a disallowed reduction")
        if split > 1 and name == COMPUTE_TYPE:
            candidates.append(plan)
    if not candidates:
        raise NoEligibleFP32Split("No actual split-K >1 with verified compute-type FP32 reduction was returned")
    candidates.sort(key=lambda plan: plan.metadata()["heuristic_position"])
    if heuristic_position is None:
        return candidates[0]
    chosen = [plan for plan in candidates if plan.metadata()["heuristic_position"] == heuristic_position]
    if len(chosen) != 1:
        raise NoEligibleFP32Split("Requested heuristic position is not an actual compute-type FP32 split")
    return chosen[0]


def build_native_probe(*, compile_requested=False):
    if compile_requested is not True:
        raise RuntimeError("native build requires an explicit separately supervised action")
    if CUDA_SOURCE != derive_source(FROZEN_SOURCE):
        raise ValueError("Derived FP32 split source mismatch")
    from torch.utils.cpp_extension import load_inline

    return load_inline(
        name="okc_lt_fp32_split_probe_" + source_sha256()[:16],
        cpp_sources=CUDA_SOURCE,
        cuda_sources=None,
        functions=None,
        with_cuda=True,
        extra_cflags=["-O3"],
        extra_ldflags=["-lcublasLt"],
        verbose=False,
    )
