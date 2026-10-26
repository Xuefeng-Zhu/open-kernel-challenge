"""P056: local source-only cuBLASLt interface probe, not a submission.

Importing this module does not import Torch, compile, load a native library, or
touch a GPU. ``build_native_probe(compile_requested=True)`` is a separate future
action; it has NOT been exercised by the CPU tests. No third-party source copied.

The narrow contract is A BF16 row-major [M,K], original W BF16 row-major [N,K]
viewed by cuBLASLt as column-major [K,N], and BF16 row-major output [M,N].
FP32 compute/scalars and a local no-split reduction preference are mandatory.
This deliberately excludes possibly useful FP32 split-K plans. It is not an
assertion that split-K is forbidden or that any chosen heuristic is fast.

Plans contain descriptors, the FULL heuristic algorithm object and metadata,
never inputs, outputs or workspaces. Selection is metadata-only; there is no
benchmark in preparation. Retain the selected plan through all graph replays:
cuBLASLt handle destruction may synchronize. Retain each graph's own output and
workspace tensors through its lifetime; sharing those buffers between concurrent
calls is forbidden. Launch itself performs no allocation, tuning, sync or state
mutation. Actual compile/API support, correctness, capture and concurrent replay
remain unverified until a separately supervised native test.

API provenance: NVIDIA CUDA 13.0 cuBLASLt documentation (matmul descriptors,
algorithm configuration, reduction schemes, current-stream launch), and PyTorch
v2.14.1 aten/src/ATen/cuda/CUDABlas.cpp local reduction preference. Installed
headers remain authoritative; compilation errors must not be bypassed by guessed
constant values or globally relaxed precision settings.
"""

import hashlib
from dataclasses import dataclass


MAX_WORKSPACE_BYTES = 64 * 1024 * 1024
MAX_HEURISTICS = 32


@dataclass(frozen=True)
class TensorMetadata:
    """Pure CPU contract fixture; CUDA validation is repeated in native code."""

    shape: tuple
    strides: tuple
    dtype: str
    device: int
    pointer: int
    conjugate: bool = False
    negative: bool = False


def dimensions(m, n, k):
    if any(type(v) is not int or not 0 < v <= 2**31 - 1 for v in (m, n, k)):
        raise ValueError("M, N and K must be positive signed-32-bit dimensions")
    return {
        "a": {"shape": (m, k), "strides": (k, 1), "order": "row", "ld": k},
        "w": {"shape": (n, k), "strides": (k, 1)},
        "b_descriptor": {"shape": (k, n), "order": "column", "ld": k},
        "out": {"shape": (m, n), "strides": (n, 1), "order": "row", "ld": n},
    }


def validate_contract(m, n, k, device, required_workspace, a, w, out, workspace):
    """Validate addresses/metadata only, without reading tensor contents."""
    layout = dimensions(m, n, k)
    if type(device) is not int or device < 0:
        raise ValueError("device must be a CUDA device index")
    if type(required_workspace) is not int or not 0 <= required_workspace <= MAX_WORKSPACE_BYTES:
        raise ValueError("workspace capacity exceeds the probe bound")
    for name, tensor in (("a", a), ("w", w), ("out", out)):
        if (tensor.shape != layout[name]["shape"] or tensor.strides != layout[name]["strides"]
                or tensor.dtype != "bfloat16" or tensor.device != device
                or tensor.conjugate or tensor.negative or type(tensor.pointer) is not int
                or tensor.pointer <= 0 or tensor.pointer % 256):
            raise ValueError(f"incompatible {name} metadata or 256-byte alignment")
    if (len(workspace.shape) != 1 or workspace.strides != (1,)
            or type(workspace.shape[0]) is not int
            or not max(1, required_workspace) <= workspace.shape[0] <= MAX_WORKSPACE_BYTES
            or workspace.dtype != "uint8" or workspace.device != device
            or workspace.conjugate or workspace.negative
            or type(workspace.pointer) is not int or workspace.pointer <= 0
            or workspace.pointer % 256):
        raise ValueError("workspace must be an aligned invocation-local byte buffer")
    spans = {
        "a": (a.pointer, a.pointer + m * k * 2),
        "w": (w.pointer, w.pointer + n * k * 2),
        "out": (out.pointer, out.pointer + m * n * 2),
        "workspace": (workspace.pointer, workspace.pointer + workspace.shape[0]),
    }
    for left, right in (("out", "a"), ("out", "w"), ("out", "workspace"),
                        ("workspace", "a"), ("workspace", "w")):
        if max(spans[left][0], spans[right][0]) < min(spans[left][1], spans[right][1]):
            raise ValueError(f"storage overlap: {left}/{right}")
    return layout


def select_plan(inventory, heuristic_position):
    """Choose an explicit returned position, never benchmark or rewrite a tactic."""
    if type(heuristic_position) is not int or heuristic_position < 0:
        raise ValueError("an explicit nonnegative heuristic position is required")
    selected = [p for p in inventory["plans"]
                if p.metadata()["heuristic_position"] == heuristic_position]
    if len(selected) != 1:
        raise ValueError("position must identify exactly one supported no-split plan")
    return selected[0]


CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#include <cublasLt.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

namespace py = pybind11;
constexpr int64_t kMaxWorkspace = 64LL * 1024 * 1024;

static void lt_ok(cublasStatus_t status, const char* operation) {
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, operation, " failed: ", int(status));
}
static void cuda_ok(cudaError_t status, const char* operation) {
  TORCH_CHECK(status == cudaSuccess, operation, " failed: ", cudaGetErrorString(status));
}
static void not_capturing(int device) {
  cudaStreamCaptureStatus capture;
  cuda_ok(cudaStreamIsCapturing(c10::cuda::getCurrentCUDAStream(device).stream(), &capture),
          "cudaStreamIsCapturing");
  TORCH_CHECK(capture == cudaStreamCaptureStatusNone, "create/select plans before capture");
}

struct Context {
  int64_t m, n, k, workspace_limit;
  int device;
  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, out = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  Context(int64_t m_, int64_t n_, int64_t k_, int device_, int64_t ws)
      : m(m_), n(n_), k(k_), workspace_limit(ws), device(device_) {}
  // Destroy only outside capture/timed calls and after graph use is finished.
  ~Context() noexcept {
    int old_device = -1;
    cudaGetDevice(&old_device);
    cudaSetDevice(device);
    if (preference) cublasLtMatmulPreferenceDestroy(preference);
    if (out) cublasLtMatrixLayoutDestroy(out);
    if (b) cublasLtMatrixLayoutDestroy(b);
    if (a) cublasLtMatrixLayoutDestroy(a);
    if (op) cublasLtMatmulDescDestroy(op);
    if (handle) cublasLtDestroy(handle);
    if (old_device >= 0) cudaSetDevice(old_device);
  }
  void init() {
    lt_ok(cublasLtCreate(&handle), "cublasLtCreate");
    lt_ok(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F), "MatmulDescCreate");
    cublasOperation_t no_transpose = CUBLAS_OP_N;
    lt_ok(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA,
          &no_transpose, sizeof(no_transpose)), "TRANSA");
    lt_ok(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB,
          &no_transpose, sizeof(no_transpose)), "TRANSB");
    cublasLtPointerMode_t pointer_mode = CUBLASLT_POINTER_MODE_HOST;
    lt_ok(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_POINTER_MODE,
          &pointer_mode, sizeof(pointer_mode)), "POINTER_MODE");
    lt_ok(cublasLtMatrixLayoutCreate(&a, CUDA_R_16BF, m, k, k), "A layout");
    lt_ok(cublasLtMatrixLayoutCreate(&b, CUDA_R_16BF, k, n, k), "B layout");
    lt_ok(cublasLtMatrixLayoutCreate(&out, CUDA_R_16BF, m, n, n), "C/D layout");
    cublasLtOrder_t row = CUBLASLT_ORDER_ROW, column = CUBLASLT_ORDER_COL;
    lt_ok(cublasLtMatrixLayoutSetAttribute(a, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)), "A order");
    lt_ok(cublasLtMatrixLayoutSetAttribute(b, CUBLASLT_MATRIX_LAYOUT_ORDER, &column, sizeof(column)), "B order");
    lt_ok(cublasLtMatrixLayoutSetAttribute(out, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)), "C/D order");
    lt_ok(cublasLtMatmulPreferenceCreate(&preference), "PreferenceCreate");
    size_t workspace_bytes = size_t(workspace_limit);
    lt_ok(cublasLtMatmulPreferenceSetAttribute(preference,
          CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes)), "workspace limit");
    uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_NONE;
    lt_ok(cublasLtMatmulPreferenceSetAttribute(preference,
          CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &reduction, sizeof(reduction)), "no-split preference");
  }
};

struct AlgorithmMetadata {
  int32_t id = 0;
  uint32_t tile = 0, stages = 0, split_k = 0, reduction = 0, swizzle = 0, custom = 0;
};
template <typename T>
static bool read_attribute(const cublasLtMatmulAlgo_t& algorithm,
                           cublasLtMatmulAlgoConfigAttributes_t attribute, T& value) {
  size_t written = 0;
  cublasStatus_t status = cublasLtMatmulAlgoConfigGetAttribute(
      &algorithm, attribute, &value, sizeof(T), &written);
  return status == CUBLAS_STATUS_SUCCESS && written == sizeof(T);
}
static bool read_metadata(const cublasLtMatmulAlgo_t& algorithm, AlgorithmMetadata& meta) {
  return read_attribute(algorithm, CUBLASLT_ALGO_CONFIG_ID, meta.id)
      && read_attribute(algorithm, CUBLASLT_ALGO_CONFIG_TILE_ID, meta.tile)
      && read_attribute(algorithm, CUBLASLT_ALGO_CONFIG_STAGES_ID, meta.stages)
      && read_attribute(algorithm, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, meta.split_k)
      && read_attribute(algorithm, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, meta.reduction)
      && read_attribute(algorithm, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, meta.swizzle)
      && read_attribute(algorithm, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, meta.custom);
}

static void check_bf16(const at::Tensor& t, int64_t rows, int64_t cols,
                       int device, const char* label) {
  TORCH_CHECK(t.is_cuda() && t.get_device() == device, label, " device");
  TORCH_CHECK(t.scalar_type() == at::kBFloat16 && t.layout() == at::kStrided,
              label, " must be strided BF16");
  TORCH_CHECK(t.dim() == 2 && t.size(0) == rows && t.size(1) == cols,
              label, " shape");
  TORCH_CHECK(t.is_contiguous() && t.stride(0) == cols && t.stride(1) == 1,
              label, " requires exact row-major strides");
  TORCH_CHECK(!t.is_conj() && !t.is_neg(), label, " unresolved view bits");
  TORCH_CHECK(uintptr_t(t.data_ptr()) % 256 == 0, label, " requires 256-byte alignment");
}
static bool overlap(const at::Tensor& a, const at::Tensor& b) {
  uintptr_t ab = uintptr_t(a.data_ptr()), bb = uintptr_t(b.data_ptr());
  return std::max(ab, bb) < std::min(ab + size_t(a.numel()) * a.element_size(),
                                    bb + size_t(b.numel()) * b.element_size());
}

struct Plan {
  const std::shared_ptr<Context> context;
  const cublasLtMatmulAlgo_t algorithm;  // FULL heuristic configuration, not just ID.
  const AlgorithmMetadata attributes;
  const size_t workspace_required;
  const int position;
  const float waves_count;
  Plan(std::shared_ptr<Context> c, const cublasLtMatmulAlgo_t& algo,
       AlgorithmMetadata attr, size_t workspace, int index, float waves)
      : context(std::move(c)), algorithm(algo), attributes(attr),
        workspace_required(workspace), position(index), waves_count(waves) {}
  py::dict metadata() const {
    py::dict d;
    d["m"] = context->m; d["n"] = context->n; d["k"] = context->k;
    d["device"] = context->device;
    d["workspace_limit_bytes"] = context->workspace_limit;
    d["workspace_required_bytes"] = workspace_required;
    d["heuristic_position"] = position; d["algorithm_id"] = attributes.id;
    d["tile_id"] = attributes.tile; d["stages_id"] = attributes.stages;
    d["split_k"] = attributes.split_k; d["reduction_scheme"] = attributes.reduction;
    d["cta_swizzle"] = attributes.swizzle; d["custom_option"] = attributes.custom;
    d["waves_count_estimate"] = waves_count;
    d["cublaslt_version"] = cublasLtGetVersion();
    d["compute"] = "CUBLAS_COMPUTE_32F"; d["scale"] = "CUDA_R_32F";
    d["input_output"] = "CUDA_R_16BF";
    d["measured"] = false;
    return d;
  }
  void run(const at::Tensor& x, const at::Tensor& w,
           const at::Tensor& out, const at::Tensor& workspace) const {
    check_bf16(x, context->m, context->k, context->device, "A");
    check_bf16(w, context->n, context->k, context->device, "W");
    check_bf16(out, context->m, context->n, context->device, "output");
    TORCH_CHECK(workspace.is_cuda() && workspace.get_device() == context->device
                && workspace.scalar_type() == at::kByte && workspace.layout() == at::kStrided
                && workspace.dim() == 1 && workspace.is_contiguous() && workspace.stride(0) == 1,
                "workspace must be contiguous CUDA uint8 on the plan device");
    TORCH_CHECK(workspace.numel() >= int64_t(std::max(size_t(1), workspace_required))
                && workspace.numel() <= kMaxWorkspace, "workspace byte capacity");
    TORCH_CHECK(!workspace.is_conj() && !workspace.is_neg()
                && uintptr_t(workspace.data_ptr()) % 256 == 0, "workspace alignment/view bits");
    TORCH_CHECK(!overlap(out, x) && !overlap(out, w) && !overlap(out, workspace)
                && !overlap(workspace, x) && !overlap(workspace, w), "mutable buffer alias");
    c10::cuda::CUDAGuard guard(x.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
    const float alpha = 1.0f, beta = 0.0f;
    lt_ok(cublasLtMatmul(context->handle, context->op, &alpha,
          x.data_ptr(), context->a, w.data_ptr(), context->b, &beta,
          out.data_ptr(), context->out, out.data_ptr(), context->out, &algorithm,
          workspace.data_ptr(), size_t(workspace.numel()), stream), "cublasLtMatmul");
  }
};

static py::dict enumerate_plans(int64_t m, int64_t n, int64_t k, int device,
                                int64_t workspace_limit, int max_results) {
  TORCH_CHECK(m > 0 && n > 0 && k > 0 && m <= INT32_MAX && n <= INT32_MAX && k <= INT32_MAX,
              "positive signed-32-bit dimensions required");
  TORCH_CHECK(device >= 0, "CUDA device index required");
  TORCH_CHECK(workspace_limit >= 0 && workspace_limit <= kMaxWorkspace, "workspace bound");
  TORCH_CHECK(max_results > 0 && max_results <= 32, "heuristic result bound");
  c10::cuda::CUDAGuard guard(c10::Device(c10::DeviceType::CUDA, device));
  not_capturing(device);
  auto context = std::make_shared<Context>(m, n, k, device, workspace_limit);
  context->init();
  std::vector<cublasLtMatmulHeuristicResult_t> results(size_t(max_results), cublasLtMatmulHeuristicResult_t{});
  int returned = 0;
  lt_ok(cublasLtMatmulAlgoGetHeuristic(context->handle, context->op,
        context->a, context->b, context->out, context->out, context->preference,
        max_results, results.data(), &returned), "AlgoGetHeuristic");
  TORCH_CHECK(returned >= 0 && returned <= max_results, "invalid heuristic count");
  py::list plans, rejected;
  for (int index = 0; index < returned; ++index) {
    const auto& result = results[size_t(index)];
    AlgorithmMetadata meta;
    const char* reason = nullptr;
    if (result.state != CUBLAS_STATUS_SUCCESS) reason = "heuristic_state";
    else if (!read_metadata(result.algo, meta)) reason = "metadata_unavailable_or_wrong_size";
    else if (meta.split_k > 1 || meta.reduction != CUBLASLT_REDUCTION_SCHEME_NONE)
      reason = "outside_no_split_precision_policy";
    else if (result.workspaceSize > size_t(workspace_limit)) reason = "workspace_limit";
    cublasLtMatmulHeuristicResult_t checked{};
    if (!reason) {
      cublasStatus_t status = cublasLtMatmulAlgoCheck(context->handle, context->op,
          context->a, context->b, context->out, context->out, &result.algo, &checked);
      if (status != CUBLAS_STATUS_SUCCESS || checked.state != CUBLAS_STATUS_SUCCESS)
        reason = "algorithm_check_failed";
      else if (checked.workspaceSize > size_t(workspace_limit)) reason = "checked_workspace_limit";
    }
    if (reason) {
      py::dict failure; failure["heuristic_position"] = index; failure["reason"] = reason;
      rejected.append(failure);
    } else {
      plans.append(std::make_shared<Plan>(context, result.algo, meta,
          std::max(result.workspaceSize, checked.workspaceSize), index, result.wavesCount));
    }
  }
  py::dict inventory;
  inventory["plans"] = plans; inventory["rejected"] = rejected;
  inventory["requested"] = max_results; inventory["returned"] = returned;
  inventory["policy"] = "metadata-only; FP32 compute; no split-K; no benchmark";
  return inventory;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  py::class_<Plan, std::shared_ptr<Plan>>(module, "Plan")
      .def("metadata", &Plan::metadata)
      .def("run", &Plan::run);
  module.def("enumerate_plans", &enumerate_plans, py::arg("m"), py::arg("n"),
             py::arg("k"), py::arg("device"), py::arg("workspace_limit_bytes"),
             py::arg("max_results") = 32);
}
'''


def source_sha256():
    return hashlib.sha256(CUDA_SOURCE.encode()).hexdigest()


def build_native_probe(*, compile_requested=False):
    """Explicit future native build. Never called by this module's CPU tests."""
    if compile_requested is not True:
        raise RuntimeError("native build requires an explicit separately supervised action")
    from torch.utils.cpp_extension import load_inline

    return load_inline(
        name="okc_lt_probe_" + source_sha256()[:16],
        cpp_sources="",
        cuda_sources=CUDA_SOURCE,
        functions=None,
        with_cuda=True,
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90,code=sm_90"],
        extra_ldflags=["-lcublasLt"],
        verbose=False,
    )
