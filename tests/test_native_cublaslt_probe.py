"""P056 source/metadata tests only: never import Torch or compile native code."""

import ast
from dataclasses import FrozenInstanceError, replace
import hashlib
from pathlib import Path
import random
import struct
import unittest
from unittest.mock import patch

import native_cublaslt_probe as probe


class FakePlan:
    def __init__(self, position):
        self.position = position

    def metadata(self):
        return {"heuristic_position": self.position}


class NativeCuBLASLtContract(unittest.TestCase):
    def fixtures(self, m=3, n=5, k=7, workspace=4096):
        return (
            probe.TensorMetadata((m, k), (k, 1), "bfloat16", 0, 0x10000),
            probe.TensorMetadata((n, k), (k, 1), "bfloat16", 0, 0x20000),
            probe.TensorMetadata((m, n), (n, 1), "bfloat16", 0, 0x30000),
            probe.TensorMetadata((workspace,), (1,), "uint8", 0, 0x40000),
        )

    def validate(self, values=None, required=4096):
        return probe.validate_contract(3, 5, 7, 0, required, *(values or self.fixtures()))

    def test_module_import_has_no_top_level_native_or_torch_import(self):
        tree = ast.parse(Path(probe.__file__).read_text())
        for node in tree.body:
            if isinstance(node, ast.Import):
                self.assertTrue(all(x.name == "hashlib" for x in node.names))
            if isinstance(node, ast.ImportFrom):
                self.assertEqual(node.module, "dataclasses")
            self.assertFalse(isinstance(node, ast.Expr) and isinstance(node.value, ast.Call))
        self.assertNotIn("import torch", Path(probe.__file__).read_text())

    def test_build_is_explicit_and_default_does_not_import_torch(self):
        with patch("builtins.__import__", side_effect=AssertionError("no package imports")):
            for value in (False, None, 0, 1, "yes"):
                with self.assertRaises(RuntimeError):
                    probe.build_native_probe(compile_requested=value)

    def test_source_digest_and_native_build_flags(self):
        self.assertEqual(probe.source_sha256(), hashlib.sha256(probe.CUDA_SOURCE.encode()).hexdigest())
        self.assertEqual(len(probe.source_sha256()), 64)
        tree = ast.parse(Path(probe.__file__).read_text())
        build = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name == "build_native_probe")
        call = next(n for n in ast.walk(build) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name) and n.func.id == "load_inline")
        kw = {node.arg: node.value for node in call.keywords}
        self.assertEqual(ast.literal_eval(kw["extra_ldflags"]), ["-lcublasLt"])
        self.assertTrue(ast.literal_eval(kw["with_cuda"]))
        self.assertNotIn("--use_fast_math", ast.literal_eval(kw["extra_cuda_cflags"]))

    def test_all_thirty_projection_shapes_and_nearby_layouts(self):
        layers = ((2304, 8192), (8192, 1024), (4096, 8192), (8192, 4096), (256, 8192))
        count = 0
        for m in (1, 8, 16, 32, 128, 256):
            for full_n, k in layers:
                n = 1024 if (full_n, k) == (8192, 4096) and m <= 8 else full_n
                layout = probe.dimensions(m, n, k)
                self.assertEqual(layout["a"]["strides"], (k, 1))
                self.assertEqual(layout["b_descriptor"], {"shape": (k, n), "order": "column", "ld": k})
                self.assertEqual(layout["out"]["strides"], (n, 1))
                count += 1
        self.assertEqual(count, 30)
        # Layout only: router FP32 weights deliberately fail this BF16 interface.
        for shape in ((7, 13, 19), (127, 1153, 131), (129, 1, 1)):
            self.assertEqual(probe.dimensions(*shape)["out"]["shape"], shape[:2])

    def test_invalid_dimensions(self):
        for value in (0, -1, 1.0, True, 2**31):
            for axis in range(3):
                shape = [3, 5, 7]
                shape[axis] = value
                with self.assertRaises(ValueError):
                    probe.dimensions(*shape)

    def test_column_major_weight_addressing_is_lossless_for_bf16_bits(self):
        # Pure integer address model; no conversion, packing or native execution.
        n, k = 256, 256
        bits = list(range(65536))
        for output_column in range(n):
            for reduction_index in range(k):
                row_major_w = output_column * k + reduction_index
                column_major_b = reduction_index + output_column * k
                self.assertEqual(bits[row_major_w], bits[column_major_b])

    def test_small_odd_shape_matmul_address_model(self):
        gen = random.Random(56001)
        for m, n, k in ((1, 1, 1), (3, 5, 7), (8, 13, 9)):
            x = [gen.randrange(-16, 17) / 8 for _ in range(m * k)]
            w = [gen.randrange(-16, 17) / 8 for _ in range(n * k)]
            ref = [[sum(x[i * k + p] * w[j * k + p] for p in range(k))
                    for j in range(n)] for i in range(m)]
            modeled = [[sum(x[i * k + p] * w[p + j * k] for p in range(k))
                        for j in range(n)] for i in range(m)]
            self.assertEqual(modeled, ref)
            # These fractions are exactly representable in BF16. This is not a
            # native accumulation, rounding, CUDA correctness or tolerance test.
            for value in x + w:
                word = struct.unpack("<I", struct.pack("<f", value))[0]
                self.assertEqual(word & 65535, 0)

    def test_valid_contract_and_zero_workspace_plan_requires_dummy_storage(self):
        self.assertEqual(self.validate()["out"]["shape"], (3, 5))
        values = self.fixtures(workspace=1)
        self.validate(values, required=0)
        with self.assertRaises(ValueError):
            self.validate(self.fixtures(workspace=0), required=0)

    def test_operand_shape_stride_dtype_device_and_view_bits_fail_closed(self):
        values = self.fixtures()
        for index in range(3):
            for changes in ({"shape": (2, 3)}, {"strides": (1, 7)},
                            {"dtype": "float32"}, {"device": 1},
                            {"conjugate": True}, {"negative": True}):
                changed = list(values)
                changed[index] = replace(values[index], **changes)
                with self.assertRaises(ValueError):
                    self.validate(changed)

    def test_operand_alignment_storage_offsets_fail_closed(self):
        values = self.fixtures()
        for index in range(4):
            for pointer in (0, -256, values[index].pointer + 2):
                changed = list(values)
                changed[index] = replace(values[index], pointer=pointer)
                with self.assertRaises(ValueError):
                    self.validate(changed)

    def test_workspace_type_size_layout_and_device(self):
        values = self.fixtures()
        for changes in ({"shape": (4095,)}, {"shape": (probe.MAX_WORKSPACE_BYTES + 1,)},
                        {"shape": (64, 64)}, {"strides": (2,)}, {"dtype": "bfloat16"},
                        {"device": 1}, {"negative": True}, {"conjugate": True}):
            changed = list(values)
            changed[3] = replace(values[3], **changes)
            with self.assertRaises(ValueError):
                self.validate(changed)
        for required in (-1, True, probe.MAX_WORKSPACE_BYTES + 1):
            with self.assertRaises(ValueError):
                self.validate(required=required)

    def test_mutable_aliases_are_rejected(self):
        values = self.fixtures()
        for left, right in ((2, 0), (2, 1), (2, 3), (3, 0), (3, 1)):
            changed = list(values)
            changed[left] = replace(values[left], pointer=values[right].pointer)
            with self.assertRaisesRegex(ValueError, "storage overlap"):
                self.validate(changed)

    def test_exact_adjacency_is_not_overlap(self):
        # 128 BF16 values occupy 256 bytes; adjacent output is independently owned.
        values = (
            probe.TensorMetadata((1, 128), (128, 1), "bfloat16", 0, 0x10000),
            probe.TensorMetadata((1, 128), (128, 1), "bfloat16", 0, 0x20000),
            probe.TensorMetadata((1, 1), (1, 1), "bfloat16", 0, 0x10100),
            probe.TensorMetadata((1,), (1,), "uint8", 0, 0x30000),
        )
        probe.validate_contract(1, 1, 128, 0, 0, *values)

    def test_explicit_selection_never_falls_back_or_benchmarks(self):
        p1, p4 = FakePlan(1), FakePlan(4)
        inventory = {"plans": [p1, p4]}
        self.assertIs(probe.select_plan(inventory, 4), p4)
        for invalid in (0, 2, -1, None, True):
            with self.assertRaises(ValueError):
                probe.select_plan(inventory, invalid)
        with self.assertRaises(ValueError):
            probe.select_plan({"plans": [p1, FakePlan(1)]}, 1)

    def test_fixture_metadata_is_immutable(self):
        with self.assertRaises(FrozenInstanceError):
            self.fixtures()[0].pointer = 0

    def test_precision_and_algorithm_check_source_contract(self):
        source = probe.CUDA_SOURCE
        for required in ("CUBLAS_COMPUTE_32F, CUDA_R_32F", "CUDA_R_16BF",
                         "CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK",
                         "uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_NONE;",
                         "meta.split_k > 1 || meta.reduction != CUBLASLT_REDUCTION_SCHEME_NONE",
                         "written == sizeof(T)", "cublasLtMatmulAlgoCheck"):
            self.assertIn(required, source)
        for forbidden in ("CUBLAS_COMPUTE_32F_FAST", "CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE",
                          "allow_tf32", "allow_bf16_reduced_precision_reduction",
                          "cudaEventElapsedTime", "cudaDeviceSynchronize", "cudaStreamSynchronize"):
            self.assertNotIn(forbidden, source)

    def test_launch_current_stream_no_allocation_tuning_or_hidden_copy(self):
        run = probe.CUDA_SOURCE.split("  void run(", 1)[1].split("\n};", 1)[0]
        self.assertIn("c10::cuda::CUDAGuard guard(x.device())", run)
        self.assertIn("c10::cuda::getCurrentCUDAStream(x.get_device()).stream()", run)
        self.assertIn("size_t(workspace.numel()), stream)", run)
        for forbidden in ("AlgoGetHeuristic", "AlgoCheck", "malloc", "at::empty",
                          ".contiguous(", "cudaMemcpy", "synchronize", "Create(", "Destroy("):
            self.assertNotIn(forbidden, run)

    def test_full_algorithm_and_immutable_plan_not_id_reconstruction(self):
        source = probe.CUDA_SOURCE
        self.assertIn("const cublasLtMatmulAlgo_t algorithm;", source)
        self.assertIn("algorithm(algo)", source)
        self.assertIn("const std::shared_ptr<Context> context;", source)
        self.assertNotIn("cublasLtMatmulAlgoInit", source)
        self.assertNotIn("cublasLtMatmulAlgoConfigSetAttribute", source)
        self.assertIn("not_capturing(device);", source)
        self.assertIn("Retain the selected plan through all graph replays", probe.__doc__)


if __name__ == "__main__":
    unittest.main()
