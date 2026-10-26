"""P058 source/address/mock tests only, with no real Torch import or native build."""
import ast
import builtins
import difflib
import hashlib
import importlib.util
from pathlib import Path
import random
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import native_cublaslt_column_probe as column
import native_cublaslt_probe as frozen
import native_cublaslt_host_probe as host


class ColumnDescriptorTests(unittest.TestCase):
    def test_original_files_and_native_body_are_unchanged(self):
        self.assertIs(column.FROZEN_SOURCE, frozen.CUDA_SOURCE)
        self.assertEqual(hashlib.sha256(frozen.CUDA_SOURCE.encode()).hexdigest(), column.FROZEN_BINDING_SHA256)
        self.assertEqual(hashlib.sha256(Path(frozen.__file__).read_bytes()).hexdigest(),
                         "a40916423e9eb7da4b93258e7e986b55777a56ca93268e9c2e3105f437b409cc")
        self.assertEqual(hashlib.sha256(Path(host.__file__).read_bytes()).hexdigest(),
                         "a91663d9b64291657d7b64e7bba7d7c4ffaeb2777964f8357dd2225adea9a300")

    def test_exact_descriptor_only_native_diff_and_no_other_body_change(self):
        expected = frozen.CUDA_SOURCE
        self.assertEqual(len(column.DESCRIPTOR_REPLACEMENTS), 9)
        for before, after in column.DESCRIPTOR_REPLACEMENTS:
            self.assertEqual(expected.count(before), 1)
            expected = expected.replace(before, after, 1)
        self.assertEqual(column.CUDA_SOURCE, expected)
        changed = [line for line in difflib.ndiff(frozen.CUDA_SOURCE.splitlines(), column.CUDA_SOURCE.splitlines())
                   if line.startswith(("+ ", "- "))]
        # One added declaration and eight replaced native lines.
        self.assertEqual(len(changed), 17)
        self.assertTrue(all(any(marker in line for marker in (
            "transpose_a", '"TRANSA"', "MatrixLayoutCreate", "cublasLtOrder_t",
            "MatrixLayoutSetAttribute", "x.data_ptr()", "w.data_ptr()")) for line in changed))
        for segment in ("struct AlgorithmMetadata", "static void check_bf16", "static py::dict enumerate_plans"):
            original = frozen.CUDA_SOURCE.split(segment, 1)[1]
            actual = column.CUDA_SOURCE.split(segment, 1)[1]
            if segment == "static py::dict enumerate_plans":
                self.assertEqual(actual, original)
        self.assertIn("CUBLAS_COMPUTE_32F, CUDA_R_32F", column.CUDA_SOURCE)
        self.assertIn("CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK", column.CUDA_SOURCE)

    def test_hash_and_count_guard_reject_ambiguous_or_changed_source(self):
        with self.assertRaises(ValueError):
            column.derive_source(frozen.CUDA_SOURCE + "\n")
        with patch.object(column, "DESCRIPTOR_REPLACEMENTS", (("absent", "replacement"),)):
            with self.assertRaisesRegex(ValueError, "exactly one"):
                column.derive_source(frozen.CUDA_SOURCE)
        with patch.object(column, "DESCRIPTOR_REPLACEMENTS", (("TORCH_CHECK", "x"),)):
            with self.assertRaisesRegex(ValueError, "exactly one"):
                column.derive_source(frozen.CUDA_SOURCE)

    def test_original_validation_metadata_and_selection_helpers_are_shared(self):
        for name in ("TensorMetadata", "dimensions", "validate_contract", "select_plan"):
            self.assertIs(getattr(column, name), getattr(frozen, name))
        self.assertEqual(column.MAX_WORKSPACE_BYTES, frozen.MAX_WORKSPACE_BYTES)
        self.assertEqual(column.MAX_HEURISTICS, frozen.MAX_HEURISTICS)

    def test_descriptor_shapes_transposes_orders_and_leading_dimensions(self):
        for m, n, k in ((1, 1, 1), (7, 13, 19), (128, 2304, 8192),
                        (16, 8192, 4096), (127, 1153, 131)):
            d = column.descriptor_contract(m, n, k)
            self.assertEqual(d["a"], dict(buffer="w", shape=(k, n), order="column", ld=k, operation="T"))
            self.assertEqual(d["b"], dict(buffer="x", shape=(k, m), order="column", ld=k, operation="N"))
            self.assertEqual(d["out"], dict(buffer="out", shape=(n, m), order="column", ld=n))
            self.assertEqual(d["logical_output"], (m, n))

    def test_exact_integer_gemm_equation_rectangular_and_tails(self):
        gen = random.Random(58001)
        for m, n, k in ((1, 1, 1), (3, 5, 7), (7, 13, 19), (8, 11, 3)):
            x = [gen.randrange(-100, 101) for _ in range(m * k)]
            w = [gen.randrange(-100, 101) for _ in range(n * k)]
            expected = [sum(x[row * k + p] * w[col * k + p] for p in range(k))
                        for row in range(m) for col in range(n)]
            actual = [None] * (m * n)
            for i in range(n):
                for j in range(m):
                    # opT(A)[i,p]=A[p,i]=W[i,p], B[p,j]=X[j,p].
                    actual[i + j * n] = sum(w[p + i * k] * x[p + j * k] for p in range(k))
            self.assertEqual(actual, expected)

    def test_all_bf16_bit_patterns_preserve_input_and_output_byte_addresses(self):
        # Bit-address proof includes signed zero, subnormals, infinities and NaNs;
        # it makes no floating-point arithmetic or native-correctness claim.
        words = list(range(65536))
        m, n, k = 256, 256, 256
        for i in range(n):
            for p in range(k):
                self.assertEqual(words[i * k + p], words[p + i * k])
        for j in range(m):
            for p in range(k):
                self.assertEqual(words[j * k + p], words[p + j * k])
        physical = set()
        for j in range(m):
            for i in range(n):
                self.assertEqual(2 * (j * n + i), 2 * (i + j * n))
                physical.add(i + j * n)
        self.assertEqual(physical, set(range(m * n)))

    def test_official_and_nearby_byte_address_samples(self):
        gen = random.Random(58002)
        shapes = [(m, n, k) for m in (1, 8, 16, 32, 128, 256)
                  for n, k in ((2304, 8192), (8192, 1024), (4096, 8192), (8192, 4096))]
        shapes += [(7, 13, 19), (127, 2304, 8192), (129, 2304, 8192)]
        for m, n, k in shapes:
            d = column.descriptor_contract(m, n, k)
            for row, col, p in [(0, 0, 0), (m - 1, n - 1, k - 1)] + [
                    (gen.randrange(m), gen.randrange(n), gen.randrange(k)) for _ in range(10)]:
                self.assertEqual(2 * (col * k + p), 2 * (p + col * d["a"]["ld"]))
                self.assertEqual(2 * (row * k + p), 2 * (p + row * d["b"]["ld"]))
                self.assertEqual(2 * (row * n + col), 2 * (col + row * d["out"]["ld"]))

    def test_inert_import_and_explicit_build_gate(self):
        original_import = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split(".")[0] == "torch":
                raise AssertionError("No Torch import")
            return original_import(name, *args, **kwargs)
        spec = importlib.util.spec_from_file_location("p058_inert_test", column.__file__)
        module = importlib.util.module_from_spec(spec)
        with patch("builtins.__import__", side_effect=guarded):
            spec.loader.exec_module(module)
            for value in (False, None, 0, 1):
                with self.assertRaises(RuntimeError):
                    module.build_native_probe(compile_requested=value)
        self.assertEqual(module.source_sha256(), column.source_sha256())

    def test_mocked_host_builder_preserves_supported_route_and_linkage(self):
        loader = Mock(return_value=object())
        torch, utils, cpp = ModuleType("torch"), ModuleType("torch.utils"), ModuleType("torch.utils.cpp_extension")
        cpp.load_inline = loader
        torch.utils, utils.cpp_extension = utils, cpp
        with patch.dict(sys.modules, {"torch": torch, "torch.utils": utils, "torch.utils.cpp_extension": cpp}):
            self.assertIs(column.build_native_probe(compile_requested=True), loader.return_value)
        loader.assert_called_once_with(name="okc_lt_column_probe_" + column.source_sha256()[:16],
            cpp_sources=column.CUDA_SOURCE, cuda_sources=None, functions=None, with_cuda=True,
            extra_cflags=["-O3"], extra_ldflags=["-lcublasLt"], verbose=False)

    def test_no_transpose_copy_precision_or_stream_mutation_added(self):
        for token in ("cudaMemcpy", "at::empty", "contiguous()", "transpose(", "setCurrentCUDAStream"):
            self.assertEqual(column.CUDA_SOURCE.count(token), frozen.CUDA_SOURCE.count(token))
        self.assertEqual(column.CUDA_SOURCE.count("CUBLASLT_REDUCTION_SCHEME_NONE"),
                         frozen.CUDA_SOURCE.count("CUBLASLT_REDUCTION_SCHEME_NONE"))
        self.assertIn("c10::cuda::getCurrentCUDAStream(x.get_device()).stream()", column.CUDA_SOURCE)
        with patch.object(column, "CUDA_SOURCE", column.CUDA_SOURCE + "\n"):
            with self.assertRaisesRegex(ValueError, "source mismatch"):
                column.build_native_probe(compile_requested=True)


if __name__ == "__main__":
    unittest.main()
