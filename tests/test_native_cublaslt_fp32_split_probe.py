"""P059 CPU source, metadata and FP32 partial models; no native/GPU validation."""
import builtins
import hashlib
import importlib.util
import math
from pathlib import Path
import random
import struct
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import native_cublaslt_fp32_split_probe as split
import native_cublaslt_probe as frozen


# Deliberately synthetic encodings: the selector must use the inventory map,
# not assumed CUDA enum integers from a particular header/runtime.
ENUMS = {split.NONE: 71, split.COMPUTE_TYPE: 503, split.INPLACE: 101, split.OUTPUT_TYPE: 997}


class Plan:
    def __init__(self, position=0, parts=1, name=split.NONE, **changed):
        self.values = dict(heuristic_position=position, split_k=parts,
            reduction_scheme_name=name, reduction_scheme=ENUMS[name],
            compute="CUBLAS_COMPUTE_32F", scale="CUDA_R_32F", input_output="CUDA_R_16BF")
        self.values.update(changed)

    def metadata(self):
        return dict(self.values)


def inventory(*plans):
    return dict(plans=list(plans), reduction_scheme_constants=dict(ENUMS))


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bf16(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    rounded = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
    return struct.unpack("<f", struct.pack("<I", rounded))[0]


def partial_model(x, w, parts, round_partial=f32):
    width = (len(x) + parts - 1) // parts
    partials = []
    visited = []
    for part in range(parts):
        acc = f32(0)
        for index in range(part * width, min((part + 1) * width, len(x))):
            visited.append(index)
            acc = f32(acc + f32(x[index] * w[index]))
        partials.append(round_partial(acc))
    result = f32(0)
    for value in partials:
        result = f32(result + value)
    return result, visited


class FP32SplitProbeTests(unittest.TestCase):
    def test_frozen_parent_and_exact_replacement_isolation(self):
        self.assertEqual(hashlib.sha256(Path(frozen.__file__).read_bytes()).hexdigest(),
                         "a40916423e9eb7da4b93258e7e986b55777a56ca93268e9c2e3105f437b409cc")
        self.assertEqual(hashlib.sha256(split.FROZEN_SOURCE.encode()).hexdigest(), split.FROZEN_BINDING_SHA256)
        source = frozen.CUDA_SOURCE
        self.assertEqual(len(split.REPLACEMENTS), 7)
        for before, after in split.REPLACEMENTS:
            self.assertEqual(source.count(before), 1)
            source = source.replace(before, after, 1)
        self.assertEqual(split.CUDA_SOURCE, source)
        original_run = frozen.CUDA_SOURCE.split("  void run(", 1)[1].split("\n};", 1)[0]
        changed_run = split.CUDA_SOURCE.split("  void run(", 1)[1].split("\n};", 1)[0]
        self.assertEqual(changed_run, original_run)
        self.assertEqual(split.CUDA_SOURCE.split("struct AlgorithmMetadata", 1)[1].split("static const char* reduction_name", 1)[0],
                         frozen.CUDA_SOURCE.split("struct AlgorithmMetadata", 1)[1].split("struct Plan", 1)[0])

    def test_hash_and_count_guard(self):
        with self.assertRaises(ValueError):
            split.derive_source(frozen.CUDA_SOURCE + "\n")
        with patch.object(split, "REPLACEMENTS", (("absent", "x"),)):
            with self.assertRaisesRegex(ValueError, "exactly one"):
                split.derive_source(frozen.CUDA_SOURCE)

    def test_only_new_binding_is_module_local_and_native_body_is_unchanged(self):
        global_binding = '  py::class_<Plan, std::shared_ptr<Plan>>(module, "Plan")'
        local_binding = '  py::class_<Plan, std::shared_ptr<Plan>>(module, "Plan", py::module_local())'
        self.assertEqual(frozen.CUDA_SOURCE.count(global_binding), 1)
        self.assertNotIn("py::module_local", frozen.CUDA_SOURCE)
        self.assertEqual(split.CUDA_SOURCE.count(local_binding), 1)
        self.assertEqual(split.CUDA_SOURCE.count("py::module_local"), 1)
        policy_only = frozen.CUDA_SOURCE
        for before, after in split.REPLACEMENTS[:-1]:
            policy_only = policy_only.replace(before, after, 1)
        self.assertEqual(split.CUDA_SOURCE.replace(local_binding, global_binding, 1), policy_only)
        self.assertEqual(split.CUDA_SOURCE.split("PYBIND11_MODULE", 1)[0],
                         policy_only.split("PYBIND11_MODULE", 1)[0])

    def test_native_precision_mask_filter_and_names_are_header_based(self):
        source = split.CUDA_SOURCE
        self.assertIn("CUBLASLT_REDUCTION_SCHEME_NONE | CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE", source)
        self.assertIn("meta.split_k <= 1 && meta.reduction == CUBLASLT_REDUCTION_SCHEME_NONE", source)
        self.assertIn("meta.split_k > 1 && meta.reduction == CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE", source)
        self.assertIn("CUBLAS_COMPUTE_32F, CUDA_R_32F", source)
        for name in ENUMS:
            self.assertIn(f'schemes["{name}"] = uint32_t({name});', source)
            self.assertIn(f'case {name}: return "{name}";', source)
        self.assertNotIn("cublasLtMatmulAlgoConfigSetAttribute", source)

    def test_real_split_selected_without_no_split_fallback(self):
        no_split, later, first = Plan(), Plan(9, 8, split.COMPUTE_TYPE), Plan(3, 2, split.COMPUTE_TYPE)
        records = inventory(no_split, later, first)
        self.assertIs(split.select_actual_fp32_split(records), first)
        self.assertIs(split.select_actual_fp32_split(records, 9), later)
        with self.assertRaises(split.NoEligibleFP32Split):
            split.select_actual_fp32_split(records, 0)
        for records in (inventory(), inventory(Plan()), inventory(Plan(parts=0))):
            with self.assertRaises(split.NoEligibleFP32Split):
                split.select_actual_fp32_split(records)

    def test_output_type_inplace_and_inconsistent_pair_rejected(self):
        for parts, name in ((8, split.OUTPUT_TYPE), (8, split.INPLACE),
                            (8, split.NONE), (1, split.COMPUTE_TYPE)):
            with self.assertRaisesRegex(ValueError, "disallowed"):
                split.select_actual_fp32_split(inventory(Plan(0, parts, name)))

    def test_compiled_enum_map_mandatory_and_not_guessed(self):
        p = Plan(2, 4, split.COMPUTE_TYPE)
        self.assertIs(split.select_actual_fp32_split(inventory(p)), p)
        for constants in (None, {}, {**ENUMS, "extra": 1}, {**ENUMS, split.NONE: True},
                          {**ENUMS, split.NONE: ENUMS[split.COMPUTE_TYPE]}):
            with self.assertRaises(ValueError):
                split.select_actual_fp32_split(dict(plans=[p], reduction_scheme_constants=constants))
        with self.assertRaises(ValueError):
            split.select_actual_fp32_split(inventory(Plan(2, 4, split.COMPUTE_TYPE, reduction_scheme=0)))

    def test_precision_and_metadata_inconsistency_fail_closed(self):
        for changed in ({"compute": "CUBLAS_COMPUTE_16F"}, {"scale": "CUDA_R_16BF"},
                        {"input_output": "CUDA_R_32F"}, {"split_k": True},
                        {"reduction_scheme_name": "UNKNOWN"}, {"heuristic_position": -1}):
            with self.assertRaises(ValueError):
                split.select_actual_fp32_split(inventory(Plan(2, 4, split.COMPUTE_TYPE, **changed)))
        with self.assertRaises(ValueError):
            split.select_actual_fp32_split(inventory(Plan(), Plan()))
        for position in (True, -1, "3"):
            with self.assertRaises(ValueError):
                split.select_actual_fp32_split(inventory(Plan(2, 4, split.COMPUTE_TYPE)), position)

    def test_fp32_partial_model_tails_and_empty_partitions(self):
        gen = random.Random(59001)
        for k in (1, 2, 7, 33, 131):
            x = [gen.randrange(-32, 33) / 8 for _ in range(k)]
            w = [gen.randrange(-32, 33) / 16 for _ in range(k)]
            expected = math.fsum(a * b for a, b in zip(x, w))
            for parts in (2, 3, 8, 16):
                result, visited = partial_model(x, w, parts)
                self.assertEqual(visited, list(range(k)))
                self.assertEqual(result, expected)

    def test_fp32_partial_cancellation_witness_exposes_bf16_partial_loss(self):
        x, w = [1.0, 1.0, -1.0], [256.0, 1.0, 256.0]
        precise, _ = partial_model(x, w, 2)
        reduced, _ = partial_model(x, w, 2, bf16)
        self.assertEqual(precise, 1.0)
        self.assertEqual(reduced, 0.0)
        # A model illustrates policy necessity; it is not cuBLAS implementation
        # emulation, an oracle test of native code or an accuracy guarantee.

    def test_original_contract_helpers_layout_and_stream_remain_unchanged(self):
        for name in ("TensorMetadata", "dimensions", "select_plan", "validate_contract"):
            self.assertIs(getattr(split, name), getattr(frozen, name))
        for token in ("MatrixLayoutCreate", "MatrixLayoutSetAttribute", "getCurrentCUDAStream",
                      "const cublasLtMatmulAlgo_t algorithm", "check_bf16", "overlap("):
            self.assertEqual(split.CUDA_SOURCE.count(token), frozen.CUDA_SOURCE.count(token))

    def test_inert_import_and_explicit_build_gate(self):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split(".")[0] == "torch":
                raise AssertionError("Actual Torch import forbidden")
            return original(name, *args, **kwargs)
        spec = importlib.util.spec_from_file_location("p059_inert_test", split.__file__)
        module = importlib.util.module_from_spec(spec)
        with patch("builtins.__import__", side_effect=guarded):
            spec.loader.exec_module(module)
            for flag in (False, None, 0, 1):
                with self.assertRaises(RuntimeError):
                    module.build_native_probe(compile_requested=flag)
        self.assertEqual(module.source_sha256(), split.source_sha256())

    def test_mocked_host_builder_and_mutated_source_rejection(self):
        loader = Mock(return_value=object())
        torch, utils, cpp = ModuleType("torch"), ModuleType("torch.utils"), ModuleType("torch.utils.cpp_extension")
        cpp.load_inline = loader
        torch.utils, utils.cpp_extension = utils, cpp
        with patch.dict(sys.modules, {"torch": torch, "torch.utils": utils, "torch.utils.cpp_extension": cpp}):
            self.assertIs(split.build_native_probe(compile_requested=True), loader.return_value)
        loader.assert_called_once_with(name="okc_lt_fp32_split_probe_" + split.source_sha256()[:16],
            cpp_sources=split.CUDA_SOURCE, cuda_sources=None, functions=None, with_cuda=True,
            extra_cflags=["-O3"], extra_ldflags=["-lcublasLt"], verbose=False)
        with patch.object(split, "CUDA_SOURCE", split.CUDA_SOURCE + "\n"):
            with self.assertRaises(ValueError):
                split.build_native_probe(compile_requested=True)


if __name__ == "__main__":
    unittest.main()
