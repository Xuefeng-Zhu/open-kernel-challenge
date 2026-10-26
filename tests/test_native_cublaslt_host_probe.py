"""P057 mock/source tests. Never import actual Torch or compile native code."""
import ast
import builtins
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import native_cublaslt_host_probe as host
import native_cublaslt_probe as frozen


class HostProbeTests(unittest.TestCase):
    def test_exact_frozen_native_body_and_original_file(self):
        self.assertIs(host.CUDA_SOURCE, frozen.CUDA_SOURCE)
        self.assertEqual(host.source_sha256(), host.FROZEN_BINDING_SHA256)
        self.assertEqual(hashlib.sha256(Path(frozen.__file__).read_bytes()).hexdigest(),
                         "a40916423e9eb7da4b93258e7e986b55777a56ca93268e9c2e3105f437b409cc")
        for device_token in ("__global__", "__device__", "<<<", "threadIdx", "blockIdx"):
            self.assertNotIn(device_token, host.CUDA_SOURCE)

    def test_every_contract_helper_is_the_original_object(self):
        for name in ("TensorMetadata", "dimensions", "select_plan", "source_sha256", "validate_contract"):
            self.assertIs(getattr(host, name), getattr(frozen, name))
        self.assertEqual(host.MAX_WORKSPACE_BYTES, frozen.MAX_WORKSPACE_BYTES)
        self.assertEqual(host.MAX_HEURISTICS, frozen.MAX_HEURISTICS)

    def test_import_is_inert_and_has_no_top_level_build(self):
        source = Path(host.__file__).read_text()
        tree = ast.parse(source)
        imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertEqual(len(imports), 1)
        self.assertEqual(imports[0].module, "native_cublaslt_probe")
        self.assertFalse(any(isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                             for node in tree.body))
        original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name.split(".")[0] == "torch":
                raise AssertionError("Actual Torch import forbidden by CPU test")
            return original_import(name, *args, **kwargs)
        spec = importlib.util.spec_from_file_location("p057_inert_fixture", host.__file__)
        module = importlib.util.module_from_spec(spec)
        with patch("builtins.__import__", side_effect=guarded_import):
            spec.loader.exec_module(module)
        self.assertEqual(module.FROZEN_BINDING_SHA256, host.FROZEN_BINDING_SHA256)

    def test_explicit_gate_precedes_any_torch_import(self):
        with patch("builtins.__import__", side_effect=AssertionError("No imports allowed")):
            for flag in (False, None, 0, 1, "yes"):
                with self.assertRaises(RuntimeError):
                    host.build_native_probe(compile_requested=flag)

    def test_binding_mismatch_precedes_any_torch_import(self):
        with patch.object(host, "source_sha256", return_value="0" * 64), \
             patch("builtins.__import__", side_effect=AssertionError("No imports allowed")):
            with self.assertRaisesRegex(ValueError, "source mismatch"):
                host.build_native_probe(compile_requested=True)

    def test_mocked_loader_gets_only_host_translation_unit_and_cuda_linkage(self):
        loader = Mock(return_value=object())
        torch = ModuleType("torch")
        utils = ModuleType("torch.utils")
        cpp_extension = ModuleType("torch.utils.cpp_extension")
        cpp_extension.load_inline = loader
        torch.utils, utils.cpp_extension = utils, cpp_extension
        modules = {"torch": torch, "torch.utils": utils, "torch.utils.cpp_extension": cpp_extension}
        with patch.dict(sys.modules, modules):
            result = host.build_native_probe(compile_requested=True)
        self.assertIs(result, loader.return_value)
        loader.assert_called_once_with(
            name="okc_lt_host_probe_" + host.FROZEN_BINDING_SHA256[:16],
            cpp_sources=frozen.CUDA_SOURCE,
            cuda_sources=None,
            functions=None,
            with_cuda=True,
            extra_cflags=["-O3"],
            extra_ldflags=["-lcublasLt"],
            verbose=False,
        )

    def test_no_compiler_override_precision_change_or_environment_mutation(self):
        source = Path(host.__file__).read_text()
        for forbidden in ("extra_cuda_cflags", "-std=", "--use_fast_math", "os.environ",
                          "subprocess", "allow_tf32", "allow_bf16_reduced_precision_reduction"):
            self.assertNotIn(forbidden, source)
        self.assertIn('name="okc_lt_host_probe_"', source)
        self.assertNotIn('name="okc_lt_probe_"', source)


if __name__ == "__main__":
    unittest.main()
