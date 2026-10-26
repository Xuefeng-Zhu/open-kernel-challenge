"""P057 identity/contract mock tests; no native build, Torch import or GPU."""
import ast
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import projection_p056_native_screen as original
import projection_p057_native_screen as repaired


class HostWorkerTests(unittest.TestCase):
    def test_all_source_bindings_include_original_worker_and_both_builders(self):
        hashes = repaired.verify_sources()
        self.assertEqual(len(hashes), 11)
        self.assertEqual(hashes["projection_p056_native_screen.py"], repaired.FROZEN_WORKER_SHA)
        self.assertEqual(hashes["native_cublaslt_probe.py"], original.NATIVE_SHA)
        self.assertEqual(hashes["native_cublaslt_host_probe.py"], repaired.BRIDGE_SHA)
        self.assertEqual(hashes["projection_p057_native_screen.py"],
                         hashlib.sha256(Path(repaired.__file__).read_bytes()).hexdigest())

    def test_entire_worker_ast_unchanged_except_explicit_identity_and_builder(self):
        source = Path(repaired.__file__).read_text()
        source = source.replace('P057 host-built native cuBLASLt diagnostic', 'P056 bounded native cuBLASLt diagnostic', 1)
        source = source.replace('P057 host-built cuBLASLt interface and two-case screen', 'P056 native cuBLASLt interface and two-case screen', 1)
        source = source.replace('"p057_native_source"', '"p056_native_source"')
        source = source.replace('"p057_reference"', '"p056_reference"').replace('"p057_parent"', '"p056_parent"')
        source = source.replace('native_source = load_module(ROOT / "native_cublaslt_host_probe.py",',
                                'native_source = load_module(ROOT / "native_cublaslt_probe.py",', 1)
        source = source.replace('"projection_p057_native_screen.py"', '"projection_p056_native_screen.py"')
        new_tree, old_tree = ast.parse(source), ast.parse(Path(original.__file__).read_text())
        new_tree.body = [node for node in new_tree.body if not (
            isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {
                "BRIDGE_SHA", "FROZEN_WORKER_SHA"} for t in node.targets))]
        verify = next(node for node in new_tree.body if isinstance(node, ast.FunctionDef) and node.name == "verify_sources")
        expected_update = verify.body[1].value.args[0]
        keep = [(key, value) for key, value in zip(expected_update.keys, expected_update.values)
                if not (isinstance(key, ast.Constant) and key.value in {
                    "native_cublaslt_host_probe.py", "projection_p056_native_screen.py"})]
        expected_update.keys, expected_update.values = map(list, zip(*keep))
        self.assertEqual(ast.dump(new_tree, include_attributes=False), ast.dump(old_tree, include_attributes=False))

    def test_cases_precision_protocol_and_deadlines_are_exact_parent_values(self):
        for name in ("CASES", "NEARBY", "WORKSPACE_LIMIT", "MAX_SELECTED", "CONFIG_KEYS", "NATIVE_SHA", "PARENT_SHA", "SAFE_SHA"):
            self.assertEqual(getattr(repaired, name), getattr(original, name))
        args = SimpleNamespace(compile_native=True, deadline_seconds=240, outer_timeout_seconds=270)
        repaired.validate_options(args)
        args.deadline_seconds = 241
        with self.assertRaises(ValueError):
            repaired.validate_options(args)

    def test_import_is_inert_and_no_original_worker_monkeypatch(self):
        tree = ast.parse(Path(repaired.__file__).read_text())
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [entry.name for entry in node.names] if isinstance(node, ast.Import) else [node.module]
                self.assertFalse(any(name.startswith(("torch", "native_cublaslt", "projection_p056_native_screen")) for name in names))
        self.assertNotIn("setattr(", Path(repaired.__file__).read_text())
        self.assertNotIn("exec(", Path(repaired.__file__).read_text())

    def test_source_failure_prevents_build_and_preserves_new_identity(self):
        with TemporaryDirectory() as folder:
            output = Path(folder) / "report.json"
            with patch.object(repaired, "verify_sources", side_effect=ValueError("bridge hash mismatch")), \
                 patch.object(repaired, "execute") as execute, patch("builtins.print"):
                code = repaired.main(["--compile-native", "--output", str(output), "--outer-timeout-seconds", "270"])
            execute.assert_not_called()
            record = json.loads(output.read_text())
            self.assertEqual(code, 1)
            self.assertIn("P057", record["kind"])
            self.assertFalse(record["native_build_complete"])
            self.assertFalse(record["native_execution"])

    def test_failure_after_partial_compile_keeps_exact_failure_stage(self):
        with TemporaryDirectory() as folder:
            output = Path(folder) / "report.json"
            def fail(args, report, save):
                report["phase"] = "native_build"
                report["compile"] = {"complete": False}
                save()
                raise RuntimeError("host compiler failure")
            with patch.object(repaired, "execute", side_effect=fail), patch("builtins.print"):
                code = repaired.main(["--compile-native", "--output", str(output), "--outer-timeout-seconds", "270"])
            record = json.loads(output.read_text())
            self.assertEqual(code, 1)
            self.assertEqual(record["phase"], "native_build")
            self.assertFalse(record["complete"])
            self.assertEqual(record["oracle_checks"], 0)
            self.assertFalse(output.with_name("report.json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
