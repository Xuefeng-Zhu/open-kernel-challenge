"""Fixture-only tests; installed FlashInfer/Torch/CUDA packages are never imported."""
import ast
import hashlib
import importlib.metadata as metadata
import json
import sys
import tempfile
import unittest
from pathlib import Path

from flashinfer_capability_inventory import DISTRIBUTIONS, MAX_FILES, Reader, inventory


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.distributions = {}

    def fixture(self, name="flashinfer-python", sources=None, extra_record=(), version="0.9.9", headers="", entry_points=None):
        folder = self.root / (name.replace("-", "_") + "-0.9.9.dist-info")
        folder.mkdir()
        (folder / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n{headers}\n")
        paths = [folder.name + "/METADATA", folder.name + "/RECORD"]
        if entry_points is not None:
            (folder / "entry_points.txt").write_text(entry_points)
            paths.append(folder.name + "/entry_points.txt")
        for relative, content in (sources or {}).items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            paths.append(relative)
        (folder / "RECORD").write_text("\n".join(path + ",," for path in [*paths, *extra_record]))
        self.distributions[name] = metadata.PathDistribution(folder)
        return folder

    def lookup(self, name):
        if name not in self.distributions:
            raise metadata.PackageNotFoundError(name)
        return self.distributions[name]

    def run_inventory(self, **kwargs):
        return inventory(self.lookup, **kwargs)

    def test_missing_distributions_explicit(self):
        out = self.run_inventory()
        self.assertEqual([r["distribution"] for r in out["distributions"]], list(DISTRIBUTIONS))
        self.assertTrue(all(r["status"] == "missing_distribution" for r in out["distributions"]))
        self.assertEqual(out["reads"]["bytes_read"], 0)

    def test_versions_hashes_and_literal_markers_without_imports(self):
        content = "def get_mm_bf16_cublaslt_module():\n    return build_and_load()\n# cublaslt_bf16_gemm_runner get_valid_tactics\n"
        self.fixture(sources={"flashinfer/gemm/gemm_base.py": content,
                              "flashinfer/jit/gemm/core.py": "# AOT or jit_cache\n",
                              "flashinfer/__init__.py": "raise RuntimeError('MUST NOT EXECUTE')\n",
                              "flashinfer/aot_manifest.json": '{"example": true}'})
        before = set(sys.modules)
        out = self.run_inventory()
        self.assertFalse(any(x == "torch" or x.startswith("flashinfer") for x in set(sys.modules) - before))
        package = out["distributions"][0]
        self.assertEqual(package["version"], "0.9.9")
        record = next(r for r in package["files"] if r["path"] == "flashinfer/gemm/gemm_base.py")
        self.assertEqual(record["sha256"], hashlib.sha256(content.encode()).hexdigest())
        self.assertEqual(record["size"], len(content))
        self.assertEqual(record["literal_markers"]["get_valid_tactics"], [3])
        self.assertNotIn("aot_manifest.json", package["missing_manifest_names"])
        self.assertFalse(out["api_callability_verified"])
        self.assertFalse(out["jit_free_verified"])
        self.assertNotIn(str(self.root), json.dumps(out))

    def test_unsafe_record_paths_and_symlink_not_read(self):
        self.fixture(extra_record=("../gemm_base.py", "/tmp/flashinfer/gemm_base.py", "flashinfer/gemm_base.py"))
        (self.root / "secret-fixture.txt").write_text("PRIVATE_FIXTURE_DO_NOT_READ")
        (self.root / "flashinfer").mkdir()
        (self.root / "flashinfer/gemm_base.py").symlink_to(self.root / "secret-fixture.txt")
        out = self.run_inventory()
        package = out["distributions"][0]
        self.assertEqual(package["unsafe_record_rows"], 2)
        self.assertTrue(any(r["status"] == "symlink_rejected" for r in package["files"]))
        self.assertNotIn("PRIVATE_FIXTURE", json.dumps(out))

    def test_file_byte_limit_prevents_source_read(self):
        self.fixture(sources={"flashinfer/gemm_base.py": "x" * 2000})
        out = self.run_inventory(reader=Reader(max_file_bytes=1000))
        record = next(r for r in out["distributions"][0]["files"] if r["path"].endswith("gemm_base.py"))
        self.assertEqual(record["status"], "file_byte_limit")
        self.assertIsNone(record["sha256"])
        self.assertLess(out["reads"]["bytes_read"], 1000)

    def test_total_byte_and_file_count_limits(self):
        self.fixture(sources={"flashinfer/gemm_base.py": "# get_valid_tactics\n"})
        out = self.run_inventory(reader=Reader(max_total_bytes=10))
        self.assertEqual(out["reads"]["bytes_read"], 0)
        self.assertEqual(out["distributions"][0]["status"], "metadata_unavailable")
        out = self.run_inventory(reader=Reader(max_files=2))
        self.assertEqual(out["reads"]["files_attempted"], 2)
        self.assertTrue(out["distributions"][0]["file_count_limit_reached"])

    def test_record_and_metadata_read_under_same_bounds(self):
        folder = self.fixture()
        (folder / "RECORD").write_text("x" * 2000)
        package = self.run_inventory(reader=Reader(max_file_bytes=1000))["distributions"][0]
        self.assertEqual(package["version"], "0.9.9")
        self.assertEqual(package["inventory_status"], "record_unavailable")
        (folder / "METADATA").write_text("x" * 2000)
        self.assertEqual(self.run_inventory(reader=Reader(max_file_bytes=1000))["distributions"][0]["status"], "metadata_unavailable")

    def test_record_row_limit_bounds_file_enumeration(self):
        self.fixture(extra_record=("flashinfer/other.py",) * 10000 + ("flashinfer/gemm_base.py",))
        package = self.run_inventory()["distributions"][0]
        self.assertEqual(package["record_rows_inspected"], 10000)
        self.assertTrue(package["record_row_limit_reached"])
        self.assertEqual(len(package["files"]), 2)

    def test_identity_mismatch_and_missing_files_explicit(self):
        folder = self.fixture(extra_record=("flashinfer/jit/core.py",))
        package = self.run_inventory()["distributions"][0]
        self.assertTrue(any(r["status"] == "missing_or_not_regular" for r in package["files"]))
        self.assertIn("jit/core.py", package["missing_source_suffixes"])
        (folder / "METADATA").write_text("Name: different-package\nVersion: 1\n\n")
        self.assertEqual(self.run_inventory()["distributions"][0]["status"], "metadata_identity_mismatch")

    def test_unlisted_paths_and_binaries_are_not_read(self):
        self.fixture(sources={"flashinfer/extension.so": "NEVER_LOAD_BINARY", "torch/gemm_base.py": "NEVER_READ_OTHER_PACKAGE",
                              "flashinfer/other.py": "NOT_ALLOWLISTED"})
        paths = [r["path"] for r in self.run_inventory()["distributions"][0]["files"]]
        self.assertEqual(len(paths), 2)

    def test_context_output_is_bounded(self):
        self.fixture(sources={"flashinfer/gemm_base.py": ("get_valid_tactics " + "x" * 500 + "\n") * 100})
        file = next(r for r in self.run_inventory()["distributions"][0]["files"] if r["path"].endswith("gemm_base.py"))
        self.assertEqual(len(file["literal_context"]), 8)
        self.assertTrue(all(len(r["text"]) <= 240 for r in file["literal_context"]))

    def test_named_prebuilt_artifacts_hash_without_decoding_and_precision_markers(self):
        markers = ("CUBLAS_COMPUTE_32F", "CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK",
                   "CUBLASLT_ALGO_CONFIG_SPLITK_NUM")
        self.fixture(sources={"flashinfer/gemm_base.py": "\n".join(markers),
                              "flashinfer/lib/mm_bf16_cublaslt.so": "placeholder",
                              "flashinfer_cubin/mm_bf16_cublaslt_sm90.cubin": "placeholder"})
        binary = b"\x00\xff\xfeget_valid_tactics\x80"
        for relative in ("flashinfer/lib/mm_bf16_cublaslt.so", "flashinfer_cubin/mm_bf16_cublaslt_sm90.cubin"):
            (self.root / relative).write_bytes(binary)
        files = self.run_inventory()["distributions"][0]["files"]
        artifacts = [r for r in files if r.get("kind") == "prebuilt_mm_bf16_cublaslt_artifact"]
        self.assertEqual(len(artifacts), 2)
        for record in artifacts:
            self.assertEqual(record["sha256"], hashlib.sha256(binary).hexdigest())
            self.assertEqual(record["size"], len(binary))
            self.assertTrue(record["binary_bytes_not_loaded"])
            self.assertTrue(record["binary_bytes_not_decoded"])
            self.assertNotIn("literal_markers", record)
            self.assertNotIn("literal_context", record)
        source = next(r for r in files if r["path"].endswith("gemm_base.py"))
        self.assertEqual(set(source["literal_markers"]), set(markers))

    def test_oversized_named_artifact_stat_only_and_other_binaries_excluded(self):
        self.fixture(sources={"flashinfer/mm_bf16_cublaslt.so": "x" * 2000,
                              "flashinfer/other.so": "MUST_NOT_READ",
                              "other/mm_bf16_cublaslt.cubin": "MUST_NOT_READ",
                              "flashinfer/mm_bf16_cublaslt.dll": "MUST_NOT_READ"})
        out = self.run_inventory(reader=Reader(max_file_bytes=1000))
        files = out["distributions"][0]["files"]
        artifacts = [r for r in files if r.get("kind") == "prebuilt_mm_bf16_cublaslt_artifact"]
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0]["status"], "file_byte_limit")
        self.assertEqual(artifacts[0]["size"], 2000)
        self.assertIsNone(artifacts[0]["sha256"])
        self.assertTrue(artifacts[0]["binary_bytes_not_loaded"])
        self.assertEqual(len(files), 3)
        self.assertLess(out["reads"]["bytes_read"], 1000)

    def test_unsupported_distributions_and_limit_expansion_fail_closed(self):
        self.distributions["flashinfer-python"] = object()
        self.assertEqual(self.run_inventory()["distributions"][0]["status"], "unsupported_metadata_layout")
        with self.assertRaises(ValueError):
            Reader(max_files=MAX_FILES + 1)

    def test_pinned_env_and_provider_declarations_read_without_factory(self):
        version = "0.7.0.post1"
        self.fixture(version=version, sources={"flashinfer/jit/env.py": "# get_aot_artifacts cuda_architectures\n"})
        self.fixture(name="flashinfer-jit-cache", version=version + "+cu130",
                     headers="Requires-Dist: flashinfer-jit-cache-sm100a (==0.7.0.post1+cu130)\n")
        self.fixture(name="flashinfer-jit-cache-sm90a", version=version + "+cu130",
            entry_points="[flashinfer.jit_cache.providers]\nsm90a = flashinfer_jit_cache_sm90a:provider\n",
            sources={"flashinfer_jit_cache_sm90a/__init__.py": "raise RuntimeError('MUST NOT EXECUTE')\n",
                     "flashinfer_jit_cache_sm90a/provider_manifest.json": json.dumps(dict(schema_version=1,
                         distribution="flashinfer-jit-cache-sm90a", version=version + "+cu130",
                         cuda_architectures=["sm_90a"], modules=["mm_bf16_cublaslt"],
                         jit_cache_dir="/private/DO_NOT_FOLLOW_OR_EMIT"))})
        before = set(sys.modules)
        output = self.run_inventory()
        by_name = {r["distribution"]: r for r in output["distributions"]}
        self.assertIn("flashinfer-jit-cache-sm100a", by_name)
        self.assertEqual(by_name["flashinfer-jit-cache-sm100a"]["status"], "missing_distribution")
        self.assertNotIn("jit/env.py", by_name["flashinfer-python"]["missing_source_suffixes"])
        provider = by_name["flashinfer-jit-cache-sm90a"]
        self.assertEqual(provider["provider_entry_points"]["entries"][0]["function"], "provider")
        self.assertTrue(provider["provider_version_evidence"]["matches_installed_shim"])
        self.assertTrue(provider["provider_version_evidence"]["matches_installed_python"])
        self.assertEqual(provider["provider_manifests"][0]["sm90_compatibility"], "declares_sm90_family")
        self.assertTrue(provider["provider_manifests"][0]["module_declared"])
        self.assertNotIn("DO_NOT_FOLLOW_OR_EMIT", json.dumps(output))
        self.assertFalse(output["provider_factories_executed"])
        self.assertFalse(output["binary_loadability_verified"])
        self.assertFalse(any(n.startswith("flashinfer_jit_cache_sm") for n in set(sys.modules) - before))

    def test_provider_mismatch_non_sm90_and_unknown_schema(self):
        self.fixture(version="0.7.0.post1")
        self.fixture(name="flashinfer-jit-cache", version="0.7.0.post1+cu130")
        self.fixture(name="flashinfer-jit-cache-sm90a", version="0.7.1+cu130", sources={
            "flashinfer_jit_cache_sm90a/provider_manifest.json": json.dumps(dict(schema_version=1,
                distribution="flashinfer-jit-cache-sm90a", version="0.7.1+cu130", cuda_architectures=["sm100a"], modules=[])),
            "flashinfer_jit_cache_sm90a/aot_manifest.json": '{"schema_version": 99}'})
        provider = next(r for r in self.run_inventory()["distributions"] if r["distribution"] == "flashinfer-jit-cache-sm90a")
        self.assertFalse(provider["provider_version_evidence"]["matches_installed_python"])
        self.assertFalse(provider["provider_version_evidence"]["matches_installed_shim"])
        self.assertEqual({r["sm90_compatibility"] for r in provider["provider_manifests"]}, {"unknown", "no_declared_sm90_family"})

    def test_unlisted_dynamic_factory_and_other_entry_groups_not_followed(self):
        self.fixture(name="flashinfer-jit-cache-sm90a", entry_points=
            "[console_scripts]\nsecret = unrelated:main\n[flashinfer.jit_cache.providers]\nsm90a = flashinfer_jit_cache_sm90a:provider\n",
            sources={"unrelated.py": "NEVER_READ", "flashinfer_jit_cache_sm90a/not_named.py": "NEVER_READ"})
        path = self.root / "flashinfer_jit_cache_sm90a/__init__.py"
        path.write_text("UNLISTED_MUST_NOT_READ")
        output = self.run_inventory()
        provider = next(r for r in output["distributions"] if r["distribution"] == "flashinfer-jit-cache-sm90a")
        self.assertEqual(len(provider["files"]), 3)
        self.assertEqual(provider["provider_manifests"], [])
        self.assertNotIn("NEVER_READ", json.dumps(output))
        self.assertNotIn("UNLISTED", json.dumps(output))

    def test_entry_and_manifest_corruption_safe_unknown(self):
        self.fixture(name="flashinfer-jit-cache-sm90a", entry_points="broken ini",
            sources={"flashinfer_jit_cache_sm90a/provider_manifest.json": "{ broken"})
        provider = next(r for r in self.run_inventory()["distributions"] if r["distribution"] == "flashinfer-jit-cache-sm90a")
        self.assertEqual(provider["provider_entry_points"]["status"], "invalid_entry_points")
        self.assertEqual(provider["provider_manifests"][0]["sm90_compatibility"], "unknown")

    def test_provider_discovery_and_reads_remain_globally_bounded(self):
        headers = "".join(f"Requires-Dist: flashinfer-jit-cache-sm{i}\n" for i in range(100, 120))
        headers += "Requires-Dist: arbitrary-package\n"
        self.fixture(name="flashinfer-jit-cache", headers=headers)
        output = self.run_inventory()
        names = [r["distribution"] for r in output["distributions"]]
        self.assertNotIn("arbitrary-package", names)
        self.assertEqual(sum(n.startswith("flashinfer-jit-cache-sm") for n in names), 8)
        shim = next(r for r in output["distributions"] if r["distribution"] == "flashinfer-jit-cache")
        self.assertTrue(shim["provider_distribution_limit_reached"])
        self.assertLessEqual(output["reads"]["files_attempted"], 64)
        self.assertLessEqual(output["reads"]["bytes_read"], 4 * 1024 * 1024)

    def test_entry_path_symlink_rejected_and_byte_budget_enforced(self):
        folder = self.fixture(name="flashinfer-jit-cache-sm90a", entry_points="x" * 2000)
        provider = next(r for r in self.run_inventory(reader=Reader(max_file_bytes=1000))["distributions"] if r["distribution"] == "flashinfer-jit-cache-sm90a")
        self.assertTrue(any(r["path"].endswith("entry_points.txt") and r["status"] == "file_byte_limit" for r in provider["files"]))
        (folder / "entry_points.txt").unlink()
        (self.root / "private.txt").write_text("MUST_NOT_READ")
        (folder / "entry_points.txt").symlink_to(self.root / "private.txt")
        provider = next(r for r in self.run_inventory()["distributions"] if r["distribution"] == "flashinfer-jit-cache-sm90a")
        self.assertTrue(any(r["path"].endswith("entry_points.txt") and r["status"] == "symlink_rejected" for r in provider["files"]))

    def test_script_has_no_package_imports_process_network_or_environment_reads(self):
        source = Path(__file__).parents[1] / "flashinfer_capability_inventory.py"
        tree = ast.parse(source.read_text())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import): imports.extend(n.name for n in node.names)
            if isinstance(node, ast.ImportFrom): imports.append(node.module)
        self.assertTrue(set(imports) <= {"csv", "configparser", "hashlib", "importlib.metadata", "io", "json", "re", "sys", "pathlib"})
        for token in ("find_spec(", "import_module(", "__import__(", "os.environ", "subprocess", "ctypes", ".glob(", ".rglob("):
            self.assertNotIn(token, source.read_text())


if __name__ == "__main__":
    unittest.main()
