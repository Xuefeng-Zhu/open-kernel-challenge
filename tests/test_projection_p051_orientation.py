"""P051 stdlib algebra/AST/harness tests: no Torch import, GPU or JIT execution."""
import argparse
import ast
import copy
import hashlib
from pathlib import Path
import random
import tempfile
import types
import unittest
from unittest.mock import patch

import projection_p051_orientation as p


def functions(path):
    return {node.name: node for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, ast.FunctionDef)}


def normalized(node):
    node = copy.deepcopy(node)
    node.decorator_list = []
    return ast.dump(node, include_attributes=False)


class OrientationTests(unittest.TestCase):
    def test_exact_r06_control_and_reducer_ast(self):
        old = functions(p.ROOT / "candidates/pooler_triton_split8.py")
        new = functions(Path(p.__file__))
        for name in ("_partial", "_reduce"):
            self.assertEqual(normalized(old[name]), normalized(new[name]), name)
        self.assertEqual((p.SPLITS, p.BLOCK_M, p.BLOCK_N, p.BLOCK_K), (8, 16, 64, 64))
        self.assertEqual(p.PARTIAL_LAUNCH, {"num_warps": 4, "num_stages": 2})
        self.assertEqual(p.REDUCE_LAUNCH, {"num_warps": 4})

    def test_only_four_orientation_statements_differ(self):
        fs = functions(Path(p.__file__))
        control = copy.deepcopy(fs["_partial"])
        candidate = copy.deepcopy(fs["_partial_weight_first"])
        candidate.name = control.name
        self.assertEqual(len(control.body), len(candidate.body))
        differences = [i for i, (a, b) in enumerate(zip(control.body, candidate.body))
                       if ast.dump(a) != ast.dump(b)]
        self.assertEqual(differences, [5, 6, 7, 8])
        self.assertEqual(ast.unparse(candidate.body[5]), "acc = tl.full((BN, 16), 0.0, tl.float32)")
        # The loop bounds, K order and both loads are bit-for-bit the same AST.
        self.assertEqual(ast.dump(candidate.body[6].iter), ast.dump(control.body[6].iter))
        self.assertEqual(ast.dump(ast.Module(body=candidate.body[6].body[:-1], type_ignores=[])),
                         ast.dump(ast.Module(body=control.body[6].body[:-1], type_ignores=[])))
        self.assertEqual(ast.unparse(candidate.body[6].body[-1]), "acc = tl.dot(w, tl.trans(x), acc)")
        self.assertEqual(ast.unparse(candidate.body[7]),
                         "offsets = (split * M + ms[None, :]) * N + ns[:, None]")
        self.assertEqual(ast.unparse(candidate.body[8]),
                         "tl.store(P + offsets, acc, (ms[None, :] < M) & (ns[:, None] < N))")
        for index in differences:
            candidate.body[index] = control.body[index]
        self.assertEqual(normalized(candidate), normalized(control))

    def test_partial_addresses_one_writer_and_identical_masked_coverage(self):
        for m in (1, 7, 8, 15, 16):
            for n in (1, 7, 63, 64, 65, 129):
                old = list(p.address_records(m, n))
                new = list(p.address_records(m, n, True))
                self.assertEqual(sorted(old), sorted(new), (m, n))
                offsets = [r[3] for r in new]
                self.assertEqual(sorted(offsets), list(range(8 * m * n)))
                self.assertEqual(len(offsets), len(set(offsets)))

    def test_split_coverage_boundaries_and_empty_neutral_splits(self):
        for k in (1, 63, 64, 65, 131, 511, 512, 513, 521, 8191, 8192, 8193):
            splits = p.split_indices(k)
            self.assertEqual(len(splits), 8)
            self.assertEqual([i for chunk in splits for i in chunk], list(range(k)))
            self.assertTrue(all(chunk == sorted(chunk) for chunk in splits))
        self.assertEqual(p.split_indices(1), [[0], [], [], [], [], [], [], []])

    def test_integer_algebra_reduction_and_adversarial_patterns(self):
        gen = random.Random(51051)
        for m, n, k in ((1, 1, 1), (7, 65, 63), (8, 67, 65), (15, 17, 131), (16, 9, 521)):
            w = [[gen.randint(-3, 3) for _ in range(k)] for _ in range(n)]
            for pattern in ("normal", "sparse", "zero"):
                x = [[gen.randint(-3, 3) for _ in range(k)] for _ in range(m)]
                if pattern != "normal":
                    x = [[0] * k for _ in range(m)]
                if pattern == "sparse":
                    for row in range(m):
                        x[row][(row * 131 + 7) % k] = 8 if row % 2 == 0 else -32
                before = copy.deepcopy((x, w))
                a, b = p.algebra_partials(x, w), p.algebra_partials(x, w, True)
                self.assertEqual(a, b)
                self.assertEqual((x, w), before)
                actual = [sum(b[s * m * n + i] for s in range(8)) for i in range(m * n)]
                expected = [sum(x[row][j] * w[col][j] for j in range(k))
                            for row in range(m) for col in range(n)]
                self.assertEqual(actual, expected)

    def test_out_of_envelope_and_ragged_inputs_rejected(self):
        for shape in ((0, 64, 64), (17, 64, 64), (32, 64, 64), (8, 0, 64),
                      (8, 64, 0), (8.0, 64, 64), (True, 64, 64)):
            with self.assertRaises(ValueError):
                p.validate_shape(*shape)
        with self.assertRaises(ValueError):
            p.algebra_partials([[1, 2]], [[1]])

    def test_source_hashes_and_compile_only_omitted_oracles(self):
        actual = p.verify_sources()
        for name, digest in p.FROZEN.items():
            self.assertEqual(actual[name], digest)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "projection_p051_orientation.py").write_bytes(Path(p.__file__).read_bytes())
            self.assertEqual(set(p.verify_sources(root, compile_only=True)), {"projection_p051_orientation.py"})
            with self.assertRaises(FileNotFoundError):
                p.verify_sources(root)
            original = "candidates/pooler_triton_split8.py"
            (root / original).parent.mkdir()
            (root / original).write_text("altered")
            with self.assertRaisesRegex(ValueError, "Frozen source mismatch"):
                p.verify_sources(root, compile_only=True)

    def test_ptx_summary_is_mnemonic_evidence_not_dynamic_metrics(self):
        ptx = """// wgmma.mma_async.fake is not an instruction
.version 8.0
 @%p1 cp.async.ca.shared.global [%r1], [%rd1], 16;
 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes foo;
 wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 foo;
 mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 foo;
 ld.local.u32 %r2, [%rd2];
 st.local.u32 [%rd3], %r3;
"""
        result = p.summarize_ptx(ptx)
        self.assertEqual(result["instruction_counts"], dict(wgmma=1, mma_sync=1, async_copy=2,
                                                          tma_tensor=1, local_load=1, local_store=1))
        self.assertEqual(result["ptx_sha256"], hashlib.sha256(ptx.encode()).hexdigest())
        self.assertEqual(len(result["mma_mnemonics"]), 2)
        self.assertIn("not SASS", result["qualification"])

    def test_lazy_binary_metadata_and_unavailable_resources_preserved(self):
        metadata = types.SimpleNamespace(name="kernel", shared=4096, num_warps=4, num_stages=2)
        class Compiled:
            asm = {"ptx": "mma.sync.fake;\n"}
            def _init_handles(self):
                self.n_regs, self.n_spills = 42, 0
        kernel = Compiled()
        kernel.metadata = metadata
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.ptx"
            result = p.compiled_record(kernel, path)
            self.assertTrue(result["binary_handle_initialization"]["loaded"])
            self.assertEqual((result["n_regs_per_thread"], result["shared_bytes"], result["n_spills_driver_raw"]), (42, 4096, 0))
            self.assertEqual(path.read_text(), kernel.asm["ptx"])
            bare = types.SimpleNamespace(asm={"ptx": "ret;\n"}, metadata={})
            absent = p.compiled_record(bare, path)
            self.assertIsNone(absent["n_regs_per_thread"])
            self.assertIsNone(absent["n_spills_driver_raw"])
            self.assertFalse(absent["binary_handle_initialization"]["attempted"])
            broken = types.SimpleNamespace(asm={"ptx": "ret;\n"}, metadata={})
            def fail():
                raise RuntimeError("driver unavailable")
            broken._init_handles = fail
            failed = p.compiled_record(broken, path)
            self.assertFalse(failed["binary_handle_initialization"]["loaded"])
            self.assertIn("driver unavailable", failed["binary_handle_initialization"]["error"])
            with self.assertRaisesRegex(RuntimeError, "no PTX"):
                p.compiled_record(types.SimpleNamespace(asm={}), path)

    def test_h200_guard_and_timeout_validation(self):
        cuda = types.SimpleNamespace(is_available=lambda: True, get_device_name=lambda: "NVIDIA H200",
                                     get_device_capability=lambda: (9, 0))
        p.require_h200(types.SimpleNamespace(cuda=cuda))
        for name, capability, available in (("H100", (9, 0), True), ("H200", (10, 0), True), ("H200", (9, 0), False)):
            cuda.get_device_name = lambda: name
            cuda.get_device_capability = lambda: capability
            cuda.is_available = lambda: available
            with self.assertRaises(RuntimeError):
                p.require_h200(types.SimpleNamespace(cuda=cuda))
        p.validate_options(argparse.Namespace(deadline_seconds=75, outer_timeout_seconds=90))
        for internal, outer in ((0, 90), (121, 180), (75, 80), (75, 181)):
            with self.assertRaises(ValueError):
                p.validate_options(argparse.Namespace(deadline_seconds=internal, outer_timeout_seconds=outer))

    def test_native_imports_are_lazy_and_compile_only_has_no_fixture_or_launch(self):
        module = ast.parse(Path(p.__file__).read_text())
        top_imports = [n for n in module.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any("torch" in ast.unparse(n) or "triton" in ast.unparse(n) for n in top_imports))
        native = functions(Path(p.__file__))["_native"]
        source = ast.unparse(native)
        self.assertIn("selected_cases = CASES[:1] if args.compile_only else CASES", source)
        self.assertIn("reference = None if args.compile_only else _load_oracle()", source)
        loop = next(n for n in native.body if isinstance(n, ast.For))
        gate_index = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.If)
                          and ast.unparse(n.test) == "args.compile_only")
        prefix = ast.Module(body=loop.body[:gate_index], type_ignores=[])
        calls = [ast.unparse(n.func) for n in ast.walk(prefix) if isinstance(n, ast.Call)]
        self.assertFalse(any("randn" in c or "native_input" in c or "launch" == c for c in calls))
        self.assertTrue(any(c.endswith(".warmup") for c in calls))
        self.assertTrue(isinstance(loop.body[gate_index].body[-1], ast.Continue))

    def test_short_k_aba_fixture_sign_flip_and_bitwise_guard(self):
        native = functions(Path(p.__file__))["_native"]
        loop = next(n for n in native.body if isinstance(n, ast.For))
        index = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.If)
                     and ast.unparse(n.test) == "k == 1")
        flip, guard = loop.body[index:index + 2]
        self.assertEqual(ast.unparse(flip.body[0]), "b.neg_()")
        self.assertEqual(ast.unparse(guard.test), "torch.equal(a.view(torch.uint8), b.view(torch.uint8))")
        self.assertIsInstance(guard.body[0], ast.Raise)
        own_branch = compile(ast.fix_missing_locations(ast.Module(body=[flip, guard], type_ignores=[])),
                             "<P051 fixture-only branch>", "exec")
        class Fixture:
            def __init__(self, values):
                self.values = list(values)
            def neg_(self):
                self.values = [-value for value in self.values]
            def view(self, dtype):
                return self.values
        fake_torch = types.SimpleNamespace(uint8=object(), equal=lambda a, b: a == b)
        a, b = Fixture([8]), Fixture([8])
        exec(own_branch, {"torch": fake_torch, "a": a, "b": b, "k": 1})
        self.assertEqual(a.values, [8])
        self.assertEqual(b.values, [-8])
        with self.assertRaisesRegex(RuntimeError, "must differ bitwise"):
            exec(own_branch, {"torch": fake_torch, "a": Fixture([8, 0]), "b": Fixture([8, 0]), "k": 2})
        exec(own_branch, {"torch": fake_torch, "a": Fixture([8, 0]), "b": Fixture([0, 8]), "k": 2})

    def test_failure_report_preserves_partial_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "timing.json"
            def failed(args, report, save, stop):
                report["cases"].append(dict(name="case", complete=False,
                    correctness=[dict(passed=False, label="normal")]))
                save()
                raise RuntimeError("diagnostic failed")
            with patch.object(p, "_native", failed), patch("builtins.print"):
                code = p.main(["--output", str(output), "--compile-only"])
            import json
            result = json.loads(output.read_text())
            self.assertEqual(code, 1)
            self.assertFalse(result["complete"])
            self.assertFalse(result["cases"][0]["correctness"][0]["passed"])
            self.assertEqual(result["error"]["type"], "RuntimeError")
            self.assertNotIn("official/reference.py", result["sources_sha256"])


if __name__ == "__main__":
    unittest.main()
