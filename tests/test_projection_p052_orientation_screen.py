"""Stdlib-only source, receipt, measurement and failure gates; no CUDA run."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import projection_p052_orientation_screen as probe


def valid_receipt(folder):
    path = Path(folder) / "correctness.json"
    record = dict(complete=True, compile_only=False, native_imports_completed=True,
        sources_sha256={**probe.orientation.FROZEN, "projection_p051_orientation.py": probe.P051_SHA},
        environment=dict(capability=[9, 0], gpu="NVIDIA H200", torch="test", triton="test", cuda_runtime="test"), cases=[])
    for name, m, n, k, _ in probe.orientation.CASES:
        item = dict(name=name, m=m, n=n, k=k, complete=True, generated_kernel_executed=True,
                    correctness_status="passed", same_address_sequential_graph_ABA=True,
                    partial_buffers_isolated=True, compiled={}, correctness=[])
        for runner in ("control", "weight_first", "reduce"):
            filename = f"correctness.{name}.{runner}.ptx"
            content = ("fixture " + name + runner).encode()
            path.with_name(filename).write_bytes(content)
            item["compiled"][runner] = dict(ptx_file=filename, ptx_sha256=hashlib.sha256(content).hexdigest())
        for label in ("normal", "outliers", "sparse", "zero", "graph_A1", "graph_B", "graph_A2"):
            for runner in ("control", "weight_first"):
                item["correctness"].append(dict(label=label, orientation=runner, passed=True))
        record["cases"].append(item)
    path.write_text(json.dumps(record))
    return path, record


def samples(comparison="matched", ratio=1.2):
    return [dict(round=i, order=list(probe.safe.runner_order(i)), us=dict(parent=12 * ratio, candidate=12),
                 comparison=comparison, comparison_sequence_index=probe.comparison_order(i).index(comparison),
                 excluded_warmup=False, correctness=dict(parent=True, candidate=True)) for i in range(24)]


def cases(matched=1 / .9, selected=1 / 1.05, wins=10):
    return [dict(m=m, passed=True, series=dict(
        matched=dict(summary=dict(median_paired_raw_ratio=matched, winning_pairs=wins, total_pairs=12)),
        selected=dict(summary=dict(median_paired_raw_ratio=selected)))) for m in (8, 7)]


class ScreenTests(unittest.TestCase):
    def test_frozen_sources_match(self):
        hashes = probe.verify_sources()
        for name, digest in probe.FROZEN.items():
            self.assertEqual(hashes[name], digest)

    def test_seventy_checked_receipt_and_ptx_bytes(self):
        with TemporaryDirectory() as folder:
            path, _ = valid_receipt(folder)
            result = probe.validate_correctness_receipt(path)
            self.assertEqual(result["cases"], 5)
            self.assertEqual(result["original_oracle_checks"], 70)
            self.assertEqual(result["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_receipt_rejects_compile_only_incomplete_wrong_source_and_failed_check(self):
        with TemporaryDirectory() as folder:
            path, good = valid_receipt(folder)
            mutations = [lambda r: r.update(compile_only=True), lambda r: r.update(complete=False),
                         lambda r: r["sources_sha256"].update({"projection_p051_orientation.py": "bad"}),
                         lambda r: r["cases"][0]["correctness"][0].update(passed=False),
                         lambda r: r["cases"].pop(),
                         lambda r: r["cases"][0].update(generated_kernel_executed=False),
                         lambda r: r["cases"][0]["correctness"].reverse(),
                         lambda r: r["environment"].update(capability=[10, 0])]
            for mutate in mutations:
                record = copy.deepcopy(good)
                mutate(record)
                path.write_text(json.dumps(record))
                with self.assertRaises(ValueError):
                    probe.validate_correctness_receipt(path)

    def test_ptx_missing_corrupt_or_path_escape_rejected(self):
        with TemporaryDirectory() as folder:
            path, record = valid_receipt(folder)
            artifact = path.with_name(record["cases"][0]["compiled"]["control"]["ptx_file"])
            artifact.write_text("changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                probe.validate_correctness_receipt(path)
            artifact.unlink()
            with self.assertRaises(FileNotFoundError):
                probe.validate_correctness_receipt(path)
            record["cases"][0]["compiled"]["control"]["ptx_file"] = "../outside.ptx"
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "artifact name"):
                probe.validate_correctness_receipt(path)

    def test_order_balanced_pairs_and_raw_distributions(self):
        raw = samples()
        for i, sample in enumerate(raw):
            sample["us"]["parent"] *= 2 if i % 2 == 0 else .5
        result = probe.summarize(raw, "matched")
        self.assertEqual(result["total_pairs"], 12)
        self.assertEqual(result["winning_pairs"], 12)
        self.assertAlmostEqual(result["median_paired_raw_ratio"], 1.2)
        self.assertEqual(result["raw_us"]["parent"]["count"], 24)
        self.assertEqual(result["raw_us"]["candidate"]["minimum"], 12)

    def test_incomplete_or_reordered_samples_never_summarized(self):
        bad = [samples()[:-1]]
        for key, value in (("comparison_sequence_index", 1), ("excluded_warmup", True),
                           ("correctness", {"parent": True}), ("order", ["candidate", "parent"]),
                           ("comparison", "selected")):
            raw = samples()
            raw[0][key] = value
            bad.append(raw)
        for raw in bad:
            with self.assertRaises(ValueError):
                probe.summarize(raw, "matched")

    def test_latency_cut_boundary_is_not_speedup_percent(self):
        self.assertTrue(probe.assess(cases())["screen_passed"])
        self.assertFalse(probe.assess(cases(matched=1.1))["screen_passed"])
        self.assertFalse(probe.assess(cases(matched=1 / .900001))["screen_passed"])
        self.assertFalse(probe.assess(cases(selected=.95))["screen_passed"])
        self.assertFalse(probe.assess(cases(selected=1 / 1.050001))["screen_passed"])
        self.assertFalse(probe.assess(cases(wins=9))["screen_passed"])

    def test_both_shapes_must_pass_and_no_automatic_promotion(self):
        data = cases()
        data[1]["series"]["matched"]["summary"]["median_paired_raw_ratio"] = 1
        self.assertFalse(probe.assess(data)["screen_passed"])
        self.assertFalse(probe.assess(cases())["automatic_selection"])
        self.assertIsNone(probe.assess(cases())["official_score"])
        with self.assertRaises(ValueError):
            probe.assess(cases()[:1])

    def test_cloned_functions_preserve_constexprs_without_global_mutation(self):
        triton = SimpleNamespace(jit=lambda fn: fn)
        old = dict(vars(probe.orientation))
        language = object()
        kernels = probe.build_kernels(triton, language)
        for key, name in (("control", "_partial"), ("weight_first", "_partial_weight_first"), ("reduce", "_reduce")):
            fn, original = kernels[key], getattr(probe.orientation, name)
            self.assertIs(fn.__code__, original.__code__)
            self.assertEqual(fn.__annotations__, original.__annotations__)
            self.assertTrue(fn.__annotations__)
            self.assertEqual(fn.__defaults__, original.__defaults__)
            self.assertEqual(fn.__kwdefaults__, original.__kwdefaults__)
            self.assertIs(fn.__globals__["tl"], language)
        self.assertEqual(vars(probe.orientation), old)

    def test_whole_call_preserves_grid_precision_and_fresh_scratch(self):
        torch = Mock()
        torch.float32, torch.bfloat16 = "fp32", "bf16"
        torch.empty.side_effect = ["partial", "out"]
        class Kernel:
            def __init__(self):
                self.grids, self.call = [], Mock()
            def __getitem__(self, grid):
                self.grids.append(grid)
                return self.call
        kernels = {name: Kernel() for name in ("control", "weight_first", "reduce")}
        x = SimpleNamespace(shape=(7, 8192), device="cuda")
        output = probe.orientation_call(torch, kernels, "weight_first", {"x": x, "config": {"n": 4096}}, {"pooler": "W"})
        self.assertEqual(output, "out")
        self.assertEqual(torch.empty.call_args_list[0].args, ((8, 7, 4096),))
        self.assertEqual(torch.empty.call_args_list[0].kwargs["dtype"], "fp32")
        self.assertEqual(kernels["weight_first"].grids, [(64, 8)])
        self.assertEqual(kernels["weight_first"].call.call_args.args[-6:], (7, 4096, 8192, 8, 64, 64))
        self.assertEqual(kernels["weight_first"].call.call_args.kwargs, probe.orientation.PARTIAL_LAUNCH)
        self.assertEqual(kernels["reduce"].grids, [(224,)])
        self.assertFalse(kernels["control"].call.called)

    def test_source_or_receipt_failure_never_enters_native_stage(self):
        with TemporaryDirectory() as folder:
            path, _ = valid_receipt(folder)
            output = Path(folder) / "timing.json"
            for target in ("verify_sources", "validate_correctness_receipt"):
                with patch.object(probe, target, side_effect=ValueError("rejected")), patch.object(probe, "execute") as execute, patch("builtins.print"):
                    code = probe.main(["--correctness-receipt", str(path), "--output", str(output), "--outer-timeout-seconds", "180"])
                self.assertEqual(code, 1)
                execute.assert_not_called()
                self.assertFalse(json.loads(output.read_text())["complete"])

    def test_partial_timeout_retains_all_observed_samples(self):
        with TemporaryDirectory() as folder:
            path, _ = valid_receipt(folder)
            output = Path(folder) / "timing.json"
            def fail(args, report, save):
                report["cases"].append(dict(m=8, passed=False, series={"matched": {"samples": [{"us": {"parent": 12}}]}}))
                save()
                raise TimeoutError("bounded")
            with patch.object(probe, "execute", side_effect=fail), patch("builtins.print"):
                code = probe.main(["--correctness-receipt", str(path), "--output", str(output), "--outer-timeout-seconds", "180"])
            result = json.loads(output.read_text())
            self.assertEqual(code, 1)
            self.assertEqual(result["cases"][0]["series"]["matched"]["samples"][0]["us"], {"parent": 12})
            self.assertEqual(result["failure"]["type"], "TimeoutError")
            self.assertFalse(result["complete"])

    def test_environment_match_does_not_invent_physical_identity(self):
        env = dict(torch="2", triton="3", cuda_runtime="13", gpu="NVIDIA H200", capability=[9, 0])
        self.assertTrue(probe.environment_matches(env, dict(env)))
        self.assertFalse(probe.environment_matches(env, dict(env, triton="different")))

    def test_static_native_imports_and_no_extra_profiler_or_calibration(self):
        tree = ast.parse(Path(probe.__file__).read_text())
        top = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any(isinstance(node, ast.Import) and any(alias.name in {"torch", "triton"} for alias in node.names) for node in top))
        text = Path(probe.__file__).read_text()
        self.assertIn("timer = safe.ColdGraphTimer(torch)", text)
        self.assertNotIn("replay_us(", text)
        self.assertNotIn("torch.profiler", text)
        self.assertIn("expected_checked_screen_replays=320", text)
        self.assertIn("capture_calls=6", text)
        self.assertIn("untimed_capture_warmups=18", text)
        self.assertIn("warmup, count, offset in ((True, 6, 100), (False, 24, 200))", text)


if __name__ == "__main__":
    unittest.main()
