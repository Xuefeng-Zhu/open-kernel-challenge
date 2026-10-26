"""P059 bounded local source/metadata/mock tests; no Torch or native execution."""
import ast
import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import projection_p059_native_screen as screen


class Plan:
    def __init__(self, algorithm=41, tile=21, position=0, **changed):
        self.values = dict(m=128, n=2304, k=8192, heuristic_position=position,
            algorithm_id=algorithm, tile_id=tile, stages_id=35, split_k=1,
            reduction_scheme=0, cta_swizzle=0, custom_option=0,
            workspace_required_bytes=0, compute="CUBLAS_COMPUTE_32F",
            scale="CUDA_R_32F", input_output="CUDA_R_16BF")
        self.values.update(changed)

    def metadata(self):
        return dict(self.values)


def case_receipt():
    def rows(n):
        return [dict(correctness={"parent": True, "candidate": True}) for _ in range(n)]
    return dict(passed=True, correctness=[dict(passed=True) for _ in range(48)],
        independent_graphs={"checks": [dict(passed=True) for _ in range(6)]},
        series={name: dict(warmup=rows(6), samples=rows(24)) for name, _, _ in screen.SERIES},
        activities={name: dict(correctness_passed=True) for name in screen.RUNNERS})


class SplitScreenTests(unittest.TestCase):
    def test_source_bindings_provenance_and_exact_parent(self):
        hashes = screen.verify_sources()
        self.assertEqual(len(hashes), 13)
        self.assertEqual(hashes[screen.PARENT], screen.PARENT_SHA)
        self.assertEqual(hashes["native_cublaslt_fp32_split_probe.py"], screen.SPLIT_SHA)
        self.assertEqual(hashes["projection_p057_native_screen.py"], screen.FROZEN_P057_SHA)
        self.assertEqual(hashes["projection_p056_native_screen.py"], screen.FROZEN_WORKER_SHA)
        self.assertEqual(hashes["projection_p059_native_screen.py"], hashlib.sha256(Path(screen.__file__).read_bytes()).hexdigest())

    def test_hash_mismatch_is_fail_closed(self):
        with TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                screen.verify_sources(Path(folder))

    def test_one_case_three_native_configurations_four_named_comparisons(self):
        self.assertEqual(screen.CASES, (("qkv_gate", 128),))
        self.assertEqual(screen.RUNNERS, ("p046", "row41", "row66", "split"))
        self.assertEqual(screen.SERIES, (("anchor_row41", "p046", "row41"),
            ("anchor_row66", "p046", "row66"), ("anchor_split", "p046", "split"),
            ("row66_vs_split", "row66", "split")))
        self.assertEqual(screen.MAX_SELECTED, 3)
        self.assertEqual(screen.WORKSPACE_LIMIT, 64 * 1024 * 1024)

    def test_controls_require_exact_historical_metadata_not_first_ids(self):
        row41 = Plan(position=3)
        row66 = Plan(66, 20, 7)
        distractor = Plan(41, 22, 0)
        result = screen.choose_controls({"plans": [row66, distractor, row41]}, 0)
        self.assertEqual(result, [row41, row66])
        # No heuristic-position or waves-estimate selection based on timing.
        duplicate = Plan(position=8, waves_count_estimate=200)
        self.assertEqual(screen.choose_controls({"plans": [duplicate, row66, row41]}, 0)[0], row41)

    def test_missing_control_is_capability_blocked_without_substitution(self):
        for changes in ({"stages_id": 36}, {"split_k": 0}, {"cta_swizzle": 1}, {"custom_option": 1}):
            with self.assertRaises(screen.CapabilityBlocked):
                screen.choose_controls({"plans": [Plan(), Plan(66, 20, 6, **changes)]}, 0)
        with self.assertRaises(screen.CapabilityBlocked):
            screen.choose_controls({"plans": []}, 0)

    def test_unsafe_control_inventory_rejected_even_if_exact_plans_exist(self):
        for changes in ({"split_k": 8}, {"reduction_scheme": 2}, {"compute": "FAST_16F"},
                        {"scale": "CUDA_R_16BF"}, {"input_output": "CUDA_R_16F"},
                        {"m": 127}, {"workspace_required_bytes": 64 * 1024 * 1024 + 1}):
            with self.assertRaises(ValueError):
                screen.choose_controls({"plans": [Plan(), Plan(66, 20, 6), Plan(99, 1, 8, **changes)]}, 0)
        for invalid in (None, True, -1):
            with self.assertRaises(ValueError):
                screen.choose_controls({"plans": [Plan()]}, invalid)

    def test_none_enum_is_consumed_from_actual_compiled_map(self):
        a, b = Plan(reduction_scheme=7), Plan(66, 20, 6, reduction_scheme=7)
        self.assertEqual(screen.choose_controls({"plans": [a, b]}, 7), [a, b])

    def test_inventory_report_contains_no_native_objects(self):
        row = screen.inventory_record(dict(requested=32, returned=2, plans=[Plan()], rejected=[],
            reduction_scheme_constants={"actual_enum": 7}))
        self.assertEqual(json.loads(json.dumps(row))["eligible"][0]["algorithm_id"], 41)
        self.assertEqual(row["reduction_scheme_constants"], {"actual_enum": 7})

    def test_limits_and_explicit_native_action(self):
        good = dict(compile_native=True, deadline_seconds=240, outer_timeout_seconds=270)
        screen.validate_options(SimpleNamespace(**good))
        for changes in ({"compile_native": False}, {"deadline_seconds": 241},
                        {"deadline_seconds": 0}, {"outer_timeout_seconds": 254}, {"outer_timeout_seconds": 271}):
            with self.assertRaises(ValueError):
                screen.validate_options(SimpleNamespace(**(good | changes)))

    def test_actual_oracle_count_and_partial_failure_retained(self):
        report = dict(cases=[case_receipt()], nearby=[])
        self.assertEqual(screen.count_records(report), 298)
        report["cases"][0]["activities"]["split"]["correctness_passed"] = False
        report["cases"][0]["series"]["anchor_split"]["samples"][-1]["correctness"]["candidate"] = False
        self.assertEqual(screen.count_records(report), 296)
        self.assertEqual(screen.count_records(dict(cases=[], nearby=[])), 0)

    def test_gate_is_ten_percent_latency_cut_not_ten_percent_speedup(self):
        def assessment(ratio, wins=12):
            return screen.assess(dict(passed=True, series={"anchor_split": {"summary": {
                "median_paired_raw_ratio": ratio,
                "reverse_order_pair_ratios": [ratio] * wins + [.99] * (12 - wins)}}}))
        self.assertFalse(assessment(1.10)["screen_passed"])
        self.assertFalse(assessment(math.nextafter(1 / .9, 0))["screen_passed"])
        self.assertTrue(assessment(1 / .9, 10)["screen_passed"])
        self.assertFalse(assessment(1.2, 9)["screen_passed"])
        self.assertFalse(assessment(.5)["screen_passed"])
        self.assertFalse(assessment(1.2)["automatic_selection"])
        with self.assertRaises(ValueError):
            screen.assess(dict(passed=False))

    def test_raw_pairing_retains_tail_and_rejects_missing_sample(self):
        rows = [dict(round=i, order=list(screen.safe.runner_order(i)), us={"parent": 20., "candidate": 10.}) for i in range(24)]
        rows[-1]["us"]["candidate"] = 100.
        summary = screen.safe.summarize_samples(rows)
        self.assertEqual(len(summary["reverse_order_pair_ratios"]), 12)
        self.assertEqual(summary["raw_us"]["candidate"]["maximum"], 100.)
        self.assertEqual(summary["median_paired_raw_ratio"], 2.)
        with self.assertRaises(ValueError):
            screen.safe.summarize_samples(rows[:-1])
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                screen.measured_us(value)

    def test_activity_has_names_durations_only_and_empty_is_unavailable(self):
        events = [SimpleNamespace(device_type="CUDA", device_time_total=2.5, name="gemm"),
                  SimpleNamespace(device_type="CUDA", device_time_total=1., name="memcpy"),
                  SimpleNamespace(device_type="CPU", device_time_total=0., name="secret-host-event")]
        report = screen.activity_summary(SimpleNamespace(events=lambda: events), "CUDA")
        self.assertTrue(report["available"])
        self.assertEqual(len(report["cuda_events"]), 2)
        self.assertEqual(len(report["copy_events"]), 1)
        self.assertNotIn("secret", json.dumps(report))
        self.assertEqual(set(report["cuda_events"][0]), {"symbol", "duration_us", "kind"})
        self.assertFalse(screen.activity_summary(SimpleNamespace(events=lambda: []), "CUDA")["available"])

    def test_activity_is_bounded_without_silent_truncation(self):
        event = SimpleNamespace(device_type="CUDA", device_time_total=1., name="gemm")
        with self.assertRaises(ValueError):
            screen.activity_summary(SimpleNamespace(events=lambda: [event] * 257), "CUDA")
        for name, duration in (("x" * 4097, 1.), ("gemm", float("nan")), ("gemm", -1.)):
            bad = SimpleNamespace(device_type="CUDA", device_time_total=duration, name=name)
            with self.assertRaises(ValueError):
                screen.activity_summary(SimpleNamespace(events=lambda: [bad]), "CUDA")

    def test_import_and_local_tests_do_not_build_native_or_import_torch(self):
        tree = ast.parse(Path(screen.__file__).read_text())
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module]
                self.assertFalse(any(n.split(".")[0] in {"torch", "native_cublaslt_probe", "native_cublaslt_fp32_split_probe", "benchmark", "modal", "subprocess"} for n in names))
        calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertNotIn("safe.verify_sources", calls)
        self.assertFalse(any(call.startswith(("subprocess.", "os.system", "urllib.")) for call in calls))
        self.assertNotIn("setattr", calls)

    def test_native_selection_precedes_all_execution_and_profiling_is_last(self):
        source = Path(screen.__file__).read_text()
        self.assertLess(source.index("select_actual_fp32_split("), source.index("reference.generate_weights("))
        self.assertLess(source.index("choose_controls(inventories"), source.index("native_graph(plan, data)"))
        before = source.index('report["phase"] = "correctness"')
        timing = source.index('report["phase"] = "paired_timing"')
        activity = source.index('report["phase"] = "activity"')
        self.assertLess(before, timing)
        self.assertLess(timing, activity)
        self.assertIn('correctness(bundle, "before")', source[before:timing])
        self.assertIn('correctness(bundle, "after")', source[timing:activity])
        self.assertIn('first = bundle["runners"]["split"]', source[before:timing])
        self.assertIn('native_graph(bundle["plans"][2], other_data)', source[before:timing])
        self.assertIn('kernel_overlap_verified=False', source)
        self.assertIn('record_shapes=False, with_stack=False, profile_memory=False', source[activity:])
        self.assertIn('row["correctness_passed"] = passed["passed"]', source[activity:])

    def test_missing_capability_is_complete_information_not_fake_execution(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            def blocked(args, report, save):
                report["phase"] = "interface_and_inventory"
                report["native_build_complete"] = True
                report["inventories"] = {"fp32_split": {"eligible": []}}
                save()
                raise screen.CapabilityBlocked("No actual compute-type split returned")
            with patch.object(screen, "execute", side_effect=blocked), patch("builtins.print"):
                code = screen.main(["--compile-native", "--output", str(output), "--outer-timeout-seconds", "270"])
            record = json.loads(output.read_text())
            self.assertEqual(code, 0)
            self.assertTrue(record["complete"])
            self.assertFalse(record["passed"])
            self.assertEqual(record["status"], "capability_blocked")
            self.assertFalse(record["native_execution"])
            self.assertEqual(record["oracle_checks"], 0)
            self.assertEqual(record["cases"], [])

    def test_failure_retains_partial_record_and_sanitizes_path(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            def fail(args, report, save):
                report["phase"] = "paired_timing"
                report["cases"] = [case_receipt()]
                report["cases"][0]["activities"] = {}
                save()
                raise RuntimeError("failure at /private/sensitive/file.cpp")
            with patch.object(screen, "execute", side_effect=fail), patch("builtins.print"):
                code = screen.main(["--compile-native", "--output", str(output), "--outer-timeout-seconds", "270"])
            record = json.loads(output.read_text())
            self.assertEqual(code, 1)
            self.assertFalse(record["complete"])
            self.assertEqual(record["oracle_checks"], 294)
            self.assertIn("<path>", record["failure"]["message"])
            self.assertFalse(output.with_name("report.json.tmp").exists())

    def test_source_failure_prevents_execution(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            with patch.object(screen, "verify_sources", side_effect=ValueError("hash mismatch")), patch.object(screen, "execute") as execute, patch("builtins.print"):
                result = screen.main(["--compile-native", "--output", str(output), "--outer-timeout-seconds", "270"])
            self.assertEqual(result, 1)
            execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
