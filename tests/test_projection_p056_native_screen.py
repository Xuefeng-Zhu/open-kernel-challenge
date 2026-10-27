"""P056 CPU/source/mock tests only. No Torch import or native compilation."""
import ast
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import projection_p056_native_screen as screen


class Plan:
    def __init__(self, position=0, algorithm=1, **changed):
        self.values = dict(heuristic_position=position, algorithm_id=algorithm, tile_id=2,
            stages_id=3, split_k=0, reduction_scheme=0, cta_swizzle=0, custom_option=0,
            workspace_required_bytes=0, waves_count_estimate=1.0)
        self.values.update(changed)

    def metadata(self):
        return dict(self.values)


def receipt_case(plans=2):
    runners = 1 + plans
    series_count = 3 if plans == 2 else 1
    def rows(n):
        return [dict(correctness={"parent": True, "candidate": True}) for _ in range(n)]
    return dict(correctness=[dict(passed=True) for _ in range(12 * runners)],
                independent_graphs={"checks": [dict(passed=True) for _ in range(6)]},
                series={str(i): dict(warmup=rows(6), samples=rows(24)) for i in range(series_count)})


class NativeScreenTests(unittest.TestCase):
    def test_actual_frozen_native_parent_and_helper_sources(self):
        hashes = screen.verify_sources()
        self.assertEqual(len(hashes), 9)
        self.assertEqual(hashes["native_cublaslt_probe.py"], screen.NATIVE_SHA)
        self.assertEqual(hashes[screen.PARENT], screen.PARENT_SHA)
        self.assertNotIn("candidates/projection_router_m1_fp32_gemv.py", hashes)

    def test_source_hash_failure_is_fail_closed(self):
        with TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                screen.verify_sources(Path(directory))

    def test_limits_explicit_compile_and_hard_timeout(self):
        good = dict(compile_native=True, deadline_seconds=240, outer_timeout_seconds=270)
        screen.validate_options(SimpleNamespace(**good))
        for changes in ({"compile_native": False}, {"deadline_seconds": 241},
                        {"deadline_seconds": 0}, {"outer_timeout_seconds": 254},
                        {"outer_timeout_seconds": 271}):
            with self.assertRaises(ValueError):
                screen.validate_options(SimpleNamespace(**(good | changes)))

    def test_first_control_then_prefer_different_algorithm_family(self):
        first, same_family, different = Plan(0, 1), Plan(1, 1, tile_id=4), Plan(8, 2)
        chosen, reason = screen.choose_plans({"plans": [different, same_family, first]})
        self.assertEqual(chosen, [first, different])
        self.assertEqual(reason, "different_algorithm_id")

    def test_same_family_configuration_fallback_and_duplicate_skip(self):
        first, duplicate, alternate = Plan(0), Plan(1, waves_count_estimate=7,
            workspace_required_bytes=1024), Plan(2, tile_id=4)
        chosen, reason = screen.choose_plans({"plans": [first, duplicate, alternate]})
        self.assertEqual(chosen, [first, alternate])
        self.assertEqual(reason, "same_family_distinct_recorded_configuration")
        chosen, reason = screen.choose_plans({"plans": [first, duplicate]})
        self.assertEqual(chosen, [first])
        self.assertEqual(reason, "only_one_distinct_recorded_configuration")

    def test_zero_eligible_or_unsafe_inventory_cannot_start_timing(self):
        with self.assertRaises(RuntimeError):
            screen.choose_plans({"plans": []})
        for bad in (Plan(split_k=8), Plan(reduction_scheme=2)):
            with self.assertRaises(ValueError):
                screen.choose_plans({"plans": [bad]})
        for limit in (0, 3):
            with self.assertRaises(ValueError):
                screen.choose_plans({"plans": [Plan()]}, limit)

    def test_recorded_oracle_counts_actual_selected_plan_cardinality(self):
        for counts, expected in (((2, 2), 444), ((1, 1), 180), ((1, 2), 312)):
            report = dict(cases=[receipt_case(n) for n in counts], nearby=[])
            self.assertEqual(screen.count_records(report), expected)
        report["cases"][0]["correctness"][0]["passed"] = False
        self.assertEqual(screen.count_records(report), 311)

    def test_partial_record_count_does_not_invent_comparisons(self):
        report = dict(cases=[dict(correctness=[{"passed": True}, {"passed": False}],
            independent_graphs={"checks": []}, series={"x": {
                "warmup": [], "samples": [dict(correctness={"parent": True})]}})], nearby=[])
        self.assertEqual(screen.count_records(report), 2)

    def test_nonfinite_or_nonpositive_samples_rejected_without_replacement(self):
        for value in (float("nan"), float("inf"), 0, -1):
            with self.assertRaises(ValueError):
                screen.measured_us(value)
        self.assertEqual(screen.measured_us(12.5), 12.5)

    def test_gross_pairing_preserves_every_sample_and_tail(self):
        rows = [dict(round=i, order=list(screen.safe.runner_order(i)),
                     us={"parent": 20.0, "candidate": 10.0}) for i in range(24)]
        rows[-1]["us"]["candidate"] = 100.0
        summary = screen.safe.summarize_samples(rows)
        self.assertEqual(len(summary["reverse_order_pair_ratios"]), 12)
        self.assertEqual(summary["raw_us"]["candidate"]["maximum"], 100.0)
        self.assertEqual(summary["median_paired_raw_ratio"], 2.0)
        with self.assertRaises(ValueError):
            screen.safe.summarize_samples(rows[:-1])

    def test_import_has_no_torch_native_or_external_execution(self):
        tree = ast.parse(Path(screen.__file__).read_text())
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module]
                self.assertFalse(any(n.split(".")[0] in {
                    "torch", "triton", "native_cublaslt_probe", "benchmark", "subprocess", "modal", "requests"
                } for n in names))
        calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertNotIn("safe.verify_sources", calls)
        self.assertFalse(any(name.startswith(("subprocess.", "os.system", "urllib.")) for name in calls))

    def test_correctness_precedes_timing_and_beta_zero_poison_stays_untimed(self):
        source = Path(screen.__file__).read_text()
        before = source.index('report["phase"] = "correctness"')
        timing = source.index('report["phase"] = "paired_timing"')
        self.assertLess(before, timing)
        self.assertIn('correctness(bundle, "before")', source[before:timing])
        measure = source.index('row["us"][runner_name] = measured_us(')
        poison = source.rindex('selected[runner_name]["output"].fill_(float("nan"))', 0, measure)
        self.assertLess(poison, measure)
        self.assertIn('kernel_overlap_verified=False', source)
        self.assertIn('correctness(bundle, "after")', source[timing:])

    def test_failure_receipt_is_atomic_partial_and_bounded(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            def failure(args, report, save):
                report["phase"] = "native_build"
                report["compile"] = {"complete": False}
                save()
                raise RuntimeError("failure reading /private/some/path/file.cpp")
            with patch.object(screen, "execute", side_effect=failure), patch("builtins.print"):
                result = screen.main(["--compile-native", "--output", str(output), "--outer-timeout-seconds", "270"])
            saved = json.loads(output.read_text())
            self.assertEqual(result, 1)
            self.assertFalse(saved["complete"])
            self.assertEqual(saved["phase"], "native_build")
            self.assertIn("<path>", saved["failure"]["message"])
            self.assertNotIn("/private", saved["failure"]["message"])
            self.assertFalse(output.with_name("report.json.tmp").exists())

    def test_source_failure_prevents_any_execute(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            with patch.object(screen, "verify_sources", side_effect=ValueError("hash mismatch")), \
                 patch.object(screen, "execute") as execute, patch("builtins.print"):
                result = screen.main(["--compile-native", "--output", str(output), "--outer-timeout-seconds", "270"])
            self.assertEqual(result, 1)
            execute.assert_not_called()
            self.assertFalse(json.loads(output.read_text())["native_execution"])


if __name__ == "__main__":
    unittest.main()
