"""P050 local mock/static protocol tests; no Torch, network or CUDA execution."""
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import warnings

import projection_p050_qkv_screen as probe


def sample_rows(ratio=1.1, pairs=None):
    pairs = pairs or [ratio] * 12
    return [dict(round=i, order=list(probe.safe.runner_order(i)),
                 us=dict(parent=11 * pairs[i // 2], candidate=11)) for i in range(24)]


def activity(symbol="main", duration=10, copy=False, available=True):
    events = [dict(symbol=symbol, duration_us=duration,
                   kind="copy" if copy else "kernel_or_other")] if available else []
    return dict(available=available, cuda_events=events, copy_events=events if copy else [])


def cases(target_ratio=1.1):
    result = []
    for layer, m, role in probe.CASES:
        profiles = {"parent": activity("original"), "candidate": activity("bias")}
        result.append(dict(layer=layer, m=m, role=role, passed=True,
            series={"main": dict(passed=True, samples=sample_rows(target_ratio if role == "target" else 1))},
            capture={"candidate": {"explicit_unfused_fallback_warning": False}},
            activities=profiles if (layer, m) in probe.PROFILE_CASES else {}))
    return result


class QkvScreenTests(unittest.TestCase):
    def test_frozen_sources_helper_and_no_receipt_dependency(self):
        hashes = probe.verify_sources()
        self.assertEqual(hashes["projection_p046_confirmation.py"], probe.SAFE_SHA256)
        self.assertEqual(hashes[probe.SOURCES["candidate"][0]], probe.SOURCES["candidate"][1])
        self.assertNotIn(probe.MODEL_RECEIPT, hashes)

    def test_exact_cases_profile_subset_and_counts(self):
        self.assertEqual(probe.CASES, (("qkv_gate", 8, "target"), ("qkv_gate", 16, "target"),
            ("qkv_gate", 32, "target"), ("qkv_gate", 128, "target"), ("qkv_gate", 256, "target"),
            ("qkv_gate", 127, "nearby"), ("qkv_gate", 129, "nearby"),
            ("qkv_gate", 1, "control"), ("unpooler", 16, "control")))
        self.assertEqual(probe.PROFILE_CASES, (("qkv_gate", 8), ("qkv_gate", 128),
            ("qkv_gate", 256), ("unpooler", 16)))
        count = probe.protocol_counts()
        self.assertEqual((count["measured_replays"], count["warmup_replays"],
            count["before_after_probe_replays"], count["activity_replays"]), (432, 108, 180, 8))
        self.assertEqual(count["timing_and_probe_oracle_comparisons"], 720)
        self.assertEqual(count["total_oracle_comparisons"], 728)
        self.assertEqual((probe.safe.WARMUP_ROUNDS, probe.safe.MEASURED_ROUNDS), (6, 24))

    def test_lazy_imports_no_execution_tools_or_helper_mutation(self):
        tree = ast.parse(Path(probe.__file__).read_text())
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module]
                self.assertFalse(any(name.split(".")[0] in {
                    "torch", "triton", "benchmark", "subprocess", "modal", "requests"} for name in names))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
                self.assertFalse(isinstance(node.value, ast.Name) and node.value.id == "safe")
        source = Path(probe.__file__).read_text()
        for disallowed in ("export_chrome_trace", "allow_tf32", "allow_bf16_reduced_precision", "beta0_kernel"):
            self.assertNotIn(disallowed, source)

    def test_embedded_model_matches_exact_frozen_official_receipt(self):
        path = probe.ROOT / probe.MODEL_RECEIPT
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), probe.MODEL_SHA256)
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 30)
        self.assertTrue(all(int(row["weight"]) == 62 for row in rows))
        for m in probe.TARGET_M:
            load = [row for row in rows if int(row["m"]) == m]
            qkv = next(row for row in load if row["layer"] == "qkv_gate")
            fraction = float(qkv["candidate_us"]) / sum(float(row["candidate_us"]) for row in load)
            self.assertAlmostEqual(probe.MODEL_QKV_FRACTIONS[m], fraction, places=15)

    def test_model_uses_six_load_geomean_and_only_five_qkv_changes(self):
        neutral = probe.modeled_sensitivity({m: 1 for m in probe.TARGET_M})
        self.assertEqual(neutral["hypothetical_relative_score_gain"], 0)
        ratios = {m: 1.25 for m in probe.TARGET_M}
        expected = math.prod(1 - fraction * .2 for fraction in probe.MODEL_QKV_FRACTIONS.values()) ** (-1 / 6) - 1
        result = probe.modeled_sensitivity(ratios)
        self.assertAlmostEqual(result["hypothetical_relative_score_gain"], expected)
        self.assertEqual((result["shape_weight"], result["equally_weighted_loads"]), (62, 6))
        self.assertTrue(result["other_25_scored_shapes_held_fixed"])
        self.assertIn("not an achieved", result["qualification"])
        for bad in ({8: 1}, {m: float("nan") for m in probe.TARGET_M},
                    {m: 0 for m in probe.TARGET_M}, {m: float("inf") for m in probe.TARGET_M}):
            with self.assertRaisesRegex(ValueError, "five positive finite"):
                probe.modeled_sensitivity(bad)

    def test_predeclared_gates_pass_only_with_complete_five_target_evidence(self):
        result = probe.assess(cases())
        self.assertTrue(result["screen_passed"])
        self.assertTrue(all(result["gates"].values()))
        self.assertEqual(result["target_median_winners"], list(probe.TARGET_M))
        self.assertEqual(result["target_winning_pairs"], {str(m): 12 for m in probe.TARGET_M})
        self.assertFalse(result["automatic_selection"])
        self.assertTrue(result["selected_parent_unchanged"])
        self.assertIn("M512", result["qualification"])
        for partial in (cases()[:2], cases()[:-1], list(reversed(cases()))):
            with self.assertRaisesRegex(ValueError, "nine predeclared"):
                probe.assess(partial)
        for location in ("case", "series"):
            data = cases()
            if location == "case":
                data[0]["passed"] = False
            else:
                data[0]["series"]["main"]["passed"] = False
            with self.assertRaisesRegex(ValueError, "Incomplete correctness"):
                probe.assess(data)

    def test_model_and_geomean_gates_are_not_replaced_by_single_fast_case(self):
        result = probe.assess(cases(1.05))
        self.assertTrue(result["gates"]["target_geomean_raw_cut_at_least4pct"])
        self.assertFalse(result["gates"]["hypothetical_score_sensitivity_gain_at_least1pct"])
        self.assertFalse(result["screen_passed"])
        data = cases(1)
        data[0]["series"]["main"]["samples"] = sample_rows(2)
        result = probe.assess(data)
        self.assertFalse(result["gates"]["at_least_four_target_median_wins"])
        self.assertFalse(result["screen_passed"])
        self.assertFalse(probe.assess(cases(1.03))["gates"]["target_geomean_raw_cut_at_least4pct"])

    def test_each_of_four_median_winners_needs_eight_winning_pairs(self):
        data = cases()
        # Seven winning pairs still produce a winning median, but miss the gate.
        for row in data[:2]:
            row["series"]["main"]["samples"] = sample_rows(pairs=[1.2] * 7 + [.99] * 5)
        result = probe.assess(data)
        self.assertEqual(len(result["target_median_winners"]), 5)
        self.assertEqual(len(result["target_median_winners_with_eight_pairs"]), 3)
        self.assertFalse(result["gates"]["at_least_four_median_winners_win_eight_of_twelve_pairs"])
        data[0]["series"]["main"]["samples"] = sample_rows(pairs=[1.2] * 8 + [.99] * 4)
        self.assertTrue(probe.assess(data)["screen_passed"])

    def test_individual_regression_gate_and_small_nonwin_allowed(self):
        data = cases(1.15)
        data[0]["series"]["main"]["samples"] = sample_rows(1 / 1.03)
        self.assertFalse(probe.assess(data)["gates"]["no_target_regresses_more_than2pct"])
        data[0]["series"]["main"]["samples"] = sample_rows(1 / 1.01)
        self.assertTrue(probe.assess(data)["screen_passed"])

    def test_nearby_improvement_allowed_regression_rejects_and_both_are_recorded(self):
        for index in (5, 6):
            data = cases()
            data[index]["series"]["main"]["samples"] = sample_rows(1.04)
            result = probe.assess(data)
            self.assertTrue(result["screen_passed"])
            self.assertEqual(result["nearby_or_control_flags_requiring_investigation"], [])
            self.assertEqual(len(result["nearby_raw_latency_movements"]), 2)
            movement = next(row for row in result["nearby_raw_latency_movements"] if row["m"] == probe.CASES[index][1])
            self.assertAlmostEqual(movement["raw_latency_movement"], 1 / 1.04 - 1)
            data[index]["series"]["main"]["samples"] = sample_rows(.97)
            result = probe.assess(data)
            self.assertFalse(result["screen_passed"])
            flags = result["nearby_or_control_flags_requiring_investigation"]
            self.assertEqual(len(flags), 1)
            self.assertEqual((flags[0]["layer"], flags[0]["m"]), probe.CASES[index][:2])
            self.assertGreater(flags[0]["raw_latency_movement"], .02)
            movement = next(row for row in result["nearby_raw_latency_movements"] if row["m"] == probe.CASES[index][1])
            self.assertAlmostEqual(movement["raw_latency_movement"], 1 / .97 - 1)

    def test_unchanged_control_movement_both_directions_prevents_clean_readiness(self):
        for index in (7, 8):
            for ratio in (.97, 1.04):
                data = cases()
                data[index]["series"]["main"]["samples"] = sample_rows(ratio)
                result = probe.assess(data)
                self.assertFalse(result["screen_passed"])
                flag = result["nearby_or_control_flags_requiring_investigation"]
                self.assertEqual(len(flag), 1)
                self.assertEqual((flag[0]["layer"], flag[0]["m"]), probe.CASES[index][:2])
                self.assertFalse(result["gates"]["no_nearby_regression_or_absolute_control_movement_over2pct"])

    def test_profile_availability_candidate_copy_and_any_case_fallback_reject(self):
        for index, row in enumerate(cases()):
            if probe.CASES[index][:2] in probe.PROFILE_CASES:
                for runner in probe.SOURCES:
                    data = cases()
                    data[index]["activities"][runner] = activity(available=False)
                    self.assertFalse(probe.assess(data)["screen_passed"])
                data = cases()
                data[index]["activities"]["candidate"] = activity("Memcpy DtoD", copy=True)
                self.assertFalse(probe.assess(data)["screen_passed"])
            data = cases()
            data[index]["capture"]["candidate"]["explicit_unfused_fallback_warning"] = True
            self.assertFalse(probe.assess(data)["screen_passed"])

    def test_assessment_recomputes_all_samples_and_rejects_partial_or_invalid_timings(self):
        data = cases()
        data[0]["series"]["main"]["summary"] = {"median_paired_raw_ratio": .01}
        self.assertTrue(probe.assess(data)["screen_passed"])
        data[0]["series"]["main"]["samples"].pop()
        with self.assertRaisesRegex(ValueError, "24 predefined"):
            probe.assess(data)
        for value in (0, -1, float("nan"), float("inf")):
            data = cases()
            data[0]["series"]["main"]["samples"][0]["us"]["candidate"] = value
            with self.assertRaisesRegex(ValueError, "Invalid raw timing"):
                probe.assess(data)

    def test_capture_pair_same_input_own_states_recognized_warning_and_no_alias(self):
        torch = Mock()
        modules = {name: Mock() for name in probe.SOURCES}
        states = {name: object() for name in probe.SOURCES}
        data, received = object(), []

        def capture(fn, actual_data, state):
            received.append((fn, actual_data, state))
            if len(received) == 2:
                warnings.warn("gemm_and_bias failed; using unfused path")
            output = Mock()
            output.data_ptr.return_value = len(received)
            return Mock(), output

        graphs, outputs, observations = probe.capture_pair(torch, capture, modules, data, states, Mock())
        self.assertEqual(set(graphs), set(probe.SOURCES))
        for index, name in enumerate(probe.SOURCES):
            self.assertIs(received[index][0], modules[name].kernel_fn)
            self.assertIs(received[index][1], data)
            self.assertIs(received[index][2], states[name])
        self.assertTrue(observations["candidate"]["explicit_unfused_fallback_warning"])
        output = Mock()
        output.data_ptr.return_value = 42
        with self.assertRaisesRegex(RuntimeError, "outputs unexpectedly alias"):
            probe.capture_pair(Mock(), Mock(return_value=(Mock(), output)), modules, data, states, Mock())

    def test_activity_names_and_durations_only_no_cpu_names_or_ids(self):
        prof = Mock()
        prof.events.return_value = [SimpleNamespace(name="private CPU label", device_type="cpu", device_time_total=9),
            SimpleNamespace(name="nvjet main", device_type="cuda", device_time_total=10, id=123),
            SimpleNamespace(name="Memcpy DtoD", device_type="cuda", device_time_total=2, id=456)]
        result = probe.activity_summary(prof, "cuda")
        self.assertTrue(result["available"])
        self.assertEqual(len(result["copy_events"]), 1)
        self.assertTrue(all(set(item) == {"symbol", "duration_us", "kind"} for item in result["cuda_events"]))
        self.assertNotIn("private", json.dumps(result))
        prof.events.return_value = []
        self.assertFalse(probe.activity_summary(prof, "cuda")["available"])
        for value in (-1, float("nan"), float("inf")):
            prof.events.return_value = [SimpleNamespace(name="main", device_type="cuda", device_time_total=value)]
            with self.assertRaisesRegex(ValueError, "activity duration"):
                probe.activity_summary(prof, "cuda")

    def test_series_oracle_order_all_samples_and_safe_pre_post_phases(self):
        torch, reference, timer = Mock(), Mock(), Mock()
        data = {"x": Mock()}
        graphs = outputs = {name: name for name in probe.SOURCES}
        log = []
        timer.measure.side_effect = lambda graph, *a: (log.append("measure:" + graph), 10 if graph == "candidate" else 11)[1]
        item = dict(correctness=[], warmup=[], samples=[])
        make = lambda *a: {"x": object()}
        check_states = Mock()
        with patch.object(probe.safe, "correctness_phase") as phase, \
                patch.object(probe.safe, "assert_input", side_effect=lambda *a: log.append("input")), \
                patch.object(probe.safe, "check_output", side_effect=lambda *a: log.append("oracle:" + a[3])):
            probe.run_series(torch, reference, data, graphs, outputs, {}, {}, (), make, 1, item,
                             timer, None, None, 0, Mock(), Mock(), check_states)
        self.assertEqual([call.args[10] for call in phase.call_args_list], ["before", "after"])
        self.assertEqual(check_states.call_count, 3)
        self.assertEqual((len(item["warmup"]), len(item["samples"])), (6, 24))
        self.assertEqual(len(item["summary"]["reverse_order_pair_ratios"]), 12)
        self.assertEqual(log[:6], ["measure:parent", "input", "measure:candidate", "input", "oracle:parent", "oracle:candidate"])
        self.assertEqual(log[6:12], ["measure:candidate", "input", "measure:parent", "input", "oracle:candidate", "oracle:parent"])
        self.assertTrue(all(len(row["correctness"]) == 2 for row in item["samples"] + item["warmup"]))
        self.assertTrue(item["passed"])

    def test_partial_timing_failure_keeps_all_completed_and_started_observations(self):
        torch, timer = Mock(), Mock()
        timer.measure.side_effect = [11, RuntimeError("device failure")]
        item = dict(passed=False, correctness=[], warmup=[], samples=[])
        with patch.object(probe.safe, "correctness_phase"), patch.object(probe.safe, "assert_input"):
            with self.assertRaisesRegex(RuntimeError, "device failure"):
                probe.run_series(torch, Mock(), {"x": Mock()}, {name: name for name in probe.SOURCES}, {},
                    {}, {}, (), lambda *a: {"x": object()}, 1, item, timer, None, None, 0,
                    Mock(), Mock(), Mock())
        self.assertFalse(item["passed"])
        sample = item["warmup"][0]
        self.assertEqual(sample["us"], {"parent": 11})
        self.assertEqual([row["status"] for row in sample["observations"]], ["measured", "started"])
        self.assertNotIn("summary", item)

    def test_all_timing_precedes_activity_and_frozen_helpers_are_used(self):
        tree = ast.parse(Path(probe.__file__).read_text())
        execute = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "execute")
        timing_lines = [node.lineno for node in ast.walk(execute) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == "run_series"]
        profile_lines = [node.lineno for node in ast.walk(execute) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and node.func.attr == "profile"]
        self.assertLess(max(timing_lines), min(profile_lines))
        helper_calls = {node.func.attr for node in ast.walk(execute) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "safe"}
        self.assertTrue({"ColdGraphTimer", "fresh_input", "freeze_tensors", "assert_frozen",
                         "check_output", "require_h200"}.issubset(helper_calls))

    def test_hash_failure_prevents_execute_and_preserves_failure_report(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "timing.json"
            with patch.object(probe, "verify_sources", side_effect=ValueError("Frozen source hash mismatch")), \
                    patch.object(probe, "execute") as execute, patch("builtins.print"):
                self.assertEqual(probe.main(["--output", str(output), "--outer-timeout-seconds", "180"]), 1)
            execute.assert_not_called()
            report = json.loads(output.read_text())
            self.assertFalse(report["passed"])
            self.assertEqual(report["failure"]["type"], "ValueError")
            self.assertEqual(report["protocol"]["counts"]["total_oracle_comparisons"], 728)

    def test_main_preserves_partial_samples_and_separates_execution_from_screen_pass(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "timing.json"

            def fail(args, report, save):
                report["cases"].append(dict(layer="qkv_gate", m=8, passed=False,
                    series={"main": {"samples": [{"round": 0, "us": {"parent": 12}}]}}))
                save()
                raise TimeoutError("bounded stop")

            with patch.object(probe, "execute", side_effect=fail), patch("builtins.print"):
                self.assertEqual(probe.main(["--output", str(output), "--outer-timeout-seconds", "180"]), 1)
            report = json.loads(output.read_text())
            self.assertEqual(report["cases"][0]["series"]["main"]["samples"][0]["us"], {"parent": 12})
            self.assertNotIn("assessment", report)

            def completed(args, report, save):
                report.update(passed=True, cases=cases(1), assessment=probe.assess(cases(1)))

            with patch.object(probe, "execute", side_effect=completed), patch("builtins.print"):
                self.assertEqual(probe.main(["--output", str(output), "--outer-timeout-seconds", "180"]), 0)
            report = json.loads(output.read_text())
            self.assertTrue(report["passed"])
            self.assertFalse(report["assessment"]["screen_passed"])

    def test_internal_and_supervisor_timeouts_are_bounded(self):
        for internal, outer in ((121, 180), (0, 180), (120, 129), (120, 181)):
            with self.assertRaises(ValueError):
                probe.safe.validate_options(SimpleNamespace(deadline_seconds=internal, outer_timeout_seconds=outer))
        probe.safe.validate_options(SimpleNamespace(deadline_seconds=120, outer_timeout_seconds=180))


if __name__ == "__main__":
    unittest.main()
