"""Local mock/static tests only; no Torch import, native execution or network."""
import ast
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import projection_p049_screen as probe


def sample_rows(ratio=1.1):
    return [dict(round=i, order=list(probe.safe.runner_order(i)),
                 us=dict(parent=11 * ratio, candidate=11)) for i in range(24)]


def activity(symbol="main", duration=10, copy=False, available=True):
    events = [dict(symbol=symbol, duration_us=duration, kind="copy" if copy else "kernel_or_other")] if available else []
    return dict(available=available, cuda_events=events, copy_events=events if copy else [])


def cases(ratios=(1.1, 1.1)):
    return [dict(layer=layer, m=m, passed=True,
                 series={"main": {"summary": probe.safe.summarize_samples(sample_rows(ratio))}},
                 capture={"candidate": {"explicit_unfused_fallback_warning": False}},
                 activities={"parent": activity("original"), "candidate": activity("bias"),
                             "beta0_control": activity("original")})
            for (layer, m), ratio in zip(probe.CASES, ratios)]


class ScreenTests(unittest.TestCase):
    def test_exact_sources_and_helper_hash(self):
        result = probe.verify_sources()
        self.assertEqual(result["projection_p046_confirmation.py"], probe.SAFE_SHA256)
        for path, digest in probe.SOURCES.values():
            self.assertEqual(result[path], digest)

    def test_two_cases_and_series_are_frozen(self):
        self.assertEqual(probe.CASES, (("qkv_gate", 128), ("unpooler", 16)))
        self.assertEqual(probe.SERIES, (("main", "candidate"), ("negative_control", "beta0_control")))
        self.assertEqual(probe.safe.WARMUP_ROUNDS, 6)
        self.assertEqual(probe.safe.MEASURED_ROUNDS, 24)

    def test_imports_are_lazy_and_no_execution_tools_present(self):
        tree = ast.parse(Path(probe.__file__).read_text())
        top_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        for node in top_imports:
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module]
            self.assertFalse(any(name.split(".")[0] in {"torch", "triton", "benchmark", "subprocess", "modal", "requests"}
                                 for name in names))
        self.assertNotIn("export_chrome_trace", Path(probe.__file__).read_text())

    def test_hash_failure_prevents_execution_and_is_reported(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "timing.json"
            with patch.object(probe, "verify_sources", side_effect=ValueError("Frozen source hash mismatch")), \
                    patch.object(probe, "execute") as execute, patch("builtins.print"):
                self.assertEqual(probe.main(["--output", str(output), "--outer-timeout-seconds", "180"]), 1)
            execute.assert_not_called()
            report = json.loads(output.read_text())
            self.assertFalse(report["passed"])
            self.assertEqual(report["failure"]["type"], "ValueError")

    def test_partial_failure_preserves_samples_and_separates_screen_status(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "timing.json"

            def fail(args, report, save):
                report["cases"].append(dict(layer="qkv_gate", m=128, passed=False,
                    series={"main": {"samples": [{"round": 0, "us": {"parent": 12}}]}}))
                save()
                raise TimeoutError("bounded stop")

            with patch.object(probe, "execute", side_effect=fail), patch("builtins.print"):
                self.assertEqual(probe.main(["--output", str(output), "--outer-timeout-seconds", "180"]), 1)
            report = json.loads(output.read_text())
            self.assertEqual(report["cases"][0]["series"]["main"]["samples"][0]["us"], {"parent": 12})
            self.assertNotIn("assessment", report)

    def test_beta0_uses_same_operands_and_ignored_self_aliases_fresh_output(self):
        torch, x, weight, transposed, output = Mock(), Mock(), Mock(), object(), object()
        x.shape, x.dtype, x.device = (128, 8192), "bf16", "cuda"
        weight.__getitem__ = Mock(return_value=SimpleNamespace(T=transposed))
        torch.empty.return_value = output
        torch.addmm.return_value = output
        data = dict(x=x, config=dict(layer="qkv_gate", n=2304))
        self.assertIs(probe.beta0_kernel(torch, data, {"qkv_gate": weight}), output)
        weight.__getitem__.assert_called_once_with(slice(None, 2304))
        torch.addmm.assert_called_once_with(output, x, transposed, beta=0, alpha=1, out=output)
        self.assertEqual(torch.empty.call_args.kwargs, dict(dtype="bf16", device="cuda"))
        x.shape = (1, 8192)
        with self.assertRaisesRegex(ValueError, "two diagnostic cases"):
            probe.beta0_kernel(torch, data, {"qkv_gate": weight})

    def test_capture_three_outputs_same_input_and_negative_control_uses_parent_state(self):
        torch = Mock()
        modules = {name: Mock() for name in probe.SOURCES}
        states = {name: object() for name in probe.SOURCES}
        data = object()
        received = []

        def capture(fn, actual_data, state):
            received.append((actual_data, state))
            output = Mock()
            output.data_ptr.return_value = len(received)
            return Mock(), output

        graphs, outputs, durations = probe.capture_runners(torch, capture, modules, data, states, Mock())
        self.assertEqual(set(graphs), set(probe.RUNNERS))
        self.assertTrue(all(value[0] is data for value in received))
        self.assertIs(received[2][1], states["parent"])
        self.assertTrue(all(not row["explicit_unfused_fallback_warning"] for row in durations.values()))

    def test_alias_output_rejected(self):
        output = Mock()
        output.data_ptr.return_value = 42
        with self.assertRaisesRegex(RuntimeError, "outputs unexpectedly alias"):
            probe.capture_runners(Mock(), Mock(return_value=(Mock(), output)),
                {name: Mock() for name in probe.SOURCES}, {}, {name: {} for name in probe.SOURCES}, Mock())

    def test_profile_allowlist_omits_ids_and_cpu_names_and_retains_copies(self):
        prof = Mock()
        prof.events.return_value = [SimpleNamespace(name="cpu secret", device_type="cpu", device_time_total=9),
            SimpleNamespace(name="nvjet main", device_type="cuda", device_time_total=10, device_index=99, id=123),
            SimpleNamespace(name="Memcpy DtoD", device_type="cuda", device_time_total=2, id=456)]
        result = probe.activity_summary(prof, "cuda")
        self.assertTrue(result["available"])
        self.assertEqual([item["symbol"] for item in result["cuda_events"]], ["nvjet main", "Memcpy DtoD"])
        self.assertEqual(len(result["copy_events"]), 1)
        self.assertTrue(all(set(item) == {"symbol", "duration_us", "kind"} for item in result["cuda_events"]))
        self.assertNotIn("secret", json.dumps(result))

    def test_empty_profile_is_unavailable_invalid_duration_rejected(self):
        prof = Mock()
        prof.events.return_value = []
        self.assertFalse(probe.activity_summary(prof, "cuda")["available"])
        prof.events.return_value = [SimpleNamespace(name="main", device_type="cuda", device_time_total=float("nan"))]
        with self.assertRaisesRegex(ValueError, "activity duration"):
            probe.activity_summary(prof, "cuda")

    def test_gates_and_negative_control_do_not_promote(self):
        data = cases()
        result = probe.assess(data)
        self.assertTrue(result["screen_passed"])
        self.assertAlmostEqual(result["target_geomean_raw_latency_cut"], 1 - 1 / 1.1)
        self.assertFalse(result["automatic_selection"])
        self.assertFalse(result["negative_control_used_for_selection"])
        data[0]["series"]["negative_control"] = {"summary": {"median_paired_raw_ratio": .001}}
        self.assertEqual(probe.assess(data), result)
        data[0]["activities"]["beta0_control"] = activity(available=False)
        no_control_trace = probe.assess(data)
        self.assertTrue(no_control_trace["screen_passed"])
        self.assertFalse(no_control_trace["negative_control_activity_available"])
        self.assertFalse(probe.assess(cases((1.04, 1.04)))["screen_passed"])
        self.assertFalse(probe.assess(cases((1.3, .97)))["screen_passed"])

    def test_empty_profile_copy_fallback_and_identical_nonwin_reject(self):
        data = cases()
        data[0]["activities"]["candidate"] = activity(available=False)
        self.assertFalse(probe.assess(data)["screen_passed"])
        data = cases()
        data[0]["activities"]["candidate"] = activity("Memcpy DtoD", copy=True)
        self.assertFalse(probe.assess(data)["screen_passed"])
        data = cases()
        data[0]["capture"]["candidate"]["explicit_unfused_fallback_warning"] = True
        self.assertFalse(probe.assess(data)["screen_passed"])
        data = cases((1, 1.3))
        data[0]["activities"]["candidate"] = activity("original")
        result = probe.assess(data)
        self.assertTrue(result["identical_kernel_without_benefit"])
        self.assertFalse(result["screen_passed"])

    def test_missing_case_and_correctness_failure_cannot_pass(self):
        with self.assertRaisesRegex(ValueError, "two predeclared"):
            probe.assess(cases()[:1])
        data = cases()
        data[0]["passed"] = False
        with self.assertRaisesRegex(ValueError, "Incomplete correctness"):
            probe.assess(data)

    def test_series_checks_both_outputs_after_both_replays_and_keeps_all_samples(self):
        torch, reference, timer = Mock(), Mock(), Mock()
        data = {"x": Mock()}
        graphs, outputs = {name: name for name in ("parent", "candidate")}, {name: name for name in ("parent", "candidate")}
        log = []
        timer.measure.side_effect = lambda graph, *a: (log.append("measure:" + graph), 10 if graph == "candidate" else 11)[1]
        item = dict(correctness=[], warmup=[], samples=[])
        make = lambda *a: {"x": object()}
        with patch.object(probe.safe, "correctness_phase") as phase, \
                patch.object(probe.safe, "assert_input", side_effect=lambda *a: log.append("input")), \
                patch.object(probe.safe, "check_output", side_effect=lambda *a: log.append("oracle:" + a[3])):
            probe.run_series(torch, reference, data, graphs, outputs, {}, {}, (), make, 1, item,
                             timer, None, None, 0, Mock(), Mock(), Mock())
        self.assertEqual(phase.call_count, 2)
        self.assertEqual(len(item["warmup"]), 6)
        self.assertEqual(len(item["samples"]), 24)
        self.assertEqual(len(item["summary"]["reverse_order_pair_ratios"]), 12)
        self.assertEqual(log[:6], ["measure:parent", "input", "measure:candidate", "input", "oracle:parent", "oracle:candidate"])
        self.assertEqual(log[6:12], ["measure:candidate", "input", "measure:parent", "input", "oracle:candidate", "oracle:parent"])
        self.assertTrue(all(len(s["correctness"]) == 2 for s in item["samples"] + item["warmup"]))
        self.assertTrue(item["passed"])

    def test_activity_collection_follows_all_paired_timing(self):
        tree = ast.parse(Path(probe.__file__).read_text())
        execute = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "execute")
        timing_lines = [node.lineno for node in ast.walk(execute) if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name) and node.func.id == "run_series"]
        profile_lines = [node.lineno for node in ast.walk(execute) if isinstance(node, ast.Call)
                         and isinstance(node.func, ast.Attribute) and node.func.attr == "profile"]
        self.assertLess(max(timing_lines), min(profile_lines))


if __name__ == "__main__":
    unittest.main()
