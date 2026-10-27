"""Mock-only P053 protocol and phase attribution checks; no GPU or SDK calls."""
import ast
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import projection_p053_phase_screen as probe


def samples(comparison="partial", ratio=1.2):
    return [dict(round=i, order=list(probe.safe.runner_order(i)),
                 us=dict(parent=12 * ratio, candidate=12), comparison=comparison,
                 comparison_sequence_index=probe.comparison_order(i).index(comparison),
                 excluded_warmup=False, correctness=dict(parent=True, candidate=True))
            for i in range(24)]


def cases(partial=1.2, whole=1.02, selected_partial=1.02, selected_whole=.9, wins=10):
    ratios = dict(partial=partial, whole=whole, selected_partial=selected_partial,
                  selected_whole=selected_whole)
    return [dict(m=m, passed=True, series={name: dict(summary=dict(
        median_paired_raw_ratio=ratio, winning_pairs=wins, total_pairs=12))
        for name, ratio in ratios.items()}) for m in (8, 7)]


def timing_record():
    record = dict(mode="timing", complete=True, oracle_checks=580,
        sources_sha256={"fixture": "sha256"}, inherited_p052={"sha256": probe.INHERITED_SHA},
        environment={"gpu": "fixture H200"}, cases=[])
    for m in probe.CASES:
        checks = [dict(phase=phase, label=label, runner=runner, passed=True)
                  for phase in ("before", "after") for label in probe.LABELS for runner in probe.RUNNERS]
        series = {}
        for name in probe.COMPARISONS:
            warmup = samples(name)[:6]
            for row in warmup:
                row["excluded_warmup"] = True
            series[name] = dict(warmup=warmup, samples=samples(name))
        record["cases"].append(dict(m=m, passed=True, correctness=checks, series=series))
    return record


def cuda_event(name, start=10, duration=4, category="kernel"):
    return dict(cat=category, ph="X", name=name, ts=start, dur=duration)


class PhaseScreenTests(unittest.TestCase):
    def test_frozen_sources_and_inherited_execution_receipt(self):
        hashes = probe.verify_sources()
        for name, digest in probe.FROZEN.items():
            self.assertEqual(hashes[name], digest)
        self.assertEqual(probe.FROZEN["projection_p052_orientation_screen.py"], probe.P052_SHA)
        inherited = probe.inherited_receipt()
        self.assertEqual(inherited["sha256"], probe.INHERITED_SHA)
        self.assertEqual(inherited["correctness_receipt_sha256"], probe.CORRECTNESS_SHA)
        self.assertEqual(inherited["prior_original_oracle_checks"], 390)
        self.assertFalse(inherited["prior_screen_passed"])

    def test_changed_inherited_receipt_is_rejected_before_parsing(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / probe.INHERITED_PATH
            path.parent.mkdir(parents=True)
            path.write_text("changed receipt")
            with self.assertRaisesRegex(ValueError, "Exact frozen P052"):
                probe.inherited_receipt(Path(folder))

    def test_counts_include_every_replay_and_untimed_partial_validation(self):
        count = probe.protocol_counts()
        self.assertEqual(count["measured_replays"], 2 * 4 * 24 * 2)
        self.assertEqual(count["warmup_replays"], 2 * 4 * 6 * 2)
        self.assertEqual(count["prepost_replays"], 2 * 2 * 5 * 5)
        self.assertEqual(count["oracle_checks"], sum(count[name] for name in
            ("measured_replays", "warmup_replays", "prepost_replays")))
        self.assertEqual(count["untimed_partial_validator_reducers"], 2 * (30 * 3 + 2 * 5 * 2))
        self.assertEqual(count["capture_calls"], 2 * 5)
        self.assertEqual(count["untimed_capture_warmups"], 3 * count["capture_calls"])
        self.assertEqual(count["activity_replays"], 2 * 3 * 3)
        self.assertEqual(count["activity_profiles"], count["activity_replays"])
        self.assertEqual(count["activity_capture_calls"], 2 * 3)
        self.assertEqual(count["activity_untimed_capture_warmups"], 3 * count["activity_capture_calls"])

    def test_exact_cases_graphs_and_comparisons(self):
        self.assertEqual(probe.CASES, (8, 7))
        self.assertEqual(probe.RUNNERS, ("control_whole", "weight_first_whole",
            "control_partial", "weight_first_partial", "p046_whole"))
        self.assertEqual(probe.COMPARISONS, {
            "partial": ("control_partial", "weight_first_partial"),
            "whole": ("control_whole", "weight_first_whole"),
            "selected_partial": ("p046_whole", "weight_first_partial"),
            "selected_whole": ("p046_whole", "weight_first_whole")})

    def test_rotation_balances_positions_and_reverses_each_pair(self):
        names = tuple(probe.COMPARISONS)
        positions = {name: Counter() for name in names}
        for i in range(24):
            actual = tuple(probe.comparison_order(i))
            shift = (i // 2) % len(names)
            expected = names[shift:] + names[:shift]
            self.assertEqual(actual, expected[::-1] if i % 2 else expected)
            self.assertEqual(set(actual), set(names))
            for position, name in enumerate(actual):
                positions[name][position] += 1
            if i % 2:
                self.assertEqual(actual, tuple(probe.comparison_order(i - 1))[::-1])
        for counts in positions.values():
            self.assertEqual(counts, Counter({0: 6, 1: 6, 2: 6, 3: 6}))

    def test_summary_uses_all_twelve_fixed_pairs(self):
        raw = samples()
        for i, sample in enumerate(raw):
            sample["us"]["parent"] *= 2 if i % 2 == 0 else .5
        result = probe.summarize(raw, "partial")
        self.assertEqual(result["total_pairs"], 12)
        self.assertEqual(result["winning_pairs"], 12)
        self.assertAlmostEqual(result["median_paired_raw_ratio"], 1.2)
        self.assertEqual(result["raw_us"]["parent"]["count"], 24)

    def test_incomplete_reordered_or_unchecked_samples_rejected(self):
        bad = [samples()[:-1]]
        for key, value in (("comparison_sequence_index", 1), ("excluded_warmup", True),
                           ("correctness", {"parent": True}), ("order", ["candidate", "parent"]),
                           ("comparison", "whole")):
            raw = samples()
            raw[0][key] = value
            bad.append(raw)
        for raw in bad:
            with self.subTest(raw=raw[0]):
                with self.assertRaises(ValueError):
                    probe.summarize(raw, "partial")

    def test_pair_measurements_precede_every_partial_validator(self):
        for comparison, runners in probe.COMPARISONS.items():
            for i in (0, 1):
                events = []
                graphs = {name: name + " graph" for name in probe.RUNNERS}
                mapping = dict(zip(("parent", "candidate"), runners))
                sample = dict(round=i, order=list(probe.safe.runner_order(i)), us={},
                              correctness={}, observations=[])
                timer = SimpleNamespace(measure=lambda graph, *args: events.append(("measure", graph)) or 12)
                def validate(*args):
                    events.append(("validate", args))
                probe.measure_pair(timer, graphs, mapping, lambda *args: events.append(("input", args)),
                    validate, sample, "eviction", "eviction_out", lambda: None, lambda: None)
                measured = [event[1] for event in events if event[0] == "measure"]
                self.assertEqual(measured, [graphs[mapping[role]] for role in sample["order"]])
                first_validation = next(j for j, event in enumerate(events) if event[0] == "validate")
                self.assertEqual(sum(event[0] == "measure" for event in events[:first_validation]), 2)
                self.assertEqual(sum(event[0] == "validate" for event in events), 2)
                self.assertEqual(sample["us"], dict(parent=12, candidate=12))
                self.assertEqual(sample["correctness"], dict(parent=True, candidate=True))

    def test_second_measurement_failure_retains_first_and_skips_validation(self):
        sample = dict(round=0, order=["parent", "candidate"], us={}, correctness={}, observations=[])
        timer = SimpleNamespace(measure=Mock(side_effect=[12, TimeoutError("bounded")]))
        validate, save = Mock(), Mock()
        with self.assertRaises(TimeoutError):
            probe.measure_pair(timer, {"control_partial": "control", "weight_first_partial": "weight"},
                {"parent": "control_partial", "candidate": "weight_first_partial"}, Mock(), validate,
                sample, None, None, lambda: None, save)
        self.assertEqual(sample["us"], {"parent": 12})
        self.assertEqual(sample["correctness"], {})
        validate.assert_not_called()
        self.assertTrue(save.called)

    def test_validator_failure_preserves_both_latencies_and_prior_check(self):
        sample = dict(round=0, order=["parent", "candidate"], us={}, correctness={}, observations=[])
        validate = Mock(side_effect=[None, RuntimeError("oracle failed")])
        with self.assertRaisesRegex(RuntimeError, "oracle failed"):
            probe.measure_pair(SimpleNamespace(measure=Mock(side_effect=[12, 10])),
                {"a": "graph a", "b": "graph b"}, {"parent": "a", "candidate": "b"},
                Mock(), validate, sample, None, None, lambda: None, Mock())
        self.assertEqual(sample["us"], {"parent": 12, "candidate": 10})
        self.assertEqual(sample["correctness"], {"parent": True})
        self.assertEqual([row["status"] for row in sample["observations"]], ["measured", "measured"])

    def test_original_kernel_clone_annotations_and_reducer_are_reused(self):
        language = object()
        old = dict(vars(probe.orientation))
        kernels = probe.prior.build_kernels(SimpleNamespace(jit=lambda fn: fn), language)
        for key, name in (("control", "_partial"), ("weight_first", "_partial_weight_first"),
                          ("reduce", "_reduce")):
            fn, original = kernels[key], getattr(probe.orientation, name)
            self.assertIs(fn.__code__, original.__code__)
            self.assertEqual(fn.__annotations__, original.__annotations__)
            self.assertTrue(fn.__annotations__)
            self.assertEqual(fn.__defaults__, original.__defaults__)
            self.assertEqual(fn.__kwdefaults__, original.__kwdefaults__)
            self.assertIs(fn.__globals__["tl"], language)
        self.assertEqual(vars(probe.orientation), old)

    def test_partial_call_owns_fresh_fp32_scratch_and_has_no_reducer(self):
        torch = Mock()
        torch.float32 = "fp32"
        torch.empty.side_effect = [object(), object()]
        class Kernel:
            def __init__(self):
                self.grids, self.call = [], Mock()
            def __getitem__(self, grid):
                self.grids.append(grid)
                return self.call
        kernels = {name: Kernel() for name in ("control", "weight_first", "reduce")}
        outputs = []
        for m, arm in ((8, "control"), (7, "weight_first")):
            x = SimpleNamespace(shape=(m, 8192), device="cuda")
            outputs.append(probe.partial_call(torch, kernels, arm,
                {"x": x, "config": {"n": 4096}}, {"pooler": "W"}))
            allocation = torch.empty.call_args
            self.assertEqual(allocation.args, ((8, m, 4096),))
            self.assertEqual(allocation.kwargs, {"dtype": "fp32", "device": "cuda"})
            self.assertEqual(kernels[arm].grids, [(64, 8)])
            self.assertEqual(kernels[arm].call.call_args.args, (x, "W", outputs[-1], m, 4096, 8192, 8, 64, 64))
            self.assertEqual(kernels[arm].call.call_args.kwargs, probe.orientation.PARTIAL_LAUNCH)
        self.assertIsNot(outputs[0], outputs[1])
        kernels["reduce"].call.assert_not_called()

    def test_dilution_gate_uses_latency_cut_pair_wins_and_whole_comparison(self):
        result = probe.assess(cases(partial=1 / .9, whole=1 / .950001))
        self.assertTrue(all(row["dilution_followup_warranted"] for row in result["cases"]))
        for values in (dict(partial=1.1), dict(partial=1 / .900001), dict(wins=9),
                       dict(whole=1 / .949999)):
            with self.subTest(values=values):
                result = probe.assess(cases(**values))
                self.assertFalse(any(row["dilution_followup_warranted"] for row in result["cases"]))
        data = cases()
        data[0]["series"]["partial"]["summary"]["total_pairs"] = 11
        result = probe.assess(data)
        self.assertFalse(result["cases"][0]["dilution_followup_warranted"])
        self.assertTrue(result["cases"][1]["dilution_followup_warranted"])

    def test_partial_rescue_question_and_shapes_are_kept_separate(self):
        for ratio, expected in ((1 / .950001, True), (1 / .95, False), (1 / .949999, False)):
            result = probe.assess(cases(selected_partial=ratio))
            self.assertEqual([row["no_reducer_only_rescue_lead"] for row in result["cases"]], [expected] * 2)
        data = cases()
        data[1]["series"]["selected_partial"]["summary"]["median_paired_raw_ratio"] = 2
        result = probe.assess(data)
        self.assertTrue(result["cases"][0]["no_reducer_only_rescue_lead"])
        self.assertFalse(result["cases"][1]["no_reducer_only_rescue_lead"])
        self.assertFalse(result["automatic_selection"])
        self.assertIsNone(result["official_score"])
        self.assertAlmostEqual(result["cases"][1]["raw_latency_cuts"]["selected_partial"], .5)
        for bad in (data[:1], data[::-1], [dict(data[0], passed=False), data[1]]):
            with self.assertRaises(ValueError):
                probe.assess(bad)

    def test_reducer_followup_requires_both_shapes_five_percent_and_ten_of_twelve_pairs(self):
        good = cases(selected_partial=1 / .95)
        result = probe.assess(good)
        self.assertTrue(result["reducer_only_followup_warranted"])
        for row in result["cases"]:
            self.assertEqual(row["selected_partial_winning_pairs"], 10)
            self.assertEqual(row["selected_partial_total_pairs"], 12)
            self.assertEqual(row["reducer_only_case_gates"], {
                "partial_at_least5pct_faster_than_p046_whole": True,
                "at_least10_of12_pairs_win": True})
        for key, value in (("median_paired_raw_ratio", 1 / .950001), ("winning_pairs", 9),
                           ("total_pairs", 11)):
            data = copy.deepcopy(good)
            data[1]["series"]["selected_partial"]["summary"][key] = value
            result = probe.assess(data)
            self.assertFalse(result["reducer_only_followup_warranted"])
            self.assertFalse(result["cases"][0]["no_reducer_only_rescue_lead"])
            self.assertTrue(result["cases"][1]["no_reducer_only_rescue_lead"])

    def test_activity_requires_complete_580_check_source_bound_timing(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "timing.json"
            record = timing_record()
            path.write_text(json.dumps(record))
            result = probe.validate_timing_receipt(path, record["sources_sha256"])
            self.assertEqual(result["oracle_checks"], 580)
            self.assertEqual(result["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(result["environment"], record["environment"])

    def test_timing_receipt_rejects_missing_failed_duplicate_and_reordered_evidence(self):
        def first_series(record):
            return record["cases"][0]["series"]["partial"]
        mutations = [lambda r: r.update(mode="activity"), lambda r: r.update(complete=False),
            lambda r: r.update(failure={"type": "TimeoutError"}), lambda r: r.update(oracle_checks=579),
            lambda r: r.update(sources_sha256={"fixture": "wrong"}),
            lambda r: r["inherited_p052"].update(sha256="wrong"), lambda r: r["cases"].reverse(),
            lambda r: r["cases"][0].update(passed=False),
            lambda r: r["cases"][0]["correctness"].pop(),
            lambda r: r["cases"][0]["correctness"].__setitem__(0, r["cases"][0]["correctness"][1]),
            lambda r: r["cases"][0]["correctness"][0].update(passed=False),
            lambda r: r["cases"][0]["series"].pop("whole"),
            lambda r: first_series(r)["samples"].pop(),
            lambda r: first_series(r)["samples"][0].update(correctness={"parent": True}),
            lambda r: first_series(r)["warmup"].pop(),
            lambda r: first_series(r)["warmup"].reverse(),
            lambda r: first_series(r)["warmup"][0].update(excluded_warmup=False),
            lambda r: first_series(r)["warmup"][0].update(comparison_sequence_index=1),
            lambda r: first_series(r)["warmup"][0].update(correctness={"parent": True}),
            lambda r: first_series(r)["warmup"][0]["us"].update(candidate=float("nan"))]
        with TemporaryDirectory() as folder:
            path = Path(folder) / "timing.json"
            good = timing_record()
            for index, mutate in enumerate(mutations):
                record = copy.deepcopy(good)
                mutate(record)
                path.write_text(json.dumps(record))
                with self.subTest(mutation=index), self.assertRaises(ValueError):
                    probe.validate_timing_receipt(path, good["sources_sha256"])

    def test_trace_names_are_fixed_and_replay_specific(self):
        names = set()
        for m in (8, 7):
            for runner in ("control", "weight_first", "p046"):
                for repeat in range(3):
                    expected = f"activity.pooler_m{m}.{runner}.repeat{repeat}.trace.json"
                    self.assertEqual(probe.trace_name(m, runner, repeat), expected)
                    names.add(expected)
        self.assertEqual(len(names), 18)
        self.assertFalse(any("/" in name or ".." in name for name in names))
        for identity in ((16, "control", 0), (8, "control_partial", 0), (8, "p046", 3)):
            with self.assertRaises(ValueError):
                probe.trace_name(*identity)

    def test_trace_uses_sorted_gpu_intervals_and_signed_gaps_only(self):
        events = [cuda_event("_reduce", 13, 2), cuda_event("_partial", 10, 4),
                  cuda_event("copy", 1, 1, "gpu_memcpy"), cuda_event("clear", 3, 1, "gpu_memset"),
                  cuda_event("_partial CPU annotation", 0, 100, "cpu_op")]
        with TemporaryDirectory() as folder:
            path = Path(folder) / "trace.json"
            path.write_text(json.dumps({"traceEvents": events}))
            result = probe.trace_intervals(path, "control")
        self.assertEqual([row["role"] for row in result["kernel_intervals"]], ["partial", "reducer"])
        self.assertEqual([row["duration_us"] for row in result["kernel_intervals"]], [4, 2])
        self.assertEqual(result["inter_kernel_gaps_us"], [-1])
        self.assertEqual(len(result["cuda_copy_intervals"]), 2)

    def test_orientation_trace_rejects_missing_extra_wrong_or_invalid_kernels(self):
        variants = [[], [cuda_event("_partial")], [cuda_event("_partial_weight_first"), cuda_event("_reduce", 20)],
            [cuda_event("_partial"), cuda_event("_reduce", 20), cuda_event("library", 30)],
            [cuda_event("_partial", duration=-1), cuda_event("_reduce", 20)],
            [cuda_event("_partial", start=float("inf")), cuda_event("_reduce", 20)],
            [cuda_event("_reduce", 10), cuda_event("_partial", 20)]]
        with TemporaryDirectory() as folder:
            path = Path(folder) / "trace.json"
            for index, events in enumerate(variants):
                path.write_text(json.dumps({"traceEvents": events}))
                with self.subTest(variant=index), self.assertRaises(ValueError):
                    probe.trace_intervals(path, "control")
            path.write_text(json.dumps({"traceEvents": [cuda_event("_partial_weight_first"), cuda_event("_reduce", 20)]}))
            self.assertEqual(len(probe.trace_intervals(path, "weight_first")["kernel_intervals"]), 2)
            path.write_text(json.dumps({"traceEvents": [cuda_event("library_bmm")]}))
            self.assertEqual(probe.trace_intervals(path, "p046")["kernel_intervals"][0]["role"], "library_or_other")

    def test_source_or_receipt_failure_never_enters_native_stage(self):
        with TemporaryDirectory() as folder:
            output = Path(folder) / "result.json"
            for target in ("verify_sources", "inherited_receipt", "validate_timing_receipt"):
                with patch.object(probe, "verify_sources", return_value={}), \
                     patch.object(probe, "inherited_receipt", return_value={}), \
                     patch.object(probe, "validate_timing_receipt", return_value={}), \
                     patch.object(probe, target, side_effect=ValueError("rejected")), \
                     patch.object(probe, "execute") as execute, patch("builtins.print"):
                    code = probe.main(["--mode", "activity", "--output", str(output),
                        "--timing-receipt", "unused", "--outer-timeout-seconds", "150"])
                self.assertEqual(code, 1)
                execute.assert_not_called()
                record = json.loads(output.read_text())
                self.assertFalse(record["complete"])
                self.assertEqual(record["failure"]["type"], "ValueError")

    def test_timeout_retains_observed_sample_and_raw_trace_hash(self):
        with TemporaryDirectory() as folder:
            output = Path(folder) / "timing.json"
            observed = {"m": 8, "passed": False, "series": {"partial": {"samples": [{"us": {"parent": 12}}]}}}
            profile = {"trace_file": "observed.trace.json", "trace_sha256": "fixture", "passed": False}
            def fail(args, report, save):
                report["cases"].append(observed)
                report["profiles"].append(profile)
                save()
                raise TimeoutError("bounded")
            with patch.object(probe, "verify_sources", return_value={}), \
                 patch.object(probe, "inherited_receipt", return_value={}), \
                 patch.object(probe, "execute", side_effect=fail), patch("builtins.print"):
                code = probe.main(["--mode", "timing", "--output", str(output), "--outer-timeout-seconds", "150"])
            self.assertEqual(code, 1)
            record = json.loads(output.read_text())
            self.assertEqual(record["cases"], [observed])
            self.assertEqual(record["profiles"], [profile])
            self.assertFalse(record["complete"])
            self.assertEqual(record["failure"]["type"], "TimeoutError")
            self.assertFalse(output.with_suffix(".json.tmp").exists())

    def test_invalid_deadlines_and_missing_activity_receipt_fail_before_execution(self):
        with TemporaryDirectory() as folder, patch.object(probe, "execute") as execute:
            base = ["--mode", "timing", "--output", str(Path(folder) / "result.json")]
            for deadline, outer in ((0, 150), (121, 150), (120, 129), (120, 151)):
                with self.subTest(deadline=deadline, outer=outer), self.assertRaises(ValueError):
                    probe.main(base + ["--deadline-seconds", str(deadline), "--outer-timeout-seconds", str(outer)])
            with self.assertRaisesRegex(ValueError, "Activity requires"):
                probe.main(["--mode", "activity", "--output", str(Path(folder) / "result.json"),
                            "--outer-timeout-seconds", "150"])
            execute.assert_not_called()

    def test_top_level_imports_are_lazy_and_no_ncu(self):
        source = Path(probe.__file__).read_text()
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.Import):
                self.assertFalse(any(alias.name.split(".")[0] in {"torch", "triton", "modal"}
                                     for alias in node.names))
            elif isinstance(node, ast.ImportFrom):
                self.assertNotIn((node.module or "").split(".")[0], {"torch", "triton", "modal"})
        self.assertFalse(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess"
            for node in ast.walk(tree)))


if __name__ == "__main__":
    unittest.main()
