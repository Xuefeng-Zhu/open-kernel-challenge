"""CPU mock/static tests only: neither candidates nor CUDA are executed."""
import ast
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import projection_p046_confirmation as probe


class Tensor:
    counter = 0

    def __init__(self, values):
        Tensor.counter += 1
        self.pointer = Tensor.counter
        self.values = list(values)
        self.shape = (1, len(values))
        self.dtype, self.device = "bf16", "cuda"

    def stride(self):
        return (len(self.values), 1)

    def data_ptr(self):
        return self.pointer

    def is_contiguous(self):
        return True

    def clone(self):
        return Tensor(self.values)

    def view(self, _):
        return self

    def copy_(self, other):
        self.values = other.values.copy()


def tensor_torch():
    result = Mock()
    result.is_tensor.side_effect = lambda t: isinstance(t, Tensor)
    result.equal.side_effect = lambda a, b: a.values == b.values
    return result


def samples(ratio=1.1):
    return [dict(round=i, order=list(probe.runner_order(i)),
                 us=dict(parent=(10 + i % 2) * ratio, candidate=10 + i % 2))
            for i in range(probe.MEASURED_ROUNDS)]


class ConfirmationTests(unittest.TestCase):
    def test_exact_source_and_oracle_hashes(self):
        hashes = probe.verify_sources()
        for path, digest in probe.SOURCES.values():
            self.assertEqual(hashes[path], digest)
        for path, digest in probe.ORACLES.items():
            self.assertEqual(hashes[path], digest)

    def test_source_failure_blocks_execution_and_keeps_failure_report(self):
        with TemporaryDirectory() as folder:
            output = Path(folder) / "timing.json"
            with patch.object(probe, "verify_sources", side_effect=ValueError("Frozen source hash mismatch")), \
                    patch.object(probe, "execute") as execute, patch("builtins.print"):
                result = probe.main(["--output", str(output), "--outer-timeout-seconds", "180"])
            self.assertEqual(result, 1)
            execute.assert_not_called()
            report = json.loads(output.read_text())
            self.assertFalse(report["passed"])
            self.assertEqual(report["failure"]["type"], "ValueError")
            self.assertEqual(report["cases"], [])

    def test_partial_failure_is_saved_not_discarded(self):
        with TemporaryDirectory() as folder:
            output = Path(folder) / "timing.json"

            def fail(args, report, save):
                report["cases"].append(dict(layer="pooler", m=8, passed=False,
                                            samples=[dict(round=0, us={"parent": 12})]))
                save()
                raise TimeoutError("deadline")

            with patch.object(probe, "execute", side_effect=fail), patch("builtins.print"):
                self.assertEqual(probe.main(["--output", str(output), "--outer-timeout-seconds", "180"]), 1)
            report = json.loads(output.read_text())
            self.assertFalse(report["passed"])
            self.assertEqual(report["cases"][0]["samples"][0]["us"], {"parent": 12})
            self.assertNotIn("assessment", report)

    def test_h200_and_deadline_guards(self):
        torch = Mock()
        torch.cuda.is_available.return_value = False
        with self.assertRaisesRegex(RuntimeError, "CUDA unavailable"):
            probe.require_h200(torch)
        torch.cuda.is_available.return_value = True
        torch.cuda.get_device_name.return_value = "NVIDIA H100"
        torch.cuda.get_device_capability.return_value = (9, 0)
        with self.assertRaisesRegex(RuntimeError, "H200 SM90"):
            probe.require_h200(torch)
        torch.cuda.get_device_name.return_value = "NVIDIA H200"
        self.assertEqual(probe.require_h200(torch), "NVIDIA H200")
        probe.validate_options(SimpleNamespace(deadline_seconds=120, outer_timeout_seconds=180))
        for deadline, outer in [(121, 180), (120, 125), (0, 180), (120, 181)]:
            with self.assertRaises(ValueError):
                probe.validate_options(SimpleNamespace(deadline_seconds=deadline, outer_timeout_seconds=outer))

    def test_timer_events_are_created_and_initialized_only_once(self):
        log, events = [], []
        torch = Mock()

        def create(**kwargs):
            index = len(events)
            event = Mock()
            event.record.side_effect = lambda: log.append(("record", index))
            event.synchronize.side_effect = lambda: log.append(("sync", index))
            event.elapsed_time.side_effect = lambda other: (log.append(("elapsed", index)), .012)[1]
            events.append(event)
            log.append(("create", index))
            return event

        torch.cuda.Event.side_effect = create
        torch.sum.side_effect = lambda *args, **kwargs: log.append(("evict", args, kwargs))
        timer = probe.ColdGraphTimer(torch)
        self.assertEqual(log, [("create", 0), ("create", 1), ("record", 0),
                               ("record", 1), ("sync", 1), ("elapsed", 0)])
        log.clear()
        graph = Mock()
        graph.replay.side_effect = lambda: log.append(("replay",))
        for _ in range(2):
            self.assertEqual(timer.measure(graph, "buffer", "scratch"), 12)
        self.assertEqual(torch.cuda.Event.call_count, 2)
        expected = [("evict", ("buffer",), {"dim": 0, "out": "scratch"}),
                    ("record", 0), ("replay",), ("record", 1), ("sync", 1), ("elapsed", 0)]
        self.assertEqual(log, expected * 2)

    def test_capture_both_runners_uses_identical_input_object(self):
        torch = Mock()
        data = {"x": Tensor([1, 2])}
        modules = {name: Mock() for name in probe.SOURCES}
        states = {name: object() for name in probe.SOURCES}
        received = []

        def capture(fn, actual, state):
            received.append((fn, actual, state))
            return Mock(), Tensor([2, 4])

        graphs, outputs, durations = probe.capture_pair(torch, capture, modules, data, states, Mock())
        self.assertEqual(set(graphs), set(probe.SOURCES))
        self.assertIs(received[0][1], data)
        self.assertIs(received[1][1], data)
        self.assertNotEqual(outputs["parent"].data_ptr(), outputs["candidate"].data_ptr())
        self.assertTrue(all(value >= 0 for value in durations.values()))

    def test_capture_refuses_shared_output(self):
        shared = Tensor([1])
        with self.assertRaisesRegex(RuntimeError, "outputs unexpectedly alias"):
            probe.capture_pair(Mock(), Mock(return_value=(Mock(), shared)),
                               {n: Mock() for n in probe.SOURCES}, {},
                               {n: {} for n in probe.SOURCES}, Mock())

    def test_original_and_derived_state_mutation_detected(self):
        torch = tensor_torch()
        state = {"pooler": Tensor([1, 2]), "pooler_column_batch2": Tensor([2, 1])}
        snapshot = probe.freeze_tensors(torch, state)
        probe.assert_frozen(torch, state, snapshot)
        state["pooler_column_batch2"].values[0] = 9
        with self.assertRaisesRegex(RuntimeError, "pooler_column_batch2"):
            probe.assert_frozen(torch, state, snapshot)
        state["pooler_column_batch2"].values[0] = 2
        state["pooler"] = Tensor([1, 2])
        with self.assertRaisesRegex(RuntimeError, "pooler"):
            probe.assert_frozen(torch, state, snapshot)
        with self.assertRaisesRegex(RuntimeError, "keys changed"):
            probe.assert_frozen(torch, {}, snapshot)

    def test_input_address_content_and_config_are_checked(self):
        torch = tensor_torch()
        fresh = {"x": Tensor([1, 2])}
        data = {"x": Tensor([1, 2]), "config": {"m": 1}}
        metadata = probe.tensor_metadata(data["x"])
        probe.assert_input(torch, data, fresh, {"m": 1}, metadata)
        data["config"]["m"] = 2
        with self.assertRaisesRegex(RuntimeError, "metadata/config"):
            probe.assert_input(torch, data, fresh, {"m": 1}, metadata)
        data["config"]["m"] = 1
        data["x"].values[1] = 3
        with self.assertRaisesRegex(RuntimeError, "content changed"):
            probe.assert_input(torch, data, fresh, {"m": 1}, metadata)

    def test_aba_and_sparse_zero_coverage_preserves_shared_address(self):
        torch = tensor_torch()
        reference = Mock()
        reference.check_implementation.return_value = (True, "mock pass")
        data = {"x": Tensor([0, 0]), "config": {"m": 1}}
        metadata = probe.tensor_metadata(data["x"])
        outputs = {name: Tensor([0, 0]) for name in probe.SOURCES}
        graphs = {name: Mock() for name in probe.SOURCES}
        for name in probe.SOURCES:
            graphs[name].replay.side_effect = lambda n=name: setattr(outputs[n], "values", [x * 2 for x in data["x"].values])

        def make(seed, dist):
            values = [0, 0] if dist == "zero" else [8, 0] if dist == "sparse" else [seed, 30 if dist == "outliers" else 1]
            return {"x": Tensor(values)}

        records = []
        probe.correctness_phase(torch, reference, data, graphs, outputs, {}, {"m": 1}, metadata,
                                make, 100, "before", records, Mock(), Mock())
        self.assertEqual(len(records), 10)
        self.assertEqual(reference.check_implementation.call_count, 10)
        self.assertTrue(all(r["passed"] and r["same_input_address"] for r in records))
        self.assertEqual({r["label"] for r in records}, {"A", "B", "A-return", "sparse", "zero"})
        self.assertEqual(probe.tensor_metadata(data["x"]), metadata)

    def test_oracle_failure_is_not_counted_as_a_pass(self):
        torch, reference = tensor_torch(), Mock()
        reference.check_implementation.return_value = (False, "bad output")
        data = {"x": Tensor([1]), "config": {}}
        with self.assertRaisesRegex(RuntimeError, "official FP64 oracle failed"):
            probe.check_output(torch, reference, data, Tensor([3]), {"x": Tensor([1])},
                               {}, probe.tensor_metadata(data["x"]), {})

    def test_sparse_spec_covers_one_nonzero_for_m1_and_bounds(self):
        for m, k in [(1, 1024), (1, 4096), (8, 8192), (128, 8192)]:
            cols, values = probe.sparse_spec(m, k, 4601000)
            self.assertEqual(len(cols), m)
            self.assertTrue(all(0 <= col < k for col in cols))
            self.assertEqual(values, [8 if row % 2 == 0 else -32 for row in range(m)])

    def test_pair_formula_and_malformed_samples(self):
        summary = probe.summarize_samples(samples(1.25))
        self.assertEqual(summary["reverse_order_pair_ratios"], [1.25] * 12)
        self.assertAlmostEqual(summary["candidate_raw_latency_cut"], .2)
        for mutation in (lambda x: x.pop(), lambda x: x[0].update(order=["candidate", "parent"]),
                         lambda x: x[0]["us"].update(parent=float("nan")),
                         lambda x: x[1]["us"].update(candidate=0)):
            values = samples()
            mutation(values)
            with self.assertRaises(ValueError):
                probe.summarize_samples(values)

    def test_target_gate_and_control_flag_do_not_select_automatically(self):
        cases = [dict(layer=l, m=m, target=t, passed=True,
                      summary=probe.summarize_samples(samples(1.1 if t else 1))) for l, m, t in probe.CASES]
        cases[3]["summary"] = probe.summarize_samples(samples(.97))
        report = probe.summarize_cases(cases)
        self.assertTrue(all(report["target_gates"].values()))
        self.assertFalse(report["automatic_selection"])
        self.assertEqual(len(report["controls_requiring_investigation"]), 1)
        self.assertEqual(report["controls_requiring_investigation"][0]["layer"], "o")
        cases[0]["passed"] = False
        with self.assertRaises(ValueError):
            probe.summarize_cases(cases)

    def test_no_provisioning_no_eager_torch_and_fixed_protocol(self):
        tree = ast.parse(Path(probe.__file__).read_text())
        top_imports = [a.name for n in tree.body if isinstance(n, ast.Import) for a in n.names]
        self.assertFalse(set(top_imports) & {"torch", "triton", "modal"})
        imports = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        imports += [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        self.assertFalse(set(imports) & {"modal", "subprocess", "requests", "urllib", "socket"})
        self.assertEqual(probe.CASES, (("pooler", 8, True), ("pooler", 16, True),
                                     ("pooler", 32, True), ("o", 1, False),
                                     ("unpooler", 1, False), ("pooler", 128, False)))
        self.assertEqual(probe.MEASURED_ROUNDS, 24)
        self.assertEqual(probe.WARMUP_ROUNDS, 6)
        self.assertEqual(probe.EVICTION_BYTES, 256 * 1024**2)

    def test_large_oracle_is_outside_inner_paired_measurement_loop(self):
        tree = ast.parse(Path(probe.__file__).read_text())
        timed_runner_loops = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.For) or not isinstance(node.target, ast.Name) or node.target.id != "name":
                continue
            calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)]
            if any(isinstance(c.func, ast.Attribute) and c.func.attr == "measure" for c in calls):
                timed_runner_loops.append(node)
                self.assertFalse(any(isinstance(c.func, ast.Name) and c.func.id == "check_output" for c in calls))
                self.assertTrue(any(isinstance(c.func, ast.Name) and c.func.id == "assert_input" for c in calls))
        self.assertEqual(len(timed_runner_loops), 1)


if __name__ == "__main__":
    unittest.main()
