"""CPU boundary/model tests only; no Torch, Quack, CUDA or native compiler."""
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import projection_p070_quack_preflight as p


class Tensor:
    def __init__(self, shape, dtype="bf16", device="cuda:0", contiguous=True, storage=None):
        self.shape = tuple(shape)
        self.dtype, self.device, self.contiguous = dtype, device, contiguous
        self.storage = object() if storage is None else storage

    def is_contiguous(self):
        return self.contiguous

    def unsqueeze(self, axis):
        assert axis == 0
        return Tensor((1,) + self.shape, self.dtype, self.device, self.contiguous, self.storage)

    def squeeze(self, axis):
        assert axis == 0 and self.shape[0] == 1
        return Tensor(self.shape[1:], self.dtype, self.device, self.contiguous, self.storage)


class Torch:
    bfloat16 = "bf16"

    def __init__(self):
        self.allocations = []

    def empty(self, shape, *, dtype, device):
        tensor = Tensor(shape, dtype, device)
        self.allocations.append(tensor)
        return tensor


def operands():
    return {"x": Tensor((128, 8192)), "config": dict(p.CONFIG)}, Tensor((2304, 8192))


def samples():
    return [dict(round=i, order=list(p.order(i)), excluded_warmup=False,
                 us={"p046": 12.0, "separate": 11.0, "serial": 10.0},
                 correctness={arm: True for arm in p.ARMS}) for i in range(p.ROUNDS)]


class QuackArmTests(unittest.TestCase):
    def test_singleton_views_and_exact_mode_only_dispatch(self):
        torch, gemm = Torch(), Mock()
        data, weight = operands()
        outputs = [p.make_quack_arm(torch, gemm, mode)(data, weight)
                   for mode in ("SEPARATE", "SERIAL")]
        first, second = gemm.call_args_list
        for call, output in zip((first, second), outputs):
            a, b, out, c, scheduler = call.args
            self.assertEqual((a.shape, b.shape, out.shape),
                             ((1, 128, 8192), (1, 2304, 8192), (1, 128, 2304)))
            self.assertIs(a.storage, data["x"].storage)
            self.assertIs(b.storage, weight.storage)
            self.assertIs(output.storage, out.storage)
            self.assertEqual(output.shape, (128, 2304))
            self.assertIsNone(c)
            self.assertIsNone(scheduler)
            self.assertEqual(out.dtype, torch.bfloat16)
            self.assertEqual(out.device, data["x"].device)
        expected = dict(tile_M=128, tile_N=128, tile_K=64, cluster_M=1,
                        cluster_N=1, cluster_K=1, split_k=2, persistent=True,
                        pingpong=False, is_dynamic_persistent=False,
                        max_swizzle_size=8, b_kn=False, alpha=1.0, beta=0.0)
        for call, mode in zip((first, second), ("SEPARATE", "SERIAL")):
            self.assertEqual(call.kwargs, dict(expected, split_k_mode=mode))

    def test_every_invocation_allocates_output_and_reenters_high_level_gemm(self):
        # Scratch belongs to the native high-level API. These sentinels prove
        # that our wrapper reenters that API, rather than caching a native call.
        torch, invocations = Torch(), []
        def gemm(*args, **kwargs):
            invocations.append((args, object(), object()))
        kernel = p.make_quack_arm(torch, gemm, "SERIAL")
        data, weight = operands()
        first = kernel(data, weight)
        replacement = Tensor(data["x"].shape)
        data["x"] = replacement
        second = kernel(data, weight)
        third = kernel(data, weight)
        self.assertEqual(len(invocations), 3)
        self.assertEqual(len(torch.allocations), 3)
        self.assertEqual(len({id(t.storage) for t in (first, second, third)}), 3)
        self.assertIs(invocations[1][0][0].storage, replacement.storage)
        self.assertEqual(len({id(row[1]) for row in invocations}), 3)
        self.assertEqual(len({id(row[2]) for row in invocations}), 3)
        self.assertTrue(all(row[0][4] is None for row in invocations))

    def test_invalid_operands_rejected_before_allocation_or_native_call(self):
        mutations = {
            "config layer": lambda d, w: d["config"].update(layer="pooler"),
            "config m": lambda d, w: d["config"].update(m=127),
            "x shape": lambda d, w: setattr(d["x"], "shape", (127, 8192)),
            "weight shape": lambda d, w: setattr(w, "shape", (2304, 8191)),
            "x dtype": lambda d, w: setattr(d["x"], "dtype", "fp32"),
            "weight dtype": lambda d, w: setattr(w, "dtype", "fp32"),
            "x strides": lambda d, w: setattr(d["x"], "contiguous", False),
            "weight strides": lambda d, w: setattr(w, "contiguous", False),
            "different device": lambda d, w: setattr(w, "device", "cuda:1"),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                torch, gemm = Torch(), Mock()
                data, weight = operands()
                mutate(data, weight)
                with self.assertRaises(ValueError):
                    p.make_quack_arm(torch, gemm, "SERIAL")(data, weight)
                self.assertEqual(torch.allocations, [])
                gemm.assert_not_called()


class ProtocolTests(unittest.TestCase):
    def test_fixed_balanced_rounds_reverse_each_adjacent_pair(self):
        self.assertEqual((p.WARMUP, p.ROUNDS, p.COHORTS), (6, 24, (7001000, 7002000)))
        for i in range(0, p.ROUNDS, 2):
            self.assertEqual(p.order(i + 1), p.order(i)[::-1])
        for slot in range(3):
            self.assertEqual(Counter(p.order(i)[slot] for i in range(p.ROUNDS)),
                             {arm: 8 for arm in p.ARMS})
        for i in range(p.ROUNDS):
            self.assertEqual(set(p.order(i)), set(p.ARMS))

    def test_partition_disjoint_complete_and_balanced_including_empty_splits(self):
        for total in range(130):
            for count in (1, 2, 3, 8, 133):
                with self.subTest(total=total, count=count):
                    spans = [p.partition(total, s, count) for s in range(count)]
                    flattened = [k for start, end in spans for k in range(start, end)]
                    self.assertEqual(flattened, list(range(total)))
                    self.assertEqual(spans[0][0], 0)
                    self.assertEqual(spans[-1][1], total)
                    self.assertTrue(all(left[1] == right[0] for left, right in zip(spans, spans[1:])))
                    sizes = [end - start for start, end in spans]
                    self.assertLessEqual(max(sizes) - min(sizes), 1)
                    self.assertEqual(sizes, sorted(sizes, reverse=True))
        self.assertEqual([p.partition(1, i, 3) for i in range(3)], [(0, 1), (1, 1), (1, 1)])

    def test_partition_rejects_invalid_domains(self):
        for args in ((-1, 0, 2), (1, 0, 0), (1, -1, 2), (1, 2, 2)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                p.partition(*args)


class SummaryTests(unittest.TestCase):
    def test_ratio_direction_and_latency_cut(self):
        result = p.summarize(samples())["p046_over_serial"]
        self.assertAlmostEqual(result["median"], 1.2)
        self.assertAlmostEqual(result["second_latency_reduction"], 1 / 6)
        self.assertEqual(len(result["paired_ratios"]), 12)

    def test_order_pair_geometric_ratio_before_median(self):
        data = samples()
        for i, row in enumerate(data):
            row["us"]["p046"], row["us"]["serial"] = (36., 9.) if i % 2 == 0 else (9., 36.)
        result = p.summarize(data)["p046_over_serial"]
        self.assertEqual(result["paired_ratios"], [1.] * 12)
        self.assertEqual(result["median"], 1.)
        self.assertEqual(result["second_latency_reduction"], 0.)

    def test_missing_failed_or_invalid_samples_are_not_summarized(self):
        mutations = {
            "missing round": lambda x: x.pop(),
            "extra round": lambda x: x.append(copy.deepcopy(x[0])),
            "wrong index": lambda x: x[0].update(round=1),
            "wrong order": lambda x: x[0].update(order=list(reversed(x[0]["order"]))),
            "missing arm": lambda x: x[0]["us"].pop("serial"),
            "extra arm": lambda x: x[0]["us"].update(other=1.),
            "failed oracle": lambda x: x[0]["correctness"].update(serial=False),
            "missing oracle": lambda x: x[0]["correctness"].pop("serial"),
            "nan": lambda x: x[0]["us"].update(serial=float("nan")),
            "infinity": lambda x: x[0]["us"].update(serial=float("inf")),
            "zero": lambda x: x[0]["us"].update(serial=0.),
            "negative": lambda x: x[0]["us"].update(serial=-1.),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                data = samples()
                mutate(data)
                with self.assertRaises(ValueError):
                    p.summarize(data)

    def test_warmup_cannot_substitute_for_measured_round(self):
        data = samples()
        data[0]["excluded_warmup"] = True
        with self.assertRaises(ValueError):
            p.summarize(data)


class CompileAndAssessmentTests(unittest.TestCase):
    def test_runtime_object_exports_must_match_reviewed_bytes(self):
        class Compiled:
            def __init__(self, payload):
                self.payload = payload

            def export_to_c(self, *, object_file_path, function_name):
                self.asserted_name = function_name
                Path(object_file_path).write_bytes(self.payload)

        functions = {name: Compiled(("inert test bytes: " + name).encode())
                     for name in ("serial", "separate", "separate_reduce")}
        plans = {name: Mock(compiled_fn=functions[name]) for name in ("serial", "separate")}
        compile_reduce = Mock(return_value=functions["separate_reduce"])
        hashes = {name + ".o": hashlib.sha256(obj.payload).hexdigest()
                  for name, obj in functions.items()}
        with tempfile.TemporaryDirectory() as folder:
            actual = p.verify_runtime_objects(plans, compile_reduce, "bf16",
                                               Path(folder) / "matching", {"objects": hashes})
            self.assertEqual(actual, hashes)
            self.assertTrue(all(obj.asserted_name == "func" for obj in functions.values()))
            compile_reduce.assert_called_once_with(out_dtype="bf16", split_k=2,
                add_to_output=False, vec_elems=4, alpha_mode=0, beta_mode=1,
                c_dtype=None, rowvec_dtype=None, colvec_dtype=None)
            functions["serial"].payload += b" changed"
            with self.assertRaisesRegex(RuntimeError, "timing not started"):
                p.verify_runtime_objects(plans, compile_reduce, "bf16",
                                         Path(folder) / "mismatch", {"objects": hashes})

    def test_compile_signatures_differ_only_at_intended_finalization_fields(self):
        serial = p.compile_arguments("bf16", "fp32", "SERIAL", "RN")
        separate = p.compile_arguments("bf16", "fp32", "SEPARATE", "RN", separate=True)
        self.assertEqual({key for key in serial if serial[key] != separate[key]},
                         {"d_dtype", "beta_mode", "split_k_mode"})
        self.assertEqual((serial["d_dtype"], serial["beta_mode"]), ("bf16", 1))
        self.assertEqual((separate["d_dtype"], separate["beta_mode"]), ("fp32", 0))
        self.assertEqual(serial["a_dtype"], serial["b_dtype"])
        self.assertEqual(serial["a_dtype"], "bf16")
        self.assertEqual(serial["rounding_mode"], "RN")
        self.assertEqual(serial["device_capacity"], (9, 0))
        self.assertEqual(serial["tile_shape_mn"], (128, 128, 64))
        self.assertEqual(serial["split_k"], 2)
        self.assertTrue(serial["batched"])

    @staticmethod
    def cohorts(p046=100., separate=100., serial=94.):
        result = [dict(seed=seed, passed=True, samples=samples()) for seed in p.COHORTS]
        for cohort in result:
            for row in cohort["samples"]:
                row["us"] = dict(p046=p046, separate=separate, serial=serial)
        return result

    def test_assessment_requires_each_cohort_and_never_promotes(self):
        result = p.assessment(self.cohorts())
        self.assertTrue(result["performance_screen_passed"])
        self.assertFalse(result["automatic_promotion"])
        self.assertEqual(result["selected"], "P046")
        for arm, time in (("p046", 98.), ("separate", 95.)):
            data = self.cohorts()
            for row in data[1]["samples"]:
                row["us"][arm] = time
            self.assertFalse(p.assessment(data)["performance_screen_passed"])
        for change in (lambda c: c.pop(), lambda c: c.reverse(),
                       lambda c: c[1].update(passed=False)):
            data = self.cohorts()
            change(data)
            with self.assertRaises(ValueError):
                p.assessment(data)

    def test_assessment_five_percent_inclusive_two_percent_exclusive(self):
        for kwargs, expected in ((dict(serial=95.), True),
                                 (dict(p046=110., serial=98.), False)):
            with self.subTest(kwargs=kwargs):
                self.assertEqual(p.assessment(self.cohorts(**kwargs))["performance_screen_passed"], expected)

    def test_compile_gate_binds_objects_sources_and_complete_review(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(p, "source_hashes", return_value={"frozen": "sha"}), \
             patch.object(p, "dependencies", return_value={"versions": {"fixture": "1"}}):
            root = Path(folder)
            objects = {}
            for name in ("serial.o", "separate.o", "separate_reduce.o"):
                payload = ("mock object, never executable: " + name).encode()
                (root / name).write_bytes(payload)
                objects[name] = hashlib.sha256(payload).hexdigest()
            receipt = dict(quack_commit=p.PIN, geometry=p.GEOMETRY,
                           dependencies={"versions": {"fixture": "1"}},
                           source_hashes={"frozen": "sha"}, objects=objects,
                           implementation_sha256=hashlib.sha256(Path(p.__file__).read_bytes()).hexdigest(),
                           review=dict(ready_for_execution=True,
                               mainloop_instructions="mock inspection",
                               stages_and_layout_differences="mock inspection",
                               register_spill_shared_resources="mock inspection",
                               synchronization_and_initialization="mock inspection"))
            path = root / "receipt.json"
            path.write_text(json.dumps(receipt))
            self.assertEqual(p.require_compile_review(path), receipt)
            changes = (
                lambda r: r.pop("review"),
                lambda r: r["review"].update(ready_for_execution=False),
                lambda r: r["review"].update(register_spill_shared_resources=" "),
                lambda r: r.update(quack_commit="different"),
                lambda r: r.update(source_hashes={"frozen": "changed"}),
                lambda r: r.update(dependencies={"versions": {"fixture": "2"}}),
                lambda r: r.update(implementation_sha256="changed"),
                lambda r: r["geometry"].update(split_k=3),
                lambda r: r["objects"].pop("separate_reduce.o"),
                lambda r: r["objects"].update(serial_o="extra"),
            )
            for change in changes:
                data = copy.deepcopy(receipt)
                change(data)
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    p.require_compile_review(path)
            path.write_text(json.dumps(receipt))
            (root / "serial.o").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Compiled object changed"):
                p.require_compile_review(path)


if __name__ == "__main__":
    unittest.main()
