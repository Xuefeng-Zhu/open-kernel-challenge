"""Fast offline arithmetic and fail-closed receipt tests; no GPU execution."""
import copy
import json
import math
import unittest

from assess_p045 import ROOT, PARENT_SHA, SOURCE_SHA, PARENT_JOB, LOADS, LAYERS, TARGET_M, assess


def receipt(sha, target_ratio=1.0):
    records = []
    for load, m in LOADS.items():
        for layer in LAYERS:
            ratio = target_ratio if layer == "pooler" and m in TARGET_M else 1.0
            records.append(dict(event="benchmark", status="pass", case=dict(load=load, layer=layer, m=m, weight=62),
                                median_ns=10000 * ratio, baseline_ns=10000, median_raw_ns=14000 * ratio,
                                baseline_raw_ns=14000, paired_ratio=1 / ratio))
    return dict(admission=dict(submission_sha256=sha, job_id=PARENT_JOB if sha == PARENT_SHA else "synthetic-p045",
                               source_commit="synthetic-fixture"), status="done",
                result=dict(passed=True, speedup=1.0, records=records, evaluator={"fixture": True}, speedup_ci=[.98, 1.02]),
                prerequisite_tests={"passed": 9, "total": 9}, shape_checks={"passed": 30, "total": 30})


class P045Assessment(unittest.TestCase):
    def setUp(self):
        self.parent = receipt(PARENT_SHA)
        self.candidate = receipt(SOURCE_SHA)

    def test_known_20pct_target_cut_and_six_load_model(self):
        out = assess(self.parent, receipt(SOURCE_SHA, .8))
        self.assertAlmostEqual(out["target_geomean_normalized_latency_cut"], .2)
        self.assertAlmostEqual(out["target_only_frozen_parent_model_factor"], (50 / 48)**(4 / 6))
        self.assertTrue(out["performance_gates_passed"])
        self.assertEqual(sorted(row["m"] for row in out["targets"]), [8, 16, 32, 128])
        self.assertEqual(out["controls_count"], 26)
        self.assertFalse(out["automatic_selection"])
        self.assertEqual(out["execution_profile_gate"]["status"], "unverified")

    def test_geomean_gate_not_every_target_at10pct(self):
        candidate = receipt(SOURCE_SHA, .7)
        row = next(r for r in candidate["result"]["records"] if r["case"]["layer"] == "pooler" and r["case"]["m"] == 8)
        row.update(median_ns=10000, paired_ratio=1.0)
        out = assess(self.parent, candidate)
        self.assertTrue(out["performance_gates_passed"])
        self.assertAlmostEqual(out["target_geomean_normalized_latency_cut"], 1 - (.7**3)**.25)

    def test_target_regression_fails_despite_geomean_gain(self):
        candidate = receipt(SOURCE_SHA, .7)
        row = next(r for r in candidate["result"]["records"] if r["case"]["layer"] == "pooler" and r["case"]["m"] == 8)
        row.update(median_ns=10600, paired_ratio=1 / 1.06)
        out = assess(self.parent, candidate)
        self.assertTrue(out["gates"]["target_geomean_latency_cut_at_least10pct"])
        self.assertFalse(out["gates"]["no_target_latency_increase_over5pct"])
        self.assertFalse(out["performance_gates_passed"])

    def test_10pct_target_cut_can_miss_whole_score_gate(self):
        out = assess(self.parent, receipt(SOURCE_SHA, .9))
        self.assertTrue(out["gates"]["target_geomean_latency_cut_at_least10pct"])
        self.assertFalse(out["gates"]["target_only_frozen_parent_score_gain_at_least1p5pct"])

    def test_all_control_movements_reported_but_excluded_from_model(self):
        candidate = receipt(SOURCE_SHA, .8)
        for row in candidate["result"]["records"]:
            if row["case"]["layer"] != "pooler" or row["case"]["m"] not in TARGET_M:
                ratio = 1.1 if row["case"]["m"] == 1 else .9
                row.update(median_ns=10000 * ratio, paired_ratio=1 / ratio)
        out = assess(self.parent, candidate)
        self.assertEqual(len(out["controls_over5pct_latency_movement"]), 26)
        self.assertEqual(len(out["controls_over5pct_latency_regression"]), 5)
        self.assertAlmostEqual(out["target_only_frozen_parent_model_factor"], (50 / 48)**(4 / 6))

    def test_reference_normalization_not_raw_time_ratio(self):
        candidate = receipt(SOURCE_SHA, .8)
        for row in candidate["result"]["records"]:
            row["median_ns"] *= 2
            row["baseline_ns"] *= 2
        out = assess(self.parent, candidate)
        self.assertAlmostEqual(out["targets"][0]["normalized_latency_ratio"], .8)
        self.assertAlmostEqual(out["target_geomean_normalized_latency_cut"], .2)

    def test_source_or_parent_identity_mismatch_rejected(self):
        for role, key, value in (("candidate", "submission_sha256", PARENT_SHA),
                                 ("parent", "submission_sha256", SOURCE_SHA),
                                 ("parent", "job_id", "different-job")):
            parent, candidate = copy.deepcopy(self.parent), copy.deepcopy(self.candidate)
            (candidate if role == "candidate" else parent)["admission"][key] = value
            with self.assertRaises(ValueError):
                assess(parent, candidate)

    def test_missing_duplicate_and_unexpected_rows_rejected(self):
        for kind in ("missing", "duplicate", "unexpected", "wrong_m", "wrong_weight"):
            candidate = copy.deepcopy(self.candidate)
            rows = candidate["result"]["records"]
            if kind == "missing": rows.pop()
            elif kind == "duplicate": rows.append(copy.deepcopy(rows[0]))
            elif kind == "unexpected": rows[0]["case"]["layer"] = "attention"
            elif kind == "wrong_m": rows[0]["case"]["m"] = 8
            else: rows[0]["case"]["weight"] = 46
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                assess(self.parent, candidate)

    def test_nonfinite_zero_negative_and_inconsistent_measurements_rejected(self):
        for field, value in (("paired_ratio", math.nan), ("median_ns", math.inf),
                             ("baseline_ns", -1), ("median_raw_ns", 0),
                             ("baseline_raw_ns", True), ("runs_ns", [1, math.nan]),
                             ("paired_ratio", 2)):
            candidate = copy.deepcopy(self.candidate)
            candidate["result"]["records"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                assess(self.parent, candidate)

    def test_correctness_failure_never_passes_gates(self):
        candidate = receipt(SOURCE_SHA, .7)
        candidate["result"]["passed"] = False
        failed = assess(self.parent, candidate)
        self.assertFalse(failed["performance_gates_passed"])
        self.assertIn("correctness did not pass", failed["decision"])
        self.assertIn("correctness passed", assess(self.parent, self.candidate)["decision"])
        candidate["result"]["passed"] = True
        candidate["shape_checks"]["passed"] = 29
        self.assertFalse(assess(self.parent, candidate)["performance_gates_passed"])

    def test_profile_text_does_not_prove_execution(self):
        candidate = receipt(SOURCE_SHA, .8)
        for load, m in LOADS.items():
            if m in TARGET_M:
                candidate["result"]["records"].append(dict(event="profile", case=dict(load=load, layer="pooler", m=m), report="synthetic bmm kernel"))
        gate = assess(self.parent, candidate)["execution_profile_gate"]
        self.assertEqual(len(gate["target_profiles"]), 4)
        self.assertEqual(gate["status"], "unverified")

    def test_real_p039_timings_as_synthetic_no_change_negative_control(self):
        parent = json.loads((ROOT / "results/remote-P2-04-router-m1-fp32-gemv.json").read_text())
        candidate = copy.deepcopy(parent)
        # Explicit synthetic fixture: reuse real P039 timings, not a real P045 run.
        candidate["admission"].update(submission_sha256=SOURCE_SHA, job_id="synthetic-no-change")
        out = assess(parent, candidate)
        self.assertEqual(out["target_geomean_normalized_latency_cut"], 0)
        self.assertEqual(out["target_only_frozen_parent_model_factor"], 1)
        self.assertFalse(out["performance_gates_passed"])
        self.assertEqual(out["controls_over5pct_latency_movement"], [])
        for load, values in parent["weighted_loads"].items():
            self.assertAlmostEqual(out["frozen_parent_load_us"][load], values["candidate_load_us"], places=8)


if __name__ == "__main__":
    unittest.main()
