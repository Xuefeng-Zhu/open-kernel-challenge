"""Small synthetic and saved-receipt tests; no GPU or new measurements."""
import copy
import json
import math
import unittest

from assess_p046 import ROOT, PARENT_SHA, PARENT_JOB, SOURCE_SHA, P045_SHA, LOADS, LAYERS, TARGET_M, assess


def receipt(sha, target_ratio=1.0):
    rows = []
    for load, m in LOADS.items():
        for layer in LAYERS:
            ratio = target_ratio if layer == "pooler" and m in TARGET_M else 1.0
            rows.append(dict(event="benchmark", status="pass", case=dict(load=load, layer=layer, m=m, weight=62),
                             median_ns=10000 * ratio, baseline_ns=10000, median_raw_ns=14000,
                             baseline_raw_ns=14000, paired_ratio=1 / ratio))
    return dict(admission=dict(submission_sha256=sha, source_commit="synthetic", job_id=PARENT_JOB if sha == PARENT_SHA else "synthetic"),
                status="done", result=dict(passed=True, speedup=1.0, speedup_ci=[.98, 1.02], records=rows),
                prerequisite_tests={"passed": 9, "total": 9}, shape_checks={"passed": 30, "total": 30})


class P046Assessment(unittest.TestCase):
    def test_three_targets_and_frozen_six_load_arithmetic(self):
        out = assess(receipt(PARENT_SHA), receipt(SOURCE_SHA, .8))
        self.assertEqual(sorted(row["m"] for row in out["targets"]), [8, 16, 32])
        self.assertEqual(out["controls_count"], 27)
        self.assertAlmostEqual(out["target_geomean_normalized_latency_cut"], .2)
        self.assertAlmostEqual(out["target_only_frozen_parent_model_factor"], (50 / 48)**.5)
        self.assertTrue(out["performance_gates_passed"])
        self.assertFalse(out["restored_m128_control"]["target"])
        self.assertFalse(out["automatic_selection"])
        self.assertTrue(out["ci_overlap"])

    def test_five_pct_cut_can_miss_one_pct_score_gate(self):
        out = assess(receipt(PARENT_SHA), receipt(SOURCE_SHA, .95))
        self.assertTrue(out["gates"]["target_geomean_latency_cut_at_least5pct"])
        self.assertFalse(out["gates"]["target_only_frozen_parent_score_gain_at_least1pct"])

    def test_one_target_regression_fails_even_if_geomean_wins(self):
        candidate = receipt(SOURCE_SHA, .7)
        row = next(r for r in candidate["result"]["records"] if r["case"]["layer"] == "pooler" and r["case"]["m"] == 8)
        row.update(median_ns=10010, paired_ratio=1 / 1.001)
        out = assess(receipt(PARENT_SHA), candidate)
        self.assertTrue(out["gates"]["target_geomean_latency_cut_at_least5pct"])
        self.assertFalse(out["gates"]["no_individual_target_latency_regression"])

    def test_ratio_direction_and_reference_normalization(self):
        candidate = receipt(SOURCE_SHA, .8)
        for row in candidate["result"]["records"]:
            row["median_ns"] *= 2
            row["baseline_ns"] *= 2
        out = assess(receipt(PARENT_SHA), candidate)
        self.assertAlmostEqual(out["targets"][0]["normalized_latency_ratio"], .8)
        self.assertAlmostEqual(out["targets"][0]["normalized_latency_cut"], .2)

    def test_all27_control_shifts_including_restored_m128_reported(self):
        candidate = receipt(SOURCE_SHA, .8)
        for row in candidate["result"]["records"]:
            if row["case"]["layer"] != "pooler" or row["case"]["m"] not in TARGET_M:
                row.update(median_ns=11000, paired_ratio=1 / 1.1)
        out = assess(receipt(PARENT_SHA), candidate)
        self.assertEqual(len(out["controls_over5pct_latency_movement"]), 27)
        self.assertAlmostEqual(out["restored_m128_control"]["normalized_latency_increase"], .1)
        self.assertAlmostEqual(out["target_only_frozen_parent_model_factor"], (50 / 48)**.5)

    def test_bad_sources_shapes_duplicate_nonfinite_and_ratio_rejected(self):
        for kind in ("source", "missing", "duplicate", "wrong_m", "wrong_weight", "nan", "ratio"):
            candidate = receipt(SOURCE_SHA)
            rows = candidate["result"]["records"]
            if kind == "source": candidate["admission"]["submission_sha256"] = P045_SHA
            elif kind == "missing": rows.pop()
            elif kind == "duplicate": rows.append(copy.deepcopy(rows[0]))
            elif kind == "wrong_m": rows[0]["case"]["m"] = 9
            elif kind == "wrong_weight": rows[0]["case"]["weight"] = 46
            elif kind == "nan": rows[0]["median_raw_ns"] = math.nan
            else: rows[0]["paired_ratio"] = 2
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                assess(receipt(PARENT_SHA), candidate)

    def test_correctness_and_profile_gate_not_replaced_by_timing(self):
        candidate = receipt(SOURCE_SHA, .8)
        candidate["prerequisite_tests"]["passed"] = 8
        candidate["result"]["records"].append(dict(event="profile", case=dict(layer="pooler", m=8), report="synthetic bmm"))
        out = assess(receipt(PARENT_SHA), candidate)
        self.assertFalse(out["performance_gates_passed"])
        self.assertEqual(out["execution_profile_gate"]["status"], "unverified")

    def test_optional_p045_comparison_preserves_original_gates(self):
        out = assess(receipt(PARENT_SHA), receipt(SOURCE_SHA, .8), receipt(P045_SHA, .9))
        comparison = out["p045_comparison"]
        self.assertTrue(comparison["available"])
        self.assertTrue(comparison["original_gates_unchanged"])
        self.assertEqual(len(comparison["pooler_target_cuts"]), 4)
        row = next(r for r in comparison["pooler_target_cuts"] if r["m"] == 8)
        self.assertAlmostEqual(row["p045_normalized_latency_cut"], .1)
        self.assertAlmostEqual(row["p046_normalized_latency_cut"], .2)
        failed = receipt(P045_SHA)
        failed["result"].update(passed=False, records=[])
        self.assertFalse(assess(receipt(PARENT_SHA), receipt(SOURCE_SHA), failed)["p045_comparison"]["available"])

    def test_ci_absent_disjoint_or_malformed(self):
        candidate = receipt(SOURCE_SHA)
        candidate["result"]["speedup_ci"] = [1.1, 1.2]
        self.assertFalse(assess(receipt(PARENT_SHA), candidate)["ci_overlap"])
        candidate["result"].pop("speedup_ci")
        self.assertIsNone(assess(receipt(PARENT_SHA), candidate)["ci_overlap"])
        for value in ([2, 1], [math.nan, 1], [1]):
            candidate["result"]["speedup_ci"] = value
            with self.assertRaises(ValueError):
                assess(receipt(PARENT_SHA), candidate)

    def test_saved_p039_no_change_and_real_p045_comparison(self):
        parent = json.loads((ROOT / "results/remote-P2-04-router-m1-fp32-gemv.json").read_text())
        candidate = copy.deepcopy(parent)
        # Synthetic no-change fixture using saved P039 timings; not a P046 run.
        candidate["admission"].update(submission_sha256=SOURCE_SHA, job_id="synthetic-no-change")
        previous = json.loads((ROOT / "results/remote-P2-08-pooler-column-batch2.json").read_text())
        out = assess(parent, candidate, previous)
        self.assertEqual(out["target_geomean_normalized_latency_cut"], 0)
        self.assertEqual(out["target_only_frozen_parent_model_factor"], 1)
        self.assertFalse(out["performance_gates_passed"])
        self.assertTrue(out["p045_comparison"]["available"])
        self.assertEqual(out["controls_over5pct_latency_movement"], [])


if __name__ == "__main__":
    unittest.main()
