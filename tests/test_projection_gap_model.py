"""Tests of score arithmetic and immutable evidence, not GPU correctness."""
import copy
import math
import unittest
import projection_gap_model as gap


class GapModelTests(unittest.TestCase):
    def test_receipt_reconstruction_and_weighted_groups(self):
        result = gap.build()
        self.assertAlmostEqual(result["current_score"], 1.1841399408925375, places=13)
        self.assertEqual(len(result["six_load_totals"]), 6)
        for values in result["fractions"].values():
            self.assertAlmostEqual(sum(values[f] for f in gap.FAMILIES), 1.)
        self.assertAlmostEqual(result["equal_load_mean_fractions"]["three_main_families"], .7580078137983209)

    def test_uniform_cut_scales_all_six_loads_reciprocally(self):
        rows = gap.read_rows()
        self.assertAlmostEqual(gap.score(gap.change(rows, gap.FAMILIES, .2))[0]/gap.score(rows)[0], 1.25)

    def test_single_load_changes_only_one_geometric_factor(self):
        rows = gap.read_rows()
        original, loads = gap.score(rows)
        altered = [{**r, "candidate_us": r["candidate_us"]*.5} if r["load"] == "decode 1" else r for r in rows]
        self.assertAlmostEqual(gap.score(altered)[0]/original, 2**(1/6))

    def test_inverse_matches_leader_and_is_bounded(self):
        rows = gap.read_rows()
        required = gap.required_cut(rows, gap.MAIN, 1.6099)
        self.assertAlmostEqual(gap.score(gap.change(rows, gap.MAIN, required))[0], 1.6099)
        self.assertAlmostEqual(required, .348864, places=5)
        self.assertIsNone(gap.required_cut(rows, ("qkv_gate",), 1.6099))
        self.assertIsNone(gap.required_cut(rows, ("unpooler",), 1.6099))

    def test_rounded_floor_is_explicit_and_no_hard_bound(self):
        value = gap.build()["rounded_task_floor_model"]
        self.assertAlmostEqual(value["score"], 1.608190, places=5)
        self.assertIn("not a strict bound", value["qualification"])
        self.assertEqual(len(gap.task_floors()), 30)

    def test_target_selection_does_not_mutate_input(self):
        rows = gap.read_rows()
        saved = copy.deepcopy(rows)
        changed = gap.change(rows, gap.MAIN, .2, 2)
        self.assertEqual(rows, saved)
        self.assertEqual(sum(a != b for a, b in zip(rows, changed)), 15)
        for invalid in (-.01, 1.01):
            with self.assertRaises(ValueError):
                gap.change(rows, gap.MAIN, invalid)


if __name__ == "__main__":
    unittest.main()
