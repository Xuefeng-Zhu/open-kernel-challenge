"""Checks the independent timing reduction, pairing and no-filter invariant."""
import itertools
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from modal_analysis import summarize_case


class ModalAnalysisTests(unittest.TestCase):
    def fixture(self):
        orders=list(itertools.permutations(["candidate","r10","sglang"]))
        return dict(case=dict(layer="qkv_gate",m=1),samples=[dict(seed=i,order=list(order),
                    us=dict(candidate=10.,r10=20.,sglang=30.)) for i,order in enumerate(orders)])

    def test_reversed_orders_pair_exactly_once(self):
        result=summarize_case(self.fixture())
        self.assertEqual([p["indices"] for p in result["opposite_order_pairs"]],[[0,5],[1,3],[2,4]])
        self.assertEqual(result["median_paired_gross_ratio"],dict(r10=2.,sglang=3.))

    def test_first_outlier_is_retained_and_changes_a_pair(self):
        data=self.fixture();data["samples"][0]["us"]["candidate"]=1000.
        result=summarize_case(data)
        self.assertEqual(result["gross_us_distribution"]["candidate"]["maximum"],1000.)
        self.assertEqual(result["gross_us_distribution"]["candidate"]["n"],6)
        self.assertAlmostEqual(result["opposite_order_pairs"][0]["geometric_control_over_candidate"]["r10"],.2)

    def test_rejects_missing_order_or_duplicate_seed(self):
        data=self.fixture();data["samples"][0]["order"]=data["samples"][1]["order"]
        with self.assertRaises(ValueError):summarize_case(data)
        data=self.fixture();data["samples"][0]["seed"]=data["samples"][1]["seed"]
        with self.assertRaises(ValueError):summarize_case(data)
