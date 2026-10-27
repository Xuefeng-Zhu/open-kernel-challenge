"""Offline score sensitivity from saved receipts; no candidate execution/network.

All altered-time scenarios are conditional arithmetic, not performance forecasts.
Run with --self-test, then without arguments to regenerate evidence.json.
"""
import hashlib
import json
import math
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "strategy"))
from weighted_analysis import read_run, score, load_totals


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cut_factor(rows, keys, cut):
    assert 0 <= cut < 1
    times = {k: r["candidate_adjusted_us"] * (1-cut if k in keys else 1)
             for k, r in rows.items()}
    return score(rows, times) / score(rows)


def analyze():
    runs = {n: read_run(n) for n in ("A2-11", "A2-13")}
    pair_path = ROOT / "a215/paired-analysis.json"
    pair = json.loads(pair_path.read_text())
    assert sha(ROOT / "a215/paired-result.json") == pair["result_sha256"]
    rows = runs["A2-13"]["shapes"]
    key = "decode 32/full"
    w = rows[key]["share_of_own_load_candidate"]
    cut = 1 - 1 / pair["raw_ratio_parent_over_candidate"]
    factor = cut_factor(rows, {key}, cut)
    assert math.isclose(factor, (1-w*cut)**(-1/6), rel_tol=1e-12)
    per_pair = [100*(cut_factor(rows, {key}, 1-1/r)-1)
                for r in pair["reverse_order_pair_ratios"]]
    paths = []
    for k, r in rows.items():
        paths.append({field: r[field] for field in (
            "key", "candidate_adjusted_us", "reference_adjusted_us",
            "candidate_raw_median_us", "reference_raw_median_us", "weight",
            "share_of_own_load_candidate", "harness_floor_us", "profile_activities")}
            | {"modeled_score_gain_percent_for_path_cut": {
                str(int(100*c)): 100*(cut_factor(rows, {k}, c)-1)
                for c in (.1, .2, .3)}})
    paths.sort(key=lambda r: -r["share_of_own_load_candidate"])
    groups = {
        "FA3_full_verify_B16_B32": ["verify 16/full", "verify 32/full"],
        "FA3_large_full_including_decode_B16": ["decode 16/full", "verify 16/full", "verify 32/full"],
        "FA3_sliding_verify_all_batches": [f"verify {b}/sliding" for b in (1,16,32)],
        "FP8_fallback_four_paths": ["decode 1/sliding", "decode 16/sliding", "decode 32/sliding", key],
    }
    scenarios = []
    for name, run in runs.items():
        for group, keys in groups.items():
            for c in (.1, .2, .3):
                f = cut_factor(run["shapes"], keys, c)
                scenarios.append(dict(basis=name, group=group, keys=keys,
                    assumed_whole_path_latency_cut_percent=100*c,
                    score_factor=f, score_gain_percent=100*(f-1), measured=False))
    prologue = []
    for saving in (5,7):
        keys = groups["FA3_large_full_including_decode_B16"]
        times = {k:r["candidate_adjusted_us"]-(saving if k in keys else 0)
                 for k,r in rows.items()}
        prologue.append(dict(assumed_saving_us_each=saving, keys=keys,
            modeled_gain_percent=100*(score(rows,times)/score(rows)-1), measured=False))
    return dict(kind="Local planning evidence; scenarios are not achieved scores",
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (
            Path(__file__), ROOT/"A2-11-official-result.json", ROOT/"A2-13-official-result.json",
            ROOT/"a215/paired-analysis.json", ROOT/"a215/paired-result.json",
            ROOT/"a215/native-analysis.json")},
        preserved={n:dict(source=v["provenance"], measured=v["measured"], loads=v["loads"])
                   for n,v in runs.items()},
        scoring=dict(load="46*sliding + 16*full", score="geomean of six reference/candidate load ratios"),
        a215=dict(measured_kind="same-session whole kernel_fn graph replay, one changed path; not a subkernel benchmark",
            measured_raw_medians_us=pair["raw_median_us"], paired_raw_ratio=pair["raw_ratio_parent_over_candidate"],
            paired_latency_cut_percent=100*cut, historical_A13_official_adjusted_target_us=rows[key]["candidate_adjusted_us"],
            historical_target_own_load_share=w, modeled_score_gain_percent=100*(factor-1),
            modeled_score=score(rows)*factor, measured_score=False,
            calibrated_ratio_sensitivity_gain_percent=100*(cut_factor(rows,{key},1-1/pair["calibrated_ratio_parent_over_candidate"])-1),
            per_pair_transfer_gain_percent_range=[min(per_pair), max(per_pair)],
            range_is_confidence_interval=False,
            target_cut_to_model_1_5_percent_aggregate=(1-1.015**-6)/w*100,
            target_cut_to_model_3_percent_aggregate=(1-1.03**-6)/w*100,
            official_check_decision="Defer performance-only confirmation: expected modeled effect is smaller than minimum reported interval; broad correctness coverage remains valuable before any adoption"),
        ranked_path_sensitivity=paths, grouped_scenarios=scenarios, prologue_saving_models=prologue)


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = read_run("A2-13")["shapes"]

    def test_score_reconstructs_receipt(self):
        self.assertAlmostEqual(score(self.rows), 1.210990803459979, places=8)

    def test_six_loads_with_explicit_weights(self):
        totals = load_totals(self.rows)
        self.assertEqual(len(totals),6)
        for load,t in totals.items():
            self.assertAlmostEqual(t["candidate_weighted_us"],
                46*self.rows[load+"/sliding"]["candidate_adjusted_us"]+
                16*self.rows[load+"/full"]["candidate_adjusted_us"])

    def test_uniform_cut(self):
        self.assertAlmostEqual(cut_factor(self.rows,set(self.rows),.2),1.25)

    def test_single_path_analytic_factor(self):
        for key,r in self.rows.items():
            self.assertAlmostEqual(cut_factor(self.rows,{key},.2),
                (1-.2*r["share_of_own_load_candidate"])**(-1/6))

    def test_paired_transfer_is_not_ratio_of_medians(self):
        d=analyze()["a215"]
        self.assertAlmostEqual(d["modeled_score_gain_percent"],.749614434429,places=9)
        self.assertFalse(d["measured_score"])
        self.assertFalse(d["range_is_confidence_interval"])

    def test_no_change_has_no_gain(self):
        self.assertAlmostEqual(cut_factor(self.rows,{"verify 32/full"},0),1)


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        unittest.main(argv=[sys.argv[0]])
    else:
        assert not sys.argv[1:]
        result=analyze()
        (HERE/"evidence.json").write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(result["a215"],indent=2))
