"""Deterministic LOCAL-only six-load analysis; never imports/runs candidates.

Measured inputs are completed official receipts. Scenarios hold same-receipt
reference latencies fixed and alter specified adjusted candidate latencies only.
Profile activities, raw graph medians, and adjusted score latencies are distinct.
No model is a measured result, a hardware diagnosis, or a leaderboard forecast.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WEIGHTS = {"sliding": 46, "full": 16}
LOADS = ("decode 1", "decode 16", "decode 32", "verify 1", "verify 16", "verify 32")
SOURCES = {
    "A2-11": dict(source="mixed_fa3_decode_parallel_merge.py", sha256="c9a0847502a0b2fe2be55f546a2b3ea7eefa931c851e328e680302ebd828baf0", commit="7e302d2366b194c32c433f6d6f9cc7b0b59b3928", job="cfa8a8670e394f80"),
    "A2-13": dict(source="mixed_fa3_fp8_qk.py", sha256="e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e", commit="21fefb946524a25d8a06562dd28b631d49cd5175", job="c2fb7b23433f4888"),
}
TARGETS = {"decode 1/sliding", "decode 16/sliding", "decode 32/sliding", "decode 32/full"}
PUBLIC_LEADER = dict(attention_score=2.9096724044146387,
    observed_utc="2026-10-25T07:18:00Z",
    source_url="https://openkernelchallenge.dev/api/v1/leaderboard/muse_attention_decode",
    provenance="Public value and timestamp supplied by parent; this analysis makes no network requests.",
    qualification="Different run and aggregate reference normalization; not a paired comparison, per-path target, or guaranteed prize threshold.")
HISTORY = [
    dict(mechanism="E1 full-only separate serial merge", document="REPORT.md",
         evidence_kind="historical Modal paired diagnostic, not official score",
         weighted_paired_ratio=0.916402,
         interpretation="This particular separate serial merge regressed; not evidence against all multi-kernel designs."),
    dict(mechanism="E3 loop-scoped num_stages=3", document="E3_PIPELINE.md",
         evidence_kind="official completed experiment", aggregate_gain_percent=0.2502,
         target_normalized_changes_percent=[-0.03,0.15,-0.20,-0.16],
         interpretation="Inconclusive; generated-code/resource report was not retrieved, so actual pipeline changes are unknown."),
    dict(mechanism="A2-10 independent serial decode merge", document="A2-10-RESULT.md",
         evidence_kind="official completed experiment", sliding_target_normalized_cuts_percent=[-2.308,-5.909,-3.959],
         interpretation="Serial reduction regressed. A2-11's later parallel reduction succeeded on the affected paths."),
    dict(mechanism="A2-12 standalone query preparation", document="A2-12-RESULT.md",
         evidence_kind="official completed experiment", target_normalized_cut_percent=-8.676,
         profile_parent_producer_us=8.1, profile_new_producer_us=6.8, profile_added_preparation_us=2.5,
         interpretation="Whole-call regression despite less producer activity; profile activities are not calibrated wall-time budgets."),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geometric_mean(values):
    return math.exp(sum(math.log(x) for x in values) / len(values))


def load_totals(rows, candidate_times=None):
    times = candidate_times or {k: r["candidate_adjusted_us"] for k, r in rows.items()}
    result = {}
    for load in LOADS:
        selected = [r for r in rows.values() if r["load"] == load]
        assert len(selected) == 2 and {r["layer"] for r in selected} == set(WEIGHTS)
        c = sum(r["weight"] * times[r["key"]] for r in selected)
        s = sum(r["weight"] * r["reference_adjusted_us"] for r in selected)
        result[load] = dict(candidate_weighted_us=c, reference_weighted_us=s,
                            ratio=s / c if c else math.inf)
    return result


def score(rows, candidate_times=None):
    totals = load_totals(rows, candidate_times)
    ratios = [v["ratio"] for v in totals.values()]
    return math.inf if any(math.isinf(x) for x in ratios) else geometric_mean(ratios)


def cost_shapley(old, new):
    """Exact two-component allocation of log((old0+old1)/(new0+new1))."""
    a, b = old
    x, y = new
    return (0.5 * (math.log((a+b)/(x+b)) + math.log((a+y)/(x+y))),
            0.5 * (math.log((a+b)/(a+y)) + math.log((x+b)/(x+y))))


def parse_activity(report):
    kernels=[]
    for line in report.splitlines():
        match=re.match(r"^(.*?)\s+([0-9.]+)\s+µs\s+([0-9.]+)%\s+(\d+)\s*$",line)
        if match:
            kernels.append(dict(name=match[1].strip(),activity_us=float(match[2]),
                                reported_activity_share_percent=float(match[3]),calls=int(match[4])))
    return kernels


def read_run(name):
    spec = SOURCES[name]
    source = ROOT / spec["source"]
    assert sha(source) == spec["sha256"], f"Frozen source mismatch: {source}"
    receipt_path = ROOT / f"{name}-official-result.json"
    receipt = json.loads(receipt_path.read_text())
    result = receipt["result"]
    assert receipt["id"] == spec["job"] and receipt["status"] == "done"
    assert result["passed"] and result["status"] == "ok"
    benchmarks = [r for r in result["records"] if r["event"] == "benchmark"]
    tests = [r for r in result["records"] if r["event"] == "test"]
    assert len(benchmarks) == 12 and len(tests) == 16
    assert all(r["status"] == "pass" for r in benchmarks + tests)
    profiles = {f'{r["case"]["load"]}/{r["case"]["layer"]}': r["report"]
                for r in result["records"] if r["event"] == "profile"}
    rows = {}
    for r in benchmarks:
        case = r["case"]
        key = f'{case["load"]}/{case["layer"]}'
        assert key not in rows and case["weight"] == WEIGHTS[case["layer"]]
        rows[key] = dict(key=key, load=case["load"], layer=case["layer"], batch=case["batch"],
            q_len=case.get("q_len", 1), weight=case["weight"], case=case,
            changed_fp8_path=key in TARGETS,
            candidate_adjusted_us=r["median_ns"] / 1000,
            reference_adjusted_us=r["baseline_ns"] / 1000,
            candidate_raw_median_us=r["median_raw_ns"] / 1000,
            reference_raw_median_us=r["baseline_raw_ns"] / 1000,
            paired_ratio=r["paired_ratio"], recorded_timing_samples=r["runs"],
            harness_floor_us=r["speed_of_light_ns"] / 1000,
            reported_replay_overhead_us=r["replay_overhead"]["overhead_ns"] / 1000,
            profile_activity_report=profiles[key],profile_activities=parse_activity(profiles[key]))
        assert math.isclose(rows[key]["reference_adjusted_us"] / rows[key]["candidate_adjusted_us"],
                            r["paired_ratio"], rel_tol=1e-10)
    totals = load_totals(rows)
    reconstructed = score(rows)
    assert math.isclose(reconstructed, result["speedup"], rel_tol=1e-10)
    assert math.isclose(result["baseline_us"] / result["score_us"], result["speedup"], rel_tol=1e-10)
    for key, row in rows.items():
        load = totals[row["load"]]
        row["candidate_weighted_us"] = row["weight"] * row["candidate_adjusted_us"]
        row["reference_weighted_us"] = row["weight"] * row["reference_adjusted_us"]
        row["share_of_own_load_candidate"] = row["candidate_weighted_us"] / load["candidate_weighted_us"]
        row["score_log_elasticity_to_path_speedup"] = row["share_of_own_load_candidate"] / 6
        row["weighted_reference_minus_candidate_us"] = row["reference_weighted_us"] - row["candidate_weighted_us"]
        row["candidate_over_harness_floor"] = row["candidate_adjusted_us"] / row["harness_floor_us"]
        row["weighted_gap_to_harness_floor_us"] = row["weight"] * (row["candidate_adjusted_us"] - row["harness_floor_us"])
    return dict(name=name, provenance=dict(**spec, receipt_file=receipt_path.name,
                receipt_sha256=sha(receipt_path), evaluator=result["evaluator"],
                # Preserve useful hardware observations, not the worker identifier.
                machine={k: result["machine"][k] for k in ("gpu_before", "gpu_after")},
                receipt_content_hash=receipt["content_hash"]),
                measured=dict(speedup=result["speedup"], speedup_ci=result["speedup_ci"],
                    score_us=result["score_us"], baseline_us=result["baseline_us"],
                    prerequisite_passes=len(tests), benchmark_passes=len(benchmarks),
                    reconstructed_six_load_score=reconstructed), shapes=rows, loads=totals)


def attribution(parent, candidate):
    old, new = parent["shapes"], candidate["shapes"]
    normalized = {k: new[k]["candidate_adjusted_us"] * old[k]["reference_adjusted_us"] /
                   new[k]["reference_adjusted_us"] for k in old}
    rows = []
    for load in LOADS:
        keys = [f"{load}/{layer}" for layer in ("sliding", "full")]
        weighted = lambda data, field: [data[k]["weight"] * data[k][field] for k in keys]
        c_effect = cost_shapley(weighted(old, "candidate_adjusted_us"), weighted(new, "candidate_adjusted_us"))
        s_effect = cost_shapley(weighted(old, "reference_adjusted_us"), weighted(new, "reference_adjusted_us"))
        n_effect = cost_shapley(weighted(old, "candidate_adjusted_us"), [old[k]["weight"] * normalized[k] for k in keys])
        for i, key in enumerate(keys):
            a, b = old[key], new[key]
            rows.append(dict(key=key, changed_fp8_path=key in TARGETS, load=load, layer=a["layer"], weight=a["weight"],
                parent_candidate_adjusted_us=a["candidate_adjusted_us"], candidate_adjusted_us=b["candidate_adjusted_us"],
                parent_reference_adjusted_us=a["reference_adjusted_us"], reference_adjusted_us=b["reference_adjusted_us"],
                parent_candidate_raw_median_us=a["candidate_raw_median_us"], candidate_raw_median_us=b["candidate_raw_median_us"],
                parent_reference_raw_median_us=a["reference_raw_median_us"], reference_raw_median_us=b["reference_raw_median_us"],
                normalized_candidate_at_parent_reference_us=normalized[key],
                normalized_latency_cut_percent=100 * (1-normalized[key]/a["candidate_adjusted_us"]),
                weighted_candidate_time_change_us=a["weight"]*(b["candidate_adjusted_us"]-a["candidate_adjusted_us"]),
                candidate_time_log_score_contribution=c_effect[i]/6,
                reference_time_log_score_contribution=-s_effect[i]/6,
                actual_log_score_contribution=(c_effect[i]-s_effect[i])/6,
                normalized_log_score_contribution=n_effect[i]/6))
    actual_ratio = candidate["measured"]["speedup"] / parent["measured"]["speedup"]
    assert math.isclose(sum(r["actual_log_score_contribution"] for r in rows), math.log(actual_ratio), abs_tol=1e-12)
    normal_score = score(old, normalized)
    assert math.isclose(sum(r["normalized_log_score_contribution"] for r in rows),
                        math.log(normal_score/parent["measured"]["speedup"]), abs_tol=1e-12)
    target_only = {k: normalized[k] if k in TARGETS else old[k]["candidate_adjusted_us"] for k in old}
    grouped = {}
    for label, selected in (("four_changed_paths", TARGETS), ("eight_unchanged_paths", set(old)-TARGETS)):
        picked = [r for r in rows if r["key"] in selected]
        grouped[label] = {field: sum(r[field] for r in picked) for field in (
            "candidate_time_log_score_contribution", "reference_time_log_score_contribution",
            "actual_log_score_contribution", "normalized_log_score_contribution")}
        grouped[label]["normalized_multiplicative_factor"] = math.exp(grouped[label]["normalized_log_score_contribution"])
    return dict(measured_cross_job_gain_percent=100*(actual_ratio-1),
        confidence_intervals_overlap=max(parent["measured"]["speedup_ci"][0],candidate["measured"]["speedup_ci"][0]) <=
                                     min(parent["measured"]["speedup_ci"][1],candidate["measured"]["speedup_ci"][1]),
        target_only_counterfactual_score=score(old,target_only),
        target_only_counterfactual_gain_percent=100*(score(old,target_only)/parent["measured"]["speedup"]-1),
        all_paths_normalized_counterfactual_score=normal_score,
        actual_minus_normalized_log_score=math.log(candidate["measured"]["speedup"]/normal_score),
        interpretation="Shapley allocates each two-shape load's logarithmic ratio exactly, then divides by six. This is mathematical attribution, not source causality. Reference drift and unchanged-path drift remain visible; cross-job runs are not paired parent/candidate trials.",
        grouped=grouped, shapes=rows)


def families(rows):
    groups = {f"path:{k}": [k] for k in rows}
    groups.update({
        "family:all_paths": list(rows),
        "family:all_full": [k for k,r in rows.items() if r["layer"] == "full"],
        "family:all_sliding": [k for k,r in rows.items() if r["layer"] == "sliding"],
        "family:decode": [k for k,r in rows.items() if r["q_len"] == 1],
        "family:verify": [k for k,r in rows.items() if r["q_len"] == 8],
        "family:four_fallback_paths": sorted(TARGETS),
        "family:eight_FA3_paths": sorted(set(rows)-TARGETS),
        "family:batched_full_FA3": ["decode 16/full", "verify 16/full", "verify 32/full"],
        "family:batched_full_all": ["decode 16/full", "decode 32/full", "verify 16/full", "verify 32/full"],
        "family:sliding_decode": [f"decode {b}/sliding" for b in (1,16,32)],
        "family:sliding_verify": [f"verify {b}/sliding" for b in (1,16,32)],
    })
    return groups


def accelerate(rows, selected, factor):
    return {k: r["candidate_adjusted_us"] / factor if k in selected else r["candidate_adjusted_us"]
            for k,r in rows.items()}


def required_factor(rows, selected, target_score):
    if target_score <= score(rows):
        return 1.0
    ceiling = score(rows, accelerate(rows, selected, math.inf))
    if target_score >= ceiling:
        return None
    lo, hi = 1.0, 2.0
    while score(rows, accelerate(rows, selected, hi)) < target_score:
        hi *= 2
    for _ in range(80):
        mid=(lo+hi)/2
        if score(rows, accelerate(rows, selected, mid)) < target_score: lo=mid
        else: hi=mid
    return hi


def scenarios(run):
    rows=run["shapes"]
    base=run["measured"]["speedup"]
    output=[]
    required=[]
    for label, keys in families(rows).items():
        for factor in (1.1,1/0.9,1.25,1.5,2.0,4.0,math.inf):
            value=score(rows,accelerate(rows,keys,factor))
            output.append(dict(run=run["name"], group=label, selected_paths=";".join(keys),
                kind="modeled_uniform_path_acceleration" if math.isfinite(factor) else "modeled_zero_latency_ceiling",
                path_acceleration=factor if math.isfinite(factor) else None,
                path_latency_cut_percent=100*(1-1/factor),
                score=value if math.isfinite(value) else None, unbounded=math.isinf(value),
                score_gain_percent=100*(value/base-1) if math.isfinite(value) else None))
        floor_times={k:r["harness_floor_us"] if k in keys else r["candidate_adjusted_us"] for k,r in rows.items()}
        floor_score=score(rows,floor_times)
        output.append(dict(run=run["name"],group=label,selected_paths=";".join(keys),
            kind="modeled_harness_floor_substitution",path_acceleration=None,score=floor_score,
            path_latency_cut_percent=None,
            unbounded=False,score_gain_percent=100*(floor_score/base-1)))
        for target in (1.3,1.5,2.0,PUBLIC_LEADER["attention_score"]):
            factor=required_factor(rows,keys,target)
            required.append(dict(run=run["name"],group=label,target_score=target,
                required_uniform_path_acceleration=factor,attainable_under_zero_floor_model=factor is not None,
                score_at_reported_harness_floors=floor_score,
                target_within_reported_harness_floor_model=target <= floor_score,
                required_uniform_factor_respects_every_selected_harness_floor=(factor is not None and
                    all(rows[k]["candidate_adjusted_us"] / factor >= rows[k]["harness_floor_us"] for k in keys)),
                target_is_provided_public_leader=target == PUBLIC_LEADER["attention_score"],
                target_is_known_prize_threshold=False))
    return output,required


def write_csv(path,rows):
    assert path.name.startswith("weighted_")
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader();writer.writerows(rows)


def analyze():
    runs={name:read_run(name) for name in SOURCES}
    comparison=attribution(runs["A2-11"],runs["A2-13"])
    scenario_rows=[];required=[];shape_rows=[];load_rows=[]
    for run in runs.values():
        sc,req=scenarios(run);scenario_rows.extend(sc);required.extend(req)
        for row in run["shapes"].values():
            shape_rows.append(dict(run=run["name"],**{k:v for k,v in row.items() if k not in ("case","profile_activity_report","profile_activities")}))
        for load,row in run["loads"].items():load_rows.append(dict(run=run["name"],load=load,**row))
    history=[dict(item,document_sha256=sha(ROOT/item["document"])) for item in HISTORY]
    leader_comparison={name:dict(equal_normalization_latency_ratio=run["measured"]["speedup"]/PUBLIC_LEADER["attention_score"],
        equal_normalization_further_latency_cut_percent=100*(1-run["measured"]["speedup"]/PUBLIC_LEADER["attention_score"]),
        equal_normalization_required_uniform_all_path_acceleration=PUBLIC_LEADER["attention_score"]/run["measured"]["speedup"])
        for name,run in runs.items()}
    requested_models=[r for r in scenario_rows if r["run"]=="A2-13" and
        ((r["group"]=="path:decode 32/full" and r["path_acceleration"] in (1/0.9,1.25)) or
         (r["group"]=="family:all_full" and r["path_acceleration"] in (1.25,2.0)))]
    prologue_models=[]
    prologue_paths=["decode 16/full","verify 16/full","verify 32/full"]
    for run in runs.values():
        for saved_us in (5,7):
            times={k:r["candidate_adjusted_us"]-(saved_us if k in prologue_paths else 0)
                   for k,r in run["shapes"].items()}
            hypothetical=score(run["shapes"],times)
            prologue_models.append(dict(run=run["name"],assumed_saving_us_per_selected_call=saved_us,
                selected_paths=prologue_paths,modeled_score=hypothetical,
                modeled_gain_percent=100*(hypothetical/run["measured"]["speedup"]-1),measured=False))
    data=dict(kind="local deterministic measured-versus-modeled weighted analysis",script_sha256=sha(Path(__file__)),
        formulas=dict(load_candidate="46*C_sliding + 16*C_full",load_reference="46*S_sliding + 16*S_full",
                      score="exp(mean_over_six_loads(log(load_reference/load_candidate)))",
                      normalized_latency="(C_A13/S_A13)*S_A11",
                      local_score_elasticity="candidate_weighted_path_time / candidate_weighted_load_time / 6"),
        limitations=["No GPU, candidate imports, external requests, or new timings.",
          "Adjusted candidate median_ns and baseline_ns reproduce scoring; raw medians are retained separately.",
          "Recorded timing sample count is not an inferred count of underlying graph replays.",
          "Same evaluator/image, but different GPU nodes/driver versions and fresh inputs; cross-job causality is not established.",
          "Reported score confidence intervals overlap; no pooled/bootstrap confidence interval invented.",
          "Profile activity sums are not wall times and do not prove registers, bandwidth, occupancy, spills, or native FP8 instructions.",
          "Floor substitutions and zero-latency ceilings are optimistic arithmetic, not attainable hardware promises.",
          "Scenario target scores 1.3/1.5/2.0 are illustrative. The separately labeled leader score was supplied by the parent from a public snapshot, not fetched or paired with these runs.",
          "Equal-normalization leader latency ratios are aggregate scenario arithmetic, not per-path reduction promises or guaranteed prize qualification."],
        runs=runs,comparison=comparison,scenarios=scenario_rows,required_factors=required,
        historical_negative_evidence=history,limited_full_prologue_savings_models=prologue_models,
        public_leader_snapshot=PUBLIC_LEADER,equal_normalization_leader_comparison=leader_comparison,
        requested_full_path_models=requested_models)
    (HERE/"weighted_analysis.json").write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+"\n")
    write_csv(HERE/"weighted_shapes.csv",shape_rows)
    write_csv(HERE/"weighted_loads.csv",load_rows)
    write_csv(HERE/"weighted_attribution.csv",comparison["shapes"])
    write_csv(HERE/"weighted_scenarios.csv",scenario_rows)
    write_csv(HERE/"weighted_required_factors.csv",required)
    print(json.dumps(dict(measured_scores={k:v["measured"]["speedup"] for k,v in runs.items()},
        actual_gain_percent=comparison["measured_cross_job_gain_percent"],
        target_only_modeled_gain_percent=comparison["target_only_counterfactual_gain_percent"],
        normalized_groups=comparison["grouped"],shape_rows=len(shape_rows),scenario_rows=len(scenario_rows)),indent=2))


class WeightedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.runs={name:read_run(name) for name in SOURCES}
    def test_frozen_sources_and_actual_scores(self):
        for run in self.runs.values():self.assertAlmostEqual(score(run["shapes"]),run["measured"]["speedup"],places=10)
    def test_weighted_loads_not_unweighted_shape_average(self):
        run=self.runs["A2-13"];rows=run["shapes"]
        self.assertEqual(len(run["loads"]),6)
        self.assertGreater(abs(score(rows)-geometric_mean([r["paired_ratio"] for r in rows.values()])),1e-3)
    def test_shapley_two_component_identity(self):
        for old,new in [((2,3),(1,2)),((2,3),(3,4)),((4,8),(4,7)),((10,1),(2,4))]:
            self.assertAlmostEqual(sum(cost_shapley(old,new)),math.log(sum(old)/sum(new)),places=14)
    def test_actual_gain_decomposition(self):
        a=attribution(self.runs["A2-11"],self.runs["A2-13"])
        self.assertAlmostEqual(math.expm1(sum(r["actual_log_score_contribution"] for r in a["shapes"]))*100,a["measured_cross_job_gain_percent"],places=10)
    def test_uniform_all_paths_acceleration(self):
        rows=self.runs["A2-13"]["shapes"]
        self.assertAlmostEqual(score(rows,accelerate(rows,set(rows),2)),2*score(rows),places=12)
    def test_elasticities_sum_to_one(self):
        for run in self.runs.values():self.assertAlmostEqual(sum(r["score_log_elasticity_to_path_speedup"] for r in run["shapes"].values()),1.0,places=14)
    def test_single_path_zero_ceiling(self):
        rows=self.runs["A2-13"]["shapes"]
        for key,row in rows.items():
            ceiling=score(rows,accelerate(rows,{key},math.inf))
            expected=score(rows)*(1/(1-row["share_of_own_load_candidate"]))**(1/6)
            self.assertAlmostEqual(ceiling,expected,places=12)
    def test_scenario_inversion_and_unattainable(self):
        rows=self.runs["A2-13"]["shapes"];keys={"decode 32/full"}
        factor=required_factor(rows,keys,1.3)
        self.assertIsNotNone(factor)
        self.assertAlmostEqual(score(rows,accelerate(rows,keys,factor)),1.3,places=12)
        self.assertIsNone(required_factor(rows,keys,2.0))
    def test_raw_adjusted_are_distinct(self):
        for run in self.runs.values():
            self.assertTrue(all(r["candidate_raw_median_us"]>r["candidate_adjusted_us"] for r in run["shapes"].values()))
    def test_profile_separate_from_adjusted_time(self):
        row=self.runs["A2-13"]["shapes"]["decode 32/full"]
        self.assertEqual([x["name"] for x in row["profile_activities"]],["_fused","_merge"])
        self.assertEqual(row["profile_activities"][0]["activity_us"],311.1)
        self.assertNotAlmostEqual(sum(x["activity_us"] for x in row["profile_activities"]),row["candidate_adjusted_us"],places=1)
    def test_supplied_leader_equal_normalization_model(self):
        rows=self.runs["A2-13"]["shapes"]
        factor=PUBLIC_LEADER["attention_score"]/score(rows)
        self.assertAlmostEqual(score(rows,accelerate(rows,set(rows),factor)),PUBLIC_LEADER["attention_score"],places=12)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(WeightedTests))
        raise SystemExit(0 if result.wasSuccessful() else 1)
    analyze()
