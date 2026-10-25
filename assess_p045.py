"""Offline P045 receipt assessment. No execution, selection, or network actions."""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT_SHA = "0c0c41119ef925b03a33ccab230bc64176cc22c242e3fea77ce01e317376defc"
SOURCE_SHA = "c751d3ec5bff008e9f893f3cc6113184e3a78ae255578e6b398c9d22122d3f58"
PARENT_JOB = "d5636ac8d1404abd"
LOADS = {"decode 1": 1, "decode 16": 16, "decode 32": 32,
         "verify 1": 8, "verify 16": 128, "verify 32": 256}
LAYERS = ("qkv_gate", "o", "pooler", "unpooler", "router")
TARGET_M = (8, 16, 32, 128)


def _finite_tree(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise ValueError("Nonfinite numeric value in benchmark row")
    elif isinstance(value, dict):
        for item in value.values():
            _finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            _finite_tree(item)


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


def _rows(receipt, expected_sha):
    if receipt["admission"]["submission_sha256"] != expected_sha:
        raise ValueError("Receipt source SHA256 mismatch")
    if receipt["status"] != "done":
        raise ValueError("Receipt is not terminal/done")
    rows = {}
    for row in receipt["result"]["records"]:
        if row.get("event") != "benchmark":
            continue
        _finite_tree(row)
        case = row["case"]
        key = (case["load"], case["layer"])
        if key in rows:
            raise ValueError(f"Duplicate benchmark row: {key}")
        if case["load"] not in LOADS or case["layer"] not in LAYERS:
            raise ValueError(f"Unexpected shape: {key}")
        if case["m"] != LOADS[case["load"]] or case["weight"] != 62:
            raise ValueError(f"Shape M/weight mismatch: {key}")
        for field in ("median_ns", "baseline_ns", "median_raw_ns", "baseline_raw_ns", "paired_ratio"):
            _positive(row[field], field)
        if not math.isclose(row["median_ns"], row["baseline_ns"] / row["paired_ratio"], rel_tol=1e-6):
            raise ValueError(f"Calibrated median/paired ratio mismatch: {key}")
        rows[key] = row
    if set(rows) != {(load, layer) for load in LOADS for layer in LAYERS}:
        raise ValueError("Expected exactly all 30 unique benchmark shapes")
    return rows


def _correct(receipt, rows):
    return (receipt["result"].get("passed") is True
            and receipt.get("prerequisite_tests") == {"passed": 9, "total": 9}
            and receipt.get("shape_checks") == {"passed": 30, "total": 30}
            and all(row.get("status") == "pass" for row in rows.values()))


def assess(parent, candidate):
    """Fail closed on incomplete/malformed data; never modify the selected parent."""
    old, new = _rows(parent, PARENT_SHA), _rows(candidate, SOURCE_SHA)
    if parent["admission"]["job_id"] != PARENT_JOB or not _correct(parent, old):
        raise ValueError("Expected the passing original P039 reference receipt")
    parent_score = _positive(parent["result"]["speedup"], "parent speedup")
    score = _positive(candidate["result"]["speedup"], "candidate speedup")
    parent_loads = {load: sum(62 * old[load, layer]["median_ns"] / 1000 for layer in LAYERS)
                    for load in LOADS}
    modeled_loads = dict(parent_loads)
    rows = []
    for load in LOADS:
        for layer in LAYERS:
            p, c = old[load, layer], new[load, layer]
            ratio = p["paired_ratio"] / c["paired_ratio"]
            target = layer == "pooler" and LOADS[load] in TARGET_M
            row = dict(load=load, layer=layer, m=LOADS[load], weight=62, target=target,
                       parent_us=p["median_ns"] / 1000, candidate_us=c["median_ns"] / 1000,
                       parent_reference_us=p["baseline_ns"] / 1000, reference_us=c["baseline_ns"] / 1000,
                       parent_raw_us=p["median_raw_ns"] / 1000, candidate_raw_us=c["median_raw_ns"] / 1000,
                       parent_reference_raw_us=p["baseline_raw_ns"] / 1000,
                       reference_raw_us=c["baseline_raw_ns"] / 1000,
                       parent_paired_speedup=p["paired_ratio"], candidate_paired_speedup=c["paired_ratio"],
                       normalized_latency_ratio=ratio, normalized_latency_increase=ratio - 1,
                       normalized_latency_cut=1 - ratio)
            rows.append(row)
            if target:
                modeled_loads[load] += 62 * row["parent_us"] * (ratio - 1)
    targets = [row for row in rows if row["target"]]
    controls = [row for row in rows if not row["target"]]
    improvement = 1 - math.exp(sum(math.log(row["normalized_latency_ratio"]) for row in targets) / 4)
    model_factor = math.exp(sum(math.log(parent_loads[load] / modeled_loads[load]) for load in LOADS) / 6)
    passed = _correct(candidate, new)
    gates = dict(official_correctness_passed=passed,
                 target_geomean_latency_cut_at_least10pct=improvement >= .10 - 1e-12,
                 no_target_latency_increase_over5pct=all(row["normalized_latency_ratio"] <= 1.05 + 1e-12 for row in targets),
                 target_only_frozen_parent_score_gain_at_least1p5pct=model_factor >= 1.015 - 1e-12)
    profiles = [dict(case=row["case"], report=row["report"]) for row in candidate["result"]["records"]
                if row.get("event") == "profile" and row.get("case", {}).get("layer") == "pooler"
                and row["case"].get("m") in TARGET_M and isinstance(row.get("report"), str)]
    return dict(parent_job=PARENT_JOB, job=candidate["admission"]["job_id"],
                source_commit=candidate["admission"]["source_commit"], source_sha256=SOURCE_SHA,
                parent_score=parent_score, score=score, score_ratio=score / parent_score,
                parent_ci=parent["result"].get("speedup_ci"), ci=candidate["result"].get("speedup_ci"),
                same_evaluator=parent["result"].get("evaluator") == candidate["result"].get("evaluator"),
                passed=passed, targets=targets, all30shapes=rows, controls_count=len(controls),
                target_geomean_normalized_latency_cut=improvement, gates=gates,
                performance_gates_passed=all(gates.values()),
                frozen_parent_load_us=parent_loads, target_only_modeled_load_us=modeled_loads,
                target_only_frozen_parent_model_factor=model_factor,
                target_only_frozen_parent_model_score=parent_score * model_factor,
                controls_over5pct_latency_movement=[row for row in controls if abs(row["normalized_latency_increase"]) > .05 + 1e-12],
                controls_over5pct_latency_regression=[row for row in controls if row["normalized_latency_increase"] > .05 + 1e-12],
                execution_profile_gate=dict(status="unverified", target_profiles=profiles,
                    requirement="Review actual profiles for all four target paths, copies and reduction kernels; receipt timing or source bmm calls do not prove execution."),
                selected_parent_unchanged=True, automatic_selection=False,
                decision=("Numeric gates pass; retain P039 pending execution-profile review and independent confirmation."
                          if all(gates.values()) else
                          ("Performance gates missed; correctness passed. Do not promote; preserve P039."
                           if passed else "Official correctness did not pass. Do not promote; preserve P039.")),
                limitations=["Separate official jobs normalized to each run's reference, not a directly paired P039/candidate experiment.",
                             "Frozen-parent score is modeled: only four target latencies change; all 26 controls remain fixed.",
                             "Control shifts are reported, not automatically attributed to this isolated source change.",
                             "No performance number proves a batched library tactic, removed split-K, absent copies, or CUDA replay/concurrency coverage."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path, help="Saved sanitized official P045 receipt")
    parser.add_argument("--parent", type=Path, default=ROOT / "results/remote-P2-04-router-m1-fp32-gemv.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = assess(json.loads(args.parent.read_text()), json.loads(args.candidate.read_text()))
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
