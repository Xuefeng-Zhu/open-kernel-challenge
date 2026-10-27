"""Offline P046 hybrid assessment; P045's rejected gates remain unchanged."""
import argparse
import json
import math
from pathlib import Path

from assess_p045 import ROOT, PARENT_SHA, PARENT_JOB, LOADS, LAYERS, _rows, _correct, _positive

SOURCE_SHA = "001ea12e8764564d507de51e433769f5bd9a9916bfc4c014f0f87522c278545e"
P045_SHA = "c751d3ec5bff008e9f893f3cc6113184e3a78ae255578e6b398c9d22122d3f58"
TARGET_M = (8, 16, 32)


def _ci(receipt):
    value = receipt["result"].get("speedup_ci")
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("Malformed speedup CI")
    lo, hi = (_positive(x, "CI bound") for x in value)
    if lo > hi:
        raise ValueError("Reversed speedup CI")
    return [lo, hi]


def assess(parent, candidate, p045=None):
    old, new = _rows(parent, PARENT_SHA), _rows(candidate, SOURCE_SHA)
    if parent["admission"]["job_id"] != PARENT_JOB or not _correct(parent, old):
        raise ValueError("Expected the passing original P039 receipt")
    parent_score = _positive(parent["result"]["speedup"], "parent score")
    score = _positive(candidate["result"]["speedup"], "candidate score")
    frozen = {load: sum(62 * old[load, layer]["median_ns"] / 1000 for layer in LAYERS) for load in LOADS}
    modeled = dict(frozen)
    rows = []
    for load in LOADS:
        for layer in LAYERS:
            p, c = old[load, layer], new[load, layer]
            ratio = p["paired_ratio"] / c["paired_ratio"]
            target = layer == "pooler" and LOADS[load] in TARGET_M
            row = dict(load=load, layer=layer, m=LOADS[load], target=target, weight=62,
                       parent_us=p["median_ns"] / 1000, candidate_us=c["median_ns"] / 1000,
                       parent_reference_us=p["baseline_ns"] / 1000, reference_us=c["baseline_ns"] / 1000,
                       parent_raw_us=p["median_raw_ns"] / 1000, candidate_raw_us=c["median_raw_ns"] / 1000,
                       parent_reference_raw_us=p["baseline_raw_ns"] / 1000, reference_raw_us=c["baseline_raw_ns"] / 1000,
                       parent_paired_speedup=p["paired_ratio"], candidate_paired_speedup=c["paired_ratio"],
                       normalized_latency_ratio=ratio, normalized_latency_cut=1 - ratio,
                       normalized_latency_increase=ratio - 1)
            rows.append(row)
            if target:
                modeled[load] += 62 * row["parent_us"] * (ratio - 1)
    targets = [row for row in rows if row["target"]]
    controls = [row for row in rows if not row["target"]]
    cut = 1 - math.exp(sum(math.log(row["normalized_latency_ratio"]) for row in targets) / 3)
    factor = math.exp(sum(math.log(frozen[load] / modeled[load]) for load in LOADS) / 6)
    passed = _correct(candidate, new)
    gates = dict(official_correctness_passed=passed,
                 target_geomean_latency_cut_at_least5pct=cut >= .05 - 1e-12,
                 no_individual_target_latency_regression=all(row["normalized_latency_ratio"] <= 1 + 1e-12 for row in targets),
                 target_only_frozen_parent_score_gain_at_least1pct=factor >= 1.01 - 1e-12)
    comparison = dict(available=False, reason="No P045 receipt supplied", original_gates_unchanged=True)
    if p045 is not None:
        if p045["admission"]["submission_sha256"] != P045_SHA:
            raise ValueError("P045 comparison source mismatch")
        comparison.update(job=p045["admission"]["job_id"], reason="P045 correctness did not pass")
        if p045["result"].get("passed") is True:
            previous = _rows(p045, P045_SHA)
            if not _correct(p045, previous):
                raise ValueError("Inconsistent P045 passing correctness evidence")
            comparison.update(available=True, reason="Separate-job reference-normalized comparison only",
                              pooler_target_cuts=[dict(load=load, m=LOADS[load],
                                  p045_normalized_latency_cut=1 - old[load, "pooler"]["paired_ratio"] / previous[load, "pooler"]["paired_ratio"],
                                  p046_normalized_latency_cut=1 - old[load, "pooler"]["paired_ratio"] / new[load, "pooler"]["paired_ratio"],
                                  p046_target=LOADS[load] in TARGET_M)
                                  for load in LOADS if LOADS[load] in (*TARGET_M, 128)])
    parent_ci, ci = _ci(parent), _ci(candidate)
    overlap = None if parent_ci is None or ci is None else max(parent_ci[0], ci[0]) <= min(parent_ci[1], ci[1])
    profiles = [dict(case=row["case"], report=row["report"]) for row in candidate["result"]["records"]
                if row.get("event") == "profile" and row.get("case", {}).get("layer") == "pooler"
                and row["case"].get("m") in (*TARGET_M, 128) and isinstance(row.get("report"), str)]
    return dict(parent_job=PARENT_JOB, job=candidate["admission"]["job_id"], source_sha256=SOURCE_SHA,
                source_commit=candidate["admission"]["source_commit"], passed=passed,
                parent_score=parent_score, score=score, score_ratio=score / parent_score,
                parent_ci=parent_ci, ci=ci, ci_overlap=overlap,
                ci_qualifier="Overlapping score intervals limit confidence; neither overlap nor non-overlap proves change causality across jobs.",
                same_evaluator=parent["result"].get("evaluator") == candidate["result"].get("evaluator"),
                targets=targets, all30shapes=rows, controls_count=len(controls),
                restored_m128_control=next(row for row in controls if row["layer"] == "pooler" and row["m"] == 128),
                target_geomean_normalized_latency_cut=cut, gates=gates, performance_gates_passed=all(gates.values()),
                frozen_parent_load_us=frozen, target_only_modeled_load_us=modeled,
                target_only_frozen_parent_model_factor=factor, target_only_frozen_parent_model_score=parent_score * factor,
                controls_over5pct_latency_movement=[row for row in controls if abs(row["normalized_latency_increase"]) > .05 + 1e-12],
                controls_over5pct_latency_regression=[row for row in controls if row["normalized_latency_increase"] > .05 + 1e-12],
                p045_comparison=comparison, automatic_selection=False, selected_parent_unchanged=True,
                execution_profile_gate=dict(status="unverified", profiles=profiles,
                    requirement="Manually verify three target paths and restored M128 from actual profiles; do not infer library tactics from timings."),
                decision=("Numeric gates pass; preserve P039 pending profile review and independent confirmation."
                          if all(gates.values()) else
                          ("Performance gates missed; correctness passed. Do not promote; preserve P039."
                           if passed else "Official correctness did not pass. Do not promote; preserve P039.")),
                limitations=["Only three target times change in the six-load model; all 27 controls are frozen.",
                             "Separate jobs are reference-normalized, not directly paired P039/P046 timings.",
                             "P045 was rejected under its original gates; this assessment does not revise those gates.",
                             "No native mechanism, removed reduction, copy absence, or dedicated CUDA replay coverage is proved by this assessor."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--parent", type=Path, default=ROOT / "results/remote-P2-04-router-m1-fp32-gemv.json")
    parser.add_argument("--p045", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = assess(json.loads(args.parent.read_text()), json.loads(args.candidate.read_text()),
                    json.loads(args.p045.read_text()) if args.p045 else None)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
