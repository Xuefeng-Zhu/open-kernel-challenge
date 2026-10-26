"""Recompute P059 conclusions from saved raw receipts; no GPU or network calls."""
import csv
import hashlib
import json
from pathlib import Path

from projection_p059_modal import validate_report


def main():
    root = Path(__file__).resolve().parent
    receipt = root / "results/remote-P059-native/report.json"
    report = json.loads(receipt.read_text())
    assert validate_report(report)
    case = report["cases"][0]
    series = case["series"]
    comparisons = {}
    samples = []
    for name, value in series.items():
        summary = value["summary"]
        ratio = summary["median_paired_raw_ratio"]
        comparisons[name] = dict(
            runners=value["runner_mapping"],
            parent_median_us=summary["raw_us"]["parent"]["median"],
            candidate_median_us=summary["raw_us"]["candidate"]["median"],
            paired_parent_over_candidate=ratio,
            paired_candidate_latency_cut=1 - 1 / ratio,
            winning_pairs=summary["winning_pairs"],
            pair_distribution=summary["pair_distribution"],
            raw_us_distribution=summary["raw_us"],
        )
        for phase in ("warmup", "samples"):
            for row in value[phase]:
                samples.append(dict(series=name, phase=phase, round=row["round"],
                    seed=row["seed"], order=",".join(row["order"]),
                    parent_us=row["us"]["parent"], candidate_us=row["us"]["candidate"],
                    parent_pass=row["correctness"]["parent"],
                    candidate_pass=row["correctness"]["candidate"]))
    symbols = {name: [e["symbol"] for e in record["cuda_events"]]
               for name, record in case["activities"].items()}
    output = dict(
        source_commit="6856cc2cb72d398c88c78b63a375bb3d2e8628e0",
        app_id="ap-vbrU0sPpoRjkmN7e7OgK4R",
        raw_report_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
        status="correctness-pass-performance-gate-rejected",
        oracle_checks=report["oracle_checks"],
        assessment=report["assessment"], comparisons=comparisons,
        selected_plans=case["selected"],
        p046_split_activity_symbols_equal=symbols["p046"] == symbols["split"],
        activity_symbols=symbols,
        decision="Preserve P046; no official check or promotion from this diagnostic",
        qualifications=[
            "Raw graph times retain overhead; not official calibrated times or score",
            "Mask change admits a different tile/custom configuration as well as split count",
            "Equal activity symbols do not establish exact tactic identity or launch geometry",
            "Single QKV M128 case; nearby and complete official shapes untested",
            "Separate-stream graph checks do not establish actual hardware overlap",
            "No native assembly, occupancy, HBM or stall counters collected",
        ],
    )
    (root / "results/router-p059-assessment.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n")
    with (root / "results/router-p059-paired.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(samples[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(samples)
    print(json.dumps({"oracle_checks": report["oracle_checks"],
                      "samples_retained_including_warmup": len(samples),
                      "p046_gate_passed": report["assessment"]["screen_passed"],
                      "no_external_action": True}))


if __name__ == "__main__":
    main()
