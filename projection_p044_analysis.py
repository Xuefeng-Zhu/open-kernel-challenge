"""Receipt-bound causal limits and score sensitivity; offline, no GPU claims."""
import csv
import hashlib
import json
from pathlib import Path
from projection_p041_plan import ROOT, read_rows, score, sensitivity


def build():
    files = ["results/remote-P2-04-router-m1-fp32-gemv.csv",
             "results/remote-P2-07-pooler-lossless-codec.csv",
             "results/remote-R06-pooler-triton8.csv"]
    tables = []
    for name in files:
        with (ROOT/name).open() as handle:
            tables.append(list(csv.DictReader(handle)))
    def target(table):
        return next(r for r in table if r["layer"] == "pooler" and int(r["m"]) == 8)
    selected, codec, historical_raw = map(target, tables)
    def normalized(row):
        return float(row["candidate_us"])/float(row["sglang_us"])
    rows = read_rows()
    base = score(rows)[0]
    activity_add_us = {16: 1.5, 32: 1.5, 128: 1.6, 256: 1.6}
    scenarios = []
    for reduction in (0.5, 1.0):
        altered = [{**r, "candidate_us": r["candidate_us"] - reduction*activity_add_us[r["m"]]}
                   if r["layer"] == "router" and r["m"] in activity_add_us else r for r in rows]
        value = score(altered)[0]
        scenarios.append(dict(activity_add_eliminated_fraction=reduction,
                              hypothetical_score=value, hypothetical_relative_gain=value/base-1))
    return dict(
        source_receipt_hashes={n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in files},
        selected_score=base,
        cross_run_pooler_m8_normalized_latency_ratios=dict(
            P043_over_R06=normalized(codec)/normalized(historical_raw),
            R06_over_P039=normalized(historical_raw)/normalized(selected),
            P043_over_P039=normalized(codec)/normalized(selected)),
        attribution_limit="R06 is historical, not the matched P043 raw-control execution; decode causality remains unmeasured",
        router_activity_add_hypotheticals=scenarios,
        scenario_limit="Subtracts separate instrumented activity durations from official adjusted latency only as a sensitivity model; not measured savings or a rigorous achievable bound",
        pooler_all_m_gt1_10pct_cut_modeled_relative_gain=sensitivity(rows, "pooler", .1, 2),
        no_gpu_executed=True)


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
