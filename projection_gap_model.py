"""Offline score sensitivities, bound to P046 and an observed public leaderboard.

No network, CUDA, imports of candidate software, or performance measurements.
"""
import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CSV = "results/remote-P2-09-pooler-column-batch2-hybrid.csv"
CSV_SHA256 = "4538b39e8710fa88864127fea35e37598b361838ffcce9ca88d856894e292a35"
RECEIPT = "results/remote-P2-09-pooler-column-batch2-hybrid.json"
PUBLIC = "results/public-leader-gap-20261025.json"
FAMILIES = ("pooler", "qkv_gate", "unpooler", "o", "router")
MAIN = FAMILIES[:3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(rows):
    loads = {}
    for row in rows:
        value = loads.setdefault(row["load"], {"candidate_us": 0., "reference_us": 0.})
        for key in value:
            value[key] += row["weight"] * row[key]
    if any(value["candidate_us"] <= 0 for value in loads.values()):
        raise ValueError("Positive load latencies required")
    result = math.exp(sum(math.log(v["reference_us"] / v["candidate_us"])
                          for v in loads.values()) / len(loads))
    return result, loads


def change(rows, families, cut, minimum_m=1):
    if not 0 <= cut <= 1:
        raise ValueError("Cut must be between zero and one")
    return [{**r, "candidate_us": r["candidate_us"] * (1-cut)}
            if r["layer"] in families and r["m"] >= minimum_m else dict(r) for r in rows]


def required_cut(rows, families, target_score, minimum_m=1):
    if score(rows)[0] >= target_score:
        return 0.
    # A zero-latency family is only a sensitivity boundary, never feasible code.
    if score(change(rows, families, 1., minimum_m))[0] < target_score:
        return None
    lo, hi = 0., 1.
    for _ in range(60):
        mid = (lo+hi)/2
        if score(change(rows, families, mid, minimum_m))[0] < target_score:
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2


def read_rows(root=ROOT):
    path = root/CSV
    if sha(path) != CSV_SHA256:
        raise ValueError("Frozen P046 timing receipt changed")
    with path.open() as stream:
        rows = [dict(load=r["load"], layer=r["layer"], m=int(r["m"]),
                     weight=int(r["weight"]), candidate_us=float(r["candidate_us"]),
                     reference_us=float(r["sglang_us"])) for r in csv.DictReader(stream)]
    if len(rows) != 30 or len({(r["load"], r["layer"]) for r in rows}) != 30:
        raise ValueError("Expected 30 unique P046 cases")
    if any(r["weight"] != 62 for r in rows):
        raise ValueError("Projection shape weights must all equal 62")
    return rows


def task_floors(root=ROOT):
    result = {}
    pattern = r"\| (decode|verify) (\d+) \| m (\d+), layer (\w+) \| 62 \| ([\d.]+) µs \|"
    for kind, batch, m, layer, value in re.findall(pattern, (root/"TASK.md").read_text()):
        result[(f"{kind} {batch}", int(m), layer)] = float(value)
    if len(result) != 30:
        raise ValueError("Expected all 30 rounded official TASK floor entries")
    return result


def build(root=ROOT):
    rows = read_rows(root)
    current, loads = score(rows)
    receipt = json.loads((root/RECEIPT).read_text())
    if not math.isclose(current, receipt["result"]["speedup"], rel_tol=1e-12):
        raise ValueError("Reconstructed score disagrees with official receipt")
    snapshot = json.loads((root/PUBLIC).read_text())
    track = next(t for t in snapshot["tracks"] if t["problem"] == "muse_bf16_proj")
    leader = track["leader"]["speedup"]
    fractions = {load: {family: sum(r["weight"]*r["candidate_us"] for r in rows
                        if r["load"] == load and r["layer"] == family)/values["candidate_us"]
                        for family in FAMILIES} for load, values in loads.items()}
    for values in fractions.values():
        values["three_main_families"] = sum(values[f] for f in MAIN)
    scenarios = []
    for families in (MAIN, ("pooler",), ("qkv_gate",), ("unpooler",)):
        for cut in (.10, .20, .30, .40):
            hypothetical, _ = score(change(rows, families, cut))
            scenarios.append(dict(families=families, uniform_latency_cut=cut,
                                  modeled_score=hypothetical, relative_gain=hypothetical/current-1))
    required = {"three_main_families_all_m": required_cut(rows, MAIN, leader),
                "three_main_families_m_gt_1": required_cut(rows, MAIN, leader, 2),
                "qkv_only": required_cut(rows, ("qkv_gate",), leader),
                "unpooler_only": required_cut(rows, ("unpooler",), leader)}
    remainder = [dict(three_family_cut=cut,
                     required_o_router_cut=required_cut(change(rows, MAIN, cut), ("o", "router"), leader))
                 for cut in (.20, .30)]
    floors = task_floors(root)
    floor_rows = [{**r, "candidate_us": floors[(r["load"], r["m"], r["layer"])]}
                  if r["layer"] in MAIN else r for r in rows]
    floor_score = score(floor_rows)[0]
    return dict(kind="Conditional score sensitivity; no new measurement", parent_job="41ba778763a5433b",
        current_score=current, public_leader=track["leader"], observed_utc=track["observed_utc"],
        leader_over_current=leader/current, uniform_all_path_cut_to_match=1-current/leader,
        scoring="Six equally weighted load ratios; each of five shape timings weighted 62",
        six_load_totals=loads, fractions=fractions,
        equal_load_mean_fractions={f: sum(v[f] for v in fractions.values())/6
                                   for f in (*FAMILIES, "three_main_families")},
        conditional_scenarios=scenarios, required_uniform_cuts=required,
        required_complement_cuts=remainder,
        rounded_task_floor_model=dict(score=floor_score, leader_relative_gap=leader/floor_score-1,
            qualification="Replace only three families with rounded TASK floors; not a strict bound, attainable forecast, or evidence other families must change. Difference from leader is below reported uncertainty."),
        rows=rows, evidence_hashes={p: sha(root/p) for p in (CSV, RECEIPT, PUBLIC, "TASK.md")},
        qualifications=["Leader CLI score rounded to four decimals; provisional and cross-run, not a paired comparison",
            "All untargeted timings held fixed; modeled changes are not achieved or predicted speedups",
            "TASK floors use ideal bandwidth/compute and omit launch, scratch, rereads, scheduling and reduction costs",
            "Scoring arithmetic does not identify measured HBM, occupancy, stalls, or competitor architecture"],
        gpu_executed=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = build()
    args.output.write_text(json.dumps(value, indent=2)+"\n")
    print(json.dumps({key:value[key] for key in ("current_score", "leader_over_current",
          "uniform_all_path_cut_to_match", "required_uniform_cuts", "rounded_task_floor_model")}))
