"""Reconstruct frozen P039 evidence and a bounded diagnostic plan, offline only.

No CUDA, account, network, or process launches. Byte models describe compulsory
data and arithmetic; they do not measure HBM traffic or achieved utilization.
"""
import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = "candidates/projection_router_m1_fp32_gemv.py"
SOURCE_SHA256 = "0c0c41119ef925b03a33ccab230bc64176cc22c242e3fea77ce01e317376defc"
RECEIPT = "results/remote-P2-04-router-m1-fp32-gemv.json"
CSV = "results/remote-P2-04-router-m1-fp32-gemv.csv"
REPRESENTATIVES = (("pooler", 8), ("pooler", 128), ("pooler", 256),
                   ("qkv_gate", 128), ("unpooler", 16))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dimensions():
    tree = ast.parse((ROOT / "official/reference.py").read_text())
    names = {"LAYERS", "UNPOOL_SLICE", "SLICED_MAX_TOKENS"}
    return {node.targets[0].id: ast.literal_eval(node.value)
            for node in tree.body if isinstance(node, ast.Assign)
            and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in names}


def compulsory_model(layer, m):
    spec = dimensions()
    n, k = spec["LAYERS"][layer]
    if layer == "unpooler" and m <= spec["SLICED_MAX_TOKENS"]:
        n = spec["UNPOOL_SLICE"]
    width = 4 if layer == "router" else 2
    w, x, out = n * k * width, m * k * 2, m * n * width
    flops = 2 * m * n * k
    return dict(m=m, n=n, k=k, weights_bytes=w, activations_bytes=x,
                output_bytes=out, unique_bytes=w+x+out, flops=flops,
                flops_per_unique_byte=flops/(w+x+out),
                ideal_weight_value_reuses=m,
                ideal_input_value_reuses=n,
                official_model_floor_us=max((w+x+out)/4.8e12, flops/989e12)*1e6,
                qualification="Compulsory unique data and dense arithmetic; excludes scratch, rereads and launch cost; not measured traffic/reuse/bottleneck")


def score(rows):
    loads = {}
    for row in rows:
        item = loads.setdefault(row["load"], dict(candidate_us=0., reference_us=0.))
        item["candidate_us"] += row["weight"] * row["candidate_us"]
        item["reference_us"] += row["weight"] * row["reference_us"]
    ratios = [v["reference_us"] / v["candidate_us"] for v in loads.values()]
    return math.exp(sum(map(math.log, ratios))/len(ratios)), loads


def read_rows():
    with (ROOT / CSV).open() as file:
        raw = list(csv.DictReader(file))
    # Field names belong to the sanitized official receipt, not a new spec.
    rows = [dict(load=r["load"], layer=r["layer"], m=int(r["m"]),
                 weight=int(r["weight"]), candidate_us=float(r["candidate_us"]),
                 reference_us=float(r["sglang_us"])) for r in raw]
    if len(rows) != 30 or any(r["weight"] != 62 for r in rows):
        raise ValueError("Expected frozen projection 30 cases, each weight 62")
    if len({(r["load"], r["layer"]) for r in rows}) != 30:
        raise ValueError("Duplicate shape in receipt")
    return rows


def sensitivity(rows, family, reduction, minimum_m=1, only_m=None):
    if not 0 <= reduction < 1:
        raise ValueError("Latency reduction must be in [0, 1)")
    changed = [{**r, "candidate_us": r["candidate_us"]*(1-reduction)
                if r["layer"] == family and r["m"] >= minimum_m
                and (only_m is None or r["m"] == only_m) else r["candidate_us"]}
               for r in rows]
    return score(changed)[0]/score(rows)[0]-1


def build():
    if sha(ROOT/SOURCE) != SOURCE_SHA256:
        raise ValueError("Selected P039 source changed; do not profile a different source")
    receipt = json.loads((ROOT/RECEIPT).read_text())
    ledger = json.loads((ROOT/"experiments/projection-cycle2-budget.json").read_text())
    if ledger["selection"]["sha256"] != SOURCE_SHA256:
        raise ValueError("Selection changed; reconcile before continuing")
    rows = read_rows()
    value, loads = score(rows)
    if not math.isclose(value, receipt["result"]["speedup"], rel_tol=1e-12):
        raise ValueError("Score reconstruction does not match receipt")
    fractions = {family: sum(r["weight"]*r["candidate_us"]/loads[r["load"]]["candidate_us"]
                             for r in rows if r["layer"] == family)/len(loads)
                 for family in sorted({r["layer"] for r in rows})}
    scenarios = []
    for family in ("pooler", "qkv_gate", "unpooler"):
        for cut in (.05, .08, .10, .20):
            scenarios.append(dict(layer=family, m_scope="all m>1", latency_cut=cut,
                                  modeled_score_gain=sensitivity(rows, family, cut, 2)))
    representatives = []
    for layer, m in REPRESENTATIVES:
        row = next(r for r in rows if (r["layer"], r["m"]) == (layer, m))
        representatives.append(dict(**row, model=compulsory_model(layer, m)))
    return dict(kind="P041 local evidence/diagnostic preparation; no GPU execution",
                parent=dict(source=SOURCE, sha256=SOURCE_SHA256,
                            source_commit=receipt["admission"]["source_commit"],
                            job=receipt["admission"]["job_id"]),
                evidence_hashes={p:sha(ROOT/p) for p in (CSV, RECEIPT, "official/reference.py", "TASK.md")},
                official_passed=receipt["result"]["passed"],
                prerequisites=receipt["prerequisite_tests"], shapes=receipt["shape_checks"],
                score=value, score_ci=receipt["result"]["speedup_ci"],
                six_weighted_loads=loads, mean_within_load_family_fraction=fractions,
                rows=rows, representatives=representatives, conditional_scenarios=scenarios,
                saved_official_profile=dict(
                    job_id=receipt['admission']['job_id'],
                    qualification="Instrumented activity, not calibrated latency or hardware counters",
                    representative_activities=[dict(layer=p['case']['layer'], m=p['case']['m'], report=p['report'])
                        for p in receipt['result']['records'] if p.get('event') == 'profile'
                        and (p['case']['layer'], p['case']['m']) in REPRESENTATIVES]),
                scenario_qualification="All untargeted times frozen; modeled values, not predicted or achieved speedups",
                gpu_launched=False, counter_method="ordinary activity only; coordinator reports both bounded NCU metric attempts timed out with no values; no retry")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build()
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(score=result["score"], representatives=len(result["representatives"]),
                         source_sha256=SOURCE_SHA256, gpu_launched=False)))


if __name__ == "__main__":
    main()
