"""Sanitize one terminal official result, preserving all actual timing samples."""
import argparse
import csv
import datetime
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("experiment")
    args = p.parse_args()
    ledger = json.loads((ROOT / "experiments/remote-budget.json").read_text())
    entry = next(x for x in ledger["requests"] if x["experiment"] == args.experiment)
    assert entry["state"] == "terminal", "Never infer a result from queue state"
    job = json.loads((ROOT / "results" / (args.experiment + "-job.local.json")).read_text())
    assert job["id"] == entry["job_id"] and job["mode"] == "check"
    result = job.get("result") or {}
    keep = {k: result[k] for k in ["baseline_us", "checks", "error", "flags", "gpu", "passed",
                                   "records", "returncode", "score_us", "speedup", "speedup_ci", "status"] if k in result}
    machine = result.get("machine", {})
    keep["machine"] = {k: machine[k] for k in ["gpu_before", "gpu_after"] if k in machine}
    evaluator = result.get("evaluator", {})
    keep["evaluator"] = {"code_sha256": evaluator.get("code_sha256"), "files": evaluator.get("files"),
                         "image_digest": evaluator.get("image", "").split("@")[-1]}
    rows = [x for x in result.get("records", []) if x.get("event") == "benchmark"]
    tests = [x for x in result.get("records", []) if x.get("event") == "test"]
    loads = {}
    for row in rows:
        if row.get("status") != "pass":
            continue
        case = row["case"]
        load = loads.setdefault(case["load"], dict(sglang_load_us=0.0, candidate_load_us=0.0))
        load["sglang_load_us"] += row["baseline_ns"] * case["weight"] / 1000
        load["candidate_load_us"] += row["median_ns"] * case["weight"] / 1000
    for load in loads.values():
        load["speedup"] = load["sglang_load_us"] / load["candidate_load_us"]
    derived = None
    if result.get("passed"):
        assert len(rows) == 30 and len(tests) == 9 and all(x["status"] == "pass" for x in rows + tests)
        derived = math.exp(sum(math.log(x["speedup"]) for x in loads.values()) / len(loads))
        assert math.isclose(derived, result["speedup"], rel_tol=1e-8)
    baseline = json.loads((ROOT / "results/h200-check.json").read_text())["result"]
    out = dict(experiment=args.experiment, admission=entry, status=job["status"], result=keep,
               weighted_loads=loads, recomputed_speedup=derived,
               prerequisite_tests=dict(passed=sum(x["status"] == "pass" for x in tests), total=len(tests)),
               shape_checks=dict(passed=sum(x["status"] == "pass" for x in rows), total=len(rows)),
               comparison_note="Official within-run SGLang pairing is authoritative. Ratio to the prior starter is a reference-normalized cross-run comparison, not a paired same-input trial.",
               original_baseline_job="e5a5b2d20da54187",
               speedup_over_original_baseline=(result["speedup"] / baseline["speedup"]) if result.get("speedup") else None,
               redactions="Raw log tail, account/team, server node/cluster/call identifiers, payload and local paths omitted; exact raw reply retained locally.")
    for key in ["created_at", "started_at", "finished_at"]:
        if job.get(key):
            out[key + "_utc"] = datetime.datetime.fromtimestamp(job[key], datetime.timezone.utc).isoformat()
    text = json.dumps(out, indent=2).replace(str(ROOT), "<project>") + "\n"
    output = ROOT / "results" / ("remote-" + args.experiment + ".json")
    output.write_text(text)
    if rows:
        with output.with_suffix(".csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["job_id", "load", "m", "layer", "weight", "status", "sglang_us", "candidate_us", "speedup"], lineterminator="\n")
            w.writeheader()
            for row in rows:
                case = row["case"]
                w.writerow(dict(job_id=job["id"], load=case.get("load"), m=case["m"], layer=case["layer"],
                                weight=case.get("weight"), status=row.get("status"),
                                sglang_us=row["baseline_ns"] / 1000 if row.get("baseline_ns") is not None else None,
                                candidate_us=row["median_ns"] / 1000 if row.get("median_ns") is not None else None,
                                speedup=row.get("speedup")))
    print(json.dumps(dict(experiment=args.experiment, job_id=job["id"], status=job["status"],
                          passed=result.get("passed"), speedup=result.get("speedup"),
                          speedup_ci=result.get("speedup_ci"), tests=out["prerequisite_tests"],
                          shapes=out["shape_checks"], relative_to_original=out["speedup_over_original_baseline"])))


if __name__ == "__main__":
    main()
