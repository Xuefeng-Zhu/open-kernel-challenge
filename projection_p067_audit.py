"""Reproduce the P046 saved-evidence audit. No CUDA, candidates or network imports."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import projection_p066_decision as score

ROOT = Path(__file__).resolve().parent
FROZEN = {
    "results/remote-P2-09-pooler-column-batch2-hybrid.json": "984c9e0dcaea6bc41fff9b12103eb9a13e137f3c42b3021035f2245255288850",
    "results/remote-P049-screen/timing.json": "6e78fffd4dfdafe53eb7b5d63102bbeffdf97d84ccab5956773944a7b60fa512",
    "results/remote-P053-phase/timing.json": "f6f52fd366d74e2e912a6ea761055b2a1d2f3073b90d3d4cb27230a140a1da95",
    "results/remote-P053-phase/activity.json": "e5adaabc3ba123dc250aa5dd3c3696a808959e2c44e2931be4a6c27380f5aa2b",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    for name, expected in FROZEN.items():
        if digest(ROOT / name) != expected:
            raise ValueError("Frozen receipt mismatch: " + name)
    selected = score.run()  # Checks exact selected source, all 30 rows and score.
    official = json.loads((ROOT / next(iter(FROZEN))).read_text())
    activity = json.loads((ROOT / "results/remote-P053-phase/activity.json").read_text())
    publication = json.loads((ROOT / "results/router-p053-assessment.json").read_text())
    if not activity["complete"] or activity["oracle_checks"] != 18:
        raise ValueError("Incomplete historical activity receipt")
    if activity["environment"]["submission_sha256"] != score.SHA:
        raise ValueError("Activity belongs to a different selected source")
    profiles = [p for p in activity["profiles"] if p["runner"] == "p046"]
    if {(p["m"], p["repeat"]) for p in profiles} != {(m, r) for m in (8, 7) for r in range(3)}:
        raise ValueError("Missing exact-P046 repeats")
    traces = []
    for profile in profiles:
        name = profile["trace_file"]
        manifest = publication["artifact_manifest"][name]
        if manifest["raw_sha256"] != profile["trace_sha256"]:
            raise ValueError("Raw trace binding mismatch: " + name)
        path = ROOT / "results/remote-P053-phase" / name
        if digest(path) != manifest["published_sha256"]:
            raise ValueError("Published trace mismatch: " + name)
        trace = json.loads(path.read_text())
        events = sorted((e for e in trace["traceEvents"] if e.get("cat") == "kernel"), key=lambda e: e["ts"])
        if len(events) != (1 if profile["m"] == 8 else 2):
            raise ValueError("Unexpected selected kernel count")
        rows = []
        for event in events:
            args = event["args"]
            launch_ratio = math.prod(args["grid"]) / trace["deviceProperties"][0]["numSms"]
            if not math.isclose(args["blocks per SM"], launch_ratio, rel_tol=1e-6):
                raise ValueError("Unexpected derived launch ratio")
            rows.append(dict(symbol=event["name"], duration_us=event["dur"], stream=args["stream"],
                grid=args["grid"], block=args["block"], registers_per_thread=args["registers per thread"],
                reported_shared_bytes=args["shared memory"], grid_blocks_divided_by_sms=launch_ratio))
        traces.append(dict(m=profile["m"], repeat=profile["repeat"],
            published_trace_sha256=manifest["published_sha256"], raw_trace_sha256=profile["trace_sha256"],
            kernels=rows, inter_kernel_gaps_us=[events[i+1]["ts"]-events[i]["ts"]-events[i]["dur"] for i in range(len(events)-1)],
            scope="Instrumented intervals and static launch metadata; no counters or achieved occupancy"))
    p049 = json.loads((ROOT / "results/remote-P049-screen/timing.json").read_text())
    if not p049["passed"] or p049["environment"]["submission_sha256"] != score.SHA:
        raise ValueError("P049 selected-source binding failed")
    return dict(kind="P067 saved baseline evidence audit; no new execution", selected=selected,
        source_and_receipt_hashes=FROZEN | {str(score.SOURCE.relative_to(ROOT)): score.SHA},
        official_evaluator=official["result"]["evaluator"],
        official_profiles=[dict(case=r["case"], report=r["report"]) for r in official["result"]["records"] if r.get("event")=="profile"],
        p049_actual_p046_activity=[dict(layer=c["layer"], m=c["m"], activity=c["activities"]["parent"]) for c in p049["cases"]],
        p053_actual_p046_traces=traces, historical_modal_environment=activity["environment"],
        unknowns=["Exact P046 vendor tactic and instruction binary", "Per-CTA tile/K work assignment and persistence",
                  "Internal buffer ownership and synchronization", "Achieved occupancy, HBM/L2 bytes and stalls",
                  "Pinned Quack dependencies available in exact official image"],
        live_execution=dict(attempted=False, new_jobs=0, credential_actions=0,
            blocker="Existing Modal CLI credential expired; no renewal authorized in this audit. Official status/profile interface has no documented native artifact/counter export."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run()
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(selected_job=result["selected"]["job_id"], score=result["selected"]["score"],
        verified_p046_traces=len(result["p053_actual_p046_traces"]), official_profiles=len(result["official_profiles"]),
        new_jobs=0, native_artifacts_available=False)))
