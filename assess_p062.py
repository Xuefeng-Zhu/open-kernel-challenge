"""P062 raw-consumer screen. Cross-run ratios are not same-input causal pairs."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPERIMENT = "P2-10-p043-raw-control"
RAW_SHA = "8ee5189831d7dcd209284b14a1ed907140d45ad9beba8c266122dc07e93a43de"
TARGETS = (8, 16, 32)
BASELINES = (("128378586ebe4730", "9598c7d1ebe6373b56f9ced2c0405b855f3a6895628f771c2eb8af314003b837"),
             ("41ba778763a5433b", "001ea12e8764564d507de51e433769f5bd9a9916bfc4c014f0f87522c278545e"))


def normalized_latency(a, b):
    """Latency A/B, using each receipt's SGLang/A and SGLang/B ratios."""
    if not all(math.isfinite(x) and x > 0 for x in (a, b)):
        raise ValueError("Positive finite paired ratios required")
    return b / a


def records(receipt):
    rows = [r for r in receipt["result"].get("records", []) if r.get("event") == "benchmark"]
    keyed = {(r["case"]["load"], r["case"]["layer"]): r for r in rows}
    if len(rows) != 30 or len(keyed) != 30 or any(r.get("status") != "pass" for r in rows):
        raise ValueError("Expected 30 distinct passing scored records")
    return keyed


def assess(raw, compressed, selected):
    for receipt, (job, source) in zip((compressed, selected), BASELINES):
        if (receipt["admission"]["job_id"], receipt["admission"]["submission_sha256"]) != (job, source):
            raise ValueError("Historical baseline identity mismatch")
    if raw["admission"]["submission_sha256"] != RAW_SHA or raw["admission"]["mode"] != "check":
        raise ValueError("Wrong source or submission mode")
    if raw["status"] not in {"done", "failed", "rejected", "duplicate"}:
        raise ValueError("A pending job is not evidence")
    result = raw["result"]
    base = dict(experiment=EXPERIMENT, job_id=raw["admission"]["job_id"], raw_source_sha256=RAW_SHA,
                selected_job=selected["admission"]["job_id"], selected_preserved=True,
                compressed_job=compressed["admission"]["job_id"], new_codegen_evidence=False)
    if result.get("passed") is not True:
        return dict(**base, decision="Native correctness/compilation did not pass; no timing conclusion or promotion.",
                    passed=False, error=result.get("error"))
    for receipt in (raw, compressed, selected):
        if receipt["prerequisite_tests"] != dict(passed=9, total=9):
            raise ValueError("Missing original prerequisites")
    rr, cr, sr = records(raw), records(compressed), records(selected)
    if set(rr) != set(cr) or set(rr) != set(sr):
        raise ValueError("Case sets differ")
    rows = []
    for key, r in rr.items():
        c, s = cr[key], sr[key]
        if any(v["case"]["m"] != r["case"]["m"] or v["case"]["weight"] != 62 for v in (r,c,s)):
            raise ValueError("M/weight mismatch")
        target = r["case"]["layer"] == "pooler" and r["case"]["m"] in TARGETS
        rows.append(dict(load=key[0], layer=key[1], m=r["case"]["m"], target=target,
            raw_calibrated_us=r["median_ns"]/1000, compressed_calibrated_us=c["median_ns"]/1000,
            selected_calibrated_us=s["median_ns"]/1000,
            raw_graph_us=r["median_raw_ns"]/1000, compressed_graph_us=c["median_raw_ns"]/1000,
            selected_graph_us=s["median_raw_ns"]/1000,
            raw_over_selected=normalized_latency(r["paired_ratio"],s["paired_ratio"]),
            compressed_over_raw=normalized_latency(c["paired_ratio"],r["paired_ratio"]),
            raw_over_compressed=normalized_latency(r["paired_ratio"],c["paired_ratio"]),
            raw_graph_over_selected_graph=r["median_raw_ns"]/s["median_raw_ns"],
            raw_graph_over_compressed_graph=r["median_raw_ns"]/c["median_raw_ns"] ))
    targets = sorted([r for r in rows if r["target"]],key=lambda r:r["m"])
    if [r["m"] for r in targets] != list(TARGETS):
        raise ValueError("Expected three target shapes")
    profiles = [dict(case=r["case"],report=r["report"]) for r in result["records"]
                if r.get("event") == "profile" and r["case"]["layer"] == "pooler" and r["case"]["m"] in TARGETS]
    executed = sorted(p["case"]["m"] for p in profiles
                      if "_pooler_codec_partial" in p["report"] and "_reduce" in p["report"]) == list(TARGETS)
    controls = [r for r in rows if not r["target"]]
    drift = [r for r in controls if abs(r["raw_over_compressed"]-1)>.05 or abs(r["raw_over_selected"]-1)>.05]
    same_evaluator = result["evaluator"] == compressed["result"]["evaluator"] == selected["result"]["evaluator"]
    if not executed:
        decision = "Fallback or missing profile evidence; raw-consumer performance question remains unresolved."
    elif not same_evaluator or drift:
        decision = "Report observed differences, but changed evaluator/controls limit attribution; preserve P046."
    elif all(r["raw_over_selected"]>1.303348748235759 for r in targets):
        decision = "All raw targets exceed the optimistic storage-only parity screen; deprioritize this custom consumer, not all compression architectures. Preserve P046."
    elif all(r["raw_over_selected"]<=1.05 for r in targets):
        decision = "Raw consumer is near the selected baseline on these cross-run observations; compiler/layout capture is justified before any decoder redesign. Preserve P046 pending independent evidence."
    else:
        decision = "Mixed or moderate raw-consumer deficit; no general compression rescue or promotion. Use per-shape evidence to scope further diagnosis."
    return dict(**base, passed=True, score=result["speedup"], score_ci=result["speedup_ci"],
        targets=targets, all30shapes=rows, profiles=profiles, custom_path_execution_verified=executed,
        same_evaluator=same_evaluator, control_count=len(controls), controls_over5pct=drift,
        decision=decision,
        qualifications=["Source constant and profile jointly establish the raw path; the shared kernel symbol alone does not.",
          "Historical compressed and selected receipts are separate runs; reference normalization is not a new paired A/B experiment.",
          "Gross graph ratios are unnormalized cross-run diagnostics; calibrated medians and activity times are different quantities.",
          "1.30335 is a zero-decode/all-time-weight-traffic scenario, not a physical bound. Near-flat values are inconclusive.",
          "No PTX/SASS/register/spill/counter export is assumed; hardware causality remains unresolved.",
          "Official correctness/graph cases do not establish dedicated overlapping-stream or all-bit GPU decoder coverage."])


if __name__ == "__main__":
    raw=json.loads((ROOT/f"results/remote-{EXPERIMENT}.json").read_text())
    compressed=json.loads((ROOT/"results/remote-P2-07-pooler-lossless-codec.json").read_text())
    selected=json.loads((ROOT/"results/remote-P2-09-pooler-column-batch2-hybrid.json").read_text())
    out=assess(raw,compressed,selected)
    (ROOT/"results/projection-p062-assessment.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:out[k] for k in ("job_id","passed","decision")}))
