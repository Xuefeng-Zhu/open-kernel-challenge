"""Offline P068 validation and allowlisted publication; no native execution."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import projection_p068_collector as collector
import projection_p068_modal as launcher


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(path.read_text())


def publish(raw, dest, ledger, stop, output):
    if dest.exists() or output.exists():
        raise ValueError("Publication exists; inspect instead of replacing")
    run, shutdown = read(ledger), read(stop)
    if not (run["diagnostic_complete"] and run["shutdown_verified"]
            and run["paid_after_zero"] and shutdown["all_stopped_tasks_zero"]
            and shutdown["app"]["app_id"] == run["app_id"]
            and shutdown["app"]["state"] == "stopped" and str(shutdown["app"]["tasks"]) == "0"):
        raise ValueError("Missing completed run and independent shutdown")
    expected = collector.sources()
    if run["payload_sha256"] != expected:
        raise ValueError("Run source freeze differs from inspected files")
    manifest = run["transport_manifest"]["artifact_sha256"]
    content = {name: (raw / name).read_bytes() for name in manifest}
    if any(digest(content[name]) != value for name, value in manifest.items()):
        raise ValueError("Raw transport receipt mismatch")
    launcher.validate_complete(content, expected)
    timing, activity = json.loads(content["timing.json"]), json.loads(content["activity.json"])
    phases = []
    for case in activity["cases"]:
        events = [p["events"] for p in case["activity"]]
        identity = [(e["symbol"], e["metadata"]) for e in events[0]]
        if any([(e["symbol"], e["metadata"]) for e in seq] != identity for seq in events):
            raise ValueError("Dispatch/resources changed across repeats")
        medians = [statistics.median(seq[i]["duration_us"] for seq in events)
                   for i in range(len(events[0]))]
        phases.append(dict(layer=case["layer"], m=case["m"],
            dispatch=[dict(symbol=e["symbol"], **e["metadata"]) for e in events[0]],
            instrumented_us=[[e["duration_us"] for e in seq] for seq in events],
            instrumented_medians_us=medians, gaps_us=[p["gaps_us"] for p in case["activity"]],
            reducer_fraction_of_median_phase_sum=medians[-1]/sum(medians) if len(medians)==2 else None,
            qualification="Instrumented phase durations; not whole-call acceptance timing or removable cost"))
    rows = []
    for case in timing["cases"]:
        summary = case["graph_full_call"]
        pairs = summary["reverse_order_pair_ratios"]
        rows.append(dict(layer=case["layer"], m=case["m"],
            raw_us=summary["raw_us"], pair_distribution=summary["pair_distribution"],
            paired_b_over_a_latency_change=summary["aa_relative_movement"],
            graph_b_faster_pairs=sum(r>1 for r in pairs), pairs=len(pairs),
            absolute_median_drift_over_2pct=summary["aa_drift_exceeds_2pct"],
            qualification="A and B are identical P046, not competing candidates; all samples retained"))
    # No CPU trace events, process/host identifiers, absolute paths, or worker logs
    # leave the ignored raw directory. Kernel evidence is retained verbatim.
    publications = {name: content[name] for name in
        ("timing.json", "activity.json", "controller.json", "runtime-metadata.json", "remote.json")}
    for name in sorted(launcher.TRACES):
        trace = json.loads(content[name])
        reduced = dict(traceEvents=[dict(ph=e["ph"], cat=e["cat"], name=e.get("name"),
            ts=e["ts"], dur=e["dur"], args={k:e.get("args",{}).get(k) for k in
                ("grid", "block", "stream", "registers per thread", "shared memory", "cluster")})
            for e in trace["traceEvents"] if e.get("ph")=="X"
            and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")],
            **{k:trace.get(k) for k in ("cuda_driver_version", "cuda_runtime_version", "cupti_version")})
        if collector.trace_summary(reduced) != collector.trace_summary(trace):
            raise ValueError("Publication altered observed kernel evidence")
        publications[name] = (json.dumps(reduced, indent=2)+"\n").encode()
    dest.mkdir(parents=True, exist_ok=False)
    published = {}
    for name, value in publications.items():
        public_name = name.replace(".trace.local.json", ".trace.json")
        (dest / public_name).write_bytes(value)
        published[name] = dict(published_file=public_name, raw_sha256=manifest[name], published_sha256=digest(value),
            scope="kernel/copy/memset intervals and selected launch fields only" if name in launcher.TRACES else "unchanged structured receipt")
    result = dict(experiment="P068", source_commit=run["source_commit"],
        selected_source_sha256=expected[collector.SOURCE], app_id=run["app_id"],
        decision="Baseline acquisition completed; preserve P046, no optimization or official score",
        oracle_checks=258, timing_oracle_checks=240, activity_oracle_checks=18,
        timing=rows, activity=phases, environment=timing["environment"],
        library_identity=timing["library_identity"], runtime_metadata=json.loads(content["runtime-metadata.json"]),
        profiler_metadata_interpretation="profiler_executed=false refers to standalone commands; Torch/CUPTI activity did execute",
        artifact_manifest=published, raw_artifacts_retained_privately=True,
        worker_wall_seconds=run["worker_wall_seconds"], client_wall_seconds=run["client_wall_seconds"],
        shutdown=shutdown, cost_reserve_usd=run["cost_reserve_usd"],
        observed_workspace_gross_delta_usd=run["observed_workspace_gross_delta_usd"],
        delayed_workspace_gross_delta_usd=run.get("delayed_workspace_gross_delta_usd"),
        delayed_billing_observed_utc=run.get("delayed_billing_observed_utc"),
        cost_qualification="Billing snapshot can lag; retain reserve, zero observed delta is not a free-run claim",
        limitations=list(collector.ABSENT.values()) + [
            "Three shapes and one weight seed only; not all scored/prerequisite cases or concurrency",
            "Identical graph median drift up to 1.98%; small candidate gains need independent confirmation",
            "Tiny negative trace gaps are retained as timestamp uncertainty, not an overlap claim",
            "Driver profiling restriction value 0 and installed tools do not prove counters work",
            "Test image ID and loaded DSO hashes do not establish official judge image or vendor cubin/tactic identity"])
    output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(oracle_checks=258, published_files=len(publications),
        app_id=run["app_id"], stopped_tasks_zero=True, gpu_calls_added=0)))
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "dest", "ledger", "stop", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args=parser.parse_args()
    publish(args.raw, args.dest, args.ledger, args.stop, args.output)
