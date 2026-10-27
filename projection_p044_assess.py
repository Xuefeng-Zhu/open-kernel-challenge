"""Offline assessment of the five-shape P039 diagnostic, never an official score."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from projection_p041_plan import SOURCE, SOURCE_SHA256, REPRESENTATIVES

RUNNERS = ("candidate", "baseline")


def mapping(value):
    return value if isinstance(value, dict) else {}


def finite(value, positive=True):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (value > 0 if positive else value >= 0))


def distribution(values):
    """Linear-interpolated empirical percentiles; not confidence intervals."""
    values = sorted(values)
    if not values:
        return dict(count=0, minimum=None, p10=None, median=None, p90=None, maximum=None)
    def percentile(q):
        pos = (len(values)-1)*q
        low = math.floor(pos)
        return values[low] + (values[min(low+1, len(values)-1)]-values[low])*(pos-low)
    return dict(count=len(values), minimum=values[0], p10=percentile(.1),
                median=statistics.median(values), p90=percentile(.9), maximum=values[-1])


def summarize_condition(record):
    samples = record.get("samples", [])
    if not isinstance(samples, list):
        return dict(complete=False, errors=["samples is not a list"], raw_sample_count=0)
    errors, valid = [], []
    for i, sample in enumerate(samples):
        expected = list(RUNNERS if i % 2 == 0 else reversed(RUNNERS))
        ok = (isinstance(sample, dict) and sample.get("round") == i
              and sample.get("order") == expected and isinstance(sample.get("us"), dict)
              and all(finite(sample["us"].get(name)) for name in RUNNERS))
        valid.append(ok)
        if not ok:
            errors.append(f"Invalid round/order/timing at sample index {i}")
    pairs, pair_indices = [], []
    for i in range(0, len(samples)-1, 2):
        if valid[i] and valid[i+1]:
            ratios = [samples[j]["us"]["baseline"]/samples[j]["us"]["candidate"]
                      for j in (i, i+1)]
            pairs.append(math.sqrt(ratios[0]*ratios[1]))
            pair_indices.append([i, i+1])
    runners = {}
    for name in RUNNERS:
        raw = [sample["us"][name] if ok else None for sample, ok in zip(samples, valid)]
        runners[name] = dict(raw_gross_us=raw,
                             gross_us=distribution([v for v in raw if v is not None]))
    return dict(raw_sample_count=len(samples), invalid_sample_indices=[i for i, ok in enumerate(valid) if not ok],
                complete=len(samples) == 24 and not errors, errors=errors,
                pair_indices=pair_indices, pair_ratios=pairs, paired_ratio=distribution(pairs),
                runners=runners, expected_samples=24, expected_reverse_pairs=12,
                qualification="Baseline/candidate gross-time reverse-pair ratios; no calibration or confidence interval")


def aba_summary(case):
    rows = case.get("aba", [])
    result = {}
    for name in RUNNERS:
        selected = [row for row in rows if isinstance(row, dict) and row.get("runner") == name]
        labels = [row.get("state") for row in selected]
        passed = [row for row in selected if row.get("passed") is True]
        result[name] = dict(records=len(selected), completed_passes=len(passed), states=labels,
            complete=labels == ["A", "B", "A-return"] and len(passed) == 3
                     and all(row.get("same_input_address") is True
                             and row.get("immutable_x_weights") is True for row in selected))
    return result


def kernel_class(symbol):
    name = str(symbol).lower()
    if "reduce" in name or "reduction" in name:
        return "reduction"
    if name.startswith("nvjet_") or any(token in name for token in ("gemm", "_gemv", "_partial")):
        return "main"
    return "unknown"


def summarize_trace(directory, entry):
    trace = entry.get("trace", {})
    name = trace.get("file", "")
    result = dict(file=name, available=False, resource_metadata_available=False, kernels=[], errors=[])
    # Trace filenames are emitted beside the report. Never follow a report outside this directory.
    path = directory/name
    if not name or Path(name).name != name or path.is_symlink():
        result["errors"].append("Missing or unsafe adjacent trace filename")
        return result
    if not path.is_file():
        result["errors"].append("Raw trace file missing")
        return result
    raw = path.read_bytes()
    result["sha256"] = hashlib.sha256(raw).hexdigest()
    result["bytes"] = len(raw)
    if result["sha256"] != trace.get("sha256") or len(raw) != trace.get("bytes"):
        result["errors"].append("Raw trace hash/size mismatch")
        return result
    try:
        events = json.loads(raw)["traceEvents"]
        if not isinstance(events, list):
            raise ValueError("traceEvents is not a list")
        resource_keys = {"grid", "block", "registersperthread", "sharedmemory", "sharedmemorybytes",
                         "staticsharedmemory", "dynamicsharedmemory", "blockspersm", "warpspersm"}
        totals = dict(main=0., reduction=0., unknown=0.)
        for event in events:
            categories = {x.strip().lower() for x in str(event.get("cat", "")).split(",")}
            if event.get("ph") != "X" or not categories.intersection({"kernel", "cuda_kernel"}):
                continue
            args = event.get("args", {})
            if not isinstance(args, dict):
                raise ValueError("kernel args is not an object")
            duration = event.get("dur")
            if not finite(duration, positive=False):
                raise ValueError("kernel duration is missing/nonfinite/negative")
            resources = {key: value for key, value in args.items()
                         if "".join(c for c in key.lower() if c.isalnum()) in resource_keys}
            kind = kernel_class(event.get("name"))
            totals[kind] += duration
            result["kernels"].append(dict(symbol=event.get("name"), duration_us=duration,
                symbol_class=kind, trace_args=args, resources_as_emitted=resources))
        result["available"] = bool(result["kernels"])
        result["resource_metadata_available"] = any(k["resources_as_emitted"] for k in result["kernels"])
        result["duration_us_by_symbol_class"] = totals
        if not result["available"]:
            result["errors"].append("No device kernel activities in raw trace")
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        result["errors"].append(str(error))
        result["available"] = False
    result["qualification"] = "Classification by visible symbol only; instrumented duration, launch resources, not achieved occupancy/counters/PTX/SASS/spills"
    return result


def read_report(directory, mode):
    path = directory/(mode+".json")
    if not path.is_file():
        return {}, dict(file=path.name, error="Report missing")
    raw = path.read_bytes()
    identity = dict(file=path.name, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    try:
        report = json.loads(raw)
        if not isinstance(report, dict):
            raise ValueError("Report must be an object")
        return report, identity
    except ValueError as error:
        return {}, dict(**identity, error=str(error))


def assess(directory):
    directory = Path(directory)
    reports, identities = {}, {}
    for mode in ("timing", "activity"):
        reports[mode], identities[mode] = read_report(directory, mode)
    hashes = {mode: dict(declared=r.get("source_sha256"),
                        verified=mapping(r.get("source_hashes")).get(SOURCE),
                        environment=mapping(r.get("environment")).get("submission_sha256"))
              for mode, r in reports.items()}
    bound = all(all(value == SOURCE_SHA256 for value in fields.values()) for fields in hashes.values())
    result = dict(kind="P044 offline five-shape diagnostic assessment", source=SOURCE,
        expected_source_sha256=SOURCE_SHA256, source_binding=dict(valid=bound, modes=hashes),
        report_files=identities, modes={}, cases=[], complete=False,
        limitations=["Only source-bound summaries can be attributed to frozen P039",
                     "Five representatives, not the full official suite or official score",
                     "Gross CUDA graph timings; no calibrated score or confidence interval",
                     "Warm/cold sensitivity does not establish HBM traffic or bottleneck causality",
                     "Sequential ABA only; derived prepared-state bytes and concurrency are not exhaustively tested",
                     "Trace launch metadata is not achieved occupancy, hardware counters, PTX/SASS or spills"])
    case_maps = {}
    for mode, report in reports.items():
        case_maps[mode] = {}
        info = dict(worker_passed=report.get("passed") is True, failure=report.get("failure"),
                    preparation=report.get("preparation"), wall_seconds=report.get("wall_seconds"), errors=[])
        result["modes"][mode] = info
        if report.get("mode") != mode:
            info["errors"].append("Missing or mismatched report mode")
        preparation = mapping(mapping(report.get("preparation")).get("runners"))
        if any(not finite(mapping(preparation.get(name)).get("synchronized_wall_seconds"), positive=False)
               for name in RUNNERS):
            info["errors"].append("Missing or invalid synchronized preparation duration")
        protocol = report.get("protocol", {})
        if (protocol.get("warmup_replays") != 6 or protocol.get("eviction_bytes") != 256*1024**2
                or (mode == "timing" and protocol.get("measured_rounds") != 24)):
            info["errors"].append("Missing or changed declared timing protocol")
        for case in report.get("cases", []):
            key = (case.get("layer"), case.get("m"))
            if key not in REPRESENTATIVES or key in case_maps[mode]:
                info["errors"].append("Unexpected or duplicate case")
            else:
                case_maps[mode][key] = case
    for key in REPRESENTATIVES:
        item = dict(layer=key[0], m=key[1], modes={})
        result["cases"].append(item)
        for mode in reports:
            case = case_maps[mode].get(key, {})
            item["modes"][mode] = dict(present=bool(case), worker_passed=case.get("passed") is True,
                                      aba=aba_summary(case))
        timing = case_maps["timing"].get(key, {})
        conditions = {name: summarize_condition(record) for name, record in timing.get("conditions", {}).items()}
        item["conditions"] = conditions
        item["warmup_raw_records"] = len(timing.get("warmup", []))
        item["warmup_expected_records"] = 12  # Six per runner.
        item["warm_over_cold_gross_ratios"] = {}
        if all(name in conditions and "runners" in conditions[name] for name in ("cold", "warm")):
            for runner in RUNNERS:
                cold = conditions["cold"]["runners"][runner]["raw_gross_us"]
                warm = conditions["warm"]["runners"][runner]["raw_gross_us"]
                ratios = [w/c for c, w in zip(cold, warm) if c is not None and w is not None]
                item["warm_over_cold_gross_ratios"][runner] = dict(per_round_ratios=ratios,
                    distribution=distribution(ratios), qualification="Below one means warm replay was faster; same-round own-runner gross times")
        activities = case_maps["activity"].get(key, {}).get("activities", [])
        item["activities"] = {}
        for runner in RUNNERS:
            entries = [entry for entry in activities if entry.get("runner") == runner]
            item["activities"][runner] = summarize_trace(directory, entries[0]) if len(entries) == 1 else dict(
                available=False, resource_metadata_available=False, errors=["Missing or duplicate runner activity"])
    result["complete"] = (bound and all(info["worker_passed"] and not info["failure"] and not info["errors"] for info in result["modes"].values())
        and all(all(mode["worker_passed"] and all(a["complete"] for a in mode["aba"].values())
                    for mode in item["modes"].values())
                and item["warmup_raw_records"] == 12
                and all(name in item["conditions"] and item["conditions"][name]["complete"] for name in ("cold", "warm"))
                and all(a["available"] for a in item["activities"].values()) for item in result["cases"]))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = assess(args.artifacts_dir)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps(dict(complete=result["complete"], source_bound=result["source_binding"]["valid"], cases=len(result["cases"]))))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
