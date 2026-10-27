"""P053 phase-cost diagnostic: fixed P051 math, never a submission or score.

Partial-only graphs include FP32 scratch stores and compiler layout conversion.
Their original reducer runs only for validation, outside every timed pair.
Full/tail primitive correctness is inherited from frozen P052, not rerun here.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import projection_p052_orientation_screen as prior

safe = prior.safe
orientation = prior.orientation
ROOT = Path(__file__).resolve().parent
P052_SHA = "042c395d20ffa6b191629aa79d9b68b9766cb92ea94df56d7763e741d46cdd59"
INHERITED_PATH = "results/remote-P052-screen/timing.json"
INHERITED_SHA = "2ae054ec40d6f39a119dfe0c97fd330dfde09e315ba51d45b0cffb6cc71cd30f"
CORRECTNESS_SHA = "057400331f717576d5260c7c88bab5430dad679fa5e2f6cab3c3857b3434313e"
FROZEN = {**prior.FROZEN, "projection_p052_orientation_screen.py": P052_SHA}
CASES = (8, 7)
RUNNERS = ("control_whole", "weight_first_whole", "control_partial",
           "weight_first_partial", "p046_whole")
COMPARISONS = {
    "partial": ("control_partial", "weight_first_partial"),
    "whole": ("control_whole", "weight_first_whole"),
    "selected_partial": ("p046_whole", "weight_first_partial"),
    "selected_whole": ("p046_whole", "weight_first_whole"),
}
PROFILE_RUNNERS = ("control", "weight_first", "p046")
LABELS = ("A", "B", "A-return", "sparse", "zero")


def verify_sources(root=ROOT):
    hashes = {}
    for name, expected in FROZEN.items():
        digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError("Frozen source mismatch: " + name)
        hashes[name] = digest
    for name in ("projection_p053_phase_screen.py", "official/muse_bf16_proj.json"):
        hashes[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    return hashes


def inherited_receipt(root=ROOT):
    raw = (root / INHERITED_PATH).read_bytes()
    if hashlib.sha256(raw).hexdigest() != INHERITED_SHA:
        raise ValueError("Exact frozen P052 receipt required")
    receipt = json.loads(raw)
    if not (receipt.get("complete") is True and receipt.get("oracle_checks") == 320
            and receipt.get("correctness_receipt", {}).get("sha256") == CORRECTNESS_SHA
            and receipt.get("sources_sha256", {}).get("projection_p052_orientation_screen.py") == P052_SHA):
        raise ValueError("Invalid inherited P052 execution evidence")
    return dict(path=INHERITED_PATH, sha256=INHERITED_SHA,
                correctness_receipt_sha256=CORRECTNESS_SHA, prior_original_oracle_checks=390,
                full_tail_coverage="Inherited P052 five-case primitive execution only; not rerun by P053",
                prior_screen_passed=receipt["assessment"]["screen_passed"])


def protocol_counts():
    return dict(measured_replays=384, warmup_replays=96, prepost_replays=100,
                oracle_checks=580, untimed_partial_validator_reducers=220,
                capture_calls=10, untimed_capture_warmups=30,
                activity_replays=18, activity_profiles=18,
                activity_capture_calls=6, activity_untimed_capture_warmups=18)


def comparison_order(index):
    names = tuple(COMPARISONS)
    offset = (index // 2) % len(names)
    rotated = names[offset:] + names[:offset]
    return rotated if index % 2 == 0 else tuple(reversed(rotated))


def summarize(samples, comparison):
    if comparison not in COMPARISONS:
        raise ValueError("Unknown comparison")
    for index, sample in enumerate(samples):
        if (sample.get("comparison") != comparison
                or sample.get("comparison_sequence_index") != comparison_order(index).index(comparison)
                or sample.get("excluded_warmup") is not False
                or sample.get("correctness") != {"parent": True, "candidate": True}):
            raise ValueError("Incomplete or reordered measured comparison")
    result = safe.summarize_samples(samples)
    result["winning_pairs"] = sum(value > 1 for value in result["reverse_order_pair_ratios"])
    result["total_pairs"] = len(result["reverse_order_pair_ratios"])
    return result


def assess(cases):
    if [case.get("m") for case in cases] != list(CASES) or not all(case.get("passed") is True for case in cases):
        raise ValueError("Both complete cases required")
    results = []
    for case in cases:
        summaries = {name: case["series"][name]["summary"] for name in COMPARISONS}
        cuts = {name: 1 - 1 / summary["median_paired_raw_ratio"] for name, summary in summaries.items()}
        partial = summaries["partial"]
        strong_partial = (cuts["partial"] >= .10 - 1e-12
                          and partial["total_pairs"] == 12 and partial["winning_pairs"] >= 10)
        selected = summaries["selected_partial"]
        rescue_gates = dict(partial_at_least5pct_faster_than_p046_whole=cuts["selected_partial"] >= .05 - 1e-12,
                            at_least10_of12_pairs_win=selected["total_pairs"] == 12 and selected["winning_pairs"] >= 10)
        results.append(dict(m=case["m"], raw_latency_cuts=cuts,
            partial_winning_pairs=partial["winning_pairs"],
            dilution_followup_warranted=strong_partial and cuts["whole"] < .05,
            selected_partial_winning_pairs=selected["winning_pairs"],
            selected_partial_total_pairs=selected["total_pairs"], reducer_only_case_gates=rescue_gates,
            no_reducer_only_rescue_lead=not all(rescue_gates.values())))
    return dict(cases=results,
        reducer_only_followup_warranted=all(not case["no_reducer_only_rescue_lead"] for case in results),
        automatic_selection=False, official_score=None,
        qualification="Research decisions only. Partial-only excludes the required reducer and is not a complete implementation; it still includes layout conversion and FP32 scratch stores. No reducer latency is inferred by subtracting event medians; no achieved overlap or bottleneck claim.")


def validate_timing_receipt(path, sources):
    raw = path.read_bytes()
    record = json.loads(raw)
    if not (record.get("mode") == "timing" and record.get("complete") is True
            and not record.get("failure") and record.get("oracle_checks") == 580
            and record.get("sources_sha256") == sources
            and record.get("inherited_p052", {}).get("sha256") == INHERITED_SHA):
        raise ValueError("Completed source-bound P053 timing required before activity")
    cases = record.get("cases", [])
    if [case.get("m") for case in cases] != list(CASES):
        raise ValueError("Timing case coverage mismatch")
    count = 0
    expected = {(phase, label, runner) for phase in ("before", "after")
                for label in LABELS for runner in RUNNERS}
    for case in cases:
        checks = case.get("correctness", [])
        if not (case.get("passed") is True and len(checks) == 50
                and {(c.get("phase"), c.get("label"), c.get("runner")) for c in checks} == expected
                and all(c.get("passed") is True for c in checks)
                and set(case.get("series", {})) == set(COMPARISONS)):
            raise ValueError("Incomplete timing correctness coverage")
        count += len(checks)
        for name, series in case["series"].items():
            summarize(series["samples"], name)
            warmup = series.get("warmup", [])
            if len(warmup) != 6:
                raise ValueError("Missing warmup records")
            for index, row in enumerate(warmup):
                if (row.get("round") != index or row.get("excluded_warmup") is not True
                        or row.get("comparison") != name
                        or row.get("comparison_sequence_index") != comparison_order(index).index(name)
                        or tuple(row.get("order", [])) != safe.runner_order(index)
                        or row.get("correctness") != {"parent": True, "candidate": True}
                        or set(row.get("us", {})) != {"parent", "candidate"}
                        or not all(math.isfinite(v) and v > 0 for v in row["us"].values())):
                    raise ValueError("Invalid warmup evidence")
            count += 2 * (len(warmup) + len(series["samples"]))
    if count != 580:
        raise ValueError("Actual oracle count mismatch")
    return dict(sha256=hashlib.sha256(raw).hexdigest(), filename=path.name,
                oracle_checks=count, environment=record["environment"])


def partial_call(torch, kernels, arm, data, weights):
    x = data["x"]
    m, k = x.shape
    n = data["config"]["n"]
    orientation.validate_shape(m, n, k)
    partial = torch.empty((8, m, n), dtype=torch.float32, device=x.device)
    kernels[arm][((n + 63) // 64, 8)](x, weights["pooler"], partial, m, n, k, 8, 64, 64,
                                      **orientation.PARTIAL_LAUNCH)
    return partial


def measure_pair(timer, graphs, role_to_runner, input_check, validate, sample,
                 eviction, eviction_out, gate, save):
    """No validator reducer or oracle executes between the two timed members."""
    for role in sample["order"]:
        gate()
        observation = dict(role=role, runner=role_to_runner[role], status="started",
                           started_utc=datetime.now(timezone.utc).isoformat())
        sample["observations"].append(observation)
        save()
        sample["us"][role] = timer.measure(graphs[role_to_runner[role]], eviction, eviction_out)
        input_check()
        observation["status"] = "measured"
        save()
    for role in sample["order"]:
        gate()
        validate(role_to_runner[role])
        sample["correctness"][role] = True
        save()


def trace_name(m, runner, repeat):
    if m not in CASES or runner not in PROFILE_RUNNERS or repeat not in range(3):
        raise ValueError("Unexpected activity identity")
    return f"activity.pooler_m{m}.{runner}.repeat{repeat}.trace.json"


def trace_intervals(path, runner):
    """Use only CUPTI kernel intervals; gaps are not event-median subtraction."""
    if runner not in PROFILE_RUNNERS:
        raise ValueError("Unexpected profiled runner")
    trace = json.loads(path.read_text())
    kernels, copies = [], []
    for event in trace.get("traceEvents", []):
        category = event.get("cat")
        if category not in ("kernel", "gpu_memcpy", "gpu_memset") or event.get("ph") != "X":
            continue
        start, duration = event.get("ts"), event.get("dur")
        if not isinstance(start, (int, float)) or not isinstance(duration, (int, float)) or not (
                math.isfinite(start) and math.isfinite(duration) and duration >= 0):
            raise ValueError("Invalid CUDA activity interval")
        name = str(event.get("name", ""))
        role = ("partial" if "_partial" in name else "reducer" if "_reduce" in name else "library_or_other") if runner != "p046" else "library_or_other"
        item = dict(name=name, role=role, start_us=start, duration_us=duration, end_us=start + duration)
        (kernels if category == "kernel" else copies).append(item)
    kernels.sort(key=lambda event: event["start_us"])
    if not kernels:
        raise ValueError("No CUPTI graph-kernel intervals")
    if runner in ("control", "weight_first"):
        expected_partial = "_partial_weight_first" if runner == "weight_first" else "_partial"
        if not (len(kernels) == 2 and [event["role"] for event in kernels] == ["partial", "reducer"]
                and expected_partial in kernels[0]["name"]
                and (runner != "control" or "_partial_weight_first" not in kernels[0]["name"])):
            raise ValueError("Whole orientation graph does not match one original partial and reducer")
    return dict(kernel_intervals=kernels, cuda_copy_intervals=copies,
                inter_kernel_gaps_us=[right["start_us"] - left["end_us"] for left, right in zip(kernels, kernels[1:])],
                qualification="Separately instrumented single-replay CUPTI intervals and signed timeline gaps only. Not acceptance timing, hardware counters, pure dot time or achieved pipeline overlap.")


def execute(args, report, save):
    started = time.monotonic()
    def gate():
        if time.monotonic() - started >= args.deadline_seconds:
            raise TimeoutError("P053 internal deadline; supervisor must stop blocking native calls")

    import torch
    import triton
    import triton.language as tl
    from benchmark import capture
    from harness import environment, load_module
    safe.require_h200(torch)
    report["environment"] = environment(torch, "cuda", ROOT / prior.P046_PATH)
    report["environment"].update(triton=triton.__version__)
    report["environment"]["capability"] = list(report["environment"]["capability"])
    if args.mode == "activity" and not prior.environment_matches(report["timing_receipt"]["environment"], report["environment"]):
        raise RuntimeError("Timing/activity hardware or software environment differs")
    reference = load_module(ROOT / "official/reference.py", "p053_reference")
    parent = load_module(ROOT / prior.P046_PATH, "p053_p046")
    kernels = prior.build_kernels(triton, tl)
    with torch.inference_mode():
        gate()
        weights = reference.generate_weights(reference.WEIGHTS_SEED, device="cuda")
        frozen_original = safe.freeze_tensors(torch, weights)
        oracle_weights = {name: snapshot[1] for name, snapshot in frozen_original.items()}
        state = parent.prepare(weights)
        frozen_state = safe.freeze_tensors(torch, state)
        def check_states():
            gate()
            safe.assert_frozen(torch, weights, frozen_original)
            safe.assert_frozen(torch, state, frozen_state)
        check_states()
        report["preparation"] = dict(actual_p046_prepare_calls=1, weight_keys=list(weights),
            scope="Unmodified official weights and P046.prepare; every original/prepared tensor frozen")
        eviction = torch.ones(safe.EVICTION_BYTES // 4, dtype=torch.float32, device="cuda")
        eviction_out = torch.empty((), dtype=torch.float32, device="cuda")
        timer = safe.ColdGraphTimer(torch) if args.mode == "timing" else None
        report["event_initialization"] = "Timing uses the frozen initialized event pair; no event-median subtraction"
        checked = 0
        for case_index, m in enumerate(CASES):
            gate()
            seed = 5300000 + case_index * 10000
            make = lambda s, dist="normal", rows=m: safe.fresh_input(torch, reference, "pooler", rows, s, dist)
            data = make(seed)
            config, pointer = dict(data["config"]), safe.tensor_metadata(data["x"])
            initial = {"x": data["x"].clone()}
            if bool(parent._pooler_column_batch2_compatible(data["x"], state, 4096)) != (m == 8):
                raise RuntimeError("Original P046 M8 BMM / M7 MM dispatch not established")
            item = dict(m=m, passed=False, p046_route="column_batch2" if m == 8 else "pretransposed_mm",
                        capture={}, correctness=[], series={name: dict(warmup=[], samples=[]) for name in COMPARISONS})
            report["cases"].append(item)
            graphs, outputs = {}, {}
            active = RUNNERS if args.mode == "timing" else tuple(name + "_whole" for name in PROFILE_RUNNERS)
            for runner in active:
                gate()
                arm = "weight_first" if runner.startswith("weight_first") else "control"
                if runner == "p046_whole":
                    fn, runner_state = parent.kernel_fn, state
                elif runner.endswith("_partial"):
                    fn, runner_state = (lambda d, s, a=arm: partial_call(torch, kernels, a, d, s)), weights
                else:
                    fn, runner_state = (lambda d, s, a=arm: prior.orientation_call(torch, kernels, a, d, s)), weights
                graphs[runner], outputs[runner] = capture(fn, data, runner_state)
                torch.cuda.synchronize()
                safe.assert_input(torch, data, initial, config, pointer)
                check_states()
                item["capture"][runner] = dict(output_shape=list(outputs[runner].shape), dtype=str(outputs[runner].dtype))
                save()
            if len({value.data_ptr() for value in outputs.values()}) != len(active):
                raise RuntimeError("Captured graph outputs alias")

            def validate(runner, fresh):
                nonlocal checked
                safe.assert_input(torch, data, fresh, config, pointer)
                output = outputs[runner]
                if runner.endswith("_partial"):
                    output = torch.empty((m, config["n"]), dtype=torch.bfloat16, device=data["x"].device)
                    kernels["reduce"][((m * config["n"] + 127) // 128,)](
                        outputs[runner], output, m * config["n"], 8, 128, **orientation.REDUCE_LAUNCH)
                    torch.cuda.synchronize()
                safe.check_output(torch, reference, data, output, fresh, config, pointer, oracle_weights)
                checked += 1
                report["oracle_checks"] = checked
                return output

            def correctness_phase(phase, fixture_seed):
                a = make(fixture_seed)
                probes = (("A", a), ("B", make(fixture_seed + 1, "outliers")), ("A-return", a),
                          ("sparse", make(fixture_seed + 2, "sparse")), ("zero", make(fixture_seed + 3, "zero")))
                first = {}
                for index, (label, fresh) in enumerate(probes):
                    data["x"].copy_(fresh["x"])
                    order = RUNNERS if index % 2 == 0 else tuple(reversed(RUNNERS))
                    for runner in order:
                        gate()
                        record = dict(phase=phase, label=label, runner=runner, passed=False)
                        item["correctness"].append(record)
                        outputs[runner].fill_(float("nan"))
                        graphs[runner].replay()
                        torch.cuda.synchronize()
                        output = validate(runner, fresh)
                        if label == "A":
                            first[runner] = output.clone()
                        elif label == "A-return" and not torch.equal(first[runner].view(torch.uint8), output.view(torch.uint8)):
                            raise RuntimeError("Graph A-return differs: " + runner)
                        record["passed"] = True
                        save()
                    check_states()

            if args.mode == "timing":
                correctness_phase("before", seed + 10)
                for warmup, count, offset in ((True, 6, 100), (False, 24, 200)):
                    for index in range(count):
                        fresh = make(seed + offset + index)
                        for sequence_index, comparison in enumerate(comparison_order(index)):
                            gate()
                            data["x"].copy_(fresh["x"])
                            opponent, candidate = COMPARISONS[comparison]
                            mapping = {"parent": opponent, "candidate": candidate}
                            sample = dict(round=index, comparison=comparison, comparison_sequence_index=sequence_index,
                                seed=seed + offset + index, order=list(safe.runner_order(index)), us={}, observations=[],
                                correctness={}, excluded_warmup=warmup)
                            item["series"][comparison]["warmup" if warmup else "samples"].append(sample)
                            save()
                            measure_pair(timer, graphs, mapping,
                                lambda: safe.assert_input(torch, data, fresh, config, pointer),
                                lambda runner: validate(runner, fresh), sample, eviction, eviction_out, gate, save)
                        check_states()
                correctness_phase("after", seed + 400)
                for comparison in COMPARISONS:
                    item["series"][comparison]["summary"] = summarize(item["series"][comparison]["samples"], comparison)
            else:
                for runner in PROFILE_RUNNERS:
                    for repeat in range(3):
                        gate()
                        fresh = make(seed + 600 + repeat)
                        data["x"].copy_(fresh["x"])
                        torch.sum(eviction, dim=0, out=eviction_out)
                        torch.cuda.synchronize()
                        record = dict(m=m, runner=runner, repeat=repeat, passed=False,
                                      trace_file=trace_name(m, runner, repeat))
                        report["profiles"].append(record)
                        save()
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA], record_shapes=False,
                                with_stack=False, profile_memory=False) as prof:
                            graphs[runner + "_whole"].replay()
                            torch.cuda.synchronize()
                        path = args.output.with_name(record["trace_file"])
                        prof.export_chrome_trace(str(path))
                        record["trace_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                        save()  # Retain raw activity even if the following oracle fails.
                        # Oracle and all validator work are outside profiler scope.
                        validate(runner + "_whole", fresh)
                        record.update(trace_intervals(path, runner), passed=True)
                        check_states()
                        save()
            item["passed"] = True
            save()
        expected = 580 if args.mode == "timing" else 18
        if checked != expected:
            raise RuntimeError("Unexpected actual oracle check count")
        if args.mode == "timing":
            report["assessment"] = assess(report["cases"])
        report["complete"] = True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("timing", "activity"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timing-receipt", type=Path)
    parser.add_argument("--deadline-seconds", type=float, default=120)
    parser.add_argument("--outer-timeout-seconds", type=float, required=True)
    args = parser.parse_args(argv)
    safe.validate_options(args)
    if args.outer_timeout_seconds > 150:
        raise ValueError("P053 external timeout must be at most150 seconds")
    if args.mode == "activity" and args.timing_receipt is None:
        raise ValueError("Activity requires completed timing receipt")
    started = time.monotonic()
    report = dict(kind="P053 fixed-math phase-cost diagnostic", mode=args.mode, complete=False,
        cases=[], profiles=[], oracle_checks=0, protocol=protocol_counts(),
        comparisons={name: list(pair) for name, pair in COMPARISONS.items()},
        limits=dict(internal_seconds=args.deadline_seconds, external_seconds=args.outer_timeout_seconds),
        limitations=["Partial-only includes compiler layout conversion and FP32 scratch stores; not pure mainloop time",
            "Partial-only is incomplete math; original reducer is executed solely for untimed oracle validation",
            "No subtraction of event medians to estimate reducer cost; activity intervals are separately instrumented",
            "Primitive full/tail coverage inherited from P052; current validation covers only M8/M7",
            "No official score, promotion, NCU, hardware counters or achieved-overlap claim",
            "Untimed graph-capture setup calls not individually checked; concurrency not tested"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        report["wall_seconds"] = time.monotonic() - started
        temp = args.output.with_suffix(args.output.suffix + ".tmp")
        temp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        temp.replace(args.output)
    try:
        report["sources_sha256"] = verify_sources()
        report["inherited_p052"] = inherited_receipt()
        if args.mode == "activity":
            report["timing_receipt"] = validate_timing_receipt(args.timing_receipt, report["sources_sha256"])
        save()
        execute(args, report, save)
        code = 0
    except Exception as exc:
        report["failure"] = dict(type=type(exc).__name__, message=str(exc)[:700])
        code = 1
    finally:
        save()
    print(json.dumps(dict(mode=args.mode, complete=report["complete"], oracle_checks=report["oracle_checks"])))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
