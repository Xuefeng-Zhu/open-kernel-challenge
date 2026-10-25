"""Predeclared nine-case P046/P050 H200 screen; no provisioning or official score.

The supervisor must enforce --outer-timeout-seconds. Importing this module
imports neither Torch nor candidate code. All activity follows all timing.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
import warnings

import projection_p046_confirmation as safe

ROOT = Path(__file__).resolve().parent
SOURCES = {
    "parent": ("candidates/projection_pooler_column_batch2_hybrid.py",
               "001ea12e8764564d507de51e433769f5bd9a9916bfc4c014f0f87522c278545e"),
    "candidate": ("candidates/projection_qkv_zero_bias.py",
                  "b08b6e371fef83851268a8dbfa1159d1b6d5a8e141a410a596dfc3f159e0da8b"),
}
SAFE_SHA256 = "6c57daf6ab5aca4a73aa1a608b6ae978af284a02d41c489b4e7b62b307505029"
TARGET_M = (8, 16, 32, 128, 256)
CASES = tuple(("qkv_gate", m, "target") for m in TARGET_M) + (
    ("qkv_gate", 127, "nearby"), ("qkv_gate", 129, "nearby"),
    ("qkv_gate", 1, "control"), ("unpooler", 16, "control"))
PROFILE_CASES = (("qkv_gate", 8), ("qkv_gate", 128),
                 ("qkv_gate", 256), ("unpooler", 16))
MODEL_RECEIPT = "results/remote-P2-09-pooler-column-batch2-hybrid.csv"
MODEL_SHA256 = "4538b39e8710fa88864127fea35e37598b361838ffcce9ca88d856894e292a35"
MODEL_PARENT_JOB = "41ba778763a5433b"
MODEL_QKV_FRACTIONS = {8: .26688550149315954, 16: .2198894355618683,
                       32: .2218156346327628, 128: .2277972459885718,
                       256: .21332516350297043}


def verify_sources(root=ROOT):
    expected = {path: digest for path, digest in SOURCES.values()} | safe.ORACLES
    expected["projection_p046_confirmation.py"] = SAFE_SHA256
    hashes = {}
    for path, digest in expected.items():
        hashes[path] = hashlib.sha256((root / path).read_bytes()).hexdigest()
        if hashes[path] != digest:
            raise ValueError("Frozen source hash mismatch: " + path)
    for path in ("benchmark.py", "harness.py", "official/muse_bf16_proj.json",
                 "projection_p050_qkv_screen.py"):
        hashes[path] = hashlib.sha256((root / path).read_bytes()).hexdigest()
    return hashes


def protocol_counts():
    return dict(cases=9, target_cases=5, nearby_cases=2, unchanged_controls=2,
                measured_replays=9 * 24 * 2, warmup_replays=9 * 6 * 2,
                before_after_probe_replays=9 * 2 * 5 * 2,
                timing_and_probe_oracle_comparisons=9 * (24 + 6 + 10) * 2,
                activity_replays=4 * 2, total_oracle_comparisons=728,
                capture_warmup_calls=9 * 3 * 2, capture_calls=9 * 2)


def capture_pair(torch, capture, modules, data, states, gate):
    graphs, outputs, observations = {}, {}, {}
    for name in SOURCES:
        gate()
        torch.cuda.synchronize()
        started = time.monotonic()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            graphs[name], outputs[name] = capture(modules[name].kernel_fn, data, states[name])
            torch.cuda.synchronize()
        fallback = any("gemm_and_bias" in str(w.message) and "unfused" in str(w.message)
                       for w in caught)
        observations[name] = dict(synchronized_wall_seconds=time.monotonic() - started,
            explicit_unfused_fallback_warning=fallback,
            warning_scope="Only recognized gemm_and_bias fallback is reported; absence is not proof of Lt selection")
    if outputs["parent"].data_ptr() == outputs["candidate"].data_ptr():
        raise RuntimeError("Captured runner outputs unexpectedly alias")
    return graphs, outputs, observations


def activity_summary(prof, cuda_type):
    """Allowlisted names/durations only; no trace IDs, stacks, shapes or host data."""
    events = []
    for event in prof.events():
        if event.device_type != cuda_type:
            continue
        duration = float(event.device_time_total)
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("Invalid activity duration")
        name = str(event.name)
        events.append(dict(symbol=name, duration_us=duration,
            kind="copy" if "copy" in name.lower() or "memcpy" in name.lower() else "kernel_or_other"))
    return dict(available=bool(events), cuda_events=events,
        copy_events=[event for event in events if event["kind"] == "copy"],
        qualification="One instrumented graph replay after all timing; names/durations only, not acceptance timing, tactic identity, instructions or counters. No visible copy does not prove no hidden copy.")


def modeled_sensitivity(ratios):
    """Apply diagnostic raw cuts to fixed official P046 load fractions only."""
    if set(ratios) != set(TARGET_M) or any(
            not math.isfinite(value) or value <= 0 for value in ratios.values()):
        raise ValueError("Exactly five positive finite target ratios are required")
    factors = {m: 1 - MODEL_QKV_FRACTIONS[m] * (1 - 1 / ratios[m]) for m in TARGET_M}
    gain = math.exp(-sum(math.log(value) for value in factors.values()) / 6) - 1
    return dict(hypothetical_relative_score_gain=gain,
        modeled_load_latency_factors={str(m): factors[m] for m in TARGET_M},
        frozen_qkv_load_fractions={str(m): MODEL_QKV_FRACTIONS[m] for m in TARGET_M},
        frozen_parent_job=MODEL_PARENT_JOB, frozen_receipt_path=MODEL_RECEIPT,
        frozen_receipt_sha256=MODEL_SHA256, shape_weight=62, equally_weighted_loads=6,
        unchanged_load_m=1, other_25_scored_shapes_held_fixed=True,
        qualification="Hypothetical cross-environment sensitivity: diagnostic raw paired cuts applied to frozen calibrated official P046 load weights. This is not an achieved, predicted or newly measured official score.")


def assess(cases):
    if [(case["layer"], case["m"], case["role"]) for case in cases] != list(CASES):
        raise ValueError("Exactly the nine predeclared cases are required")
    if not all(case["passed"] and case["series"]["main"]["passed"] for case in cases):
        raise ValueError("Incomplete correctness cannot pass the screen")
    summaries = { (case["layer"], case["m"]): safe.summarize_samples(case["series"]["main"]["samples"])
                  for case in cases }
    ratios = {m: summaries["qkv_gate", m]["median_paired_raw_ratio"] for m in TARGET_M}
    target_cut = 1 - math.exp(-sum(math.log(ratios[m]) for m in TARGET_M) / len(TARGET_M))
    target_wins = [m for m in TARGET_M if ratios[m] > 1]
    pair_wins = {m: sum(ratio > 1 for ratio in summaries["qkv_gate", m]["reverse_order_pair_ratios"])
                 for m in TARGET_M}
    robust_wins = [m for m in target_wins if pair_wins[m] >= 8]
    movement_flags, nearby_movements, fallback_or_copy = [], [], []
    profiling_available = True
    for case in cases:
        layer, m = case["layer"], case["m"]
        movement = 1 / summaries[layer, m]["median_paired_raw_ratio"] - 1
        movement_record = dict(layer=layer, m=m, role=case["role"], raw_latency_movement=movement)
        if case["role"] == "nearby":
            nearby_movements.append(movement_record)
        if ((case["role"] == "nearby" and movement > .02)
                or (case["role"] == "control" and abs(movement) > .02)):
            movement_flags.append(movement_record)
        observed_fallback = case["capture"]["candidate"]["explicit_unfused_fallback_warning"]
        observed_copy = bool(case["activities"].get("candidate", {}).get("copy_events"))
        if observed_fallback or observed_copy:
            fallback_or_copy.append(dict(layer=layer, m=m, fallback=observed_fallback, copy=observed_copy))
        if (layer, m) in PROFILE_CASES:
            profiling_available &= all(case["activities"].get(name, {}).get("available", False)
                                       for name in SOURCES)
    model = modeled_sensitivity(ratios)
    gates = dict(hypothetical_score_sensitivity_gain_at_least1pct=model["hypothetical_relative_score_gain"] >= .01,
                 target_geomean_raw_cut_at_least4pct=target_cut >= .04,
                 at_least_four_target_median_wins=len(target_wins) >= 4,
                 at_least_four_median_winners_win_eight_of_twelve_pairs=len(robust_wins) >= 4,
                 no_target_regresses_more_than2pct=all(1 / ratio - 1 <= .02 for ratio in ratios.values()),
                 no_observed_candidate_fallback_or_copy=not fallback_or_copy,
                 activity_available_on_four_predeclared_cases=profiling_available,
                 no_nearby_regression_or_absolute_control_movement_over2pct=not movement_flags)
    return dict(target_geomean_raw_latency_cut=target_cut,
        target_median_winners=target_wins, target_winning_pairs={str(m): pair_wins[m] for m in TARGET_M},
        target_median_winners_with_eight_pairs=robust_wins,
        hypothetical_frozen_parent_model=model, gates=gates, screen_passed=all(gates.values()),
        nearby_raw_latency_movements=nearby_movements,
        nearby_or_control_flags_requiring_investigation=movement_flags,
        observed_candidate_fallback_or_copy=fallback_or_copy, automatic_selection=False,
        selected_parent_unchanged=True,
        qualification="Predeclared nine-case screen only. Passing warrants official comparison, never automatic selection. Nearby regressions or unchanged-control absolute movements above 2% prevent clean readiness but do not establish a code defect; nearby improvements are retained and allowed. QKV M512/full prerequisites, concurrency, unprofiled path internals and official score remain unverified.")


def run_series(torch, reference, data, graphs, outputs, oracle_weights, config,
               pointer, make, seed, item, timer, eviction, eviction_out,
               started, gate, save, check_states):
    safe.correctness_phase(torch, reference, data, graphs, outputs, oracle_weights, config, pointer,
                           make, seed + 10, "before", item["correctness"], gate, save)
    check_states()
    for warmup, count, offset in ((True, safe.WARMUP_ROUNDS, 100), (False, safe.MEASURED_ROUNDS, 200)):
        for i in range(count):
            gate()
            fresh = make(seed + offset + i)
            data["x"].copy_(fresh["x"])
            sample = dict(round=i, seed=seed + offset + i, order=list(safe.runner_order(i)),
                          us={}, observations=[], correctness={}, excluded_warmup=warmup)
            item["warmup" if warmup else "samples"].append(sample)
            save()
            for name in safe.runner_order(i):
                gate()
                observation = dict(runner=name, started_utc=datetime.now(timezone.utc).isoformat(),
                                   elapsed_start_seconds=time.monotonic() - started, status="started")
                sample["observations"].append(observation)
                save()
                sample["us"][name] = timer.measure(graphs[name], eviction, eviction_out)
                observation.update(status="measured", elapsed_end_seconds=time.monotonic() - started)
                safe.assert_input(torch, data, fresh, config, pointer)
                save()
            for name in safe.runner_order(i):
                gate()
                safe.check_output(torch, reference, data, outputs[name], fresh, config, pointer, oracle_weights)
                sample["correctness"][name] = True
                save()
    check_states()
    safe.correctness_phase(torch, reference, data, graphs, outputs, oracle_weights, config, pointer,
                           make, seed + 400, "after", item["correctness"], gate, save)
    check_states()
    item["summary"] = safe.summarize_samples(item["samples"])
    item["passed"] = True
    save()


def execute(args, report, save):
    started = time.monotonic()

    def gate():
        if time.monotonic() - started > args.deadline_seconds:
            raise TimeoutError("Internal screen deadline reached; external supervisor handles blocked CUDA calls")

    import torch
    from benchmark import capture
    from harness import environment, load_module
    safe.require_h200(torch)
    report["environment"] = environment(torch, "cuda", ROOT / SOURCES["parent"][0])
    report["environment"]["triton"] = __import__("triton").__version__
    reference = load_module(ROOT / "official/reference.py", "p050_reference")
    modules = {name: load_module(ROOT / path, "p050_" + name) for name, (path, _) in SOURCES.items()}
    with torch.inference_mode():
        gate()
        weights = reference.generate_weights(reference.WEIGHTS_SEED, device="cuda")
        original = safe.freeze_tensors(torch, weights)
        oracle_weights = {name: snapshot[1] for name, snapshot in original.items()}
        states, frozen_states = {}, {}
        report["preparation"] = {}
        for name in SOURCES:
            gate()
            torch.cuda.synchronize()
            tick = time.monotonic()
            states[name] = modules[name].prepare(weights)
            torch.cuda.synchronize()
            report["preparation"][name] = dict(synchronized_wall_seconds=time.monotonic() - tick)
            safe.assert_frozen(torch, weights, original)
            frozen_states[name] = safe.freeze_tensors(torch, states[name])
            save()

        def check_states():
            gate()
            safe.assert_frozen(torch, weights, original)
            for name in SOURCES:
                safe.assert_frozen(torch, states[name], frozen_states[name])

        eviction = torch.ones(safe.EVICTION_BYTES // 4, dtype=torch.float32, device="cuda")
        eviction_out = torch.empty((), dtype=torch.float32, device="cuda")
        timer = safe.ColdGraphTimer(torch)
        pending_profiles = []
        for case_index, (layer, m, role) in enumerate(CASES):
            gate()
            seed = 5000000 + case_index * 2000
            make = lambda s, dist="normal", l=layer, size=m: safe.fresh_input(torch, reference, l, size, s, dist)
            data = make(seed)
            initial = {"x": data["x"].clone()}
            config, pointer = dict(data["config"]), safe.tensor_metadata(data["x"])
            item = dict(layer=layer, m=m, role=role, passed=False, series={}, activities={},
                        shared_input_address=True, activity_requested=(layer, m) in PROFILE_CASES)
            report["cases"].append(item)
            save()
            graphs, outputs, item["capture"] = capture_pair(torch, capture, modules, data, states, gate)
            safe.assert_input(torch, data, initial, config, pointer)
            check_states()
            series = dict(passed=False, correctness=[], warmup=[], samples=[],
                          runner_mapping=dict(parent="parent", candidate="candidate"))
            item["series"]["main"] = series
            run_series(torch, reference, data, graphs, outputs, oracle_weights, config, pointer,
                       make, seed, series, timer, eviction, eviction_out, started, gate, save, check_states)
            item.update(timing_complete=True, immutable_original_and_prepared_tensors=True)
            if (layer, m) in PROFILE_CASES:
                pending_profiles.append((item, data, graphs, outputs, config, pointer, make, seed))
            else:
                item["passed"] = True
            save()
            del graphs, outputs
        # Every target, nearby and unchanged-control sample precedes profiling.
        for item, data, graphs, outputs, config, pointer, make, seed in pending_profiles:
            for name in SOURCES:
                gate()
                fresh = make(seed + 600)
                data["x"].copy_(fresh["x"])
                torch.sum(eviction, dim=0, out=eviction_out)
                torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA],
                                            record_shapes=False, with_stack=False, profile_memory=False) as prof:
                    graphs[name].replay()
                    torch.cuda.synchronize()
                safe.check_output(torch, reference, data, outputs[name], fresh, config, pointer, oracle_weights)
                item["activities"][name] = activity_summary(prof, torch.autograd.DeviceType.CUDA)
                item["activities"][name]["correctness_passed"] = True
                save()
            check_states()
            item["passed"] = True
            save()
        gate()
        report["assessment"] = assess(report["cases"])
        report["passed"] = True
        report["wall_seconds"] = time.monotonic() - started


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-seconds", type=float, default=120)
    parser.add_argument("--outer-timeout-seconds", type=float, required=True)
    args = parser.parse_args(argv)
    safe.validate_options(args)
    report = dict(kind="P050 QKV-only public Torch route screen; not official score", passed=False,
        sources={name: dict(path=path, sha256=digest) for name, (path, digest) in SOURCES.items()}, cases=[],
        protocol=dict(cases=[dict(layer=layer, m=m, role=role) for layer, m, role in CASES],
            counts=protocol_counts(), profile_cases=[dict(layer=layer, m=m) for layer, m in PROFILE_CASES],
            series="One independent main parent/candidate series per case; no beta-zero repeat",
            warmup_rounds=safe.WARMUP_ROUNDS, measured_rounds=safe.MEASURED_ROUNDS,
            reverse_order_pairs=12, eviction_bytes=safe.EVICTION_BYTES,
            events="One pair created, recorded, synchronized and queried before capture; sequentially reused",
            calibration="unmeasured; no subtraction", capture_untimed_calls_per_runner=3,
            same_input_address_across_runners=True,
            immutable_state_scope="after prepare/capture and before/after correctness/timing phases; input/config after every observed replay",
            oracle_order="Both distinct outputs checked after both timed replays; input/config immediately after each replay",
            activity="After all timing; one replay per selected case/runner; allowlisted names/durations only",
            internal_deadline_seconds=args.deadline_seconds, external_supervisor_timeout_seconds=args.outer_timeout_seconds,
            supervisor="must enforce hard timeout externally", raw_samples_retained=True,
            predeclared_gates="Hypothetical modeled gain >=1%; target GM cut >=4%; >=4/5 median winners each with >=8/12 winning pairs; no target >2% loss; nearby regression >2% or unchanged-control absolute movement >2% prevents clean readiness; nearby improvements retained and allowed; observed candidate fallback/copy rejects; both activities present on four profile cases"),
        limitations=["Nine selected cases; QKV M512 and the full official prerequisite suite remain unverified",
            "Sequential graph replay, not concurrent execution", "No hardware counters, instructions, tactic identity or confidence interval",
            "Activity covers four selected cases; names cannot prove hidden library behavior",
            "Hypothetical raw-to-calibrated cross-environment sensitivity is not an official score",
            "Internal checks cannot interrupt a blocking CUDA/profiler call", "P046 remains selected; no automatic promotion"])

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    try:
        report["source_hashes"] = verify_sources()
        save()
        execute(args, report, save)
    except Exception as error:
        report["failure"] = dict(type=type(error).__name__, message=str(error)[:700])
        save()
        print(json.dumps(dict(passed=False, failure=report["failure"])))
        return 1
    save()
    print(json.dumps(dict(passed=report["passed"], cases=len(report["cases"]), automatic_selection=False)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
