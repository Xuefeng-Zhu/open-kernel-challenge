"""Two-case P046/P049 H200 screen; no provisioning or official score.

The supervisor must enforce --outer-timeout-seconds. Importing this module
imports neither Torch nor candidate code. Activity collection follows all timing.
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
    "candidate": ("candidates/projection_qkv_unpooler_zero_bias.py",
                  "4c3b0ed4181cbdc70f8557fd7445b6937c1ae0b236724c87178faf9d02116ad7"),
}
SAFE_SHA256 = "6c57daf6ab5aca4a73aa1a608b6ae978af284a02d41c489b4e7b62b307505029"
CASES = (("qkv_gate", 128), ("unpooler", 16))
RUNNERS = ("parent", "candidate", "beta0_control")
SERIES = (("main", "candidate"), ("negative_control", "beta0_control"))


def verify_sources(root=ROOT):
    expected = {path: digest for path, digest in SOURCES.values()} | safe.ORACLES
    expected["projection_p046_confirmation.py"] = SAFE_SHA256
    hashes = {}
    for path, digest in expected.items():
        hashes[path] = hashlib.sha256((root / path).read_bytes()).hexdigest()
        if hashes[path] != digest:
            raise ValueError("Frozen source hash mismatch: " + path)
    for path in ("benchmark.py", "harness.py", "official/muse_bf16_proj.json",
                 "projection_p049_screen.py"):
        hashes[path] = hashlib.sha256((root / path).read_bytes()).hexdigest()
    return hashes


def beta0_kernel(torch, data, state):
    """Negative control only: same GEMM arguments; ignored self aliases fresh out."""
    x, config = data["x"], data["config"]
    layer, n = config["layer"], config["n"]
    if (layer, x.shape[0]) not in CASES:
        raise ValueError("Beta-zero control is restricted to the two diagnostic cases")
    out = torch.empty((x.shape[0], n), dtype=x.dtype, device=x.device)
    return torch.addmm(out, x, state[layer][:n].T, beta=0, alpha=1, out=out)


def capture_runners(torch, capture, modules, data, states, gate):
    graphs, outputs, observations = {}, {}, {}
    for name in RUNNERS:
        gate()
        torch.cuda.synchronize()
        started = time.monotonic()
        fn = (lambda d, s: beta0_kernel(torch, d, s)) if name == "beta0_control" else modules[name].kernel_fn
        state = states["parent"] if name == "beta0_control" else states[name]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            graphs[name], outputs[name] = capture(fn, data, state)
            torch.cuda.synchronize()
        fallback = any("gemm_and_bias" in str(w.message) and "unfused" in str(w.message)
                       for w in caught)
        observations[name] = dict(synchronized_wall_seconds=time.monotonic() - started,
                                  explicit_unfused_fallback_warning=fallback,
                                  warning_scope="Only recognized gemm_and_bias fallback is reported; absence is not proof of Lt selection")
    if len({output.data_ptr() for output in outputs.values()}) != len(RUNNERS):
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
                qualification="One instrumented graph replay; names/durations only, not timing acceptance data, API/tactic identity, instructions or counters. No visible copy does not prove no hidden copy.")


def assess(cases):
    if [(case["layer"], case["m"]) for case in cases] != list(CASES):
        raise ValueError("Exactly the two predeclared cases are required")
    if not all(case["passed"] for case in cases):
        raise ValueError("Incomplete correctness cannot pass the screen")
    ratios = [case["series"]["main"]["summary"]["median_paired_raw_ratio"] for case in cases]
    cut = 1 - 1 / math.prod(ratios) ** (1 / len(ratios))
    profiling_available = all(case["activities"][name]["available"] for case in cases for name in SOURCES)
    control_profiling_available = all(case["activities"]["beta0_control"]["available"] for case in cases)
    fallback_or_copy = []
    identical_without_benefit = []
    for case, ratio in zip(cases, ratios):
        candidate = case["activities"]["candidate"]
        if case["capture"]["candidate"]["explicit_unfused_fallback_warning"] or candidate["copy_events"]:
            fallback_or_copy.append(dict(layer=case["layer"], m=case["m"]))
        parent_events = case["activities"]["parent"]["cuda_events"]
        candidate_events = candidate["cuda_events"]
        if parent_events and candidate_events and [e["symbol"] for e in parent_events] == [e["symbol"] for e in candidate_events] and ratio <= 1:
            identical_without_benefit.append(dict(layer=case["layer"], m=case["m"]))
    gates = dict(target_geomean_raw_cut_at_least5pct=cut >= .05,
                 neither_case_regresses_more_than2pct=all(1 / ratio - 1 <= .02 for ratio in ratios),
                 no_observed_candidate_fallback_or_copy=not fallback_or_copy,
                 no_identical_kernel_without_benefit=not identical_without_benefit,
                 activity_available=profiling_available)
    return dict(target_geomean_raw_latency_cut=cut, gates=gates,
                observed_candidate_fallback_or_copy=fallback_or_copy,
                identical_kernel_without_benefit=identical_without_benefit,
                screen_passed=all(gates.values()), automatic_selection=False,
                negative_control_used_for_selection=False,
                negative_control_activity_available=control_profiling_available,
                qualification="Two-case screen only; no full-shape/official score or confidence interval. Negative control tests route equivalence; it cannot promote the candidate. Identical names do not establish identical machine code.")


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
    reference = load_module(ROOT / "official/reference.py", "p049_reference")
    modules = {name: load_module(ROOT / path, "p049_" + name) for name, (path, _) in SOURCES.items()}
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
        for case_index, (layer, m) in enumerate(CASES):
            gate()
            seed = 4900000 + case_index * 2000
            make = lambda s, dist="normal", l=layer, size=m: safe.fresh_input(torch, reference, l, size, s, dist)
            data = make(seed)
            initial = {"x": data["x"].clone()}
            config, pointer = dict(data["config"]), safe.tensor_metadata(data["x"])
            item = dict(layer=layer, m=m, passed=False, series={}, activities={}, shared_input_address=True)
            report["cases"].append(item)
            save()
            graphs, outputs, item["capture"] = capture_runners(torch, capture, modules, data, states, gate)
            safe.assert_input(torch, data, initial, config, pointer)
            check_states()
            for series_name, alternate in SERIES:
                series = dict(passed=False, runner_mapping=dict(parent="parent", candidate=alternate),
                              correctness=[], warmup=[], samples=[], acceptance_eligible=series_name == "main")
                item["series"][series_name] = series
                run_series(torch, reference, data, {"parent": graphs["parent"], "candidate": graphs[alternate]},
                           {"parent": outputs["parent"], "candidate": outputs[alternate]}, oracle_weights,
                           config, pointer, make, seed, series, timer, eviction, eviction_out,
                           started, gate, save, check_states)
            item["timing_complete"] = True
            pending_profiles.append((item, data, graphs, outputs, config, pointer, make, seed))
            save()
        # All acceptance and control timing precedes profiler initialization.
        for item, data, graphs, outputs, config, pointer, make, seed in pending_profiles:
            for name in RUNNERS:
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
            item.update(passed=True, immutable_original_and_prepared_tensors=True)
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
    report = dict(kind="P049 zero-bias public Torch route screen; not official score", passed=False,
                  sources={name: dict(path=path, sha256=digest) for name, (path, digest) in SOURCES.items()}, cases=[],
                  protocol=dict(cases=[dict(layer=layer, m=m) for layer, m in CASES],
                    series=dict(SERIES), warmup_rounds=safe.WARMUP_ROUNDS, measured_rounds=safe.MEASURED_ROUNDS,
                    negative_control="beta=0 addmm uses fresh 2-D output as ignored self, matching mm's native call; candidate uses 1-D zero bias. This tests route equivalence, not beta as the sole changed argument.",
                    reverse_order_pairs=12, eviction_bytes=safe.EVICTION_BYTES,
                    events="One pair created/initialized before capture; sequentially reused", calibration="unmeasured; no subtraction",
                    capture_untimed_calls_per_runner=3, same_input_address_across_runners=True,
                    immutable_state_scope="after prepare/capture and before/after correctness/timing phases; input/config after every observed replay",
                    oracle_order="Both separate outputs checked after both timed replays; input/config immediately after each replay",
                    activity="After all timing; one replay per case/runner; allowlisted names/durations only",
                    internal_deadline_seconds=args.deadline_seconds, external_supervisor_timeout_seconds=args.outer_timeout_seconds,
                    supervisor="must enforce hard timeout externally", raw_samples_retained=True),
                  limitations=["Two shapes, not full official correctness or score", "Sequential replay, not concurrency",
                    "No hardware counters, PTX/SASS, clock traces, tactic identity or formal confidence interval",
                    "Empty activity means unavailable; names cannot prove hidden library behavior",
                    "Internal checks cannot interrupt a blocking CUDA/profiler call"])

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

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
