"""Unchanged-P039 H200 diagnostic; no provisioning, counters, or official score.

Invoke only beneath an external process/container supervisor enforcing the
declared --outer-timeout-seconds. Monotonic checks cannot interrupt a stuck CUDA
call. Modes are separate: correctness, paired gross timing, ordinary activity.
Warm-cache measurements are optional explanatory controls, never acceptance data.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from projection_p041_plan import ROOT, SOURCE, SOURCE_SHA256, REPRESENTATIVES

EVICTION_BYTES = 256 * 1024 * 1024
WARMUP_REPLAYS = 6
MEASURED_ROUNDS = 24  # Twelve reverse-order pairs, fixed before execution.


def verify_sources(root=ROOT):
    paths = (SOURCE, "official/reference.py", "official/baseline.py",
             "official/checks.py", "official/muse_bf16_proj.json", "harness.py",
             "benchmark.py", "projection_p041_plan.py", "projection_p041_probe.py")
    hashes = {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}
    if hashes[SOURCE] != SOURCE_SHA256:
        raise ValueError("Selected P039 source hash mismatch; execution refused")
    return hashes


def validate_options(args):
    if not 0 < args.deadline_seconds <= 300:
        raise ValueError("Internal deadline must be positive and at most 300 seconds")
    if not args.deadline_seconds + 10 <= args.outer_timeout_seconds <= 600:
        raise ValueError("External supervisor timeout must exceed internal deadline by 10 seconds and be at most 600")
    if args.include_warm_cache_diagnostic and args.mode != "timing":
        raise ValueError("Warm-cache control is available only in timing mode")


def require_h200(torch):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; H200 diagnostic not executed")
    name = torch.cuda.get_device_name()
    if "H200" not in name or tuple(torch.cuda.get_device_capability()) != (9, 0):
        raise RuntimeError("An H200 SM90 is required; diagnostic not executed")
    return name


def runner_order(round_index):
    return ("candidate", "baseline") if round_index % 2 == 0 else ("baseline", "candidate")


def condition_order(round_index, include_warm):
    if not include_warm:
        return ("cold",)
    # Opposite runner orders occur under both condition orders over four rounds.
    return ("cold", "warm") if (round_index // 2) % 2 == 0 else ("warm", "cold")


def paired_summary(samples):
    if len(samples) < 2 or len(samples) % 2:
        raise ValueError("Complete reverse-order pairs required")
    for i, sample in enumerate(samples):
        if sample["round"] != i or tuple(sample["order"]) != runner_order(i):
            raise ValueError("Unexpected measurement order; no post-hoc pairing")
        if not all(math.isfinite(sample["us"][name]) and sample["us"][name] > 0
                   for name in ("candidate", "baseline")):
            raise ValueError("Invalid timing sample")
    ratios = [r["us"]["baseline"]/r["us"]["candidate"] for r in samples]
    pairs = [math.sqrt(ratios[i]*ratios[i+1]) for i in range(0, len(ratios), 2)]
    return dict(pair_ratios=pairs, median_paired_gross_ratio=statistics.median(pairs),
                median_gross_us={name: statistics.median(r["us"][name] for r in samples)
                                 for name in ("candidate", "baseline")},
                qualification="Gross whole-call CUDA graph timing; no calibration, confidence interval, or official score")


class NoEviction:
    """Used only after an immediate untimed replay of the identical graph."""
    def sum(self):
        return None


def timed_replay(graph, eviction, condition, replay_us):
    if condition == "warm":
        graph.replay()  # Nothing touching device memory may intervene before timing.
        return replay_us(graph, NoEviction())
    if condition != "cold":
        raise ValueError("Unknown cache condition")
    return replay_us(graph, eviction)


def prepare_with_timing(torch, module, weights):
    """Host elapsed time including completion of this runner's untimed GPU work."""
    torch.cuda.synchronize()
    started = time.perf_counter()
    state = module.prepare(weights)
    torch.cuda.synchronize()
    return state, time.perf_counter() - started


def export_activity_trace(prof, path):
    """Preserve the raw trace and extract only metadata actually emitted by Kineto."""
    prof.export_chrome_trace(str(path))
    raw = path.read_bytes()
    events = json.loads(raw)["traceEvents"]
    if not isinstance(events, list):
        raise ValueError("Chrome traceEvents must be a list")
    resource_names = {
        "grid": "grid", "block": "block",
        "registersperthread": "registers_per_thread",
        "sharedmemory": "shared_memory_bytes",
        "sharedmemorybytes": "shared_memory_bytes",
        "staticsharedmemory": "static_shared_memory_bytes",
        "dynamicsharedmemory": "dynamic_shared_memory_bytes",
        "blockspersm": "blocks_per_sm",
        "warpspersm": "warps_per_sm",
    }
    kernels = []
    for event in events:
        categories = {c.strip().lower() for c in str(event.get("cat", "")).split(",")}
        if event.get("ph") != "X" or not categories.intersection({"kernel", "cuda_kernel"}):
            continue
        trace_args = event.get("args", {})
        if not isinstance(trace_args, dict):
            raise ValueError("Chrome kernel args must be an object")
        resources = {}
        for key, value in trace_args.items():
            normalized = "".join(c for c in key.lower() if c.isalnum())
            if normalized in resource_names:
                resources[resource_names[normalized]] = value
        kernels.append(dict(symbol=event.get("name"), duration_us=event.get("dur"),
                            timestamp_us=event.get("ts"), trace_args=trace_args,
                            resources=resources))
    return dict(file=path.name, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
                chrome_kernels=kernels,
                qualification="Raw ordinary activity trace; resource fields only when emitted, not PTX/SASS, spills, hardware counters, or acceptance timing")


def assert_immutable(torch, data, expected_x, weights, frozen_weights, pointer):
    if data["x"].data_ptr() != pointer or not torch.equal(data["x"], expected_x):
        raise RuntimeError("Captured input address/content mutated")
    for name in frozen_weights:
        if not torch.equal(weights[name], frozen_weights[name]):
            raise RuntimeError("Original weight tensor mutated: " + name)


def output_check(torch, reference, data, out, expected_x, weights, frozen_weights, pointer):
    assert_immutable(torch, data, expected_x, weights, frozen_weights, pointer)
    passed, message = reference.check_implementation(data, out, frozen_weights)
    if not passed:
        raise RuntimeError("Official oracle failed: " + str(message)[:400])


def execute(args, report, save):
    # Lazy imports keep local mocked/static tests from loading a CUDA candidate.
    import torch
    from benchmark import capture, replay_us
    from harness import environment, load_module
    require_h200(torch)
    report["environment"] = environment(torch, "cuda", ROOT/SOURCE)
    report["environment"]["triton"] = __import__("triton").__version__
    started = time.monotonic()

    def gate():
        if time.monotonic()-started > args.deadline_seconds:
            raise TimeoutError("Internal diagnostic deadline reached; supervisor remains responsible for stuck operations")

    reference = load_module(ROOT/"official/reference.py", "p041_reference")
    modules = dict(candidate=load_module(ROOT/SOURCE, "p041_candidate"),
                   baseline=load_module(ROOT/"official/baseline.py", "p041_baseline"))
    with torch.inference_mode():
        gate()
        weights = reference.generate_weights(reference.WEIGHTS_SEED, device="cuda")
        frozen = {name: value.clone() for name, value in weights.items()}
        states = {}
        report["preparation"] = dict(runners={},
            qualification="Untimed per-runner host elapsed time bracketed by CUDA synchronization; excludes weight generation/cloning and is not kernel latency")
        for name, module in modules.items():
            gate()
            states[name], elapsed = prepare_with_timing(torch, module, weights)
            report["preparation"]["runners"][name] = dict(synchronized_wall_seconds=elapsed)
            save()
        eviction = torch.ones(EVICTION_BYTES//4, device="cuda", dtype=torch.float32)
        for case_index, (layer, m) in enumerate(REPRESENTATIVES):
            gate()
            seed = 4100000 + case_index*1000
            make = lambda s, dist="normal": reference.generate_input(
                m=m, layer=layer, seed=s, dist=dist, device="cuda")
            a, b = make(seed), make(seed+1, "outliers")
            graphs, outputs, inputs, pointers = {}, {}, {}, {}
            item = dict(layer=layer, m=m, aba=[], passed=False)
            report["cases"].append(item)
            for name, module in modules.items():
                gate()
                inputs[name] = make(seed)
                pointers[name] = inputs[name]["x"].data_ptr()
                graphs[name], outputs[name] = capture(module.kernel_fn, inputs[name], states[name])
                assert_immutable(torch, inputs[name], a["x"], weights, frozen, pointers[name])
                first_a = None
                for label, fresh in (("A", a), ("B", b), ("A-return", a)):
                    gate()
                    inputs[name]["x"].copy_(fresh["x"])
                    graphs[name].replay()
                    torch.cuda.synchronize()
                    output_check(torch, reference, inputs[name], outputs[name], fresh["x"],
                                 weights, frozen, pointers[name])
                    if first_a is None:
                        first_a = outputs[name].clone()
                    elif label == "A-return" and not torch.equal(first_a, outputs[name]):
                        raise RuntimeError("Repeated A did not reproduce exact output bytes")
                    item["aba"].append(dict(runner=name, state=label, passed=True,
                                            distribution="outliers" if label == "B" else "normal",
                                            same_input_address=True, immutable_x_weights=True,
                                            oracle="unmodified official reference with frozen original weights"))
            if args.mode == "timing":
                item["warmup"] = []
                for i in range(WARMUP_REPLAYS):
                    fresh = make(seed+100+i)
                    for name in runner_order(i):
                        gate()
                        inputs[name]["x"].copy_(fresh["x"])
                        us = timed_replay(graphs[name], eviction, "cold", replay_us)
                        output_check(torch, reference, inputs[name], outputs[name], fresh["x"],
                                     weights, frozen, pointers[name])
                        item["warmup"].append(dict(round=i, runner=name, gross_us=us,
                                                   excluded_by_predeclared_protocol=True))
                item["conditions"] = {c: dict(samples=[], acceptance_eligible=(c == "cold"))
                                      for c in condition_order(0, args.include_warm_cache_diagnostic)}
                for i in range(MEASURED_ROUNDS):
                    fresh = make(seed+200+i)
                    for condition in condition_order(i, args.include_warm_cache_diagnostic):
                        sample = dict(round=i, seed=seed+200+i, order=list(runner_order(i)), us={})
                        for name in runner_order(i):
                            gate()
                            inputs[name]["x"].copy_(fresh["x"])
                            sample["us"][name] = timed_replay(graphs[name], eviction, condition, replay_us)
                            output_check(torch, reference, inputs[name], outputs[name], fresh["x"],
                                         weights, frozen, pointers[name])
                        item["conditions"][condition]["samples"].append(sample)
                    save()
                for condition, values in item["conditions"].items():
                    values.update(paired_summary(values["samples"]))
                    if condition == "warm":
                        values["qualification"] += "; warm-cache sensitivity only, not scored or acceptance evidence"
            elif args.mode == "activity":
                item["activities"] = []
                for name in runner_order(case_index):
                    gate()
                    fresh = make(seed+500)
                    inputs[name]["x"].copy_(fresh["x"])
                    for _ in range(WARMUP_REPLAYS):
                        graphs[name].replay()
                    eviction.sum()
                    torch.cuda.synchronize()
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                            torch.profiler.ProfilerActivity.CUDA],
                                                record_shapes=False, with_stack=False,
                                                profile_memory=False) as prof:
                        graphs[name].replay()
                        torch.cuda.synchronize()
                    trace_path = args.output.with_name(
                        f"{args.output.stem}.case{case_index:02d}.{layer}.m{m}.{name}.trace.json")
                    trace = export_activity_trace(prof, trace_path)
                    output_check(torch, reference, inputs[name], outputs[name], fresh["x"],
                                 weights, frozen, pointers[name])
                    events = [dict(symbol=e.name, duration_us=e.device_time_total)
                              for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
                    item["activities"].append(dict(runner=name, cuda_events=events, trace=trace,
                        qualification="One instrumented activity sample, not acceptance timing or score",
                        resource_metadata="See trace.chrome_kernels; missing fields are unavailable, not zero",
                        instructions_spills="PTX/SASS and spills are not captured"))
            gate()
            item["passed"] = True
            save()
        report["wall_seconds"] = time.monotonic()-started
        report["passed"] = True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("correctness", "timing", "activity"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-seconds", type=float, default=150)
    parser.add_argument("--outer-timeout-seconds", type=float, required=True,
                        help="Must actually be enforced by the external supervisor; this argument is only its declaration")
    parser.add_argument("--include-warm-cache-diagnostic", action="store_true")
    args = parser.parse_args(argv)
    validate_options(args)
    report = dict(kind="P041 unchanged-source diagnostic; not official score", mode=args.mode,
                  passed=False, cases=[], source_sha256=SOURCE_SHA256,
                  representatives=[dict(layer=l, m=m) for l, m in REPRESENTATIVES],
                  protocol=dict(eviction_bytes=EVICTION_BYTES, warmup_replays=WARMUP_REPLAYS,
                                capture_helper_untimed_warmup_calls=3,
                                measured_rounds=MEASURED_ROUNDS if args.mode == "timing" else 0,
                                paired_rounds=MEASURED_ROUNDS//2 if args.mode == "timing" else 0,
                                raw_samples_preserved=True, fixed_cost_subtraction=False,
                                internal_deadline_seconds=args.deadline_seconds,
                                external_supervisor_timeout_seconds=args.outer_timeout_seconds,
                                supervisor_enforcement="required externally; not verified/enforced by this worker",
                                warm_control="immediate identical-graph replay, then timed replay; never acceptance"),
                  limitations=["Five representatives only, not all nine prerequisites and thirty scored shapes",
                               "Sequential same-address replay, not overlapping streams/concurrent calls",
                               "No hardware counters, NCU, binary/PTX capture or measured traffic",
                               "Original official oracle/tolerances; no relaxed precision", 
                               "Monotonic deadline cannot interrupt a blocking CUDA call"])
    def save():
        args.output.write_text(json.dumps(report, indent=2)+"\n")
    try:
        report["source_hashes"] = verify_sources()
        save()
        execute(args, report, save)
    except Exception as error:
        report["failure"] = dict(type=type(error).__name__, message=str(error)[:700])
        save()
        print(json.dumps(dict(passed=False, mode=args.mode, failure=report["failure"])))
        return 1
    save()
    print(json.dumps(dict(passed=report["passed"], mode=args.mode, cases=len(report["cases"]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
