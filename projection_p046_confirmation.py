"""Direct P039/P046 H200 confirmation worker; no provisioning or official score.

Run only under an external supervisor enforcing --outer-timeout-seconds.
The internal monotonic deadline cannot interrupt a blocked CUDA operation.
Importing this module for local tests does not import Torch or the candidates.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parent
SOURCES = {
    "parent": ("candidates/projection_router_m1_fp32_gemv.py",
               "0c0c41119ef925b03a33ccab230bc64176cc22c242e3fea77ce01e317376defc"),
    "candidate": ("candidates/projection_pooler_column_batch2_hybrid.py",
                  "001ea12e8764564d507de51e433769f5bd9a9916bfc4c014f0f87522c278545e"),
}
ORACLES = {
    "official/reference.py": "e62491abc9ddcad33f5faba8b0ee74cf6370b7be5dd2dc05031742f3ebf7147d",
    "official/checks.py": "9a046c3c38db28b325dcd1557baecdf6bc9ef941d2175f9a900ba4c230bb8055",
}
CASES = (("pooler", 8, True), ("pooler", 16, True), ("pooler", 32, True),
         ("o", 1, False), ("unpooler", 1, False), ("pooler", 128, False))
EVICTION_BYTES = 256 * 1024 * 1024
WARMUP_ROUNDS = 6
MEASURED_ROUNDS = 24  # Twelve consecutive reversed-order pairs, fixed in advance.


def verify_sources(root=ROOT):
    expected = {path: digest for path, digest in SOURCES.values()} | ORACLES
    hashes = {}
    for path, digest in expected.items():
        hashes[path] = hashlib.sha256((root / path).read_bytes()).hexdigest()
        if hashes[path] != digest:
            raise ValueError("Frozen source hash mismatch: " + path)
    for path in ("benchmark.py", "harness.py", "projection_p046_confirmation.py"):
        hashes[path] = hashlib.sha256((root / path).read_bytes()).hexdigest()
    return hashes


def validate_options(args):
    if not 0 < args.deadline_seconds <= 120:
        raise ValueError("Internal deadline must be positive and at most 120 seconds")
    if not args.deadline_seconds + 10 <= args.outer_timeout_seconds <= 180:
        raise ValueError("Supervisor timeout must exceed the internal deadline by 10 seconds and be at most 180")


def require_h200(torch):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; confirmation not executed")
    name = torch.cuda.get_device_name()
    if "H200" not in name or tuple(torch.cuda.get_device_capability()) != (9, 0):
        raise RuntimeError("H200 SM90 required; confirmation not executed")
    return name


def runner_order(round_index):
    return ("parent", "candidate") if round_index % 2 == 0 else ("candidate", "parent")


def quantile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def distribution(values):
    return dict(count=len(values), minimum=min(values), p10=quantile(values, .1),
                median=statistics.median(values), p90=quantile(values, .9), maximum=max(values))


def summarize_samples(samples):
    if len(samples) != MEASURED_ROUNDS:
        raise ValueError("All 24 predefined rounds are required; partial samples are not summarized")
    for i, sample in enumerate(samples):
        if sample["round"] != i or tuple(sample["order"]) != runner_order(i):
            raise ValueError("Unexpected order; no post-hoc re-pairing")
        for name in SOURCES:
            if not math.isfinite(sample["us"][name]) or sample["us"][name] <= 0:
                raise ValueError("Invalid raw timing")
    ratios = [s["us"]["parent"] / s["us"]["candidate"] for s in samples]
    pairs = [math.sqrt(ratios[i] * ratios[i + 1]) for i in range(0, len(ratios), 2)]
    median = statistics.median(pairs)
    return dict(round_parent_over_candidate_ratios=ratios,
                reverse_order_pair_ratios=pairs, pair_distribution=distribution(pairs),
                raw_us={name: distribution([s["us"][name] for s in samples]) for name in SOURCES},
                median_paired_raw_ratio=median, candidate_raw_latency_cut=1 - 1 / median,
                qualification="Direct gross whole-call graph timing; no overhead subtraction, formal confidence interval or official score")


def summarize_cases(cases):
    if [(c["layer"], c["m"], c["target"]) for c in cases] != list(CASES):
        raise ValueError("Complete six-case set required")
    if not all(c["passed"] for c in cases):
        raise ValueError("A partial or failed correctness case cannot pass confirmation")
    ratios = [c["summary"]["median_paired_raw_ratio"] for c in cases if c["target"]]
    target_cut = 1 - 1 / math.prod(ratios) ** (1 / len(ratios))
    controls = [dict(layer=c["layer"], m=c["m"],
                     raw_latency_movement=1 / c["summary"]["median_paired_raw_ratio"] - 1)
                for c in cases if not c["target"]]
    return dict(target_geomean_raw_latency_cut=target_cut,
                target_gates=dict(geomean_cut_at_least3pct=target_cut >= .03,
                                  no_individual_regression=all(r >= 1 for r in ratios)),
                controls=controls,
                controls_requiring_investigation=[c for c in controls if abs(c["raw_latency_movement"]) > .02],
                automatic_selection=False,
                caveat="Twelve pairs are a bounded diagnostic; small movements require repeatability/quantization review. Neither target gates nor absence of a control flag establish a full official score.")


class ColdGraphTimer:
    """One initialized event pair, reused only after the preceding end completes."""
    def __init__(self, torch):
        self.torch = torch
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()
        self.end.record()
        self.end.synchronize()
        self.start.elapsed_time(self.end)  # Force lazy event initialization now.

    def measure(self, graph, eviction, eviction_out):
        self.torch.sum(eviction, dim=0, out=eviction_out)
        self.start.record()
        graph.replay()
        self.end.record()
        self.end.synchronize()
        return self.start.elapsed_time(self.end) * 1000


def tensor_metadata(tensor):
    return (tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype),
            str(tensor.device), tensor.data_ptr())


def freeze_tensors(torch, tensors):
    snapshots = {}
    for name, tensor in tensors.items():
        if not torch.is_tensor(tensor) or not tensor.is_contiguous():
            raise ValueError("Expected contiguous tensor-only prepared state: " + name)
        snapshots[name] = (tensor_metadata(tensor), tensor.clone())
    return snapshots


def assert_frozen(torch, tensors, snapshots):
    if tensors.keys() != snapshots.keys():
        raise RuntimeError("Prepared/original tensor keys changed")
    for name, tensor in tensors.items():
        metadata, frozen = snapshots[name]
        if tensor_metadata(tensor) != metadata or not torch.equal(
                tensor.view(torch.uint8), frozen.view(torch.uint8)):
            raise RuntimeError("Prepared/original tensor changed: " + name)


def assert_input(torch, data, fresh, config, pointer):
    if data["config"] != config or tensor_metadata(data["x"]) != pointer:
        raise RuntimeError("Captured input metadata/config changed")
    if not torch.equal(data["x"].view(torch.uint8), fresh["x"].view(torch.uint8)):
        raise RuntimeError("Captured input content changed")


def check_output(torch, reference, data, output, fresh, config, pointer, frozen_weights):
    assert_input(torch, data, fresh, config, pointer)
    passed, message = reference.check_implementation(data, output, frozen_weights)
    if not passed:
        raise RuntimeError("Unmodified official FP64 oracle failed: " + str(message)[:400])


def sparse_spec(m, k, seed):
    return [(seed + 131 * row) % k for row in range(m)], [8 if row % 2 == 0 else -32 for row in range(m)]


def fresh_input(torch, reference, layer, m, seed, distribution_name="normal"):
    data = reference.generate_input(m=m, layer=layer, seed=seed,
                                    dist="outliers" if distribution_name == "outliers" else "normal",
                                    device="cuda")
    if distribution_name in ("sparse", "zero"):
        data["x"].zero_()
        if distribution_name == "sparse":
            cols, values = sparse_spec(m, data["x"].shape[1], seed)
            indices = torch.tensor(cols, dtype=torch.int64, device="cuda")[:, None]
            values = torch.tensor(values, dtype=data["x"].dtype, device="cuda")[:, None]
            data["x"].scatter_(1, indices, values)
    elif distribution_name not in ("normal", "outliers"):
        raise ValueError("Unknown input distribution")
    return data


def capture_pair(torch, capture, modules, data, states, gate):
    graphs, outputs, durations = {}, {}, {}
    for name in SOURCES:
        gate()
        torch.cuda.synchronize()
        started = time.monotonic()
        graphs[name], outputs[name] = capture(modules[name].kernel_fn, data, states[name])
        torch.cuda.synchronize()
        durations[name] = time.monotonic() - started
    if outputs["parent"].data_ptr() == outputs["candidate"].data_ptr():
        raise RuntimeError("Captured outputs unexpectedly alias")
    return graphs, outputs, durations


def correctness_phase(torch, reference, data, graphs, outputs, weights, config,
                      pointer, make, seed, phase, records, gate, save):
    a = make(seed, "normal")
    probes = (("A", a, "normal"), ("B", make(seed + 1, "outliers"), "outliers"),
              ("A-return", a, "normal"), ("sparse", make(seed + 2, "sparse"), "sparse"),
              ("zero", make(seed + 3, "zero"), "zero"))
    first = {}
    for index, (label, fresh, dist) in enumerate(probes):
        data["x"].copy_(fresh["x"])
        for name in runner_order(index):
            gate()
            item = dict(phase=phase, label=label, runner=name, distribution=dist, passed=False)
            records.append(item)
            save()
            graphs[name].replay()
            torch.cuda.synchronize()
            check_output(torch, reference, data, outputs[name], fresh, config, pointer, weights)
            if label == "A":
                first[name] = outputs[name].clone()
            elif label == "A-return" and not torch.equal(
                    first[name].view(torch.uint8), outputs[name].view(torch.uint8)):
                raise RuntimeError("A-return did not reproduce exact output bytes: " + name)
            item.update(passed=True, same_input_address=True, config_unchanged=True)
            save()


def execute(args, report, save):
    started = time.monotonic()

    def gate():
        if time.monotonic() - started > args.deadline_seconds:
            raise TimeoutError("Internal confirmation deadline reached; external supervisor handles blocked CUDA calls")

    import torch
    from benchmark import capture
    from harness import environment, load_module
    require_h200(torch)
    report["environment"] = environment(torch, "cuda", ROOT / SOURCES["parent"][0])
    report["environment"]["triton"] = __import__("triton").__version__
    reference = load_module(ROOT / "official/reference.py", "p046_reference")
    modules = {name: load_module(ROOT / path, "p046_" + name) for name, (path, _) in SOURCES.items()}
    with torch.inference_mode():
        gate()
        weights = reference.generate_weights(reference.WEIGHTS_SEED, device="cuda")
        frozen_original = freeze_tensors(torch, weights)
        oracle_weights = {name: snapshot[1] for name, snapshot in frozen_original.items()}
        states, frozen_states = {}, {}
        report["preparation"] = {}
        for name in SOURCES:
            gate()
            torch.cuda.synchronize()
            tick = time.monotonic()
            states[name] = modules[name].prepare(weights)
            torch.cuda.synchronize()
            report["preparation"][name] = dict(synchronized_wall_seconds=time.monotonic() - tick)
            assert_frozen(torch, weights, frozen_original)
            frozen_states[name] = freeze_tensors(torch, states[name])
            save()

        def check_states():
            gate()
            assert_frozen(torch, weights, frozen_original)
            for name in SOURCES:
                assert_frozen(torch, states[name], frozen_states[name])

        eviction = torch.ones(EVICTION_BYTES // 4, dtype=torch.float32, device="cuda")
        eviction_out = torch.empty((), dtype=torch.float32, device="cuda")
        timer = ColdGraphTimer(torch)
        report["event_initialization"] = "Two CUDA events created, recorded, synchronized and elapsed-time queried before capture/warmup/measurement; reused sequentially"
        for case_index, (layer, m, target) in enumerate(CASES):
            gate()
            seed = 4600000 + case_index * 1000
            make = lambda s, dist="normal": fresh_input(torch, reference, layer, m, s, dist)
            data = make(seed)
            expected_initial = {"x": data["x"].clone()}
            config, pointer = dict(data["config"]), tensor_metadata(data["x"])
            item = dict(layer=layer, m=m, target=target, passed=False,
                        correctness=[], warmup=[], samples=[], shared_input_address=True)
            report["cases"].append(item)
            save()
            graphs, outputs, durations = capture_pair(torch, capture, modules, data, states, gate)
            item["capture_wall_seconds"] = durations
            assert_input(torch, data, expected_initial, config, pointer)
            check_states()
            correctness_phase(torch, reference, data, graphs, outputs, oracle_weights, config, pointer,
                              make, seed + 10, "before", item["correctness"], gate, save)
            check_states()
            for warmup, count, offset in ((True, WARMUP_ROUNDS, 100), (False, MEASURED_ROUNDS, 200)):
                for i in range(count):
                    gate()
                    fresh = make(seed + offset + i)
                    data["x"].copy_(fresh["x"])
                    sample = dict(round=i, seed=seed + offset + i, order=list(runner_order(i)),
                                  us={}, observations=[], correctness={},
                                  excluded_warmup=warmup)
                    (item["warmup"] if warmup else item["samples"]).append(sample)
                    save()
                    for name in runner_order(i):
                        gate()
                        observation = dict(runner=name, started_utc=datetime.now(timezone.utc).isoformat(),
                                           elapsed_start_seconds=time.monotonic() - started, status="started")
                        sample["observations"].append(observation)
                        save()
                        sample["us"][name] = timer.measure(graphs[name], eviction, eviction_out)
                        observation.update(status="measured", elapsed_end_seconds=time.monotonic() - started)
                        # Detect input mutation before the second runner sees the shared X,
                        # but keep the large FP64 oracle out of the paired measurement gap.
                        assert_input(torch, data, fresh, config, pointer)
                        save()
                    for name in runner_order(i):
                        gate()
                        check_output(torch, reference, data, outputs[name], fresh, config, pointer, oracle_weights)
                        sample["correctness"][name] = True
                        save()
            check_states()
            correctness_phase(torch, reference, data, graphs, outputs, oracle_weights, config, pointer,
                              make, seed + 400, "after", item["correctness"], gate, save)
            check_states()
            item["summary"] = summarize_samples(item["samples"])
            item.update(passed=True, immutable_original_and_prepared_tensors=True,
                        immutable_state_scope="after each prepare/capture and before/after correctness/timing phases; input/config checked every observed replay")
            save()
            del graphs, outputs
        gate()
        report["assessment"] = summarize_cases(report["cases"])
        report["passed"] = True
        report["wall_seconds"] = time.monotonic() - started


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-seconds", type=float, default=120)
    parser.add_argument("--outer-timeout-seconds", type=float, required=True)
    args = parser.parse_args(argv)
    validate_options(args)
    report = dict(kind="P046 direct P039/P046 confirmation; not official score", passed=False,
                  sources={name: dict(path=path, sha256=digest) for name, (path, digest) in SOURCES.items()},
                  cases=[], protocol=dict(cases=[dict(layer=l, m=m, target=t) for l, m, t in CASES],
                    eviction_bytes=EVICTION_BYTES, warmup_rounds=WARMUP_ROUNDS,
                    capture_untimed_calls_per_runner=3, measured_rounds=MEASURED_ROUNDS,
                    reverse_order_pairs=MEASURED_ROUNDS // 2, same_input_address_across_runners=True,
                    oracle_order="Check both separate outputs after both timed replays in each round; input/config checked after each replay",
                    calibration="unmeasured; no fixed-cost subtraction", raw_samples_retained=True,
                    internal_deadline_seconds=args.deadline_seconds,
                    external_supervisor_timeout_seconds=args.outer_timeout_seconds,
                    supervisor="must be enforced externally; not enforced by this worker"),
                  limitations=["Six selected cases only, not the full official suite or official score",
                    "Sequential graph replay, not concurrent execution",
                    "No hardware counters/profiler, instructions, clock traces or occupancy evidence",
                    "Event quantization and twelve-pair uncertainty require interpretation; no formal confidence interval",
                    "Internal deadline cannot interrupt a blocking CUDA operation"])

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
    print(json.dumps(dict(passed=report["passed"], cases=len(report["cases"]),
                          automatic_selection=False)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
