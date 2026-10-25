"""P052 two-case orientation execution screen, gated by P051 correctness.

This worker provisions nothing. The launcher must first run frozen P051 in full
correctness mode and enforce this worker's external timeout. No profiler, guessed
calibration, official score or automatic selection is provided.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import types

import projection_p046_confirmation as safe
import projection_p051_orientation as orientation

ROOT = Path(__file__).resolve().parent
P046_PATH = "candidates/projection_pooler_column_batch2_hybrid.py"
P046_SHA = "001ea12e8764564d507de51e433769f5bd9a9916bfc4c014f0f87522c278545e"
P051_SHA = "445ab7d44e36cdceafdf1ecdbe19d7fafa247d8033ab580ab38f74b98f3533db"
SAFE_SHA = "6c57daf6ab5aca4a73aa1a608b6ae978af284a02d41c489b4e7b62b307505029"
FROZEN = {
    P046_PATH: P046_SHA,
    "projection_p051_orientation.py": P051_SHA,
    "projection_p046_confirmation.py": SAFE_SHA,
    "harness.py": "d46ea816a0091cea8b6e429a6e8facfbe0a82c20488a8c9b89023c1561777fd8",
    "benchmark.py": "5d7f2db870a1e82142be4b5f91e319c944673b31bdb66bacf02159d2032dde78",
    **orientation.FROZEN,
}
CASES = (8, 7)
RUNNERS = ("control", "weight_first", "p046")
COMPARISONS = {"matched": "control", "selected": "p046"}


def verify_sources(root=ROOT):
    hashes = {}
    for name, expected in FROZEN.items():
        digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError("Frozen source mismatch: " + name)
        hashes[name] = digest
    for name in ("projection_p052_orientation_screen.py", "official/muse_bf16_proj.json"):
        hashes[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    return hashes


def validate_correctness_receipt(path):
    """Require every frozen P051 case/check and verify saved PTX bytes, fail shut."""
    data = path.read_bytes()
    receipt = json.loads(data)
    if receipt.get("complete") is not True or receipt.get("compile_only") is not False or receipt.get("error"):
        raise ValueError("A complete execution-mode P051 correctness receipt is required")
    expected_hashes = {**orientation.FROZEN, "projection_p051_orientation.py": P051_SHA}
    if any(receipt.get("sources_sha256", {}).get(name) != digest for name, digest in expected_hashes.items()):
        raise ValueError("P051 correctness source identity mismatch")
    cases = receipt.get("cases", [])
    expected_labels = [(label, runner) for label in
        ("normal", "outliers", "sparse", "zero", "graph_A1", "graph_B", "graph_A2")
        for runner in ("control", "weight_first")]
    if len(cases) != len(orientation.CASES):
        raise ValueError("All five P051 correctness cases are required")
    for item, (name, m, n, k, _) in zip(cases, orientation.CASES):
        if (item.get("name"), item.get("m"), item.get("n"), item.get("k")) != (name, m, n, k):
            raise ValueError("Unexpected P051 correctness case")
        if not (item.get("complete") is True and item.get("generated_kernel_executed") is True
                and item.get("correctness_status") == "passed"
                and item.get("same_address_sequential_graph_ABA") is True
                and item.get("partial_buffers_isolated") is True):
            raise ValueError("Incomplete P051 execution/correctness coverage")
        records = item.get("correctness", [])
        if [(r.get("label"), r.get("orientation")) for r in records] != expected_labels or not all(
                r.get("passed") is True for r in records):
            raise ValueError("All seventy original-oracle P051 checks must pass")
        compiled = item.get("compiled", {})
        if set(compiled) != {"control", "weight_first", "reduce"}:
            raise ValueError("Missing P051 compiled evidence")
        for runner, kernel in compiled.items():
            filename = path.stem + "." + name + "." + runner + ".ptx"
            if kernel.get("ptx_file") != filename:
                raise ValueError("Unexpected P051 PTX artifact name")
            if hashlib.sha256(path.with_name(filename).read_bytes()).hexdigest() != kernel.get("ptx_sha256"):
                raise ValueError("P051 PTX artifact hash mismatch")
    if not receipt.get("native_imports_completed") or receipt.get("environment", {}).get("capability") != [9, 0]:
        raise ValueError("P051 native H200 evidence missing")
    return dict(sha256=hashlib.sha256(data).hexdigest(), filename=path.name,
                cases=5, original_oracle_checks=70, sequential_graph_ABA_cases=5,
                source_sha256=P051_SHA, environment=receipt["environment"],
                qualification="Receipt is a separate executed correctness stage, not performance evidence")


def comparison_order(round_index):
    return ("matched", "selected") if round_index % 2 == 0 else ("selected", "matched")


def summarize(samples, comparison):
    if comparison not in COMPARISONS:
        raise ValueError("Unknown comparison")
    for i, sample in enumerate(samples):
        if (sample.get("comparison") != comparison or
                sample.get("comparison_sequence_index") != comparison_order(i).index(comparison)
                or sample.get("excluded_warmup") is not False
                or sample.get("correctness") != {"parent": True, "candidate": True}):
            raise ValueError("Incomplete or reordered measured comparison")
    result = safe.summarize_samples(samples)
    pairs = result["reverse_order_pair_ratios"]
    result["winning_pairs"] = sum(ratio > 1 for ratio in pairs)
    result["total_pairs"] = len(pairs)
    return result


def assess(cases):
    if [case.get("m") for case in cases] != list(CASES) or not all(case.get("passed") is True for case in cases):
        raise ValueError("Both complete M8 and M7 cases are required")
    details = []
    for case in cases:
        matched = case["series"]["matched"]["summary"]
        selected = case["series"]["selected"]["summary"]
        cut = 1 - 1 / matched["median_paired_raw_ratio"]
        regression = 1 / selected["median_paired_raw_ratio"] - 1
        gates = dict(matched_raw_latency_cut_at_least10pct=cut >= .10 - 1e-12,
                     matched_at_least10_of12_pairs_win=matched["total_pairs"] == 12 and matched["winning_pairs"] >= 10,
                     selected_regression_at_most5pct=regression <= .05 + 1e-12)
        details.append(dict(m=case["m"], matched_latency_cut=cut,
                            selected_latency_regression=regression, gates=gates))
    return dict(cases=details, screen_passed=all(all(case["gates"].values()) for case in details),
                automatic_selection=False, official_score=None,
                qualification="Bounded two-case mechanism screen; twelve order-balanced pairs, no formal confidence interval. Passing does not promote P046 or establish a whole-score gain.")


def build_kernels(triton, tl):
    """Bind frozen own function bodies without changing P051's module globals."""
    globals_copy = dict(vars(orientation))
    globals_copy["tl"] = tl
    kernels = {}
    for name, function_name in (("control", "_partial"), ("weight_first", "_partial_weight_first"), ("reduce", "_reduce")):
        original = getattr(orientation, function_name)
        function = types.FunctionType(original.__code__, globals_copy, original.__name__,
                                      original.__defaults__, original.__closure__)
        function.__annotations__ = dict(original.__annotations__)
        function.__kwdefaults__ = original.__kwdefaults__
        function.__qualname__ = original.__qualname__
        kernels[name] = triton.jit(function)
    return kernels


def orientation_call(torch, kernels, runner, data, weights):
    if runner not in ("control", "weight_first"):
        raise ValueError("Unknown orientation")
    x = data["x"]
    m, k = x.shape
    n = data["config"]["n"]
    orientation.validate_shape(m, n, k)
    partial = torch.empty((8, m, n), dtype=torch.float32, device=x.device)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    kernels[runner][((n + 63) // 64, 8)](x, weights["pooler"], partial, m, n, k, 8, 64, 64,
                                        **orientation.PARTIAL_LAUNCH)
    kernels["reduce"][((m * n + 127) // 128,)](partial, out, m * n, 8, 128,
                                               **orientation.REDUCE_LAUNCH)
    return out


def environment_matches(previous, current):
    keys = ("torch", "triton", "cuda_runtime", "gpu", "capability")
    return all(previous.get(key) == current.get(key) for key in keys)


def execute(args, report, save):
    started = time.monotonic()
    def gate():
        if time.monotonic() - started >= args.deadline_seconds:
            raise TimeoutError("P052 internal deadline reached; external supervisor must stop blocking calls")

    import torch
    import triton
    import triton.language as tl
    from benchmark import capture
    from harness import environment, load_module
    safe.require_h200(torch)
    report["environment"] = environment(torch, "cuda", ROOT / P046_PATH)
    report["environment"]["triton"] = triton.__version__
    report["environment"]["capability"] = list(report["environment"]["capability"])
    if not environment_matches(report["correctness_receipt"]["environment"], report["environment"]):
        raise RuntimeError("Correctness and timing environments differ")
    report["environment_match"] = "Same named hardware/software; same physical container/device is enforced by launcher, not established by a UUID in this report"
    reference = load_module(ROOT / "official/reference.py", "p052_reference")
    parent = load_module(ROOT / P046_PATH, "p052_p046")
    kernels = build_kernels(triton, tl)
    with torch.inference_mode():
        gate()
        weights = reference.generate_weights(reference.WEIGHTS_SEED, device="cuda")
        frozen_original = safe.freeze_tensors(torch, weights)
        oracle_weights = {name: snapshot[1] for name, snapshot in frozen_original.items()}
        torch.cuda.synchronize()
        tick = time.monotonic()
        state = parent.prepare(weights)
        torch.cuda.synchronize()
        report["preparation"] = dict(p046_synchronized_wall_seconds=time.monotonic() - tick,
                                     actual_p046_prepare_calls=1,
                                     weight_keys=list(weights),
                                     weight_generation="Unmodified full official generate_weights; original and all prepared tensors frozen. Router is required by unchanged P046.prepare.")
        safe.assert_frozen(torch, weights, frozen_original)
        frozen_state = safe.freeze_tensors(torch, state)
        def check_states():
            gate()
            safe.assert_frozen(torch, weights, frozen_original)
            safe.assert_frozen(torch, state, frozen_state)
        eviction = torch.ones(safe.EVICTION_BYTES // 4, dtype=torch.float32, device="cuda")
        eviction_out = torch.empty((), dtype=torch.float32, device="cuda")
        timer = safe.ColdGraphTimer(torch)
        report["event_initialization"] = "Event pair created, recorded, synchronized and elapsed-time queried before any timing; reused only after each end completes"
        for case_index, m in enumerate(CASES):
            gate()
            seed = 5200000 + case_index * 2000
            make = lambda s, dist="normal", rows=m: safe.fresh_input(torch, reference, "pooler", rows, s, dist)
            data = make(seed)
            config, pointer = dict(data["config"]), safe.tensor_metadata(data["x"])
            initial = {"x": data["x"].clone()}
            expected_packed = m == 8
            if bool(parent._pooler_column_batch2_compatible(data["x"], state, 4096)) != expected_packed:
                raise RuntimeError("P046 original M8 packed / M7 fallback route not established")
            item = dict(m=m, layer="pooler", passed=False,
                        p046_route="column_batch2" if expected_packed else "pretransposed_mm",
                        capture={}, series={name: dict(correctness=[], warmup=[], samples=[])
                                            for name in COMPARISONS})
            report["cases"].append(item)
            graphs, outputs = {}, {}
            for runner in RUNNERS:
                gate()
                fn = parent.kernel_fn if runner == "p046" else (
                    lambda d, s, arm=runner: orientation_call(torch, kernels, arm, d, s))
                runner_state = state if runner == "p046" else weights
                torch.cuda.synchronize()
                tick = time.monotonic()
                graphs[runner], outputs[runner] = capture(fn, data, runner_state)
                torch.cuda.synchronize()
                item["capture"][runner] = dict(synchronized_wall_seconds=time.monotonic() - tick)
                safe.assert_input(torch, data, initial, config, pointer)
                check_states()
            if len({out.data_ptr() for out in outputs.values()}) != 3:
                raise RuntimeError("Captured runner outputs alias")
            def paired_views(comparison):
                opponent = COMPARISONS[comparison]
                return ({"parent": graphs[opponent], "candidate": graphs["weight_first"]},
                        {"parent": outputs[opponent], "candidate": outputs["weight_first"]})
            for comparison in COMPARISONS:
                pair_graphs, pair_outputs = paired_views(comparison)
                safe.correctness_phase(torch, reference, data, pair_graphs, pair_outputs, oracle_weights,
                    config, pointer, make, seed + 10, "before", item["series"][comparison]["correctness"], gate, save)
                check_states()
            for warmup, count, offset in ((True, 6, 100), (False, 24, 200)):
                for round_index in range(count):
                    fresh = make(seed + offset + round_index)
                    for sequence_index, comparison in enumerate(comparison_order(round_index)):
                        gate()
                        data["x"].copy_(fresh["x"])
                        pair_graphs, pair_outputs = paired_views(comparison)
                        sample = dict(round=round_index, seed=seed + offset + round_index,
                                      comparison=comparison, comparison_sequence_index=sequence_index,
                                      order=list(safe.runner_order(round_index)), us={}, observations=[],
                                      correctness={}, excluded_warmup=warmup)
                        item["series"][comparison]["warmup" if warmup else "samples"].append(sample)
                        save()
                        for role in safe.runner_order(round_index):
                            gate()
                            observation = dict(role=role, started_utc=datetime.now(timezone.utc).isoformat(),
                                               elapsed_start_seconds=time.monotonic() - started, status="started")
                            sample["observations"].append(observation)
                            sample["us"][role] = timer.measure(pair_graphs[role], eviction, eviction_out)
                            observation.update(status="measured", elapsed_end_seconds=time.monotonic() - started)
                            safe.assert_input(torch, data, fresh, config, pointer)
                            save()
                        # Every replay's own output is checked before the next
                        # comparison can overwrite the shared candidate output.
                        for role in safe.runner_order(round_index):
                            gate()
                            safe.check_output(torch, reference, data, pair_outputs[role], fresh, config, pointer, oracle_weights)
                            sample["correctness"][role] = True
                            save()
                    check_states()
            for comparison in COMPARISONS:
                pair_graphs, pair_outputs = paired_views(comparison)
                series = item["series"][comparison]
                safe.correctness_phase(torch, reference, data, pair_graphs, pair_outputs, oracle_weights,
                    config, pointer, make, seed + 400, "after", series["correctness"], gate, save)
                check_states()
                series["summary"] = summarize(series["samples"], comparison)
            item.update(passed=True, shared_input_address=True,
                        immutable_original_and_prepared_state=True,
                        state_check_scope="After prepare, every capture, every comparison correctness phase and every timing round; input/config after every observed replay")
            save()
        report["assessment"] = assess(report["cases"])
        report["oracle_checks"] = sum(
            sum(record["passed"] is True for record in series["correctness"]) +
            sum(sum(value is True for value in sample["correctness"].values())
                for sample in series["warmup"] + series["samples"])
            for case in report["cases"] for series in case["series"].values())
        if report["oracle_checks"] != 320:
            raise RuntimeError("Expected exactly 320 successful screen replay checks")
        report["complete"] = True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correctness-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-seconds", type=float, default=120)
    parser.add_argument("--outer-timeout-seconds", type=float, required=True)
    args = parser.parse_args(argv)
    safe.validate_options(args)
    report = dict(kind="P052 matched orientation raw execution screen", complete=False, cases=[],
        protocol=dict(cases=list(CASES), runners=list(RUNNERS), comparisons=COMPARISONS,
                      warmup_rounds_per_comparison=6, measured_rounds_per_comparison=24,
                      reversed_pairs_per_comparison=12, comparison_order="matched/selected on even rounds; reversed on odd rounds",
                      runner_order="opponent/candidate on even rounds; reversed on odd rounds",
                      eviction_bytes=safe.EVICTION_BYTES, calibration="unmeasured; no subtraction",
                      expected_checked_screen_replays=320, expected_prior_P051_checks=70,
                      capture_calls=6, untimed_capture_warmups=18,
                      internal_deadline_seconds=args.deadline_seconds,
                      required_external_timeout_seconds=args.outer_timeout_seconds),
        limitations=["Two pooler shapes only; no official score or promotion",
                     "Every observed replay checked, but CUDA capture's untimed setup calls are not individually checked",
                     "Sequential independent graphs; concurrent execution not tested",
                     "No profiler, clock telemetry, hardware counters or achieved overlap claim",
                     "External supervisor required for blocking CUDA/compiler calls"])
    started = time.monotonic()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        report["wall_seconds"] = time.monotonic() - started
        temp = args.output.with_suffix(args.output.suffix + ".tmp")
        temp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        temp.replace(args.output)
    try:
        report["sources_sha256"] = verify_sources()
        report["correctness_receipt"] = validate_correctness_receipt(args.correctness_receipt)
        save()
        execute(args, report, save)
        code = 0
    except Exception as exc:
        report["failure"] = dict(type=type(exc).__name__, message=str(exc)[:700])
        code = 1
    finally:
        save()
    print(json.dumps(dict(complete=report["complete"], cases=len(report["cases"]),
                          screen_passed=report.get("assessment", {}).get("screen_passed"), automatic_selection=False)))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
