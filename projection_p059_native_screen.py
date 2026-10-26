"""P059 fixed-layout FP32 split-K cuBLASLt diagnostic; never an official score/submission.

No Torch/native import or build happens on import. The external supervisor MUST
enforce the stated hard timeout: a monotonic gate cannot interrupt NVCC/CUDA.
Only build_native_probe's documented load_inline route may start a compiler.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time

import projection_p046_confirmation as safe

ROOT = Path(__file__).resolve().parent
BRIDGE_SHA = "a91663d9b64291657d7b64e7bba7d7c4ffaeb2777964f8357dd2225adea9a300"
FROZEN_P057_SHA = "e22e220b69330f1b614fffd69596a945d2a8b91bee8fcd6f70f426f9b645b4af"
SPLIT_SHA = "bb2a54cd7cc14bde5ec3a170a6bf8172032abbcade890ee3d6a0262001e614af"
FROZEN_WORKER_SHA = "17baa53ba3c3c54201f570cb2a08d9415f8cd0ef4db625625ac7399962843a2f"
NATIVE_SHA = "a40916423e9eb7da4b93258e7e986b55777a56ca93268e9c2e3105f437b409cc"
PARENT = "candidates/projection_pooler_column_batch2_hybrid.py"
PARENT_SHA = "001ea12e8764564d507de51e433769f5bd9a9916bfc4c014f0f87522c278545e"
SAFE_SHA = "6c57daf6ab5aca4a73aa1a608b6ae978af284a02d41c489b4e7b62b307505029"
CASES = (("qkv_gate", 128),)
NEARBY = (("qkv_gate", 127),)
WORKSPACE_LIMIT = 64 * 1024 * 1024
MAX_SELECTED = 3
CONFIG_KEYS = ("algorithm_id", "tile_id", "stages_id", "split_k", "reduction_scheme",
               "cta_swizzle", "custom_option")


def verify_sources(root=ROOT):
    expected = dict(safe.ORACLES)
    expected.update({"native_cublaslt_probe.py": NATIVE_SHA,
                     "native_cublaslt_host_probe.py": BRIDGE_SHA,
                     "projection_p056_native_screen.py": FROZEN_WORKER_SHA,
                     "projection_p057_native_screen.py": FROZEN_P057_SHA,
                     "native_cublaslt_fp32_split_probe.py": SPLIT_SHA, PARENT: PARENT_SHA,
                     "projection_p046_confirmation.py": SAFE_SHA})
    result = {}
    for name, expected_sha in expected.items():
        result[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if result[name] != expected_sha:
            raise ValueError("Frozen source mismatch: " + name)
    for name in ("projection_p059_native_screen.py", "benchmark.py", "harness.py",
                 "official/muse_bf16_proj.json"):
        result[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    return result


def validate_options(args):
    if args.compile_native is not True:
        raise ValueError("--compile-native is required for this supervised native action")
    if not 0 < args.deadline_seconds <= 240:
        raise ValueError("Internal deadline must be positive and at most240 seconds")
    if not args.deadline_seconds + 15 <= args.outer_timeout_seconds <= 270:
        raise ValueError("External timeout must add at least15 seconds and be at most270")


RUNNERS = ("p046", "row41", "row66", "split")
SERIES = (("anchor_row41", "p046", "row41"),
          ("anchor_row66", "p046", "row66"),
          ("anchor_split", "p046", "split"),
          ("row66_vs_split", "row66", "split"))
CONTROL_KEYS = ("algorithm_id", "tile_id", "stages_id", "split_k",
                "reduction_scheme", "cta_swizzle", "custom_option")


class CapabilityBlocked(RuntimeError):
    """Required controls or an actual FP32 split are absent; no substitute."""


def choose_controls(inventory, compiled_none):
    """Match recorded configurations exactly, without executing or timing them."""
    if type(compiled_none) is not int or compiled_none < 0:
        raise ValueError("Actual compiled NONE enum is required")
    wanted = ((41, 21, 35, 1, compiled_none, 0, 0),
              (66, 20, 35, 1, compiled_none, 0, 0))
    plans = sorted(inventory["plans"], key=lambda plan: plan.metadata()["heuristic_position"])
    found = {}
    for plan in plans:
        meta = plan.metadata()
        if (meta.get("compute") != "CUBLAS_COMPUTE_32F"
                or meta.get("scale") != "CUDA_R_32F" or meta.get("input_output") != "CUDA_R_16BF"
                or type(meta.get("split_k")) is not int or meta["split_k"] > 1
                or meta.get("reduction_scheme") != compiled_none):
            raise ValueError("Frozen no-split inventory precision/policy changed")
        if (meta.get("m"), meta.get("n"), meta.get("k")) != (128, 2304, 8192):
            raise ValueError("Frozen control shape changed")
        if not 0 <= meta.get("workspace_required_bytes", -1) <= WORKSPACE_LIMIT:
            raise ValueError("Frozen control workspace is invalid")
        key = tuple(meta.get(name) for name in CONTROL_KEYS)
        if key in wanted:
            found.setdefault(key, plan)
    missing = [list(key) for key in wanted if key not in found]
    if missing:
        raise CapabilityBlocked("Exact historical no-split controls absent: " + json.dumps(missing))
    return [found[key] for key in wanted]


def inventory_record(inventory):
    result = dict(requested=inventory["requested"], returned=inventory["returned"],
                  rejected=inventory["rejected"], eligible=[plan.metadata() for plan in inventory["plans"]])
    if "reduction_scheme_constants" in inventory:
        result["reduction_scheme_constants"] = inventory["reduction_scheme_constants"]
    return result


def activity_summary(prof, cuda_type):
    """Bounded allowlisted symbols/durations only; no raw trace or host metadata."""
    events = []
    for event in prof.events():
        if event.device_type != cuda_type:
            continue
        duration = float(event.device_time_total)
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("Invalid activity duration")
        if len(events) == 256:
            raise ValueError("Unexpectedly large single-replay activity; no silent truncation")
        symbol = str(event.name)
        if len(symbol) > 4096:
            raise ValueError("Unexpectedly large activity symbol")
        events.append(dict(symbol=symbol, duration_us=duration,
                           kind="copy" if "copy" in symbol.lower() or "memcpy" in symbol.lower() else "kernel_or_other"))
    return dict(available=bool(events), cuda_events=events,
                copy_events=[row for row in events if row["kind"] == "copy"],
                qualification="One post-timing instrumented replay; names/durations only, not acceptance timing, instructions, counters or achieved occupancy. Empty events mean unavailable.")


def assess(case):
    if not case["passed"]:
        raise ValueError("Correctness must complete before assessment")
    summary = case["series"]["anchor_split"]["summary"]
    ratio = float(summary["median_paired_raw_ratio"])
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("Invalid paired ratio")
    wins = sum(value > 1 for value in summary["reverse_order_pair_ratios"])
    cut = 1 - 1 / ratio
    return dict(screen_passed=(ratio >= 1 / .9 and wins >= 10),
                target_latency_cut=cut, winning_pairs=wins,
                minimum_latency_cut=.10, minimum_winning_pairs=10,
                automatic_selection=False, official_score=False,
                qualification="Only justifies broader correctness/shape coverage; never promotes this one-case diagnostic. Same-interface contrast is descriptive, not a claim of isolated algorithm identity.")


def toolchain_metadata(torch):
    """Fixed vendor header reads only; no compiler subprocess or broad traversal."""
    from torch.utils.cpp_extension import CUDA_HOME
    output = dict(torch=torch.__version__, torch_cuda=torch.version.cuda,
                  cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI), headers=[])
    for name in ("cuda.h", "cuda_runtime_api.h", "cublasLt.h", "cublas_api.h"):
        row = dict(name=name, available=False)
        path = Path(CUDA_HOME) / "include" / name if CUDA_HOME else None
        if path is not None and path.is_file():
            row["size_bytes"] = path.stat().st_size
            if row["size_bytes"] <= 2 * 1024 * 1024:
                payload = path.read_bytes()
                row.update(available=True, sha256=hashlib.sha256(payload).hexdigest())
                text = payload.decode("utf-8", errors="replace")
                row["version_defines"] = dict(re.findall(
                    r"^\s*#\s*define\s+(CUDA_VERSION|CUBLAS_VER_MAJOR|CUBLAS_VER_MINOR|CUBLAS_VER_PATCH|CUBLAS_VER_BUILD)\s+(\d+)",
                    text, re.MULTILINE))
            else:
                row["reason"] = "header_read_cap"
        output["headers"].append(row)
    output["qualification"] = "Header hashes and framework ABI metadata; no compiler-version subprocess. Native build success establishes only this image's compile/link compatibility."
    return output


def sanitize_failure(error):
    # Preserve bounded build diagnostics without reporting arbitrary absolute paths.
    message = str(error)
    message = re.sub(r"(?:/[^\s:'\"]+)+", "<path>", message)
    return dict(type=type(error).__name__, message=message[:2500])


def measured_us(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Invalid raw graph time; no sample replacement is permitted")
    return value


def count_records(report):
    total = 0
    for case in report["cases"]:
        total += sum(row.get("passed") is True for row in case["correctness"])
        total += sum(row.get("passed") is True for row in case["independent_graphs"]["checks"])
        for series in case["series"].values():
            for row in series["warmup"] + series["samples"]:
                total += sum(value is True for value in row["correctness"].values())
        total += sum(row.get("correctness_passed") is True for row in case.get("activities", {}).values())
    for case in report["nearby"]:
        total += sum(row.get("passed") is True for row in case.get("correctness", []))
    return total


def execute(args, report, save):
    started = time.monotonic()

    def gate():
        if time.monotonic() - started >= args.deadline_seconds:
            raise TimeoutError("Internal compile+screen deadline; supervisor interrupts blocking calls")

    import torch
    from benchmark import capture
    from harness import environment, load_module
    safe.require_h200(torch)
    report["environment"] = environment(torch, "cuda", ROOT / PARENT)
    report["toolchain"] = toolchain_metadata(torch)
    no_split_source = load_module(ROOT / "native_cublaslt_host_probe.py", "p059_no_split_source")
    split_source = load_module(ROOT / "native_cublaslt_fp32_split_probe.py", "p059_split_source")
    report["native_source_string_sha256"] = dict(no_split=no_split_source.source_sha256(),
                                                fp32_split=split_source.source_sha256())
    report["phase"] = "native_build"
    report["compile"] = dict(started_utc=datetime.now(timezone.utc).isoformat(), complete=False,
                             elapsed_start_seconds=time.monotonic() - started, builders={})
    save()
    tick = time.monotonic()
    natives = {}
    for name, source in (("no_split", no_split_source), ("fp32_split", split_source)):
        gate()
        row = dict(complete=False)
        report["compile"]["builders"][name] = row
        save()
        build_tick = time.monotonic()
        natives[name] = source.build_native_probe(compile_requested=True)
        row.update(complete=True, wall_seconds=time.monotonic() - build_tick)
        save()
    report["compile"].update(complete=True, wall_seconds=time.monotonic() - tick)
    report["native_build_complete"] = True
    report["phase"] = "interface_and_inventory"
    report["inventories"] = {}
    inventories = {}
    for name, native in natives.items():
        gate()
        inventories[name] = native.enumerate_plans(128, 2304, 8192, torch.cuda.current_device(), WORKSPACE_LIMIT, 32)
        report["inventories"][name] = inventory_record(inventories[name])
        save()
    try:
        split_plan = split_source.select_actual_fp32_split(inventories["fp32_split"])
    except split_source.NoEligibleFP32Split as error:
        raise CapabilityBlocked(str(error)) from error
    controls = choose_controls(inventories["no_split"],
        inventories["fp32_split"]["reduction_scheme_constants"][split_source.NONE])
    split_meta = split_plan.metadata()
    if ((split_meta["m"], split_meta["n"], split_meta["k"]) != (128, 2304, 8192)
            or not 0 <= split_meta["workspace_required_bytes"] <= WORKSPACE_LIMIT):
        raise ValueError("Split plan shape/workspace changed")
    plans = controls + [split_plan]
    report["selected_configurations"] = len(plans)
    save()
    gate()
    reference = load_module(ROOT / "official/reference.py", "p059_reference")
    parent = load_module(ROOT / PARENT, "p059_parent")
    retained = []  # Keep contexts/graphs/private buffers alive through every replay.
    with torch.inference_mode():
        weights = reference.generate_weights(reference.WEIGHTS_SEED, device="cuda")
        frozen = safe.freeze_tensors(torch, weights)
        oracle_weights = {name: snap[1] for name, snap in frozen.items()}
        state = parent.prepare(weights)
        frozen_state = safe.freeze_tensors(torch, state)

        def immutable():
            gate()
            safe.assert_frozen(torch, weights, frozen)
            safe.assert_frozen(torch, state, frozen_state)

        def check(data, output, fresh, config, pointer, record):
            gate()
            safe.check_output(torch, reference, data, output, fresh, config, pointer, oracle_weights)
            record["passed"] = True
            report["oracle_checks"] += 1
            save()

        def make(layer, m, seed, distribution="normal"):
            if distribution != "cancel":
                return safe.fresh_input(torch, reference, layer, m, seed, distribution)
            data = safe.fresh_input(torch, reference, layer, m, seed, "zero")
            # Two exactly opposite products for output column0, without changing W.
            data["x"][:, 0] = weights[layer][0, 1] * 32
            data["x"][:, 1] = -weights[layer][0, 0] * 32
            return data

        def native_graph(plan, data):
            meta = plan.metadata()
            out = torch.empty((meta["m"], meta["n"]), dtype=torch.bfloat16, device="cuda")
            workspace = torch.empty((max(1, meta["workspace_required_bytes"]),), dtype=torch.uint8, device="cuda")
            def fn(current_data, unused):
                plan.run(current_data["x"], weights[current_data["config"]["layer"]], out, workspace)
                return out
            graph, actual_out = capture(fn, data, None)
            if actual_out is not out:
                raise RuntimeError("Native capture output changed identity")
            return dict(plan=plan, graph=graph, output=out, workspace=workspace)

        # Fixed one-case, metadata-only selection is complete before any native execution.
        for index, (layer, m) in enumerate(CASES):
            gate()
            n, k = reference.LAYERS[layer]
            seed = 5900000 + index * 10000
            data = make(layer, m, seed)
            config, pointer = dict(data["config"]), safe.tensor_metadata(data["x"])
            item = dict(layer=layer, m=m, n=n, k=k, passed=False,
                        selected=[plan.metadata() for plan in plans],
                        runner_names=list(RUNNERS), activities={},
                        correctness=[], series={}, independent_graphs=dict(checks=[], complete=False),
                        same_address_aba=False, native_private_workspaces=True)
            report["cases"].append(item)
            save()
            initial = {"x": data["x"].clone()}
            graph, output = capture(parent.kernel_fn, data, state)
            runners = {"p046": dict(graph=graph, output=output)}
            for name, plan in zip(RUNNERS[1:], plans):
                gate()
                runners[name] = native_graph(plan, data)
                report["native_execution"] = True
            if len({entry["output"].data_ptr() for entry in runners.values()}) != len(runners):
                raise RuntimeError("Captured output alias")
            safe.assert_input(torch, data, initial, config, pointer)
            immutable()
            retained.append(dict(item=item, data=data, config=config, pointer=pointer,
                                 runners=runners, plans=plans, seed=seed))
            save()

        def correctness(bundle, phase):
            item, data, config, pointer = (bundle[key] for key in ("item", "data", "config", "pointer"))
            layer, m, seed = item["layer"], item["m"], bundle["seed"] + (500 if phase == "after" else 0)
            a = make(layer, m, seed + 1)
            fixtures = (("A", a), ("B", make(layer, m, seed + 2, "outliers")),
                        ("A-return", a), ("column_probes", make(layer, m, seed + 3, "sparse")),
                        ("zero", make(layer, m, seed + 4, "zero")),
                        ("cancel", make(layer, m, seed + 5, "cancel")))
            first = {}
            for label, fresh in fixtures:
                gate()
                data["x"].copy_(fresh["x"])
                for name, runner in bundle["runners"].items():
                    gate()
                    runner["output"].fill_(float("nan"))
                    row = dict(phase=phase, label=label, runner=name, passed=False,
                               output_poison="NaN", same_input_address=True)
                    item["correctness"].append(row)
                    save()
                    runner["graph"].replay()
                    torch.cuda.synchronize()
                    check(data, runner["output"], fresh, config, pointer, row)
                    if label == "A":
                        first[name] = runner["output"].clone()
                    elif label == "A-return" and not torch.equal(
                            first[name].view(torch.uint8), runner["output"].view(torch.uint8)):
                        row["passed"] = False
                        raise RuntimeError("Native/P046 A-return bytes differ")
            item["same_address_aba"] = True
            immutable()
            save()

        # All main-case adversarial correctness gates precede all timing.
        report["phase"] = "correctness"
        for bundle in retained:
            correctness(bundle, "before")
            item = bundle["item"]
            layer, m = item["layer"], item["m"]
            first = bundle["runners"]["split"]
            other_data = make(layer, m, bundle["seed"] + 700, "outliers")
            other = native_graph(bundle["plans"][2], other_data)
            bundle["independent"] = (other_data, other)
            other_config, other_pointer = dict(other_data["config"]), safe.tensor_metadata(other_data["x"])
            pointers = [first["output"].data_ptr(), other["output"].data_ptr(),
                        first["workspace"].data_ptr(), other["workspace"].data_ptr()]
            if len(set(pointers)) != 4 or other_data["x"].data_ptr() == bundle["data"]["x"].data_ptr():
                raise RuntimeError("Independent graph inputs/mutable buffers alias")
            streams = (torch.cuda.Stream(), torch.cuda.Stream())
            for stream in streams:
                stream.wait_stream(torch.cuda.current_stream())
            for repeat in range(3):
                gate()
                fresh0 = make(layer, m, bundle["seed"] + 710 + repeat)
                fresh1 = make(layer, m, bundle["seed"] + 720 + repeat, "outliers")
                bundle["data"]["x"].copy_(fresh0["x"])
                other_data["x"].copy_(fresh1["x"])
                first["output"].fill_(float("nan"))
                other["output"].fill_(float("nan"))
                for stream in streams:
                    stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(streams[0]):
                    first["graph"].replay()
                with torch.cuda.stream(streams[1]):
                    other["graph"].replay()
                torch.cuda.current_stream().wait_stream(streams[0])
                torch.cuda.current_stream().wait_stream(streams[1])
                torch.cuda.synchronize()
                for name, data, runner, fresh, config, pointer in (
                        ("original", bundle["data"], first, fresh0, bundle["config"], bundle["pointer"]),
                        ("independent", other_data, other, fresh1, other_config, other_pointer)):
                    row = dict(repeat=repeat, runner=name, passed=False)
                    item["independent_graphs"]["checks"].append(row)
                    check(data, runner["output"], fresh, config, pointer, row)
            item["independent_graphs"].update(complete=True, separate_streams=True,
                distinct_mutable_buffers=True, kernel_overlap_verified=False,
                qualification="Distinct-stream graph submissions with shared immutable plan/W and private input/output/workspace; hardware overlap not instrumented.")
            immutable()
            save()

        report["phase"] = "paired_timing"
        eviction = torch.ones(safe.EVICTION_BYTES // 4, dtype=torch.float32, device="cuda")
        eviction_out = torch.empty((), dtype=torch.float32, device="cuda")
        timer = safe.ColdGraphTimer(torch)
        for bundle in retained:
            item, data = bundle["item"], bundle["data"]
            mappings = SERIES
            for si, (name, parent_name, candidate_name) in enumerate(mappings):
                gate()
                series = dict(runner_mapping=dict(parent=parent_name, candidate=candidate_name),
                              warmup=[], samples=[], passed=False)
                item["series"][name] = series
                selected = {"parent": bundle["runners"][parent_name], "candidate": bundle["runners"][candidate_name]}
                for warmup, count, offset in ((True, 6, 1000), (False, 24, 1100)):
                    for i in range(count):
                        gate()
                        seed = bundle["seed"] + si * 200 + offset + i
                        fresh = make(item["layer"], item["m"], seed)
                        data["x"].copy_(fresh["x"])
                        row = dict(round=i, seed=seed, order=list(safe.runner_order(i)),
                                   us={}, correctness={}, excluded_warmup=warmup)
                        series["warmup" if warmup else "samples"].append(row)
                        save()
                        for runner_name in safe.runner_order(i):
                            gate()
                            selected[runner_name]["output"].fill_(float("nan"))
                            row["us"][runner_name] = measured_us(timer.measure(selected[runner_name]["graph"], eviction, eviction_out))
                            safe.assert_input(torch, data, fresh, bundle["config"], bundle["pointer"])
                            save()
                        for runner_name in safe.runner_order(i):
                            passed = {}
                            check(data, selected[runner_name]["output"], fresh,
                                  bundle["config"], bundle["pointer"], passed)
                            row["correctness"][runner_name] = passed["passed"]
                            save()
                series["summary"] = safe.summarize_samples(series["samples"])
                pairs = series["summary"]["reverse_order_pair_ratios"]
                series["summary"]["winning_pairs"] = sum(ratio > 1 for ratio in pairs)
                series["passed"] = True
                immutable()
                save()
            correctness(bundle, "after")
            item["passed"] = True
            save()

        # Profiling starts only after every acceptance sample and post-check.
        # No trace files, stacks, shapes, memory profile, or hardware counters.
        report["phase"] = "activity"
        for bundle in retained:
            item, data = bundle["item"], bundle["data"]
            for ni, name in enumerate(RUNNERS):
                gate()
                fresh = make(item["layer"], item["m"], bundle["seed"] + 3000 + ni)
                data["x"].copy_(fresh["x"])
                runner = bundle["runners"][name]
                runner["output"].fill_(float("nan"))
                torch.sum(eviction, dim=0, out=eviction_out)
                torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA],
                                            record_shapes=False, with_stack=False, profile_memory=False) as prof:
                    runner["graph"].replay()
                    torch.cuda.synchronize()
                row = activity_summary(prof, torch.autograd.DeviceType.CUDA)
                row["correctness_passed"] = False
                item["activities"][name] = row
                save()
                passed = {}
                check(data, runner["output"], fresh, bundle["config"], bundle["pointer"], passed)
                row["correctness_passed"] = passed["passed"]
                immutable()
                save()
        for layer, m in NEARBY:
            report["nearby"].append(dict(layer=layer, m=m, executed=False,
                reason="One canonical case only; nearby/full official coverage requires a separate validation",
                correctness=[]))
        report["phase"] = "cleanup"
        torch.cuda.synchronize()
        immutable()
        report["oracle_checks"] = count_records(report)
        report["complete"] = True
        report["passed"] = all(item["passed"] for item in report["cases"])
        report["status"] = "completed"
        report["assessment"] = assess(report["cases"][0])
        report["wall_seconds"] = time.monotonic() - started
        save()
        # All retained contexts/graphs die outside capture and timing, after sync.


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-native", action="store_true")
    parser.add_argument("--deadline-seconds", type=float, default=240)
    parser.add_argument("--outer-timeout-seconds", type=float, required=True)
    args = parser.parse_args(argv)
    validate_options(args)
    report = dict(kind="P059 fixed-layout FP32 split-K capability and one-case screen", status="incomplete", complete=False,
                  passed=False, phase="source_preflight", native_build_complete=False,
                  native_execution=False, oracle_checks=0, selected_configurations=0, cases=[], nearby=[],
                  protocol=dict(cases=[dict(layer=l, m=m) for l, m in CASES], maximum_configurations=3,
                    selection="Exact historical row41 and row66 NONE controls; first actual split>1 COMPUTE_TYPE plan by heuristic position; no timing-based selection or fallback",
                    warmup_rounds=6, measured_rounds=24, reverse_order_pairs=12,
                    capture_eager_warmups_per_graph=3,
                    oracle_count_scope="Recorded oracle comparisons only; excludes three untimed eager warmups per successful captured graph (15 for five graphs on a completed screen)",
                    eviction_bytes=safe.EVICTION_BYTES, workspace_limit_bytes=WORKSPACE_LIMIT,
                    calibration="none; gross graph times only", raw_samples_retained=True,
                    original_oracle="FP64 and original column-probe tolerances, unmodified",
                    internal_deadline_seconds=args.deadline_seconds,
                    external_supervisor_timeout_seconds=args.outer_timeout_seconds),
                  limitations=["Not full official correctness or score; nearby shapes explicitly reported",
                    "Post-timing activity names/durations only; no hardware counters or native assembly inspection",
                    "Separate-stream submissions do not establish actual kernel overlap",
                    "One fixed layout and one metadata-selected actual FP32 split; other tactics, shapes and layouts untested",
                    "Internal gate cannot interrupt native compiler/CUDA; external hard timeout required"])

    def save():
        temp = args.output.with_name(args.output.name + ".tmp")
        temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temp.replace(args.output)

    try:
        report["source_hashes"] = verify_sources()
        save()
        execute(args, report, save)
    except CapabilityBlocked as error:
        report.update(complete=True, passed=False, status="capability_blocked",
                      capability_blocked=sanitize_failure(error))
        report["oracle_checks"] = count_records(report)
        save()
        print(json.dumps(dict(complete=True, passed=False, status="capability_blocked",
                              reason=report["capability_blocked"])), flush=True)
        return 0
    except Exception as error:
        report.update(complete=False, passed=False, status="incomplete")
        report["failure"] = sanitize_failure(error)
        report["oracle_checks"] = count_records(report)
        save()
        print(json.dumps(dict(complete=False, phase=report["phase"], failure=report["failure"])), flush=True)
        return 1
    save()
    print(json.dumps(dict(complete=report["complete"], passed=report["passed"],
                         oracle_checks=report["oracle_checks"], automatic_selection=False)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
