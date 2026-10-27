"""P057 host-built native cuBLASLt diagnostic; never an official score/submission.

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
FROZEN_WORKER_SHA = "17baa53ba3c3c54201f570cb2a08d9415f8cd0ef4db625625ac7399962843a2f"
NATIVE_SHA = "a40916423e9eb7da4b93258e7e986b55777a56ca93268e9c2e3105f437b409cc"
PARENT = "candidates/projection_pooler_column_batch2_hybrid.py"
PARENT_SHA = "001ea12e8764564d507de51e433769f5bd9a9916bfc4c014f0f87522c278545e"
SAFE_SHA = "6c57daf6ab5aca4a73aa1a608b6ae978af284a02d41c489b4e7b62b307505029"
CASES = (("qkv_gate", 128), ("unpooler", 16))
NEARBY = (("qkv_gate", 127), ("unpooler", 15))
WORKSPACE_LIMIT = 64 * 1024 * 1024
MAX_SELECTED = 4
CONFIG_KEYS = ("algorithm_id", "tile_id", "stages_id", "split_k", "reduction_scheme",
               "cta_swizzle", "custom_option")


def verify_sources(root=ROOT):
    expected = dict(safe.ORACLES)
    expected.update({"native_cublaslt_probe.py": NATIVE_SHA,
                     "native_cublaslt_host_probe.py": BRIDGE_SHA,
                     "projection_p056_native_screen.py": FROZEN_WORKER_SHA, PARENT: PARENT_SHA,
                     "projection_p046_confirmation.py": SAFE_SHA})
    result = {}
    for name, expected_sha in expected.items():
        result[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if result[name] != expected_sha:
            raise ValueError("Frozen source mismatch: " + name)
    for name in ("projection_p057_native_screen.py", "benchmark.py", "harness.py",
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


def choose_plans(inventory, limit=2):
    """Deterministic metadata selection, before any correctness or timing result."""
    if not 1 <= limit <= 2:
        raise ValueError("At most two plans per case")
    plans = sorted(inventory["plans"], key=lambda plan: plan.metadata()["heuristic_position"])
    unique, seen = [], set()
    for plan in plans:
        meta = plan.metadata()
        if meta["split_k"] > 1 or meta["reduction_scheme"] != 0:
            raise ValueError("Native inventory violated no-split policy")
        key = tuple(meta[field] for field in CONFIG_KEYS)
        if key not in seen:
            unique.append(plan)
            seen.add(key)
    if not unique:
        raise RuntimeError("No eligible native no-split heuristic; no timing is permitted")
    selected = [unique[0]]
    reason = "only_one_distinct_recorded_configuration"
    if limit > 1 and len(unique) > 1:
        first_id = unique[0].metadata()["algorithm_id"]
        other = next((plan for plan in unique[1:] if plan.metadata()["algorithm_id"] != first_id), None)
        reason = "different_algorithm_id" if other is not None else "same_family_distinct_recorded_configuration"
        selected.append(other if other is not None else unique[1])
    return selected, reason


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
    native_source = load_module(ROOT / "native_cublaslt_host_probe.py", "p057_native_source")
    report["native_source_string_sha256"] = native_source.source_sha256()
    report["phase"] = "native_build"
    report["compile"] = dict(started_utc=datetime.now(timezone.utc).isoformat(), complete=False,
                             elapsed_start_seconds=time.monotonic() - started)
    save()
    gate()
    tick = time.monotonic()
    native = native_source.build_native_probe(compile_requested=True)
    report["compile"].update(complete=True, wall_seconds=time.monotonic() - tick)
    report["native_build_complete"] = True
    save()
    gate()
    reference = load_module(ROOT / "official/reference.py", "p057_reference")
    parent = load_module(ROOT / PARENT, "p057_parent")
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

        # Query ALL main cases before any performance measurement. No timing in
        # preparation or in heuristic selection; first selected plan is the control.
        report["phase"] = "interface_and_inventory"
        for index, (layer, m) in enumerate(CASES):
            gate()
            n, k = reference.LAYERS[layer]
            inventory = native.enumerate_plans(m, n, k, torch.cuda.current_device(), WORKSPACE_LIMIT, 32)
            plans, reason = choose_plans(inventory)
            if report["selected_configurations"] + len(plans) > MAX_SELECTED:
                raise RuntimeError("Global configuration allowance exceeded")
            report["selected_configurations"] += len(plans)
            seed = 5600000 + index * 10000
            data = make(layer, m, seed)
            config, pointer = dict(data["config"]), safe.tensor_metadata(data["x"])
            item = dict(layer=layer, m=m, n=n, k=k, passed=False,
                        inventory=dict(requested=inventory["requested"], returned=inventory["returned"],
                            rejected=inventory["rejected"], eligible=[plan.metadata() for plan in inventory["plans"]]),
                        selected=[plan.metadata() for plan in plans], second_selection_reason=reason,
                        correctness=[], series={}, independent_graphs=dict(checks=[], complete=False),
                        same_address_aba=False, native_private_workspaces=True)
            report["cases"].append(item)
            save()
            initial = {"x": data["x"].clone()}
            graph, output = capture(parent.kernel_fn, data, state)
            runners = {"p046": dict(graph=graph, output=output)}
            for pi, plan in enumerate(plans):
                gate()
                runners["native" + str(pi)] = native_graph(plan, data)
                report["native_execution"] = True
            if len({entry["output"].data_ptr() for entry in runners.values()}) != len(runners):
                raise RuntimeError("Captured output alias")
            safe.assert_input(torch, data, initial, config, pointer)
            immutable()
            retained.append(dict(item=item, data=data, config=config, pointer=pointer,
                                 runners=runners, plans=plans, inventory=inventory, seed=seed))
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
            first = bundle["runners"]["native0"]
            other_data = make(layer, m, bundle["seed"] + 700, "outliers")
            other = native_graph(bundle["plans"][0], other_data)
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
            mappings = [("anchor_native0", "p046", "native0")]
            if "native1" in bundle["runners"]:
                mappings += [("anchor_native1", "p046", "native1"),
                             ("native_control", "native0", "native1")]
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

        # Nearby execution would require additional shape-specific plans. The
        # four-configuration ceiling takes priority; absence is explicit.
        for layer, m in NEARBY:
            report["nearby"].append(dict(layer=layer, m=m, executed=False,
                reason="Main-case configuration/deadline budget prioritized; nearby native plans not validated",
                correctness=[]))
        report["phase"] = "cleanup"
        torch.cuda.synchronize()
        immutable()
        report["oracle_checks"] = count_records(report)
        report["complete"] = True
        report["passed"] = all(item["passed"] for item in report["cases"])
        report["assessment"] = dict(automatic_selection=False, official_score=False,
            native_correctness_on_main_cases=report["passed"],
            qualification="Exploratory metadata-selected tactics. Paired gross times retain all samples; selecting the best point estimate requires an independent follow-up, full-shape validation and actual native phase evidence.")
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
    report = dict(kind="P057 host-built cuBLASLt interface and two-case screen", complete=False,
                  passed=False, phase="source_preflight", native_build_complete=False,
                  native_execution=False, oracle_checks=0, selected_configurations=0, cases=[], nearby=[],
                  protocol=dict(cases=[dict(layer=l, m=m) for l, m in CASES], maximum_configurations=4,
                    selection="First eligible native control; prefer a second different algorithm_id, otherwise distinct metadata configuration; no timing-based selection",
                    warmup_rounds=6, measured_rounds=24, reverse_order_pairs=12,
                    capture_eager_warmups_per_graph=3,
                    oracle_count_scope="Recorded oracle comparisons only; excludes three untimed eager warmups per successful captured graph (at most24 for eight graphs)",
                    eviction_bytes=safe.EVICTION_BYTES, workspace_limit_bytes=WORKSPACE_LIMIT,
                    calibration="none; gross graph times only", raw_samples_retained=True,
                    original_oracle="FP64 and original column-probe tolerances, unmodified",
                    internal_deadline_seconds=args.deadline_seconds,
                    external_supervisor_timeout_seconds=args.outer_timeout_seconds),
                  limitations=["Not full official correctness or score; nearby shapes explicitly reported",
                    "No profiler/hardware counters or native assembly inspection",
                    "Separate-stream submissions do not establish actual kernel overlap",
                    "No-split policy excludes potentially useful FP32 split-K tactics",
                    "Internal gate cannot interrupt native compiler/CUDA; external hard timeout required"])

    def save():
        temp = args.output.with_name(args.output.name + ".tmp")
        temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temp.replace(args.output)

    try:
        report["source_hashes"] = verify_sources()
        save()
        execute(args, report, save)
    except Exception as error:
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
