"""P070 diagnostic only: P046 / pinned Quack SEPARATE / SERIAL at QKV M128.

Default command is a dependency-free plan. No service, installation or submission
code. Native execution requires the pinned dependency checkout and an external
timeout; this file is not a competition candidate.
"""
import argparse
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PIN = "77a750c0d9548d4a6e23c0f64062d61b3ea420ff"
CHAMPION = "candidates/projection_pooler_column_batch2_hybrid.py"
FROZEN = {
    CHAMPION: "001ea12e8764564d507de51e433769f5bd9a9916bfc4c014f0f87522c278545e",
    "projection_p046_confirmation.py": "6c57daf6ab5aca4a73aa1a608b6ae978af284a02d41c489b4e7b62b307505029",
    "benchmark.py": "5d7f2db870a1e82142be4b5f91e319c944673b31bdb66bacf02159d2032dde78",
    "harness.py": "d46ea816a0091cea8b6e429a6e8facfbe0a82c20488a8c9b89023c1561777fd8",
    "official/reference.py": "e62491abc9ddcad33f5faba8b0ee74cf6370b7be5dd2dc05031742f3ebf7147d",
    "official/checks.py": "9a046c3c38db28b325dcd1557baecdf6bc9ef941d2175f9a900ba4c230bb8055",
    "official/muse_bf16_proj.json": "ee02afa1e848c15b16d677fc1e405617a51182fbda7ed6b21c9e5f6a7c5ef0e9",
}
PACKAGES = ("torch", "triton", "nvidia-cutlass-dsl", "apache-tvm-ffi", "einops")
ARMS = ("p046", "separate", "serial")
CONFIG = dict(layer="qkv_gate", m=128, n=2304, k=8192)
GEOMETRY = dict(tile_M=128, tile_N=128, tile_K=64, cluster_M=1,
                cluster_N=1, cluster_K=1, split_k=2, persistent=True,
                pingpong=False, is_dynamic_persistent=False,
                max_swizzle_size=8, b_kn=False)
# Each adjacent pair reverses the complete order; every arm occupies each
# position twice per six-round block. Two seed cohorts are confirmation, not
# post-hoc opportunities to choose a winning seed.
ORDERS = (ARMS, ARMS[::-1], ("separate", "serial", "p046"),
          ("p046", "serial", "separate"), ("serial", "p046", "separate"),
          ("separate", "p046", "serial"))
WARMUP, ROUNDS, COHORTS = 6, 24, (7001000, 7002000)
LIMITATIONS = [
    "No native compilation or GPU pass follows from CPU/mock tests.",
    "SEPARATE compiles FP32 partial output; SERIAL compiles BF16 final output. "
    "Automatic AB/epilogue stages, shared layouts and resources may differ.",
    "P046 vendor geometry and algorithm differ from both Quack arms; it is the "
    "real performance control, not a causal finalization-only control.",
    "SM90 Quack defaults PDL on; the high-level call has no use_pdl override.",
    "Private scratch supports sequential replay structurally; overlapping replay "
    "of one graph, independent concurrent kernels and persistent-grid liveness are unverified.",
    "One canonical shape only; no nearby/tail shapes, official score or calibrated timing.",
    "Upstream CuTe FFI converter patch needs contest eligibility review before any submission.",
]


def source_hashes(root=ROOT):
    result = {}
    for name, expected in FROZEN.items():
        result[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if result[name] != expected:
            raise ValueError("Frozen source mismatch: " + name)
    return result


def dependencies():
    versions, missing = {}, []
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(name)
    return dict(platform=platform.system(), machine=platform.machine(),
                python=platform.python_version(), versions=versions, missing=missing,
                quack="Requires a clean checkout at " + PIN + "; package version is insufficient")


def verify_quack(root):
    """Read-only Git checks; no clone, install, update or third-party execution."""
    root = Path(root).resolve()
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], check=True,
                              capture_output=True, text=True, timeout=10).stdout.strip()
    if git("rev-parse", "HEAD") != PIN:
        raise ValueError("Quack revision does not match the reviewed pin")
    if git("status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("Quack checkout must be clean, including untracked source")
    if not (root / "quack/gemm.py").is_file():
        raise ValueError("Missing pinned Quack source tree")
    return root


def order(index):
    return ORDERS[index % len(ORDERS)]


def partition(total_tiles, split, count=2):
    """CPU model of the pinned scheduler's balanced contiguous K partition."""
    if total_tiles < 0 or count < 1 or not 0 <= split < count:
        raise ValueError("Invalid partition")
    per, remainder = divmod(total_tiles, count)
    start = split * per + min(split, remainder)
    return start, start + per + (split < remainder)


def make_quack_arm(torch, gemm, mode, plans=None, label=None):
    """Only a callable is cached. D, semaphore and partials are call-local.

    The pinned high-level gemm allocates/initializes scratch inside this call.
    Capturing only its compiled_fn would incorrectly exclude that work.
    """
    def kernel(data, weight):
        x = data["x"]
        if data["config"] != CONFIG or tuple(x.shape) != (128, 8192):
            raise ValueError("P070 is restricted to canonical QKV M128")
        if tuple(weight.shape) != (2304, 8192):
            raise ValueError("Unexpected QKV weight shape")
        if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
            raise ValueError("BF16 operands required")
        if not x.is_contiguous() or not weight.is_contiguous() or x.device != weight.device:
            raise ValueError("Original contiguous colocated operands required")
        out = torch.empty((1, 128, 2304), dtype=torch.bfloat16, device=x.device)
        compiled_plan = gemm(x.unsqueeze(0), weight.unsqueeze(0), out, None, None,
                             **GEOMETRY, alpha=1.0, beta=0.0, split_k_mode=mode)
        if plans is not None:
            plans[label] = compiled_plan  # Immutable compile metadata, never runtime scratch.
        return out.squeeze(0)
    return kernel


def summarize(samples):
    if len(samples) != ROUNDS:
        raise ValueError("All predefined rounds required; no partial-result selection")
    for i, sample in enumerate(samples):
        if sample["round"] != i or tuple(sample["order"]) != order(i):
            raise ValueError("Unexpected timing order")
        if sample.get("excluded_warmup") is not False:
            raise ValueError("Warmup or unclassified samples cannot enter measured results")
        if set(sample["us"]) != set(ARMS) or not all(sample["correctness"].get(a) for a in ARMS):
            raise ValueError("Incomplete or incorrect round")
        if not all(math.isfinite(t) and t > 0 for t in sample["us"].values()):
            raise ValueError("Invalid timing")
    result = {}
    for first, second in itertools.combinations(ARMS, 2):
        ratios = [s["us"][first] / s["us"][second] for s in samples]
        paired = [math.sqrt(ratios[i] * ratios[i + 1]) for i in range(0, ROUNDS, 2)]
        median = statistics.median(paired)
        result[first + "_over_" + second] = dict(paired_ratios=paired, median=median,
            second_latency_reduction=1 - 1 / median, minimum=min(paired), maximum=max(paired))
    return result


def assessment(cohorts):
    if [c["seed"] for c in cohorts] != list(COHORTS) or not all(c["passed"] for c in cohorts):
        raise ValueError("Two complete correct fresh cohorts are required")
    comparisons = [summarize(c["samples"]) for c in cohorts]
    cuts = [s["p046_over_serial"]["second_latency_reduction"] for s in comparisons]
    advantages = [s["separate_over_serial"]["second_latency_reduction"] for s in comparisons]
    return dict(serial_cut_vs_p046=cuts, serial_cut_vs_separate=advantages,
        performance_screen_passed=all(s["p046_over_serial"]["median"] >= 1 / .95 for s in comparisons)
            and all(s["separate_over_serial"]["median"] > 1 / .98 for s in comparisons),
        criterion="Both fresh cohorts: >=5% raw whole-call cut versus P046 and >2% versus SEPARATE",
        automatic_promotion=False, selected="P046",
        qualification="Bounded diagnostic; source/compiled differences and fresh official full-suite verification remain")


def plan():
    deps = dependencies()
    return dict(kind="P070 offline preflight; not a scoring candidate", source_hashes=source_hashes(),
        quack_commit=PIN, dependencies=deps, arms=ARMS, case=CONFIG, geometry=GEOMETRY,
        protocol=dict(cohorts=COHORTS, excluded_warmup_per_cohort=WARMUP,
                      measured_rounds_per_cohort=ROUNDS, orders=ORDERS,
                      same_address_replays="normal A, outlier B, exact A-return, sparse, zero; before and after",
                      scratch="High-level gemm inside each capture; SERIAL zeroing included",
                      timing="Whole-call cold-L2 CUDA graph, gross events, no subtraction",
                      correctness="Original FP64/probe oracle for every measured/warmup output",
                      expected_oracle_checks=2 * (2 * 5 * 3 + (WARMUP + ROUNDS) * 3),
                      additional_capture_calls=24,
                      capture_call_checks="Three library warmups plus one capture per arm/cohort; "
                      "these 24 outputs are not individually oracle-checked",
                      between_arm_work="Input immutability check after each timed arm; "
                      "cold-L2 eviction before each arm; FP64 oracles after all three"),
        gate=dict(local_native_compile="blocked" if deps["missing"] else "not attempted",
                  generated_pipeline_layout_resources="unverified",
                  gpu_correctness="not executed", performance="not measured",
                  selection="Retain P046; require correctness and >=5% whole-call cut against P046 "
                  "in both fresh cohorts, plus >2% SERIAL-vs-SEPARATE cut; <=2% is inconclusive"),
        limitations=LIMITATIONS)


def compile_arguments(bf16, fp32, mode, rounding, separate=False):
    """Pinned fake-tensor signature matching alpha=1.0/beta=0.0 runtime calls.

    M/N/K/L are symbolic in this API. SEPARATE strips epilogue scalars from
    its mainloop, but its reducer and SERIAL retain beta_mode=1 for beta=0.
    """
    return dict(a_dtype=bf16, b_dtype=bf16, d_dtype=fp32 if separate else bf16,
        c_dtype=None, a_major="k", b_major="k", d_major="n", c_major=None,
        tile_shape_mn=(128, 128, 64), cluster_shape_mnk=(1, 1, 1),
        pingpong=False, persistent=True, is_dynamic_persistent=False,
        rowvec_dtype=None, colvec_dtype=None, colvec_ndim=0,
        alpha_mode=0, beta_mode=0 if separate else 1, add_to_output=False,
        concat_layout=(), varlen_m=False, varlen_k=False, gather_A=False,
        use_tma_gather=False, has_batch_idx_permute=False, device_capacity=(9, 0),
        rounding_mode=rounding, sr_seed_mode=0, num_warps=None,
        split_k=2, split_k_mode=mode, batched=True, b_kn=False)


def compile_offline(quack_root, destination):
    """Source-supported fake-tensor export only; no device bind or launch.

    Successful object export alone does not clear the generated-code gate.
    Stage/layout/resource inspection still needs supported compiler artifacts.
    """
    if dependencies()["missing"]:
        raise RuntimeError("Missing dependencies; no installation is performed")
    root = verify_quack(quack_root)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES="", QUACK_ARCH="90",
                      CUTE_DSL_ARCH="sm_90a", QUACK_CACHE_ENABLED="0")
    sys.path.insert(0, str(root))
    from cutlass import BFloat16, Float32
    from quack.gemm import _compile_gemm
    from quack.gemm_config import SplitKMode
    from quack.rounding import RoundingMode
    from quack.split_k_reduce import _compile_split_k_reduce
    import importlib
    if Path(importlib.import_module("quack.gemm").__file__).resolve() != root / "quack/gemm.py":
        raise RuntimeError("Unexpected imported Quack module")
    outputs = {}
    for name, separate, mode in (("serial", False, SplitKMode.SERIAL),
                                 ("separate", True, SplitKMode.SEPARATE)):
        compiled = _compile_gemm(**compile_arguments(BFloat16, Float32, mode,
                                                      RoundingMode.RN, separate))
        target = destination / (name + ".o")
        compiled.export_to_c(object_file_path=str(target), function_name="func")
        outputs[target.name] = hashlib.sha256(target.read_bytes()).hexdigest()
    reducer = _compile_split_k_reduce(out_dtype=BFloat16, split_k=2,
        add_to_output=False, vec_elems=4, alpha_mode=0, beta_mode=1,
        c_dtype=None, rowvec_dtype=None, colvec_dtype=None)
    target = destination / "separate_reduce.o"
    reducer.export_to_c(object_file_path=str(target), function_name="func")
    outputs[target.name] = hashlib.sha256(target.read_bytes()).hexdigest()
    receipt = dict(quack_commit=PIN, dependencies=dependencies(), objects=outputs, geometry=GEOMETRY,
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        compile_arguments="Symbolic M/N/K/L; fixed BF16 types/layout/tile/split/mode, intended canonical QKV128",
        cuda_execution=False, generated_stage_layout_resource_gate="not yet inspected",
        absent=["PTX/SASS inspection", "register/spill/shared-memory comparison",
                "native correctness", "performance", "official environment identity"],
        source_hashes=source_hashes())
    (destination / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def require_compile_review(path):
    """Fail closed until real generated artifacts have been reviewed.

    This records an engineering gate, not an authority to launch or a claim
    that a human-written review is equivalent to measurements.
    """
    path = Path(path)
    receipt = json.loads(path.read_text())
    if (receipt.get("quack_commit") != PIN or receipt.get("geometry") != GEOMETRY
            or receipt.get("source_hashes") != source_hashes()
            or receipt.get("dependencies") != dependencies()
            or receipt.get("implementation_sha256") != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()):
        raise ValueError("Compile receipt does not match current frozen inputs/implementation")
    expected = {"serial.o", "separate.o", "separate_reduce.o"}
    if set(receipt.get("objects", {})) != expected:
        raise ValueError("All three compiled objects required")
    for name, digest in receipt["objects"].items():
        if hashlib.sha256((path.parent / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Compiled object changed: " + name)
    review = receipt.get("review", {})
    if review.get("ready_for_execution") is not True or not all(
            isinstance(review.get(field), str) and review[field].strip()
            for field in ("mainloop_instructions", "stages_and_layout_differences",
                          "register_spill_shared_resources", "synchronization_and_initialization")):
        raise ValueError("Generated instruction/layout/resource/protocol review remains incomplete")
    return receipt


def verify_runtime_objects(plans, compile_reduce, bf16, directory, receipt):
    """Conservative binary-identity gate; non-identical exports stop timing.

    Main objects come from the actual high-level launch plans. The reducer is
    obtained with the exact pinned runtime signature (no execution here).
    Nondeterministic object metadata may require later inspection; it is not
    silently ignored as equivalent code.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    compiled = {name: plans[name].compiled_fn for name in ("serial", "separate")}
    compiled["separate_reduce"] = compile_reduce(out_dtype=bf16, split_k=2,
        add_to_output=False, vec_elems=4, alpha_mode=0, beta_mode=1,
        c_dtype=None, rowvec_dtype=None, colvec_dtype=None)
    actual = {}
    for name, function in compiled.items():
        path = directory / (name + ".o")
        function.export_to_c(object_file_path=str(path), function_name="func")
        actual[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != receipt["objects"]:
        raise RuntimeError("Runtime compiled exports differ from reviewed artifacts; timing not started")
    return actual


def execute(quack_root, compile_review, report, save, deadline_seconds=120):
    """Future authorized native diagnostic, never called by plan or tests.

    An external supervisor must terminate this process/container on a blocked
    CUDA call; the monotonic deadline only checks between calls.
    """
    if dependencies()["missing"]:
        raise RuntimeError("Missing dependencies; no installation is performed")
    report["compile_review"] = require_compile_review(compile_review)
    root = verify_quack(quack_root)
    os.environ.update(QUACK_ARCH="90", CUTE_DSL_ARCH="sm_90a", QUACK_CACHE_ENABLED="0")
    sys.path.insert(0, str(root))
    import torch
    from quack.gemm import gemm
    from quack.gemm_config import SplitKMode
    from quack.split_k_reduce import _compile_split_k_reduce
    from cutlass import BFloat16
    import importlib
    module = importlib.import_module("quack.gemm")
    if Path(module.__file__).resolve() != root / "quack/gemm.py":
        raise RuntimeError("Unexpected imported Quack module")
    from benchmark import capture
    from harness import environment, load_module
    import projection_p046_confirmation as helper
    helper.require_h200(torch)
    report["environment"] = environment(torch, "cuda", ROOT / CHAMPION)
    reference = load_module(ROOT / "official/reference.py", "p070_reference")
    champion = load_module(ROOT / CHAMPION, "p070_p046")
    plans = {}
    functions = dict(p046=champion.kernel_fn,
        separate=make_quack_arm(torch, gemm, SplitKMode.SEPARATE, plans, "separate"),
        serial=make_quack_arm(torch, gemm, SplitKMode.SERIAL, plans, "serial"))
    started = time.monotonic()
    def gate():
        if time.monotonic() - started > deadline_seconds:
            raise TimeoutError("Internal deadline exceeded; external watchdog required for blocked CUDA")
    report["cohorts"] = []
    with torch.inference_mode():
        eviction = torch.ones(helper.EVICTION_BYTES // 4, dtype=torch.float32, device="cuda")
        eviction_out = torch.empty((), dtype=torch.float32, device="cuda")
        timer = helper.ColdGraphTimer(torch)
        for seed in COHORTS:
            gate()
            weights = reference.generate_weights(seed, device="cuda")
            original = helper.freeze_tensors(torch, weights)
            oracle_weights = {name: item[1] for name, item in original.items()}
            state = champion.prepare(weights)
            frozen_state = helper.freeze_tensors(torch, state)
            states = dict(p046=state, separate=weights["qkv_gate"], serial=weights["qkv_gate"])
            make = lambda s, dist="normal": helper.fresh_input(torch, reference, "qkv_gate", 128, s, dist)
            data = make(seed)
            initial = {"x": data["x"].clone()}
            pointer = helper.tensor_metadata(data["x"])
            record = dict(seed=seed, passed=False, correctness=[], warmup=[], samples=[])
            report["cohorts"].append(record)
            save()
            graphs, outputs = {}, {}
            for arm in ARMS:
                gate()
                graphs[arm], outputs[arm] = capture(functions[arm], data, states[arm])
            record["runtime_objects"] = verify_runtime_objects(plans, _compile_split_k_reduce,
                BFloat16, Path(compile_review).parent / ("runtime-" + str(seed)), report["compile_review"])
            if len({v.data_ptr() for v in outputs.values()}) != len(ARMS):
                raise RuntimeError("Outputs unexpectedly alias across arms")
            helper.assert_input(torch, data, initial, CONFIG, pointer)
            def immutable():
                helper.assert_frozen(torch, weights, original)
                helper.assert_frozen(torch, state, frozen_state)
            def check(arm, fresh):
                helper.check_output(torch, reference, data, outputs[arm], fresh, CONFIG, pointer, oracle_weights)
            immutable()
            for phase in ("before", "after"):
                a = make(seed + 10)
                for label, fresh in (("A", a), ("B", make(seed + 11, "outliers")),
                                     ("A-return", a), ("sparse", make(seed + 12, "sparse")),
                                     ("zero", make(seed + 13, "zero"))):
                    data["x"].copy_(fresh["x"])
                    if label == "A": first = {}
                    for arm in ARMS:
                        gate()
                        graphs[arm].replay()
                        torch.cuda.synchronize()
                        check(arm, fresh)
                        if label == "A": first[arm] = outputs[arm].clone()
                        if label == "A-return" and not torch.equal(
                                first[arm].view(torch.uint8), outputs[arm].view(torch.uint8)):
                            raise RuntimeError("A-return output changed: " + arm)
                        record["correctness"].append(dict(phase=phase, label=label, arm=arm, passed=True))
                        save()
                immutable()
                if phase == "after": break
                for warmup, count, offset in ((True, WARMUP, 100), (False, ROUNDS, 200)):
                    for i in range(count):
                        fresh = make(seed + offset + i)
                        data["x"].copy_(fresh["x"])
                        sample = dict(round=i, order=order(i), us={}, correctness={}, excluded_warmup=warmup)
                        (record["warmup"] if warmup else record["samples"]).append(sample)
                        for arm in order(i):
                            gate()
                            sample["us"][arm] = timer.measure(graphs[arm], eviction, eviction_out)
                            helper.assert_input(torch, data, fresh, CONFIG, pointer)
                        # FP64 reference checks occur after all three timings in this round.
                        for arm in order(i):
                            gate(); check(arm, fresh)
                            sample["correctness"][arm] = True
                        save()
                immutable()
            record.update(passed=True, summary=summarize(record["samples"]))
            save()
            del graphs, outputs, states, state, frozen_state, weights, original, oracle_weights
        report["correctness_passed"] = True
        report["assessment"] = assessment(report["cohorts"])
        report["gate"].update(gpu_correctness="passed for documented canonical-shape coverage",
                             performance="measured raw diagnostic; see assessment",
                             generated_pipeline_layout_resources="see bound compile_review")
        report["wall_seconds"] = time.monotonic() - started


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "compile", "run"), nargs="?", default="plan")
    parser.add_argument("--quack-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compile-review", type=Path)
    args = parser.parse_args(argv)
    report = plan()
    if args.action != "plan" and (not args.quack_root or not args.output):
        parser.error("compile/run require --quack-root and --output; external timeout required")
    if args.action == "run" and not args.compile_review:
        parser.error("run requires --compile-review from inspected exact-source compiler artifacts")
    if args.action == "compile":
        try:
            print(json.dumps(compile_offline(args.quack_root, args.output), indent=2))
        except Exception as error:
            print(json.dumps(dict(status="compile blocked/failed", type=type(error).__name__,
                                  message=str(error)[:500])))
            return 1
        return 0
    def save():
        if args.output:
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    save()
    if args.action == "run":
        report["correctness_passed"] = False
        try:
            execute(args.quack_root, args.compile_review, report, save)
        except Exception as error:
            report["failure"] = dict(type=type(error).__name__, message=str(error)[:500])
            save()
            print(json.dumps(report["failure"]))
            return 1
        save()
    else:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
