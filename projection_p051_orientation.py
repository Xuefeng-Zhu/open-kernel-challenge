"""P051: R06 split-K orientation diagnostic, not an OKC candidate or benchmark.

Import and CPU tests use only the standard library. Native execution explicitly
requires an H200 and an external timeout supervisor. No CUDA timing is collected.
The two partial kernels differ only in accumulator orientation, dot operand order
and the corresponding scratch-store indices. Both use the original R06 reducer.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import platform
import re
import sys
import time

ROOT = Path(__file__).resolve().parent
FROZEN = {
    "candidates/pooler_triton_split8.py": "76f202f5fe9efb23ce0e10c6739871fab74512464a0e9f1229cda53c0c625406",
    "official/reference.py": "e62491abc9ddcad33f5faba8b0ee74cf6370b7be5dd2dc05031742f3ebf7147d",
    "official/checks.py": "9a046c3c38db28b325dcd1557baecdf6bc9ef941d2175f9a900ba4c230bb8055",
}
SPLITS, BLOCK_M, BLOCK_N, BLOCK_K = 8, 16, 64, 64
PARTIAL_LAUNCH = {"num_warps": 4, "num_stages": 2}
REDUCE_LAUNCH = {"num_warps": 4}  # R06 did not override reducer num_stages.
CASES = (
    ("pooler_m8", 8, 4096, 8192, "official scored shape"),
    ("pooler_m7", 7, 4096, 8192, "official prerequisite shape"),
    ("masked_edges", 15, 65, 131, "synthetic address/empty-split case"),
    ("short_k", 1, 63, 1, "synthetic short-K/empty-split case"),
    ("split_tail", 16, 67, 521, "synthetic split-tail case"),
)


def _partial(X, W, P, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
             S: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    ns = tl.program_id(0) * BN + tl.arange(0, BN)
    split = tl.program_id(1)
    ms = tl.arange(0, 16)
    ks = tl.arange(0, BK)
    part_k = tl.cdiv(K, S * BK) * BK
    acc = tl.full((16, BN), 0.0, tl.float32)
    for step in range(0, part_k, BK):
        k = split * part_k + step + ks
        x = tl.load(X + ms[:, None] * K + k[None, :],
                    (ms[:, None] < M) & (k[None, :] < K), other=0.0)
        w = tl.load(W + ns[:, None] * K + k[None, :],
                    (ns[:, None] < N) & (k[None, :] < K), other=0.0)
        acc = tl.dot(x, tl.trans(w), acc)
    offsets = (split * M + ms[:, None]) * N + ns[None, :]
    tl.store(P + offsets, acc, (ms[:, None] < M) & (ns[None, :] < N))


def _partial_weight_first(X, W, P, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                          S: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    ns = tl.program_id(0) * BN + tl.arange(0, BN)
    split = tl.program_id(1)
    ms = tl.arange(0, 16)
    ks = tl.arange(0, BK)
    part_k = tl.cdiv(K, S * BK) * BK
    acc = tl.full((BN, 16), 0.0, tl.float32)
    for step in range(0, part_k, BK):
        k = split * part_k + step + ks
        x = tl.load(X + ms[:, None] * K + k[None, :],
                    (ms[:, None] < M) & (k[None, :] < K), other=0.0)
        w = tl.load(W + ns[:, None] * K + k[None, :],
                    (ns[:, None] < N) & (k[None, :] < K), other=0.0)
        acc = tl.dot(w, tl.trans(x), acc)
    offsets = (split * M + ms[None, :]) * N + ns[:, None]
    tl.store(P + offsets, acc, (ms[None, :] < M) & (ns[:, None] < N))


def _reduce(P, OUT, COUNT: tl.constexpr, S: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    splits = tl.arange(0, S)
    values = tl.load(P + splits[:, None] * COUNT + offsets[None, :],
                     offsets[None, :] < COUNT, other=0.0)
    tl.store(OUT + offsets, tl.sum(values, axis=0), offsets < COUNT)


def validate_shape(m, n, k):
    if any(type(v) is not int for v in (m, n, k)) or not 1 <= m <= BLOCK_M or n < 1 or k < 1:
        raise ValueError("Diagnostic requires integer 1<=M<=16, N>=1, K>=1; no extra M tiling")


def split_indices(k):
    validate_shape(1, 1, k)
    part_k = ((k + SPLITS * BLOCK_K - 1) // (SPLITS * BLOCK_K)) * BLOCK_K
    return [list(range(s * part_k, min((s + 1) * part_k, k))) for s in range(SPLITS)]


def address_records(m, n, weight_first=False):
    """Valid (split, token, column, scratch offset), in each arm's lane order."""
    validate_shape(m, n, 1)
    for split in range(SPLITS):
        for column_tile in range((n + BLOCK_N - 1) // BLOCK_N):
            lanes = ((i, j) for i in range(BLOCK_M) for j in range(BLOCK_N))
            if weight_first:
                lanes = ((i, j) for j in range(BLOCK_N) for i in range(BLOCK_M))
            for token, column_lane in lanes:
                column = column_tile * BLOCK_N + column_lane
                if token < m and column < n:
                    yield split, token, column, (split * m + token) * n + column


def algebra_partials(x, w, weight_first=False):
    """Exact Python-number address model; not BF16/FP32 or Triton execution."""
    m, n, k = len(x), len(w), len(x[0]) if x else 0
    validate_shape(m, n, k)
    if any(len(row) != k for row in x + w):
        raise ValueError("Ragged inputs")
    ranges = split_indices(k)
    partials = [None] * (SPLITS * m * n)
    for split, token, column, offset in address_records(m, n, weight_first):
        partials[offset] = sum((w[column][j] * x[token][j] if weight_first
                               else x[token][j] * w[column][j]) for j in ranges[split])
    return partials


def verify_sources(root=ROOT, compile_only=False):
    hashes = {}
    for relative, expected in FROZEN.items():
        if compile_only and (relative.startswith("official/") or not (root / relative).exists()):
            continue
        digest = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError("Frozen source mismatch: " + relative)
        hashes[relative] = digest
    hashes["projection_p051_orientation.py"] = hashlib.sha256(
        (root / "projection_p051_orientation.py").read_bytes()).hexdigest()
    return hashes


def validate_options(args):
    if not 0 < args.deadline_seconds <= 120:
        raise ValueError("Internal deadline must be positive and at most120 seconds")
    if not args.deadline_seconds + 10 <= args.outer_timeout_seconds <= 180:
        raise ValueError("External supervisor timeout must be at least10 seconds longer and at most180")


def require_h200(torch):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; native diagnostic not executed")
    if "H200" not in torch.cuda.get_device_name() or tuple(torch.cuda.get_device_capability()) != (9, 0):
        raise RuntimeError("H200 SM90 required; native diagnostic not executed")


def summarize_ptx(ptx):
    # Count textual PTX instruction mnemonics, not executed instructions or SASS.
    instructions = []
    for line in ptx.splitlines():
        line = line.split("//", 1)[0].strip()
        line = re.sub(r"^@!?%\w+\s+", "", line)
        if line and not line.startswith((".", "//", "{" , "}")):
            instructions.append(line.split()[0])
    prefixes = {"wgmma": "wgmma.mma_async", "mma_sync": "mma.sync",
                "async_copy": "cp.async", "tma_tensor": "cp.async.bulk.tensor",
                "local_load": "ld.local", "local_store": "st.local"}
    return {"ptx_sha256": hashlib.sha256(ptx.encode()).hexdigest(),
            "instruction_counts": {key: sum(op.startswith(prefix) for op in instructions)
                                   for key, prefix in prefixes.items()},
            "mma_mnemonics": sorted(set(op for op in instructions if op.startswith(("wgmma.mma_async", "mma.sync")))),
            "qualification": "Static PTX text only; not SASS, achieved overlap, hardware traffic or dynamic instruction counts"}


def compiled_record(kernel, artifact_path):
    ptx = kernel.asm.get("ptx")
    if not isinstance(ptx, str) or not ptx.strip():
        raise RuntimeError("Compiled kernel returned no PTX evidence")
    artifact_path.write_text(ptx)
    # warmup compiles without running generated code. Loading the resulting
    # binary obtains driver attributes, and also does not launch its kernels.
    resource_load = {"attempted": False, "loaded": False, "error": None}
    initialize = getattr(kernel, "_init_handles", None)
    if initialize is not None:
        resource_load["attempted"] = True
        try:
            initialize()
            resource_load["loaded"] = True
        except Exception as exc:
            resource_load["error"] = type(exc).__name__ + ": " + str(exc)[:300]
    metadata = kernel.metadata
    get = metadata.get if isinstance(metadata, dict) else lambda key, default=None: getattr(metadata, key, default)
    return dict(**summarize_ptx(ptx), ptx_file=artifact_path.name,
                binary_handle_initialization=resource_load,
                n_regs_per_thread=getattr(kernel, "n_regs", None),
                n_spills_driver_raw=getattr(kernel, "n_spills", None),
                shared_bytes=get("shared"),
                compiler_metadata={key: get(key) for key in ("name", "num_warps", "num_stages", "num_ctas", "warp_size", "maxnreg")},
                resource_qualification="Reported compiler/driver metadata; missing values stay null. Triton3.8 n_spills is LOCAL_SIZE_BYTES/4, not measured spill traffic. No achieved occupancy claim.")


def _load_oracle():
    official = ROOT / "official"
    sys.path.insert(0, str(official))
    try:
        checks = importlib.import_module("checks")
        reference = importlib.import_module("reference")
    finally:
        sys.path.pop(0)
    for module, filename in ((checks, "checks.py"), (reference, "reference.py")):
        if Path(module.__file__).resolve() != (official / filename).resolve():
            raise RuntimeError("An unrelated cached oracle module was found")
    return reference


def native_input(torch, m, k, seed, distribution):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((m, k), generator=gen, device="cuda")
    if distribution == "outliers":
        x[:, ::max(1, k // 100)] *= 30
    elif distribution == "zero":
        x.zero_()
    elif distribution == "sparse":
        x.zero_()
        for row in range(m):
            x[row, (seed + row * 131) % k] = 8 if row % 2 == 0 else -32
    elif distribution != "normal":
        raise ValueError("Unknown input distribution")
    if distribution in {"normal", "outliers"}:
        # Explicit original-precision probe rows; fixtures are diagnostic, not a
        # claim to duplicate the judge's seeded generate_input implementation.
        for row in range(0, m, 8):
            x[row].zero_()
            x[row, (seed + row * 131) % k] = 8 if row % 2 == 0 else -32
    return x.bfloat16()


def _native(args, report, save, stop):
    global tl
    import torch
    import triton
    import triton.language as tl
    require_h200(torch)
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    report["environment"] = dict(python=platform.python_version(), torch=torch.__version__,
        triton=triton.__version__, cuda_runtime=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
        multiprocessor_count=props.multi_processor_count,
        bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        tf32_allowed=torch.backends.cuda.matmul.allow_tf32)
    report["native_imports_completed"] = True
    save()
    kernels = {"control": triton.jit(_partial), "weight_first": triton.jit(_partial_weight_first),
               "reduce": triton.jit(_reduce)}
    reference = None if args.compile_only else _load_oracle()
    # Compile-only admission is exactly the original scored R06 pooler M8 case.
    selected_cases = CASES[:1] if args.compile_only else CASES
    for case_index, (name, m, n, k, coverage) in enumerate(selected_cases):
        stop()
        validate_shape(m, n, k)
        record = dict(name=name, m=m, n=n, k=k, coverage=coverage,
                      compiled={}, correctness=[], complete=False)
        report["cases"].append(record)
        x = torch.empty((m, k), dtype=torch.bfloat16, device="cuda")
        w = torch.empty((n, k), dtype=torch.bfloat16, device="cuda")
        p = torch.empty((SPLITS, m, n), dtype=torch.float32, device="cuda")
        out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
        for orientation in ("control", "weight_first"):
            stop()
            compiled = kernels[orientation].warmup(x, w, p, m, n, k, SPLITS, BLOCK_N, BLOCK_K,
                grid=((n + BLOCK_N - 1) // BLOCK_N, SPLITS), **PARTIAL_LAUNCH)
            path = args.output.with_name(args.output.stem + "." + name + "." + orientation + ".ptx")
            record["compiled"][orientation] = compiled_record(compiled, path)
            save()
        compiled = kernels["reduce"].warmup(p, out, m * n, SPLITS, 128,
            grid=((m * n + 127) // 128,), **REDUCE_LAUNCH)
        path = args.output.with_name(args.output.stem + "." + name + ".reduce.ptx")
        record["compiled"]["reduce"] = compiled_record(compiled, path)
        record["lowering_differs"] = (record["compiled"]["control"]["mma_mnemonics"] !=
                                       record["compiled"]["weight_first"]["mma_mnemonics"])
        if args.compile_only:
            record.update(complete=True, generated_kernel_executed=False, correctness_status="not run")
            save()
            continue

        gen = torch.Generator(device="cuda").manual_seed(51001 + case_index)
        w.copy_((torch.randn((n, k), generator=gen, device="cuda") / k**.5).bfloat16())
        frozen_w = w.clone()
        captured_pointer = x.data_ptr()
        buffers = {orientation: (torch.empty_like(p), torch.empty_like(out))
                   for orientation in ("control", "weight_first")}
        if len({tensor.data_ptr() for pair in buffers.values() for tensor in pair}) != 4:
            raise RuntimeError("Invocation-local partial/output buffers unexpectedly alias")

        def launch(orientation):
            partial, output = buffers[orientation]
            kernels[orientation][((n + BLOCK_N - 1) // BLOCK_N, SPLITS)](
                x, w, partial, m, n, k, SPLITS, BLOCK_N, BLOCK_K, **PARTIAL_LAUNCH)
            kernels["reduce"][((m * n + 127) // 128,)](
                partial, output, m * n, SPLITS, 128, **REDUCE_LAUNCH)

        def check(orientation, label, expected_x):
            torch.cuda.synchronize()
            if x.data_ptr() != captured_pointer or not torch.equal(x.view(torch.uint8), expected_x.view(torch.uint8)):
                raise RuntimeError("Input address/content changed")
            if not torch.equal(w.view(torch.uint8), frozen_w.view(torch.uint8)):
                raise RuntimeError("Weight bits changed")
            data = {"x": expected_x, "config": dict(layer="pooler", m=m, n=n, k=k)}
            passed, message = reference.check_implementation(data, buffers[orientation][1], {"pooler": frozen_w})
            record["correctness"].append(dict(orientation=orientation, label=label, passed=bool(passed),
                                                oracle_message=str(message)[:500]))
            save()
            if not passed:
                raise RuntimeError("Original official FP64 oracle failed")

        for dist_index, distribution in enumerate(("normal", "outliers", "sparse", "zero")):
            fresh = native_input(torch, m, k, 52000 + 10 * case_index + dist_index, distribution)
            for orientation in ("control", "weight_first"):
                stop()
                x.copy_(fresh)
                buffers[orientation][0].fill_(float("nan"))
                buffers[orientation][1].fill_(float("nan"))
                launch(orientation)
                check(orientation, distribution, fresh)

        graphs = {}
        for orientation in ("control", "weight_first"):
            stop()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                launch(orientation)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graphs[orientation] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[orientation]):
                launch(orientation)
        a = native_input(torch, m, k, 53000 + case_index, "normal")
        b = native_input(torch, m, k, 54000 + case_index, "outliers")
        if k == 1:
            # The sparse probe replaces the sole entry with +8 for both seeds.
            # Negation preserves BF16/power-of-two exactness and changes B.
            b.neg_()
        if torch.equal(a.view(torch.uint8), b.view(torch.uint8)):
            raise RuntimeError("Graph ABA fixtures must differ bitwise")
        for label, fresh in (("graph_A1", a), ("graph_B", b), ("graph_A2", a)):
            for orientation in ("control", "weight_first"):
                stop()
                x.copy_(fresh)
                buffers[orientation][0].fill_(float("nan"))
                buffers[orientation][1].fill_(float("nan"))
                graphs[orientation].replay()
                check(orientation, label, fresh)
        record.update(complete=True, generated_kernel_executed=True, correctness_status="passed",
                      same_address_sequential_graph_ABA=True,
                      concurrent_calls_tested=False, partial_buffers_isolated=True)
        save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--deadline-seconds", type=float, default=120)
    parser.add_argument("--outer-timeout-seconds", type=float, default=180,
                        help="Declared external supervisor timeout, not enforced by this worker")
    args = parser.parse_args(argv)
    validate_options(args)
    started = time.monotonic()
    report = dict(kind="P051 matched orientation compiler/correctness diagnostic",
                  started_utc=datetime.now(timezone.utc).isoformat(), complete=False,
                  compile_only=args.compile_only, native_imports_completed=False, cases=[],
                  limits=dict(internal_seconds=args.deadline_seconds,
                              required_external_timeout_seconds=args.outer_timeout_seconds),
                  precision="Original BF16 operands, FP32 partials and reduction, BF16 output",
                  measurement="No timings, overhead subtraction, performance gate or official score",
                  limitations=["Not the full 9-prerequisite/30-scored official suite",
                               "M9..16 are diagnostic tile-envelope extensions, not historical R06 dispatch",
                               "No simultaneous streams/calls test; graph ABA is sequential",
                               "An internal deadline cannot interrupt blocked CUDA/compiler operations"])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        report["wall_seconds"] = time.monotonic() - started
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        temporary.replace(args.output)

    def stop():
        if time.monotonic() - started >= args.deadline_seconds:
            raise TimeoutError("Internal diagnostic deadline reached")

    try:
        report["sources_sha256"] = verify_sources(compile_only=args.compile_only)
        report["provenance_reference_hashes"] = FROZEN
        report["provenance_note"] = "sources_sha256 contains only locally verified files; provenance_reference_hashes records pinned local development references, including any omitted compile-only oracle files"
        save()
        _native(args, report, save, stop)
        report["complete"] = True
        return_code = 0
    except Exception as exc:
        report["error"] = dict(type=type(exc).__name__, message=str(exc)[:700])
        return_code = 1
    finally:
        save()
    print(json.dumps(dict(complete=report["complete"], cases_completed=sum(c["complete"] for c in report["cases"]),
                          native_imports_completed=report["native_imports_completed"],
                          error_type=report.get("error", {}).get("type"))))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
