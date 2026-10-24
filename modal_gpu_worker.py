"""Bounded H200 diagnostic worker. No Modal import, credentials or network calls.

Run only INSIDE a separately authorized ephemeral H200 function. This file does
not provision compute. Gross graph times are diagnostics, not leaderboard scores.
"""
import argparse
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def make_modal_entry():
    # Local function is serialized by value. Its code is in this allowlisted
    # worker; it captures no host globals, credentials or project directory.
    def execute(candidate_name, repeats=6, seed_offset=200000):
        import json
        import os
        from pathlib import Path
        import signal
        import subprocess
        import sys
        import time

        root = Path("/opt/okc-bundle")
        output = Path("/tmp/okc-smoke-result")
        started = time.monotonic()
        command = [sys.executable, str(root / "modal_gpu_worker.py"),
                   "--candidate", str(root / "candidates" / candidate_name),
                   "--output-dir", str(output), "--repeats", str(repeats),
                   "--seed-offset", str(seed_offset)]
        with Path("/tmp/okc-smoke-worker.log").open("w") as log:
            child = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            try:
                code = child.wait(timeout=555)
                state = "completed"
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                code, state = -signal.SIGKILL, "deadline-killed"
        records = {p.name: json.loads(p.read_text()) for p in output.glob("*.json")}
        logs = {p.name: p.read_text()[-16000:] for p in output.glob("*.log")}
        logs["worker.log"] = Path("/tmp/okc-smoke-worker.log").read_text()[-16000:]
        return dict(exit_code=code, state=state, worker_wall_seconds=time.monotonic()-started,
                    python=sys.version, python_executable=sys.executable, records=records, logs=logs)
    return execute


def save(path, report):
    Path(path).write_text(json.dumps(report, indent=2) + "\n")


def benchmark(candidate, output, repeats, seed_offset=200000):
    import torch
    from benchmark import capture, replay_us
    from harness import contract, environment, load_module
    report = dict(kind="Modal H200 diagnostic, NOT official score", cases=[],
                  timing="gross CUDA-event times around graph replays; 256MiB cold-L2 read outside events; no subtraction",
                  environment=environment(torch, "cuda", candidate), stages={}, repeats=repeats, seed_offset=seed_offset)
    if not torch.cuda.is_available() or "H200" not in torch.cuda.get_device_name():
        report["blocked"] = "Explicit H200 required; no fallback GPU"
        save(output, report)
        return 2
    report["versions"] = {name: importlib.metadata.version(name) for name in ["torch", "triton", "sglang"]}
    reference = load_module(ROOT / "official/reference.py", "modal_reference")
    started = time.monotonic()
    modules = {
        "candidate": load_module(candidate, "modal_candidate"),
        "r10": load_module(ROOT / "candidates/router_pooler_hybrid.py", "modal_r10"),
        "sglang": load_module(ROOT / "official/baseline.py", "modal_sglang"),
    }
    report["source_sha256"] = {name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                               for name, module in modules.items()}
    weights = reference.generate_weights(reference.WEIGHTS_SEED, device="cuda")
    states = {name: module.prepare(weights) for name, module in modules.items()}
    torch.cuda.synchronize()
    report["stages"]["import_weights_prepare_seconds"] = time.monotonic() - started
    eviction = torch.ones(256 * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)
    orders = list(itertools.permutations(modules))
    with torch.inference_mode():
        for case in contract()["benchmarks"]:
            data = reference.generate_input(**case, device="cuda")
            graphs, outputs = {}, {}
            started = time.monotonic()
            for name, module in modules.items():
                graphs[name], outputs[name] = capture(module.kernel_fn, data, states[name])
            torch.cuda.synchronize()
            setup_seconds = time.monotonic() - started
            samples = []
            for repeat in range(repeats):
                seed = case["seed"] + seed_offset + repeat
                fresh = reference.generate_input(**{**case, "seed": seed}, device="cuda")
                data["x"].copy_(fresh["x"])
                order = orders[repeat % len(orders)]
                times = {name: replay_us(graphs[name], eviction) for name in order}
                for name, out in outputs.items():
                    ok, message = reference.check_implementation(data, out, weights)
                    if not ok:
                        report["failed"] = dict(case=case, seed=seed, runner=name, message=message)
                        save(output, report)
                        return 1
                samples.append(dict(seed=seed, order=order, us=times))
            report["cases"].append(dict(case=case, capture_compile_warmup_seconds=setup_seconds,
                                        samples=samples,
                                        median_gross_us={name: statistics.median(s["us"][name] for s in samples)
                                                         for name in modules}))
            save(output, report)
            del graphs, outputs
    report["passed"] = True
    save(output, report)
    return 0


def supervise(candidate, output_dir, repeats, seed_offset=200000):
    output_dir.mkdir(parents=True, exist_ok=False)
    import torch
    if not torch.cuda.is_available() or "H200" not in torch.cuda.get_device_name():
        save(output_dir / "stages.json", dict(blocked="H200 required; no CUDA/fallback execution", stages=[]))
        return 2
    manifest = json.loads((ROOT / "bundle-manifest.json").read_text())
    if candidate.relative_to(ROOT).as_posix() != manifest["candidate"]:
        raise ValueError("Candidate must match frozen bundle")
    for name, expected in manifest["files"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected["sha256"]:
            raise ValueError(f"Bundle integrity failed: {name}")
    deadline = time.monotonic() + 540
    report = dict(kind="bounded GPU preflight; no official upload", source_commit=manifest["source_commit"],
                  candidate=manifest["candidate"], stages=[], deadline_seconds=540, seed_offset=seed_offset)
    steps = [("correctness-" + name, ["check.py", "--device", "cuda", "--suite", "all",
              "--submission", str(source), "--seed-offset", str(seed_offset),
              "--output", str(output_dir / (name + "-correctness.json"))])
             for name, source in [("candidate", candidate),
                                  ("r10", ROOT / "candidates/router_pooler_hybrid.py"),
                                  ("sglang", ROOT / "official/baseline.py")]]
    steps.append(("graph-benchmark", [__file__, "--phase", "benchmark", "--candidate", str(candidate),
                                     "--output-dir", str(output_dir), "--repeats", str(repeats),
                                     "--seed-offset", str(seed_offset)]))
    for name, command in steps:
        started = time.monotonic()
        remaining = deadline - started
        if remaining <= 0:
            report["blocked"] = "Internal absolute deadline exhausted"
            break
        with (output_dir / (name + ".log")).open("w") as log:
            process = subprocess.Popen([sys.executable, *command], cwd=ROOT, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=remaining)
                state = "completed"
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                code, state = -signal.SIGKILL, "deadline-killed"
        report["stages"].append(dict(name=name, command=[Path(sys.executable).name, *command],
                                     elapsed_seconds=time.monotonic()-started, exit_code=code, state=state))
        save(output_dir / "stages.json", report)
        if code:
            report["blocked"] = f"{name}: {state}, exit {code}; no retries"
            break
    report["passed"] = len(report["stages"]) == 4 and "blocked" not in report
    save(output_dir / "stages.json", report)
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["supervise", "benchmark"], default="supervise")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--seed-offset", type=int, default=200000)
    args = parser.parse_args()
    if args.repeats < 6 or args.repeats % 6:
        parser.error("Use a positive multiple of six for balanced three-runner orders")
    candidate = args.candidate.resolve()
    if args.phase == "benchmark":
        return benchmark(candidate, args.output_dir / "graph-diagnostic.json", args.repeats, args.seed_offset)
    return supervise(candidate, args.output_dir, args.repeats, args.seed_offset)


if __name__ == "__main__":
    raise SystemExit(main())
