"""H200-only diagnostic CUDA-graph comparison, not the official judge or score.

Times include graph replay overhead; no guessed fixed-cost subtraction. Use approved
`okc check` for the official evaluator, paired statistics and confidence interval.
"""
import argparse
import json
import statistics
from pathlib import Path

import torch

from harness import ROOT, contract, environment, load_module


def capture(fn, data, state):
    # Warm up libraries on a side stream before capture (PyTorch CUDA graph pattern).
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn(data, state)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn(data, state)
    return graph, out


def replay_us(graph, eviction):
    # This read is before timing and larger than H200's L2; never skip cold-L2 work.
    eviction.sum()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=12, help="even count, alternating first runner")
    parser.add_argument("--seed-offset", type=int, default=100000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 4 or args.repeats % 2:
        parser.error("--repeats must be even and at least 4")
    report = dict(kind="local H200 diagnostic; not official timing or score",
                  environment=environment(torch, "cuda"), repeats=args.repeats,
                  seed_offset=args.seed_offset, eviction_bytes=256 * 1024 * 1024,
                  timing="CUDA events around one CUDA graph replay, cold L2, gross time; no fixed-cost subtraction",
                  cases=[])
    if not torch.cuda.is_available():
        report["blocked"] = "CUDA unavailable; requires H200 SXM and the official SGLang image"
    elif "H200" not in torch.cuda.get_device_name():
        report["blocked"] = "This harness requires H200; another GPU cannot establish challenge performance"
    if "blocked" in report:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(report["blocked"])
        return 2

    reference = load_module(ROOT / "official/reference.py", "benchmark_reference")
    candidate = load_module(ROOT / "submission.py", "benchmark_candidate")
    baseline = load_module(ROOT / "official/baseline.py", "sglang_baseline")
    weights = reference.generate_weights(reference.WEIGHTS_SEED, device="cuda")
    states = dict(candidate=candidate.prepare(weights), baseline=baseline.prepare(weights))
    eviction = torch.ones(report["eviction_bytes"] // 4, device="cuda", dtype=torch.float32)
    for case in contract()["benchmarks"]:
        data = reference.generate_input(**case, device="cuda")
        graphs, outputs = {}, {}
        for name, module in [("candidate", candidate), ("baseline", baseline)]:
            graphs[name], outputs[name] = capture(module.kernel_fn, data, states[name])
        samples = []
        for i in range(args.repeats):
            fresh = reference.generate_input(**{**case, "seed": case["seed"] + args.seed_offset + i}, device="cuda")
            data["x"].copy_(fresh["x"])
            order = ["candidate", "baseline"] if i % 2 == 0 else ["baseline", "candidate"]
            times = {name: replay_us(graphs[name], eviction) for name in order}
            # Each output is checked on this repetition's fresh activation, outside timing.
            for name, output in outputs.items():
                ok, message = reference.check_implementation(data, output, weights)
                if not ok:
                    report["failed"] = dict(case=case, repetition=i, runner=name, message=message)
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    raise RuntimeError(f"{name} correctness failed: {message}")
            samples.append(dict(seed=case["seed"] + args.seed_offset + i, order=order, us=times))
        ratios = [s["us"]["baseline"] / s["us"]["candidate"] for s in samples]
        paired = [(ratios[i] * ratios[i + 1]) ** 0.5 for i in range(0, len(ratios), 2)]
        result = dict(case=case, samples=samples, median_paired_gross_ratio=statistics.median(paired),
                      median_gross_us={name: statistics.median(s["us"][name] for s in samples) for name in graphs})
        report["cases"].append(result)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"{case['load']} {case['layer']}: {result['median_gross_us']} (diagnostic)", flush=True)
        del graphs, outputs
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
