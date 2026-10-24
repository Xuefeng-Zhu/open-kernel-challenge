"""CPU correctness is useful evidence, never an H200 performance result."""
import argparse
import json
from pathlib import Path

import torch

from harness import ROOT, cases_for, environment, load_module


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--submission", type=Path, default=ROOT / "submission.py")
    parser.add_argument("--suite", choices=["prerequisites", "scored", "all"], default="all")
    parser.add_argument("--weight-seed", type=int, default=5151)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    report = dict(kind="local correctness only", environment=environment(torch, args.device, args.submission),
                  weight_seed=args.weight_seed, seed_offset=args.seed_offset, cases=[])
    if args.device == "cuda" and not torch.cuda.is_available():
        report["blocked"] = "CUDA unavailable: requires an NVIDIA CUDA host; challenge measurements require H200 SXM"
        if args.output:
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(report["blocked"])
        return 2
    reference = load_module(ROOT / "official/reference.py", "official_reference")
    submission = load_module(args.submission.resolve(), "candidate")
    weights = reference.generate_weights(args.weight_seed, device=args.device)
    snapshots = {name: w.clone() for name, w in weights.items()}
    state = submission.prepare(weights)
    for case in cases_for(args.suite):
        seed = case["seed"] + args.seed_offset
        data = reference.generate_input(**{**case, "seed": seed}, device=args.device)
        x_before = data["x"].clone()
        output = submission.kernel_fn(data, state)
        ok, message = reference.check_implementation(data, output, weights)
        if not torch.equal(data["x"], x_before):
            ok, message = False, "candidate mutated the activation"
        record = dict(case=case, actual_seed=seed, passed=ok, message=message)
        report["cases"].append(record)
        print(f"{'PASS' if ok else 'FAIL'} m={case['m']} {case['layer']} {case.get('dist', 'normal')}: {message}", flush=True)
    unchanged = all(torch.equal(w, snapshots[name]) for name, w in weights.items())
    report["weights_unchanged"] = unchanged
    report["passed"] = unchanged and all(c["passed"] for c in report["cases"])
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
