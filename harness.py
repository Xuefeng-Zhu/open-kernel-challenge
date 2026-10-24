"""Local contract checks; imports the downloaded, unmodified official oracle."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "official"))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def contract():
    return json.loads((ROOT / "official/muse_bf16_proj.json").read_text())


def correctness_cases():
    # Derive the nine prerequisite cases from the saved TASK.md, not a second spec.
    task = contract()["task_md"]
    block = task.split("Before timing, a run must pass these 9 tests", 1)[1].split("```", 2)[1]
    cases = []
    for i, line in enumerate(block.strip().splitlines()):
        match = re.fullmatch(r"m (\d+), layer (\w+)(?:, dist (\w+))?", line)
        if not match:
            raise ValueError(f"Unrecognized official correctness case: {line}")
        m, layer, dist = match.groups()
        cases.append(dict(m=int(m), layer=layer, dist=dist or "normal", seed=10001 + i))
    if len(cases) != 9:
        raise ValueError("Official prerequisite case count changed; review contract")
    return cases


def cases_for(suite):
    scored = [dict(c) for c in contract()["benchmarks"]]
    if suite == "prerequisites":
        return correctness_cases()
    if suite == "scored":
        return scored
    return correctness_cases() + scored


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment(torch, device):
    info = dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
                device=device, cuda_runtime=torch.version.cuda,
                cuda_available=torch.cuda.is_available(), submission_sha256=file_hash(ROOT / "submission.py"),
                contract_sha256=file_hash(ROOT / "official/muse_bf16_proj.json"))
    if device == "cuda" and torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name()
        info["capability"] = torch.cuda.get_device_capability()
        info["tf32_allowed"] = torch.backends.cuda.matmul.allow_tf32
        info["bf16_reduced_precision_reduction"] = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    return info
