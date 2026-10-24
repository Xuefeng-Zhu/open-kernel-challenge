"""Build a local, explicit-file bundle. No Modal SDK, authentication or network."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
PUBLIC_IMAGE = "lmsysorg/sglang@sha256:8cfc5a72cb296771f5f8a4a4688cb6d05685610a830cbb616fa4ab0eb7bb2b0a"
FIXED = (
    "harness.py", "check.py", "benchmark.py", "modal_gpu_worker.py",
    "candidates/router_pooler_hybrid.py", "LICENSE", "THIRD_PARTY_NOTICES.md",
    "licenses/June-9-Researcher-Reciprocity.txt",
    "official/AGENTS.md", "official/TASK.md", "official/baseline.py",
    "official/checks.py", "official/reference.py", "official/template.py",
    "official/muse_bf16_proj.json", "official/manifest.json",
)


def describe(candidate):
    candidate = Path(candidate)
    if candidate.is_absolute() or candidate.parent != Path("candidates"):
        raise ValueError("Select one project-relative candidates/projection_*.py file")
    if not candidate.name.startswith("projection_") or candidate.suffix != ".py":
        raise ValueError("Only isolated projection-cycle2 sources may be bundled")
    extra = ()
    if candidate.name == "projection_full_composition.py":
        extra=("candidates/projection_native_composition.py","candidates/projection_pooler_layout.py")
    if candidate.name in {"projection_qkv_layout.py","projection_unpooler_layout.py","projection_pooler_grouped.py","projection_pooler_narrow.py"}:
        extra=("candidates/projection_full_composition.py",)
    files = {}
    for relative in sorted(set(FIXED + extra + (candidate.as_posix(),))):
        source = ROOT / relative
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Missing or symlinked source: {relative}")
        raw = source.read_bytes()
        committed = subprocess.check_output(["git", "show", "HEAD:" + relative], cwd=ROOT)
        if raw != committed:
            raise ValueError(f"Commit exact bytes before bundling: {relative}")
        files[relative] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    return dict(kind="local-only Modal preparation; not permission to upload or run",
                source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                candidate=candidate.as_posix(), files=files, public_image=PUBLIC_IMAGE,
                environment_warning="Public SGLang base, not the private OKC judge overlay; record actual runtime versions",
                resource_plan=dict(gpu="H200", cpu=2, memory_mib=8192,
                                   max_containers=1, min_containers=0, buffer_containers=0,
                                   scaledown_window=2, timeout=600, startup_timeout=120, retries=0),
                internal_deadline_seconds=540,
                estimate=dict(currency="USD", h200_per_second=.001261,
                              cpu_core_per_second=.0000131, gib_per_second=.00000222,
                              base_600_second_total=(.001261+2*.0000131+8*.00000222)*600,
                              excludes="startup/build/idle/burst resources; estimate is not a hard billing guarantee"),
                gates=dict(account_verified=False, sufficient_actual_credits_verified=False,
                           payment_method_verified=False, actual_net_spend_cap_zero_verified=False,
                           cli_authorized=False, intended_source_upload_authorized=False))


def build(plan, destination):
    destination = Path(destination).resolve()
    allowed = (ROOT / "local-notes").resolve()
    if allowed not in destination.parents:
        raise ValueError("Destination must be a NEW directory inside ignored local-notes/")
    destination.mkdir(parents=True, exist_ok=False)
    for relative, expected in plan["files"].items():
        raw = (ROOT / relative).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected["sha256"]:
            raise ValueError("Source changed after manifest creation")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    (destination / "bundle-manifest.json").write_text(json.dumps(plan, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--build", type=Path, help="Optional NEW ignored local destination; never uploads")
    args = parser.parse_args()
    plan = describe(args.candidate)
    if args.build:
        build(plan, args.build)
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
