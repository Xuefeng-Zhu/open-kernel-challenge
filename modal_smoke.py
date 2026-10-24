"""One explicitly authorized ephemeral smoke. Launcher stays local; no source auto-upload."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
LEDGER = ROOT / "experiments/modal-smoke-budget.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def write(record):
    LEDGER.write_text(json.dumps(record, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--approved-initial-smoke", action="store_true")
    args = parser.parse_args()
    if not args.approved_initial_smoke:
        parser.error("An approved initial smoke is required; this file is not general run authorization")
    ledger = json.loads(LEDGER.read_text())
    if ledger["attempts"]:
        raise RuntimeError("An attempt already exists. Inspect it; never automatically retry")
    bundle = args.bundle.resolve()
    if (ROOT / "local-notes").resolve() not in bundle.parents:
        raise ValueError("Use only the explicit ignored bundle directory")
    manifest = json.loads((bundle / "bundle-manifest.json").read_text())
    actual = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()}
    if actual != set(manifest["files"]) | {"bundle-manifest.json"} or len(manifest["files"]) != 17:
        raise ValueError("Exactly the reviewed 17 source files plus generated manifest required")
    for name, expected in manifest["files"].items():
        p = bundle / name
        if p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest() != expected["sha256"]:
            raise ValueError("Frozen upload bytes changed")
    auth = ROOT / "local-notes/modal-auth.toml"
    if not auth.exists() or auth.stat().st_mode & 0o077:
        raise ValueError("Approved owner-only local credential required")
    os.environ["MODAL_CONFIG_PATH"] = str(auth)
    import modal
    from modal_gpu_worker import make_modal_entry
    attempt = dict(created_utc=now(), state="reserved", app_id=None, source_commit=manifest["source_commit"],
                   candidate=manifest["candidate"], submission_sha256=manifest["files"][manifest["candidate"]]["sha256"],
                   bundle_sha256=hashlib.sha256((bundle / "bundle-manifest.json").read_bytes()).hexdigest(),
                   resource_plan=manifest["resource_plan"], max_additional_paid_usd=0,
                   credits_before_usd=30, spend_cap_verified_usd=0,
                   note="Stripe portal pause wording unresolved; no service resume. Credit-covered Function smoke only")
    ledger["attempts"].append(attempt)
    write(ledger)
    started = time.monotonic()
    image = modal.Image.from_registry(manifest["public_image"]).entrypoint([]).add_local_dir(
        bundle, "/opt/okc-bundle", copy=False)
    app = modal.App("okc-projection-smoke", include_source=False)
    execute = app.function(image=image, gpu="H200", cpu=2, memory=8192,
                           max_containers=1, min_containers=0, buffer_containers=0,
                           scaledown_window=2, timeout=600, startup_timeout=120, retries=0,
                           serialized=True, include_source=False, block_network=True,
                           restrict_modal_access=True, name="projection-smoke")(make_modal_entry())
    try:
        with app.run(detach=False, environment_name="main"):
            attempt.update(state="accepted", app_id=app.app_id, accepted_utc=now(),
                           app_setup_seconds=time.monotonic()-started)
            write(ledger)
            print("Accepted ephemeral app:", app.app_id, flush=True)
            result = execute.remote(Path(manifest["candidate"]).name)
            target = ROOT / "local-notes/modal-initial-result.local.json"
            target.write_text(json.dumps(result, indent=2) + "\n")
            attempt.update(state="function-terminal", exit_code=result["exit_code"],
                           worker_wall_seconds=result["worker_wall_seconds"])
    except Exception as error:
        # Do not print exception contents that could include runtime identifiers.
        attempt.update(state="needs-status-reconciliation", error_type=type(error).__name__,
                       app_id=app.app_id, note="No automatic retry; query exact app state")
        raise
    finally:
        attempt.update(client_finished_utc=now(), client_wall_seconds=time.monotonic()-started)
        write(ledger)
        print("Client finished; read-only app termination verification required", flush=True)


if __name__ == "__main__":
    main()
