"""Authorized unpooler/QKV stage, cumulative credit ceiling; no automatic retries."""
import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
LEDGER = ROOT / "experiments/modal-diagnostics-budget.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def write(record):
    LEDGER.write_text(json.dumps(record, indent=2) + "\n")


def read_cli(*command):
    result = subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "modal", *command],
                            capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError("Supported read-only Modal preflight failed; no launch")
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["M19-unpooler", "M20-qkv-repeat", "M23-composition", "M26-qkv-layout", "M28-unpooler-layout"], required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    ledger = json.loads(LEDGER.read_text())
    if any(a["experiment"] == args.experiment for a in ledger["attempts"]):
        raise RuntimeError("This experiment already has a reservation: inspect, never blindly repeat")
    if any(a["state"] != "terminal" for a in ledger["attempts"]):
        raise RuntimeError("Prior app must be terminal and its termination verified")
    if args.experiment == "M20-qkv-repeat" and not any(a["experiment"] == "M19-unpooler" for a in ledger["attempts"]):
        raise RuntimeError("Inspect unpooler feedback first")
    verified = datetime.fromisoformat(ledger["billing_ui_verified_utc"])
    if (datetime.now(timezone.utc)-verified).total_seconds() > 600:
        raise RuntimeError("Fresh Mac UI credit and zero paid-cap verification required")
    if ledger["verified_net_spend_cap_usd"] != 0 or ledger["verified_credit_balance_usd"] < 2:
        raise RuntimeError("Credit/cap gate failed")
    bundle = args.bundle.resolve()
    if (ROOT / "local-notes").resolve() not in bundle.parents:
        raise ValueError("Use only explicit ignored bundle")
    manifest = json.loads((bundle / "bundle-manifest.json").read_text())
    actual = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()}
    if actual != set(manifest["files"]) | {"bundle-manifest.json"} or len(manifest["files"]) != (19 if args.experiment=="M23-composition" else (18 if args.experiment in {"M26-qkv-layout","M28-unpooler-layout"} else 17)):
        raise ValueError("Expected reviewed source/license files and manifest")
    candidate, repeats, seed, protocol = {
        "M19-unpooler": ("candidates/projection_unpooler_m1_simt.py", 12, 400000, "three-runner-v1"),
        "M20-qkv-repeat": ("candidates/projection_qkv_m1_simt.py", 30, 700000, "three-runner-v1"),
        "M23-composition": ("candidates/projection_full_composition.py", 30, 1000000, "composition-v1"),
        "M26-qkv-layout": ("candidates/projection_qkv_layout.py", 30, 1300000, "screen-v1"),
        "M28-unpooler-layout": ("candidates/projection_unpooler_layout.py", 30, 1600000, "screen-v1"),
    }[args.experiment]
    if manifest["candidate"] != candidate:
        raise ValueError("Wrong candidate for this preplanned experiment")
    cpu_reports={"M26-qkv-layout":"results/router-cycle2-qkv-layout-cpu.json", "M28-unpooler-layout":"results/router-cycle2-unpooler-layout-cpu.json"}
    if args.experiment in cpu_reports:
        cpu=json.loads((ROOT/cpu_reports[args.experiment]).read_text())
        if not cpu.get("passed") or cpu["environment"]["submission_sha256"]!=manifest["files"][candidate]["sha256"]:
            raise ValueError("Missing exact-source CPU gate")
    for name, expected in manifest["files"].items():
        p = bundle / name
        if p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest() != expected["sha256"]:
            raise ValueError("Frozen upload bytes changed")
        if (ROOT / name).read_bytes() != p.read_bytes():
            raise ValueError("Local serialized worker and bundle must match")
    auth = ROOT / "local-notes/modal-auth.toml"
    if not auth.exists() or auth.stat().st_mode & 0o077:
        raise ValueError("Existing approved owner-only credential required")
    os.environ["MODAL_CONFIG_PATH"] = str(auth)
    billing = read_cli("billing", "summary", "--json")
    if Decimal(billing["billed_cost"]) != 0:
        raise RuntimeError("Nonzero net charge observed; stop stage")
    if any(a["state"] != "stopped" or int(a["tasks"]) != 0 for a in read_cli("app", "list", "--json")):
        raise RuntimeError("Another app is active; no concurrency")
    observed = max(Decimal(billing["metered_cost"]),
                   sum((Decimal(v) for v in billing["metered_cost_breakdown"].values()), Decimal(0)))
    observed_highwater = max(Decimal(str(ledger["observed_gross_high_water_usd"])), observed)
    highwater = max(observed_highwater,Decimal(str(ledger.get("conservative_stage_spent_usd",0))))
    screen = protocol=="screen-v1"
    timeout,startup,deadline=(300,90,240) if screen else (600,120,540)
    reserve = Decimal("0.60") if screen else Decimal("1.00")
    if highwater + reserve > Decimal(str(ledger["cumulative_gross_ceiling_usd"])):
        raise RuntimeError("Conservative credit reserve exceeds USD2 stage ceiling")
    ledger["observed_gross_high_water_usd"] = str(observed_highwater)
    import modal
    from modal_gpu_worker import make_modal_entry
    attempt = dict(experiment=args.experiment, created_utc=now(), state="reserved", app_id=None,
                   source_commit=manifest["source_commit"], candidate=candidate,
                   submission_sha256=manifest["files"][candidate]["sha256"], repeats=repeats, seed_offset=seed, protocol=protocol,
                   bundle_sha256=hashlib.sha256((bundle / "bundle-manifest.json").read_bytes()).hexdigest(),
                   billing_before=billing, conservative_reserve_usd=str(reserve),
                   resource_limits=dict(gpu="H200", cpu_request_limit=[2,2], memory_mib_request_limit=[8192,8192],
                                        max_containers=1, min_containers=0, buffer_containers=0, scaledown_window=2,
                                        timeout=timeout, startup_timeout=startup, retries=0, internal_deadline=deadline),
                   base_image_id=ledger["reused_image_id"])
    ledger["attempts"].append(attempt)
    write(ledger)
    started = time.monotonic()
    image = modal.Image.from_id(ledger["reused_image_id"]).add_local_dir(bundle,"/opt/okc-bundle",copy=False)
    app = modal.App("okc-"+args.experiment.lower(),include_source=False)
    execute = app.function(image=image,gpu="H200",cpu=(2,2),memory=(8192,8192),
                           max_containers=1,min_containers=0,buffer_containers=0,scaledown_window=2,
                           timeout=timeout,startup_timeout=startup,retries=0,serialized=True,include_source=False,
                           block_network=True,restrict_modal_access=True,name="projection-diagnostic")(make_modal_entry())
    try:
        with app.run(detach=False,environment_name="main"):
            attempt.update(state="accepted",app_id=app.app_id,accepted_utc=now(),setup_seconds=time.monotonic()-started)
            write(ledger)
            print(args.experiment,"accepted",app.app_id,flush=True)
            result=execute.remote(Path(candidate).name,repeats,seed,protocol,deadline)
            (ROOT / f"local-notes/modal-{args.experiment}-result.local.json").write_text(json.dumps(result,indent=2)+"\n")
            attempt.update(state="function-terminal",exit_code=result["exit_code"],worker_wall_seconds=result["worker_wall_seconds"])
    except Exception as error:
        attempt.update(state="needs-status-reconciliation",app_id=app.app_id,error_type=type(error).__name__)
        raise
    finally:
        attempt.update(client_finished_utc=now(),client_wall_seconds=time.monotonic()-started)
        write(ledger)
        print("Client exited; verify stopped and reconcile billing before next run",flush=True)


if __name__=="__main__":
    main()
