"""Frozen single-H200 diagnostic; offline unless --launch is explicit.

Only eight named source/license files are uploaded. Existing Modal credentials remain
in the supported local config; historical accounting is reused, never refreshed.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
from decimal import Decimal
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import threading
import time
import warnings

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "attention-fusion-lookahead"
ATTENTION = REPO / "reviews/attention-v2"
ORIGINAL = Path("/Users/frank/Documents/Codex/2026-10-23/task")
AUTH = ORIGINAL / "local-notes/modal-auth.toml"
PYTHON = ORIGINAL / ".venv/bin/python"
PRIOR_RECEIPT = ATTENTION / "locality-b32l8-probe/billing.local.json"
EXPERIMENT = "attention-vjeux-v298-champion-comparison"
IMAGE = "im-cPTbGpNiX3RarKResPgX7h"
WORKER_SECONDS, CHILD_SECONDS = 540, 560
FUNCTION_SECONDS, STARTUP_SECONDS, WATCHDOG_SECONDS = 580, 20, 610
RESERVE, PRIOR_USAGE_ALLOWANCE = Decimal("2.00"), Decimal("10.00")
MAX_TOTAL, MAX_FILE = 24 * 1024 * 1024, 16 * 1024 * 1024
MAX_PACKED = MAX_TOTAL + 1024 * 1024
EXPECTED = {
    "selected.py": "3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8",
    "v298.py": "0cc4a0668acf7553c0d2c6d5c3a2e7d5c85374a70ad774463e9d2fd9e5d0f883",
    "reference.py": "386f7f67a1d788501e8fb9005d340423d098a0946e23810ba10878228c11a37b",
    "checks.py": "9a046c3c38db28b325dcd1557baecdf6bc9ef941d2175f9a900ba4c230bb8055",
    "LICENSE": "1495e1e757ef4d0925a2350563cf5754bb23c51701a8ec4fb3c5cdcbedae6747",
    "THIRD_PARTY_NOTICES.md": "c0417e70786e1094b0741a8b569927d8469cc742bf16e63a2fda2d59ad01a88b",
}
ARTIFACT_NAMES = frozenset({"summary.json", "supervisor.log"})

MAX_ENCODING = 4 * ((MAX_PACKED + 2) // 3) + 4 * len(ARTIFACT_NAMES)


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def regular_bytes(path, limit=MAX_FILE):
    path = Path(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
        raise ValueError("Regular bounded file required: " + path.name)
    raw = path.read_bytes()
    if len(raw) > limit:
        raise ValueError("File grew beyond bound: " + path.name)
    return raw


def save(path, value, exclusive=False):
    raw = (json.dumps(value, indent=2) + "\n").encode()
    path = Path(path)
    if exclusive:
        with path.open("xb") as stream:
            stream.write(raw)
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("xb") as stream:
            stream.write(raw)
        temporary.replace(path)


def payload():
    return {"worker.py": HERE / "worker.py", "selected.py": ATTENTION / "fusion-lookahead-probe/candidate.py",
        "v298.py": HERE / "upstream/muse_attention_decode.py", "reference.py": ATTENTION / "official/reference.py",
        "checks.py": ATTENTION / "official/checks.py", "LICENSE": REPO / "LICENSE",
        "THIRD_PARTY_NOTICES.md": REPO / "THIRD_PARTY_NOTICES.md", "UPSTREAM_LICENSE": HERE / "upstream/LICENSE"}


def historical_accounting():
    raw = regular_bytes(PRIOR_RECEIPT, 16384)
    receipt = json.loads(raw)
    if (receipt.get("workspace") != "xuefeng-zhu" or receipt.get("changes_made") is not False
            or Decimal(receipt["verified_net_spend_cap_usd"]) != 0
            or Decimal(receipt["total_spend_usd"]) != 0):
        raise ValueError("Prior verified paid-$0 guardrail required")
    balance = Decimal(receipt["verified_credit_balance_usd"])
    estimate = balance - PRIOR_USAGE_ALLOWANCE - RESERVE
    if estimate < 0:
        raise ValueError("Prior balance cannot cover conservative allowances")
    return {"receipt_sha256": digest(raw), "prior_observed_utc": receipt["observed_utc"],
        "historical_verified_free_balance_usd": str(balance), "prior_usage_allowance_usd": str(PRIOR_USAGE_ALLOWANCE),
        "new_job_reserve_usd": str(RESERVE), "modeled_balance_after_allowances_usd": str(estimate),
        "paid_limit_usd": "0", "fresh_billing_queried": False, "actual_charge_usd": None,
        "qualification": "Reused historical receipt; allowances and modeled remainder are not measured spend or a live balance."}


def make_manifest():
    hashes = {}
    for name, path in payload().items():
        raw = regular_bytes(path)
        hashes[name] = digest(raw)
        if name.endswith(".py"):
            compile(raw, name, "exec")
    for name, expected in EXPECTED.items():
        if hashes[name] != expected:
            raise ValueError("Original source binding mismatch: " + name)
    local = {name: digest(regular_bytes(HERE / name)) for name in ("controller.py", "test_controller.py", "PLAN.md", "test_worker.py")}
    return {"version": 1, "experiment": EXPERIMENT, "base_image": IMAGE, "payload_sha256": hashes,
        "local_sha256": local, "accounting": historical_accounting(),
        "resources": {"gpu": "H200", "cpu": [4, 4], "memory_mib": [16384, 16384], "max_containers": 1,
            "worker_deadline": WORKER_SECONDS, "supervisor_timeout": CHILD_SECONDS,
            "function_timeout": FUNCTION_SECONDS, "startup_timeout": STARTUP_SECONDS,
            "watchdog": WATCHDOG_SECONDS, "retries": 0, "invocations": 1},
        "official_check": False, "formal_entry": False, "candidate_change": False}


def verify_manifest(path):
    manifest = json.loads(regular_bytes(path, 65536))
    if manifest != make_manifest():
        raise ValueError("Frozen source, protocol, or accounting changed; no upload")
    return manifest


def configure_existing_auth():
    info = AUTH.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
        raise ValueError("Existing protected supported config required")
    if any(os.environ.get(key) for key in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "MODAL_OAUTH_REFRESH_TOKEN",
            "MODAL_OAUTH_CLIENT_ID", "MODAL_OAUTH_CLIENT_SECRET")):
        raise ValueError("Credential environment overrides forbidden")
    os.environ["MODAL_CONFIG_PATH"], os.environ["MODAL_PROFILE"] = str(AUTH), "xuefeng-zhu"


def cli(*args):
    if args != ("app", "list", "--env", "main", "--json") and not (
            len(args) == 6 and args[:2] == ("app", "stop") and re.fullmatch(r"ap-[A-Za-z0-9]+", args[2])
            and args[3:] == ("--yes", "--env", "main")):
        raise ValueError("Only read-only app-list and owned-app stop permitted")
    code = ("import warnings; from modal.exception import ServerWarning; "
        "warnings.filterwarnings('ignore',message=r'Your Modal token .* expires',category=ServerWarning); "
        "from modal.__main__ import main; main()")
    child = subprocess.run([str(PYTHON), "-c", code, *args], env={**os.environ, "TZ": "UTC"},
        capture_output=True, text=True, timeout=25)
    if child.returncode:
        raise RuntimeError("Supported Modal app operation failed; raw auth output withheld")
    return json.loads(child.stdout) if args[1] == "list" else {"exit_code": 0}


def stopped(rows):
    """Require stopped state and zero tasks in either observed CLI encoding.

    Modal emits task counts as ASCII decimal strings. Accept only integer zero
    or an all-zero ASCII decimal string; booleans and coercible values fail.
    """
    return all(row.get("state") == "stopped" and (
        (type(row.get("tasks")) is int and row["tasks"] == 0)
        or (type(row.get("tasks")) is str and re.fullmatch(r"0+", row["tasks"]) is not None)
    ) for row in rows)


def observe_shutdown(app_id, attempts=6, interval=2):
    observations = []
    for index in range(attempts):
        rows = cli("app", "list", "--env", "main", "--json")
        exact = next((row for row in rows if row.get("app_id") == app_id), None)
        observations.append({"observed_utc": now(), "app": exact})
        if exact is not None and stopped([exact]):
            return rows, exact, observations
        if index + 1 < attempts:
            time.sleep(interval)
    return rows, exact, observations


def make_remote_entry():
    # Closure-free. No credential/config paths, environment, or local receipts.
    def entry():
        import base64, gzip, hashlib, os, signal, stat, subprocess, sys, time
        from pathlib import Path
        root, output = Path("/opt/vjeux-attention-v298-trial"), Path("/tmp/vjeux-attention-v298-trial")
        output.mkdir(exist_ok=True)
        names = ("worker.py", "selected.py", "v298.py", "reference.py", "checks.py", "LICENSE", "THIRD_PARTY_NOTICES.md", "UPSTREAM_LICENSE")
        sources = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
        allowed = {"summary.json", "supervisor.log"}
        env = {key: os.environ[key] for key in ("PATH", "HOME", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES", "CUDA_MODULE_LOADING") if key in os.environ}
        env.update(PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", MAX_JOBS="2",
            TRITON_CACHE_DIR="/tmp/vjeux-attention-v298-triton", TORCH_EXTENSIONS_DIR="/tmp/vjeux-attention-v298-extensions")
        started = time.monotonic()
        with (output / "supervisor.log").open("wb") as log:
            child = subprocess.Popen([sys.executable, str(root / "worker.py"), "--deadline", "540",
                "--payload-root", str(root), "--output", str(output / "summary.json")], cwd=root,
                env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = child.wait(timeout=560)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait(timeout=3)
                code = 124
        reply = {"exit_code": code, "worker_wall_seconds": time.monotonic() - started,
            "mounted_sha256": sources, "artifacts": [], "artifact_errors": []}
        total, packed_total = 0, 0
        for name in sorted(allowed):
            path = output / name
            if not path.exists() and not path.is_symlink():
                continue
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_size > 16 * 1024 * 1024 or total + info.st_size > 24 * 1024 * 1024:
                reply["artifact_errors"].append({"name": name, "reason": "nonregular_or_oversize"})
                continue
            raw = path.read_bytes()
            if len(raw) != info.st_size:
                reply["artifact_errors"].append({"name": name, "reason": "changed_size"})
                continue
            packed = gzip.compress(raw, mtime=0)
            if packed_total + len(packed) > 25 * 1024 * 1024:
                reply["artifact_errors"].append({"name": name, "reason": "compressed_total_limit"})
                continue
            total += len(raw)
            packed_total += len(packed)
            reply["artifacts"].append({"name": name, "raw_bytes": len(raw), "raw_sha256": hashlib.sha256(raw).hexdigest(),
                "gzip_bytes": len(packed), "gzip_sha256": hashlib.sha256(packed).hexdigest(),
                "gzip_base64": base64.b64encode(packed).decode("ascii")})
        return reply
    return entry


def unpack_artifacts(reply, destination):
    if not isinstance(reply, dict) or type(reply.get("exit_code")) is not int or not isinstance(reply.get("artifacts"), list):
        raise ValueError("Invalid bounded remote reply")
    entries, names = reply["artifacts"], []
    if len(entries) > len(ARTIFACT_NAMES):
        raise ValueError("Too many artifacts")
    for row in entries:
        if not isinstance(row, dict) or row.get("name") not in ARTIFACT_NAMES or row["name"] in names:
            raise ValueError("Unknown, unsafe, or duplicate artifact name")
        names.append(row["name"])
    destination = Path(destination)
    destination.mkdir(exist_ok=False)
    if destination.is_symlink():
        raise ValueError("Artifact directory must not be a symlink")
    metadata = {row["name"]: {key: value for key, value in row.items() if key != "gzip_base64"} for row in entries}
    save(destination / "transport-metadata.json", metadata, exclusive=True)
    totals = {"raw": 0, "packed": 0, "encoded": 0}
    for row in entries:
        encoded = row.get("gzip_base64")
        if not isinstance(encoded, str):
            raise ValueError("Missing base64 artifact")
        totals["encoded"] += len(encoded)
        if totals["encoded"] > MAX_ENCODING:
            raise ValueError("Encoded artifact bound")
        packed = base64.b64decode(encoded, validate=True)
        totals["packed"] += len(packed)
        if totals["packed"] > MAX_PACKED or len(packed) != row.get("gzip_bytes") or digest(packed) != row.get("gzip_sha256"):
            raise ValueError("Compressed artifact size/digest mismatch")
        with (destination / (row["name"] + ".transport.gz")).open("xb") as stream:
            stream.write(packed)
        with gzip.GzipFile(fileobj=io.BytesIO(packed)) as stream:
            raw = stream.read(min(MAX_FILE, MAX_TOTAL - totals["raw"]) + 1)
        totals["raw"] += len(raw)
        if (len(raw) > MAX_FILE or totals["raw"] > MAX_TOTAL or len(raw) != row.get("raw_bytes")
                or digest(raw) != row.get("raw_sha256")):
            raise ValueError("Raw artifact size/digest mismatch")
        with (destination / row["name"]).open("xb") as stream:
            stream.write(raw)
    return metadata


def validate_report(report, hashes, artifacts):
    if report.get("worker_sha256") != hashes["worker.py"]:
        raise ValueError("Worker identity mismatch")
    if report.get("verified_hashes") != {k:hashes[k] for k in ("selected.py", "v298.py", "reference.py", "checks.py")}:
        raise ValueError("Frozen source binding mismatch")
    if report.get("status") != "complete":
        return False
    if any(report.get(k) is not False for k in ("candidate_changed", "official_check", "ranked_submission")):
        raise ValueError("Original candidate/official action boundary changed")
    if report.get("correctness_complete") is not True or report.get("timing_complete") is not True:
        raise ValueError("Original oracle/full paired coverage incomplete")
    checked=lambda r:all(r.get(k) is True for k in ("passed","ownership","fixed_addresses"))
    prerequisites=report.get("prerequisite_cases",[])
    if report.get("prerequisites_complete") is not True or len(prerequisites)!=16:
        raise ValueError("All sixteen prerequisite cases required")
    if any(set(row.get("arms",{}))!={"champion","v298"} or any(not checked(r) for r in row["arms"].values()) for row in prerequisites):
        raise ValueError("Both arms must pass every prerequisite")
    pairs=report.get("paired_cases",[])
    expected={(seed,b,l,k) for seed in (135901,135902) for b in (1,16,32) for l in (1,8) for k in ("sliding","full")}
    got={(r.get("seed"),r.get("case",{}).get("batch"),r.get("case",{}).get("q_len"),r.get("case",{}).get("layer")) for r in pairs}
    if len(pairs)!=24 or got!=expected:
        raise ValueError("Exact twelve shapes and two fresh fixtures required")
    for row in pairs:
        for key in ("oracle","mutation_oracle","post_oracle"):
            if set(row.get(key,{}))!={"champion","v298"} or any(not checked(r) for r in row[key].values()):
                raise ValueError("Original/replay/mutation oracle missing")
        if any(r.get("timed") is not False for r in row["mutation_oracle"].values()):
            raise ValueError("Mutation must remain untimed")
        samples=row.get("samples",[])
        if len(samples)!=12 or any(set(r)!={"champion","v298"} or any(type(v) not in (float,int) or not (0<v<float("inf")) for v in r.values()) for r in samples):
            raise ValueError("Six finite positive reverse pairs required")
    suites=report.get("whole_suite",[])
    if len(suites)!=2 or {r.get("seed") for r in suites}!={135901,135902}:
        raise ValueError("Both whole-suite reconstructions required")
    return True


def materialize_result(reply, destination, hashes):
    artifacts = unpack_artifacts(reply, destination)
    if reply.get("mounted_sha256") != hashes:
        raise ValueError("Remote mounted sources differ from frozen upload")
    path = Path(destination) / "summary.json"
    report = json.loads(path.read_text()) if path.is_file() else None
    complete = validate_report(report, hashes, artifacts) if isinstance(report, dict) else False
    return {"completion_verified": complete and reply["exit_code"] == 0 and not reply.get("artifact_errors"),
        "report": report, "artifacts_retrieved": sorted(artifacts), "artifact_errors": reply.get("artifact_errors", [])}


def launch(manifest_path):
    ledger, destination = HERE / "run.local.json", HERE / "artifacts"
    if ledger.exists() or destination.exists():
        raise ValueError("Existing request/result requires reconciliation; never duplicate")
    manifest = verify_manifest(manifest_path)
    configure_existing_auth()
    import modal
    from modal.exception import ServerWarning
    warnings.filterwarnings("ignore", message=r"Your Modal token .* expires", category=ServerWarning)
    if Path(modal.config.user_config_path).resolve() != AUTH.resolve():
        raise ValueError("Supported existing config path mismatch")
    before = cli("app", "list", "--env", "main", "--json")
    if not stopped(before):
        raise ValueError("Another Modal app/task owns the serialized slot")
    # Exclusive ledger precedes any upload or app invocation.
    record = {"state": "reserved", "created_utc": now(), "manifest_sha256": digest(regular_bytes(manifest_path)),
        "manifest": manifest, "apps_before": before, "app_id": None, "invocation_attempted": False}
    save(ledger, record, exclusive=True)
    image = modal.Image.from_id(IMAGE)
    for name, path in payload().items():
        image = image.add_local_file(path, "/opt/vjeux-attention-v298-trial/" + name, copy=False)
    app = modal.App(EXPERIMENT, include_source=False)
    fn = app.function(image=image, gpu="H200", cpu=(4, 4), memory=(16384, 16384), min_containers=0,
        max_containers=1, buffer_containers=0, scaledown_window=2, timeout=FUNCTION_SECONDS,
        startup_timeout=STARTUP_SECONDS, retries=0, serialized=True, include_source=False,
        block_network=True, restrict_modal_access=True)(make_remote_entry())
    lock, expired, stop_result = threading.Lock(), threading.Event(), {}

    def stop():
        with lock:
            if not app.app_id or stop_result.get("exit_code") == 0:
                return
            stop_result.update(app_id=app.app_id, requested_utc=now())
            try:
                stop_result.update(cli("app", "stop", app.app_id, "--yes", "--env", "main"))
            except Exception as exc:
                stop_result["error_type"] = type(exc).__name__

    def watchdog():
        expired.set()
        stop()

    started = time.monotonic()
    timer = threading.Timer(WATCHDOG_SECONDS, watchdog)
    timer.daemon = True
    try:
        timer.start()
        with app.run(detach=False, environment_name="main"):
            record.update(state="accepted", app_id=app.app_id, accepted_utc=now())
            save(ledger, record)
            print("Accepted one H200 diagnostic " + str(app.app_id), flush=True)
            if expired.is_set():
                raise TimeoutError("Watchdog expired before invocation")
            verify_manifest(manifest_path)
            record.update(state="invocation-attempted", invocation_attempted=True)
            save(ledger, record)
            reply = fn.remote()
            outcome = materialize_result(reply, destination, manifest["payload_sha256"])
            report = outcome.pop("report")
            record.update(state="function-terminal", **outcome, exit_code=reply["exit_code"],
                worker_status=(report or {}).get("status"), worker_wall_seconds=reply.get("worker_wall_seconds"))
            save(ledger, record)
    except Exception as exc:
        record.update(state="needs-reconciliation", app_id=app.app_id, error_type=type(exc).__name__)
        print("Diagnostic requires reconciliation; do not relaunch.", flush=True)
    finally:
        stop()
        timer.cancel()
        try:
            rows, exact, observations = observe_shutdown(app.app_id)
            record.update(shutdown_observations=observations, shutdown_verified=bool(exact and stopped([exact])),
                all_apps_stopped_zero_tasks=stopped(rows))
        except Exception as exc:
            record.update(shutdown_verified=False, shutdown_error_type=type(exc).__name__)
        record.update(explicit_stop=stop_result, client_finished_utc=now(), client_wall_seconds=time.monotonic() - started)
        save(ledger, record)
        print(json.dumps({key: record.get(key) for key in ("app_id", "state", "exit_code", "worker_status",
            "completion_verified", "shutdown_verified", "all_apps_stopped_zero_tasks")}), flush=True)
    return 0 if record.get("completion_verified") and record.get("shutdown_verified") else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--freeze", type=Path, help="Write an immutable offline source/accounting manifest")
    mode.add_argument("--launch", action="store_true", help="Explicit single bounded native invocation")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if args.freeze:
        save(args.freeze, make_manifest(), exclusive=True)
        print("Offline source/accounting manifest frozen; no app created.")
    elif args.manifest:
        if args.launch:
            raise SystemExit(launch(args.manifest))
        print(json.dumps(verify_manifest(args.manifest), indent=2))
    else:
        parser.error("Use --freeze PATH, --manifest PATH, or --launch --manifest PATH")
