"""Frozen single-H200 diagnostic; offline unless --launch is explicit.

Only seven named source files are uploaded. Existing Modal credentials remain
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
EXPERIMENT = "attention-kda-h200-baseline-v4-scalar-repair"
IMAGE = "im-cPTbGpNiX3RarKResPgX7h"
WORKER_SECONDS, CHILD_SECONDS = 260, 280
FUNCTION_SECONDS, STARTUP_SECONDS, WATCHDOG_SECONDS = 300, 20, 330
RESERVE, PRIOR_USAGE_ALLOWANCE = Decimal("1.20"), Decimal("10.00")
MAX_TOTAL, MAX_FILE = 24 * 1024 * 1024, 16 * 1024 * 1024
MAX_PACKED = MAX_TOTAL + 1024 * 1024
EXPECTED = {
    "selected.py": "3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8",
    "v4.py": "dcfbca2f8e1135762b08c77df931a7f53948f69746e0314797ee3dbef4147e82",
    "reference.py": "386f7f67a1d788501e8fb9005d340423d098a0946e23810ba10878228c11a37b",
    "checks.py": "9a046c3c38db28b325dcd1557baecdf6bc9ef941d2175f9a900ba4c230bb8055",
    "LICENSE": "1495e1e757ef4d0925a2350563cf5754bb23c51701a8ec4fb3c5cdcbedae6747",
    "THIRD_PARTY_NOTICES.md": "c0417e70786e1094b0741a8b569927d8469cc742bf16e63a2fda2d59ad01a88b",
}
STEMS = tuple(f"ncu-{arm}-b{batch}" for arm in ("champion", "v4") for batch in (16, 32))
ARTIFACT_NAMES = frozenset({"summary.json", "supervisor.log"}
    | {tool + suffix for tool in ("ncu-help", "ncu-version", "ncu-query")
       for suffix in (".txt", ".stdout.txt", ".stderr.txt")}
    | {stem + suffix for stem in STEMS for suffix in
       (".ncu-rep", ".csv", "-imported.csv", ".stdout.txt", ".stderr.txt", "-import.stdout.txt", "-import.stderr.txt", ".api.json")})
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
        "v4.py": ATTENTION / "user-v4-probe/submission_v4.py", "reference.py": ATTENTION / "official/reference.py",
        "checks.py": ATTENTION / "official/checks.py", "LICENSE": REPO / "LICENSE",
        "THIRD_PARTY_NOTICES.md": REPO / "THIRD_PARTY_NOTICES.md"}


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
    local = {name: digest(regular_bytes(HERE / name)) for name in ("controller.py", "test_controller.py", "PLAN.md")}
    return {"version": 1, "experiment": EXPERIMENT, "base_image": IMAGE, "payload_sha256": hashes,
        "local_sha256": local, "accounting": historical_accounting(),
        "resources": {"gpu": "H200", "cpu": [2, 2], "memory_mib": [8192, 8192], "max_containers": 1,
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
        root, output = Path("/opt/kda-h200-trial"), Path("/tmp/kda-h200-trial")
        output.mkdir(exist_ok=True)
        names = ("worker.py", "selected.py", "v4.py", "reference.py", "checks.py", "LICENSE", "THIRD_PARTY_NOTICES.md")
        sources = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
        allowed = {"summary.json", "supervisor.log"}
        for tool in ("ncu-help", "ncu-version", "ncu-query"):
            for suffix in (".txt", ".stdout.txt", ".stderr.txt"):
                allowed.add(tool + suffix)
        for arm in ("champion", "v4"):
            for batch in (16, 32):
                for suffix in (".ncu-rep", ".csv", "-imported.csv", ".stdout.txt", ".stderr.txt", "-import.stdout.txt", "-import.stderr.txt", ".api.json"):
                    allowed.add(f"ncu-{arm}-b{batch}" + suffix)
        env = {key: os.environ[key] for key in ("PATH", "HOME", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES", "CUDA_MODULE_LOADING") if key in os.environ}
        env.update(PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2",
            TRITON_CACHE_DIR="/tmp/kda-h200-triton", TORCH_EXTENSIONS_DIR="/tmp/kda-h200-extensions")
        started = time.monotonic()
        with (output / "supervisor.log").open("wb") as log:
            child = subprocess.Popen([sys.executable, str(root / "worker.py"), "--deadline", "260",
                "--payload-root", str(root), "--output", str(output / "summary.json")], cwd=root,
                env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = child.wait(timeout=280)
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
    if report.get("worker_sha256") != hashes["worker.py"] or report.get("verified_hashes") != {
            name: hashes[name] for name in ("selected.py", "v4.py", "reference.py", "checks.py")}:
        raise ValueError("Worker report source binding mismatch")
    if report.get("status") != "complete":
        return False
    if any(report.get(key) is not False for key in ("candidate_changed", "official_check", "ranked_submission")):
        raise ValueError("Diagnostic-only candidate/entry boundary changed")
    if any(report.get(key) is not True for key in ("correctness_complete", "timing_complete", "counters_complete")):
        raise ValueError("Diagnostic completion gates not all satisfied")
    if len(report.get("baseline_cases", [])) != 12 or len(report.get("paired_cases", [])) != 2:
        raise ValueError("Required whole-call coverage incomplete")
    baseline = report["baseline_cases"]
    shapes = {(row.get("case", {}).get("batch"), row.get("case", {}).get("q_len"),
               row.get("case", {}).get("layer")) for row in baseline}
    if shapes != {(batch, length, layer) for batch in (1, 16, 32) for length in (1, 8) for layer in ("sliding", "full")}:
        raise ValueError("Baseline scored-shape coverage changed")

    def checked(row):
        return all(row.get(key) is True for key in ("passed", "ownership", "fixed_addresses"))

    if any(not checked(row.get(key, {})) for row in baseline for key in ("oracle", "post_oracle")):
        raise ValueError("Baseline original oracle/cache/address checks incomplete")
    paired = report["paired_cases"]
    if {row.get("case", {}).get("batch") for row in paired} != {16, 32}:
        raise ValueError("Paired batch coverage changed")
    for row in paired:
        case = row.get("case", {})
        fixtures = row.get("fixtures", [])
        if (case.get("q_len") != 1 or case.get("layer") != "full" or case.get("page") != 1
                or len(fixtures) != 2 or [fixture.get("seed") for fixture in fixtures] != [135871, 135872]):
            raise ValueError("Declared two full-decode fixtures changed")
        for fixture in fixtures:
            for key in ("oracle", "mutation_oracle", "post_oracle"):
                rows_by_arm = fixture.get(key, {})
                if set(rows_by_arm) != {"champion", "v4"} or any(not checked(value) for value in rows_by_arm.values()):
                    raise ValueError("Paired original/mutation/post oracle evidence incomplete")
            if any(value.get("timed") is not False for value in fixture["mutation_oracle"].values()):
                raise ValueError("Mutation validation must remain untimed")
            samples = fixture.get("samples", [])
            if len(samples) != 12 or any(set(sample) != {"champion", "v4"} for sample in samples):
                raise ValueError("Six paired reverse-order measurements incomplete")
    rows = report.get("counter_reports", [])
    if len(rows) != 4 or {(r.get("arm"), r.get("batch")) for r in rows} != {
            (arm, batch) for arm in ("champion", "v4") for batch in (16, 32)}:
        raise ValueError("Required actual producers not all captured")
    for row in rows:
        stem = f"ncu-{row['arm']}-b{row['batch']}"
        name = stem + ".ncu-rep"
        expected_source = hashes["selected.py" if row["arm"] == "champion" else "v4.py"]
        artifact = artifacts.get(name, {})
        identity = row.get("identity", {})
        wanted_family = "_full2h" if row["arm"] == "v4" else "FlashAttnFwdSm90" if row["batch"] == 16 else "_fused"
        if (row.get("source_sha256") != expected_source or row.get("artifact_name") != name
                or artifact.get("raw_bytes", 0) <= 0 or row.get("artifact_bytes") != artifact.get("raw_bytes")
                or row.get("artifact_sha256") != artifact.get("raw_sha256")):
            raise ValueError("Native report source/artifact binding failed")
        if (row.get("roundtrip_passed") is not True or row.get("child_checks") != {"pre": True, "post": True}
                or row.get("capture_csv") != stem + ".csv" or row.get("import_csv") != stem + "-imported.csv"
                or row["capture_csv"] not in artifacts or row["import_csv"] not in artifacts
                or any(artifacts[row[key]].get("raw_bytes", 0) <= 0 for key in ("capture_csv", "import_csv"))
                or wanted_family not in identity.get("kernel_name", "")
                or any(not isinstance(identity.get(key), list) or len(identity[key]) != 3
                    or any(type(dim) is not int or dim <= 0 for dim in identity[key]) for key in ("grid", "block"))):
            raise ValueError("Native identity, correctness, or report roundtrip incomplete")
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
        image = image.add_local_file(path, "/opt/kda-h200-trial/" + name, copy=False)
    app = modal.App(EXPERIMENT, include_source=False)
    fn = app.function(image=image, gpu="H200", cpu=(2, 2), memory=(8192, 8192), min_containers=0,
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
