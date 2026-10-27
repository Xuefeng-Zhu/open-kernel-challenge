"""One explicitly bounded P039 diagnostic on the existing Modal image.

Local self-test imports no Modal SDK. Launch requires reviewed committed bytes,
fresh credit/cap evidence, an explicitly granted idle GPU slot, and existing auth.
No credential creation, retry, deployment, counter collection, or official check.
"""
import argparse
import base64
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import threading
import time
import zlib

ROOT = Path(__file__).resolve().parent
SOURCE = "candidates/projection_router_m1_fp32_gemv.py"
SOURCE_SHA256 = "0c0c41119ef925b03a33ccab230bc64176cc22c242e3fea77ce01e317376defc"
IMAGE = "im-cPTbGpNiX3RarKResPgX7h"
PROFILE = "xuefeng-zhu"
RATE_FLOOR = Decimal("0.00130496")
RESERVE_SECONDS = 480
MAX_RAW = 8 * 1024 * 1024
MAX_PACKED = 2 * 1024 * 1024
CHUNK = 16 * 1024
PAYLOAD = (SOURCE, "projection_p041_probe.py", "projection_p041_plan.py",
           "benchmark.py", "harness.py", "official/reference.py",
           "official/checks.py", "official/baseline.py", "official/muse_bf16_proj.json",
           "LICENSE", "THIRD_PARTY_NOTICES.md", "licenses/Triton-MIT.txt",
           "licenses/June-9-Researcher-Reciprocity.txt")


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save(path, data):
    """Atomic replacement of one record; reservation creation uses exclusive open."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def reserve(rate=RATE_FLOOR):
    return Decimal(RESERVE_SECONDS) * max(rate, RATE_FLOOR) * Decimal("1.1") + Decimal(".02")


def resource_rate(rates):
    return (Decimal(rates["gpu_hour_cost_h200"]) + 2 * Decimal(rates["cpu_hour_cost"])
            + 8 * Decimal(rates["mem_gib_hour_cost"])) / 3600


def gross(billing):
    return max(Decimal(billing["metered_cost"]), sum(
        (Decimal(v) for v in billing["metered_cost_breakdown"].values()), Decimal(0)))


def budget_guard(preflight, billing, rates):
    require(Decimal(billing["billed_cost"]) == 0, "Actual paid usage must remain zero")
    rate = max(RATE_FLOOR, resource_rate(rates))
    cost_reserve = reserve(rate)
    free_floor = Decimal(preflight["verified_credit_balance_usd"]) - max(
        Decimal(0), gross(billing) - Decimal(preflight["total_usage_usd"]))
    require(free_floor > cost_reserve, "Verified free credits cannot cover conservative reserve")
    return rate, cost_reserve, free_floor


def validate_preflight(data, observed=None):
    observed = observed or datetime.now(timezone.utc)
    stamp = datetime.fromisoformat(data["observed_utc"].replace("Z", "+00:00"))
    require(stamp.tzinfo is not None, "Preflight timestamp needs a timezone")
    require(-5 <= (observed - stamp).total_seconds() < 900, "Fresh credit/cap evidence required")
    require(data["workspace"] == PROFILE, "Wrong workspace")
    require(data["gpu_slot_granted"] is True, "Coordinator GPU slot not granted")
    require(Decimal(data["verified_net_spend_cap_usd"]) == 0, "Net paid cap must be zero")
    require(Decimal(data["total_spend_usd"]) == 0, "Paid usage must be zero")
    require(Decimal(data["verified_credit_balance_usd"]) > 0, "No verified free credits")
    require(Decimal(data["total_usage_usd"]) >= 0, "Invalid usage snapshot")


def check_owned_file(path):
    require(path.is_file() and not path.is_symlink(), "Protected file is missing or not regular")
    st = path.stat()
    require(st.st_uid == os.getuid() and st.st_mode & 0o777 == 0o600,
            "Protected file must be owner-readable mode 0600")


def configure_existing_auth(control_root):
    auth = control_root / "local-notes/modal-auth.toml"
    check_owned_file(auth)  # Deliberately never read its bytes.
    blocked = ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "MODAL_OAUTH_REFRESH_TOKEN",
               "MODAL_OAUTH_CLIENT_ID", "MODAL_OAUTH_CLIENT_SECRET")
    require(not any(os.environ.get(k) for k in blocked), "Unexpected credential environment override")
    subprocess.run(["git", "check-ignore", "-q", "local-notes/modal-auth.toml"],
                   cwd=control_root, check=True, capture_output=True)
    require(not subprocess.check_output(["git", "ls-files", "local-notes/modal-auth.toml"],
                                       cwd=control_root), "Credential file is tracked")
    metadata = json.loads((control_root / "local-notes/modal-login-20261025-1536.local.json").read_text())
    expiry = datetime.strptime(metadata["verification"]["expires_utc"],
                               "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=timezone.utc)
    require(metadata["state"] == "verified" and
            (expiry - datetime.now(timezone.utc)).total_seconds() > 600,
            "Existing authentication expiry cannot cover this diagnostic")
    os.environ["MODAL_CONFIG_PATH"] = str(auth)
    os.environ["MODAL_PROFILE"] = PROFILE
    return auth, expiry.isoformat()


def cli(control_root, *args):
    # Expiry warnings contain a credential identifier. Never forward raw stdout
    # or stderr on failure; no authentication/configuration commands are exposed.
    require(args[:2] in (("billing", "summary"), ("billing", "rates"),
                        ("app", "list"), ("app", "stop")), "Unsupported CLI action")
    code = ('import warnings; from modal.exception import ServerWarning; '
            'warnings.filterwarnings("ignore", message=r"Your Modal token .* expires", '
            'category=ServerWarning); from modal.__main__ import main; main()')
    child = subprocess.run([str(control_root / ".venv/bin/python"), "-c", code, *args],
                           capture_output=True, text=True, timeout=15)
    if child.returncode:
        raise RuntimeError("Modal operation failed: " + " ".join(args[:2]))
    return json.loads(child.stdout) if "--json" in args else {"exit_code": child.returncode}


def frozen_payload(head, root=ROOT):
    require(bool(re.fullmatch(r"[0-9a-f]{40}", head)), "Exact reviewed commit required")
    require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                    text=True).strip() == head, "HEAD differs from freeze")
    hashes = {}
    for name in (*PAYLOAD, "projection_p044_modal.py"):
        path = root / name
        require(path.is_file() and not path.is_symlink(), "Payload missing or symlink: " + name)
        raw = path.read_bytes()
        require(raw == subprocess.check_output(["git", "show", head + ":" + name], cwd=root),
                "Uncommitted or changed payload: " + name)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    require(hashes[SOURCE] == SOURCE_SHA256, "P039 source mismatch")
    return {name: root / name for name in PAYLOAD}, hashes


def artifact_name_ok(name):
    if name in ("summary.json", "timing.json", "activity.json", "timing.log", "activity.log"):
        return True
    return bool(re.fullmatch(
        r"activity\.case0[0-4]\.(pooler|qkv_gate|unpooler)\.m(8|16|128|256)\."
        r"(candidate|baseline)\.trace\.json", name))


def make_remote_entry():
    # No control-root, credentials, SDK client, config, or local state closure.
    def entry():
        import base64, hashlib, json, os, signal, subprocess, sys, time, zlib
        from pathlib import Path
        root = Path("/opt/projection-p044")
        dest = Path("/tmp/projection-p044")
        dest.mkdir(exist_ok=False)
        started = time.monotonic()
        stages = []
        for mode in ("timing", "activity"):
            command = [sys.executable, str(root / "projection_p041_probe.py"), "--mode", mode,
                       "--output", str(dest / (mode + ".json")), "--deadline-seconds", "150",
                       "--outer-timeout-seconds", "180"]
            if mode == "timing":
                command.append("--include-warm-cache-diagnostic")
            stage_start = time.monotonic()
            child = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            try:
                output, _ = child.communicate(timeout=180)
                code = child.returncode
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                output, _ = child.communicate(timeout=2)
                code = 124
            (dest / (mode + ".log")).write_bytes(output[-4000:])
            stages.append(dict(mode=mode, exit_code=code, wall_seconds=time.monotonic() - stage_start))
            if code:
                break  # Preserve failure; no replacement run or dependent stage.
        (dest / "summary.json").write_text(json.dumps(dict(stages=stages,
            complete=len(stages) == 2 and all(s["exit_code"] == 0 for s in stages),
            worker_wall_seconds=time.monotonic() - started)))
        files = sorted(dest.glob("*"))
        # Independent remote-side allowlist before any artifact is returned.
        import re
        def allowed(name):
            return name in {"summary.json", "timing.json", "activity.json", "timing.log", "activity.log"} or bool(
                re.fullmatch(r"activity\.case0[0-4]\.(pooler|qkv_gate|unpooler)\.m(8|16|128|256)\."
                             r"(candidate|baseline)\.trace\.json", name))
        if any(not p.is_file() or p.is_symlink() or not allowed(p.name) for p in files):
            yield {"kind": "failure", "reason": "artifact_path_guard"}
            return
        raw_bytes = sum(p.stat().st_size for p in files)
        if raw_bytes > 8 * 1024 * 1024:
            yield {"kind": "failure", "reason": "artifact_raw_size_guard"}
            return
        encoded = json.dumps({p.name: base64.b64encode(p.read_bytes()).decode() for p in files},
                             sort_keys=True).encode()
        packed = zlib.compress(encoded, 9)
        if len(packed) > 2 * 1024 * 1024:
            yield {"kind": "failure", "reason": "artifact_compressed_size_guard"}
            return
        count = (len(packed) + 16383) // 16384
        yield {"kind": "manifest", "chunks": count, "packed_bytes": len(packed),
               "encoded_bytes": len(encoded), "artifact_bytes": raw_bytes,
               "sha256": hashlib.sha256(packed).hexdigest(),
               "artifact_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
        for i in range(count):
            yield {"kind": "chunk", "index": i, "data": packed[i * 16384:(i + 1) * 16384]}
        yield {"kind": "complete", "chunks": count}
    return entry


def unpack_artifacts(manifest, chunks, dest):
    require(not dest.exists(), "Artifact destination exists; reconcile instead of replacing")
    require(0 < manifest["chunks"] <= MAX_PACKED // CHUNK and
            len(chunks) == manifest["chunks"], "Invalid chunk count")
    require(all(isinstance(c, bytes) and 0 < len(c) <= CHUNK for c in chunks), "Invalid chunk size")
    packed = b"".join(chunks)
    require(len(packed) == manifest["packed_bytes"] <= MAX_PACKED, "Packed size mismatch")
    require(hashlib.sha256(packed).hexdigest() == manifest["sha256"], "Packed hash mismatch")
    # Base64 plus bounded filenames/JSON overhead; raw artifact ceiling is 8MiB.
    max_encoded = 12 * 1024 * 1024
    decoder = zlib.decompressobj()
    raw = decoder.decompress(packed, max_encoded + 1)
    require(decoder.eof and not decoder.unconsumed_tail and not decoder.unused_data,
            "Invalid or oversized compressed stream")
    require(len(raw) == manifest["encoded_bytes"] <= max_encoded, "Encoded size mismatch")
    content = json.loads(raw)
    require(isinstance(content, dict) and set(content) == set(manifest["artifact_sha256"]),
            "Artifact manifest mismatch")
    require("summary.json" in content and all(artifact_name_ok(n) for n in content), "Unsafe artifact path")
    decoded = {n: base64.b64decode(v, validate=True) for n, v in content.items()}
    require(sum(map(len, decoded.values())) == manifest["artifact_bytes"] <= MAX_RAW, "Raw size mismatch")
    require(all(hashlib.sha256(v).hexdigest() == manifest["artifact_sha256"][n]
                for n, v in decoded.items()), "Artifact hash mismatch")
    summary = json.loads(decoded["summary.json"])
    dest.mkdir(parents=True, exist_ok=False)
    for name, value in decoded.items():
        (dest / name).write_bytes(value)
    return summary


def all_stopped(apps):
    return all(a["state"] == "stopped" and int(a["tasks"]) == 0 for a in apps)


def launch(head, control_root):
    control_root = control_root.resolve()
    record_path = ROOT / "experiments/projection-p044-modal.json"
    artifacts = ROOT / "local-notes/projection-p044-artifacts"
    require(not record_path.exists() and not record_path.is_symlink()
            and not artifacts.exists() and not artifacts.is_symlink(),
            "Existing diagnostic state; reconcile, never relaunch")
    subprocess.run(["git", "check-ignore", "-q", "local-notes/projection-p044-artifacts/summary.json"],
                   cwd=ROOT, capture_output=True, check=True)
    files, hashes = frozen_payload(head)
    auth, expiry = configure_existing_auth(control_root)
    pre_path = control_root / "local-notes/projection-p044-preflight.local.json"
    require(pre_path.is_file() and not pre_path.is_symlink(), "Missing local preflight")
    pre = json.loads(pre_path.read_text())
    validate_preflight(pre)
    slot = control_root / "local-notes/gpu-slot.local.json"
    prior_slot = json.loads(slot.read_text())
    require(prior_slot["state"] in ("idle", "terminal"), "Existing GPU reservation needs reconciliation")
    require(not prior_slot.get("active_reservation"), "Existing active reservation")
    billing = cli(control_root, "billing", "summary", "--json")
    require(Decimal(billing["billed_cost"]) == 0, "Actual paid usage must remain zero")
    require(all_stopped(cli(control_root, "app", "list", "--env", "main", "--json")), "An app is still active")
    rates = cli(control_root, "billing", "rates", "--json")
    rate, cost_reserve, free_floor = budget_guard(pre, billing, rates)
    import warnings
    import modal
    from modal.exception import ServerWarning
    warnings.filterwarnings("ignore", message=r"Your Modal token .* expires", category=ServerWarning)
    require(Path(modal.config.user_config_path).resolve() == auth.resolve(), "SDK uses unexpected config")
    image = modal.Image.from_id(IMAGE)
    for name, path in files.items():
        image = image.add_local_file(path, "/opt/projection-p044/" + name, copy=False)
    app = modal.App("okc-projection-p044-diagnostic", include_source=False)
    fn = app.function(image=image, gpu="H200", cpu=(2, 2), memory=(8192, 8192),
        min_containers=0, max_containers=1, buffer_containers=0, scaledown_window=2,
        timeout=390, startup_timeout=45, retries=None, serialized=True,
        include_source=False, block_network=True, restrict_modal_access=True)(make_remote_entry())
    record = dict(experiment="P044 unchanged P039 diagnostic", state="reserved", app_id=None,
        created_utc=now(), source_commit=head, source_sha256=SOURCE_SHA256, base_image=IMAGE,
        payload_sha256={n: hashes[n] for n in PAYLOAD}, controller_sha256=hashes["projection_p044_modal.py"],
        cost_reserve_usd=str(cost_reserve), verified_free_credit_floor_usd=str(free_floor),
        resource_rate_usd_per_second=str(rate), paid_limit_usd="0", billing_before=billing,
        authentication_expiry_utc=expiry, preflight_sha256=hashlib.sha256(pre_path.read_bytes()).hexdigest(),
        resources=dict(gpu="H200", cpu=[2, 2], memory_mib=[8192, 8192], max_containers=1,
            min_containers=0, buffer_containers=0, scaledown_window=2, function_timeout=390,
            startup_timeout=45, child_internal_timeout=150, child_hard_timeout=180,
            app_watchdog=420, stop_allowance=15, retry_policy=None, explicit_invocations=1),
        no_hardware_counters=True, not_official_score=True)
    # The exclusive record is the durable no-duplicate reservation. If interrupted
    # between its write and the slot update, it still prevents another launch.
    with record_path.open("x") as stream:
        stream.write(json.dumps(record, indent=2) + "\n")
    save(slot, dict(state="reserved", provider="modal", experiment="P044", app_id=None))
    started = time.monotonic()
    stop_lock = threading.Lock()
    stop_result = {}
    expired = threading.Event()

    def stop():
        with stop_lock:
            if not app.app_id or stop_result:
                return
            stop_result.update(app_id=app.app_id, requested_utc=now())
            try:
                stop_result.update(cli(control_root, "app", "stop", app.app_id, "--yes", "--env", "main"))
            except Exception as error:
                stop_result["error_type"] = type(error).__name__

    def watchdog():
        expired.set()
        stop()

    timer = threading.Timer(420, watchdog)
    timer.daemon = True
    try:
        timer.start()
        with app.run(detach=False, environment_name="main"):
            record.update(state="accepted", app_id=app.app_id, accepted_utc=now())
            save(record_path, record)
            save(slot, dict(state="accepted", provider="modal", experiment="P044", app_id=app.app_id))
            print("Accepted P044 app " + app.app_id, flush=True)
            require(not expired.is_set(), "App watchdog elapsed before invocation")
            manifest, chunks, complete = None, [], False
            for item in fn.remote_gen():
                require(not complete, "Unexpected data after complete marker")
                if item["kind"] == "manifest":
                    require(manifest is None and 0 < item["packed_bytes"] <= MAX_PACKED,
                            "Invalid or repeated manifest")
                    manifest = item
                    record["transport_manifest"] = manifest
                    save(record_path, record)
                elif item["kind"] == "chunk":
                    require(manifest is not None and item["index"] == len(chunks), "Chunk ordering mismatch")
                    require(isinstance(item["data"], bytes) and 0 < len(item["data"]) <= CHUNK, "Chunk size invalid")
                    chunks.append(item["data"])
                    require(sum(map(len, chunks)) <= MAX_PACKED, "Packed-size guard")
                elif item["kind"] == "complete":
                    require(manifest is not None and item["chunks"] == len(chunks) == manifest["chunks"],
                            "Incomplete stream")
                    complete = True
                else:
                    raise RuntimeError("Remote worker/artifact guard failed")
            require(complete and manifest is not None, "Incomplete result; reconcile same app")
            summary = unpack_artifacts(manifest, chunks, artifacts)
            record.update(state="function-terminal", artifact_retrieval_complete=True,
                          diagnostic_complete=summary["complete"], stages=summary["stages"],
                          worker_wall_seconds=summary["worker_wall_seconds"])
            save(record_path, record)
            stop()
    except Exception as error:
        record.update(state="needs-reconciliation", app_id=app.app_id, error_type=type(error).__name__)
        print("P044 needs reconciliation of the existing request; no relaunch.", flush=True)
    finally:
        stop()
        timer.cancel()
        record.update(explicit_stop=stop_result, client_finished_utc=now(),
                      client_wall_seconds=time.monotonic() - started)
        try:
            apps = cli(control_root, "app", "list", "--env", "main", "--json")
            exact = next((a for a in apps if a["app_id"] == app.app_id), None)
            record["shutdown_observation"] = ({k: exact.get(k) for k in
                ("app_id", "state", "tasks", "created_at", "stopped_at")} if exact else None)
            record["shutdown_verified"] = bool(exact and all_stopped([exact]))
            record["all_apps_stopped_tasks_zero"] = all_stopped(apps)
            after = cli(control_root, "billing", "summary", "--json")
            record["billing_after"] = after
            record["paid_after_zero"] = Decimal(after["billed_cost"]) == 0
            record["observed_workspace_gross_delta_usd"] = str(max(Decimal(0), gross(after) - gross(billing)))
            record["billing_qualification"] = "Workspace snapshot may lag; not a settled per-app invoice; reserve retained"
        except Exception as error:
            record["postcheck_error_type"] = type(error).__name__
        terminal = record.get("shutdown_verified") and record.get("all_apps_stopped_tasks_zero") and record.get("paid_after_zero")
        record["state"] = "terminal" if terminal else "needs-reconciliation"
        save(record_path, record)
        save(slot, dict(state="terminal" if terminal else "uncertain", provider="modal", experiment="P044", app_id=app.app_id))
        print(json.dumps({k: record.get(k) for k in ("app_id", "state", "diagnostic_complete", "shutdown_verified", "paid_after_zero")}), flush=True)
    return 0 if terminal and record.get("diagnostic_complete") and record.get("artifact_retrieval_complete") else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("self-test", "launch"))
    parser.add_argument("--frozen-head")
    parser.add_argument("--control-root", type=Path,
                        default=Path(os.environ.get("OKC_CONTROL_ROOT", ROOT)))
    args = parser.parse_args(argv)
    if args.action == "self-test":
        require(reserve() == Decimal(".709018880"), "Reserve calculation mismatch")
        require(len(PAYLOAD) == 13 and all("local-notes" not in p for p in PAYLOAD), "Payload allowlist changed")
        require(hashlib.sha256((ROOT / SOURCE).read_bytes()).hexdigest() == SOURCE_SHA256, "P039 changed")
        print("3 local guards passed; no Modal SDK import or external call.")
        return 0
    require(args.frozen_head is not None, "Launch requires reviewed committed freeze")
    return launch(args.frozen_head, args.control_root)


if __name__ == "__main__":
    raise SystemExit(main())
