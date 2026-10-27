"""Bounded P046/P050 QKV-only route expansion using the existing image and login.
No official check or hardware counters; separate Torch activity only; one ephemeral H200 invocation with reviewed files.
"""
import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from projection_p044_modal import (now, require, save, resource_rate, gross,
    validate_preflight, configure_existing_auth, cli, all_stopped, unpack_artifacts,
    MAX_PACKED, CHUNK)

ROOT = Path(__file__).resolve().parent
IMAGE = "im-cPTbGpNiX3RarKResPgX7h"
PROFILE = "xuefeng-zhu"
RATE_FLOOR = Decimal("0.001305166667")
SOURCE = "candidates/projection_qkv_zero_bias.py"
SOURCE_SHA256 = "b08b6e371fef83851268a8dbfa1159d1b6d5a8e141a410a596dfc3f159e0da8b"
PARENT = "candidates/projection_pooler_column_batch2_hybrid.py"
PARENT_SHA256 = "001ea12e8764564d507de51e433769f5bd9a9916bfc4c014f0f87522c278545e"
PAYLOAD = (SOURCE, PARENT, "projection_p050_qkv_screen.py", "projection_p046_confirmation.py", "benchmark.py", "harness.py",
    "official/reference.py", "official/checks.py", "official/muse_bf16_proj.json",
    "LICENSE", "THIRD_PARTY_NOTICES.md", "licenses/Triton-MIT.txt",
    "licenses/June-9-Researcher-Reciprocity.txt")


def reserve(rate=RATE_FLOOR):
    # 45s startup + 195s function hard bound + 30s teardown/resolution margin.
    return Decimal(270) * max(rate, RATE_FLOOR) * Decimal("1.1") + Decimal(".02")


def budget_guard(preflight, billing, rates):
    require(Decimal(billing["billed_cost"]) == 0, "Actual paid usage must remain zero")
    rate = max(RATE_FLOOR, resource_rate(rates))
    cost_reserve = reserve(rate)
    free_floor = Decimal(preflight["verified_credit_balance_usd"]) - max(
        Decimal(0), gross(billing) - Decimal(preflight["total_usage_usd"]))
    require(free_floor > cost_reserve, "Verified free credits cannot cover conservative reserve")
    return rate, cost_reserve, free_floor


def frozen_payload(head, root=ROOT):
    require(bool(re.fullmatch(r"[0-9a-f]{40}", head)), "Exact reviewed commit required")
    require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                    text=True).strip() == head, "HEAD differs from freeze")
    hashes = {}
    for name in (*PAYLOAD, "projection_p050_modal.py", "projection_p044_modal.py"):
        path = root / name
        require(path.is_file() and not path.is_symlink(), "Payload missing or symlink: " + name)
        raw = path.read_bytes()
        require(raw == subprocess.check_output(["git", "show", head + ":" + name], cwd=root),
                "Uncommitted or changed payload: " + name)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    require(hashes[SOURCE] == SOURCE_SHA256 and hashes[PARENT] == PARENT_SHA256,
            "Frozen P046/P050 source mismatch")
    return {name: root / name for name in PAYLOAD}, hashes


def make_remote_entry():
    # No control-root, credentials, SDK client, config, or local state closure.
    def entry():
        import base64, hashlib, json, os, signal, subprocess, sys, time, zlib
        from pathlib import Path
        root = Path("/opt/projection-p050-screen")
        dest = Path("/tmp/projection-p050-screen")
        dest.mkdir(exist_ok=False)
        started = time.monotonic()
        stages = []
        for mode in ("timing",):
            command = [sys.executable, str(root / "projection_p050_qkv_screen.py"),
                       "--output", str(dest / (mode + ".json")), "--deadline-seconds", "120",
                       "--outer-timeout-seconds", "180"]
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
            complete=len(stages) == 1 and all(s["exit_code"] == 0 for s in stages),
            worker_wall_seconds=time.monotonic() - started)))
        files = sorted(dest.glob("*"))
        # Independent remote-side allowlist before any artifact is returned.
        def allowed(name):
            return name in {"summary.json", "timing.json", "timing.log"}
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



def launch(head, control_root):
    control_root = control_root.resolve()
    record_path = ROOT / "experiments/projection-p050-screen-modal.json"
    artifacts = ROOT / "local-notes/projection-p050-screen-artifacts"
    require(not record_path.exists() and not record_path.is_symlink()
            and not artifacts.exists() and not artifacts.is_symlink(),
            "Existing diagnostic state; reconcile, never relaunch")
    subprocess.run(["git", "check-ignore", "-q", "local-notes/projection-p050-screen-artifacts/summary.json"],
                   cwd=ROOT, capture_output=True, check=True)
    files, hashes = frozen_payload(head)
    auth, expiry = configure_existing_auth(control_root)
    pre_path = control_root / "local-notes/projection-p050-screen-preflight.local.json"
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
        image = image.add_local_file(path, "/opt/projection-p050-screen/" + name, copy=False)
    app = modal.App("okc-projection-p050-screen-diagnostic", include_source=False)
    fn = app.function(image=image, gpu="H200", cpu=(2, 2), memory=(8192, 8192),
        min_containers=0, max_containers=1, buffer_containers=0, scaledown_window=2,
        timeout=195, startup_timeout=45, retries=None, serialized=True,
        include_source=False, block_network=True, restrict_modal_access=True)(make_remote_entry())
    record = dict(experiment="P050 direct P046/P050 QKV-only route expansion", state="reserved", app_id=None,
        created_utc=now(), source_commit=head, source_sha256=SOURCE_SHA256, parent_sha256=PARENT_SHA256, base_image=IMAGE,
        payload_sha256={n: hashes[n] for n in PAYLOAD}, controller_sha256=hashes["projection_p050_modal.py"],
        cost_reserve_usd=str(cost_reserve), verified_free_credit_floor_usd=str(free_floor),
        resource_rate_usd_per_second=str(rate), paid_limit_usd="0", billing_before=billing,
        authentication_expiry_utc=expiry, preflight_sha256=hashlib.sha256(pre_path.read_bytes()).hexdigest(),
        shared_controller_sha256=hashes["projection_p044_modal.py"],
        resources=dict(gpu="H200", cpu=[2, 2], memory_mib=[8192, 8192], max_containers=1,
            min_containers=0, buffer_containers=0, scaledown_window=2, function_timeout=195,
            startup_timeout=45, child_internal_timeout=120, child_hard_timeout=180,
            app_watchdog=240, stop_allowance=15, retry_policy=None, explicit_invocations=1),
        no_hardware_counters=True, not_official_score=True)
    # The exclusive record is the durable no-duplicate reservation. If interrupted
    # between its write and the slot update, it still prevents another launch.
    with record_path.open("x") as stream:
        stream.write(json.dumps(record, indent=2) + "\n")
    save(slot, dict(state="reserved", provider="modal", experiment="P050 screen", app_id=None))
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

    timer = threading.Timer(240, watchdog)
    timer.daemon = True
    try:
        timer.start()
        with app.run(detach=False, environment_name="main"):
            record.update(state="accepted", app_id=app.app_id, accepted_utc=now())
            save(record_path, record)
            save(slot, dict(state="accepted", provider="modal", experiment="P050 screen", app_id=app.app_id))
            print("Accepted P050 screen app " + app.app_id, flush=True)
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
        print("P050 screen needs reconciliation of the existing request; no relaunch.", flush=True)
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
        save(slot, dict(state="terminal" if terminal else "uncertain", provider="modal", experiment="P050 screen", app_id=app.app_id))
        print(json.dumps({k: record.get(k) for k in ("app_id", "state", "diagnostic_complete", "shutdown_verified", "paid_after_zero")}), flush=True)
    return 0 if terminal and record.get("diagnostic_complete") and record.get("artifact_retrieval_complete") else 1



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("self-test", "launch"))
    parser.add_argument("--frozen-head")
    parser.add_argument("--control-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "self-test":
        require(len(PAYLOAD) == 13 and len(set(PAYLOAD)) == 13, "Payload allowlist changed")
        require(all("local-notes" not in p and "auth" not in p for p in PAYLOAD), "Unsafe payload")
        require(hashlib.sha256((ROOT / SOURCE).read_bytes()).hexdigest() == SOURCE_SHA256, "P050 changed")
        require(hashlib.sha256((ROOT / PARENT).read_bytes()).hexdigest() == PARENT_SHA256, "P046 changed")
        print("4 local guards passed; no Modal SDK import or external call.")
        return 0
    require(args.frozen_head is not None, "Launch requires reviewed committed freeze")
    return launch(args.frozen_head, args.control_root)


if __name__ == "__main__":
    raise SystemExit(main())
