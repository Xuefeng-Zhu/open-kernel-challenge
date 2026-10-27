"""One P053 timing-then-activity diagnostic on the existing image/login.
Activity requires a validated complete timing/correctness receipt, not a win.
No official check or counters; launch requires reviewed committed payload bytes.
"""
import argparse
import base64
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
import zlib
from projection_p044_modal import (now, require, save, resource_rate, gross,
    validate_preflight, configure_existing_auth, cli, all_stopped, MAX_RAW, MAX_PACKED, CHUNK)

ROOT = Path(__file__).resolve().parent
IMAGE = "im-cPTbGpNiX3RarKResPgX7h"
PROFILE = "xuefeng-zhu"
RATE_FLOOR = Decimal("0.001305166667")
WORKER = "projection_p053_phase_screen.py"
TIMING_WORKER = "projection_p052_orientation_screen.py"
RECEIPT = "results/remote-P052-screen/timing.json"
CORRECTNESS_WORKER = "projection_p051_orientation.py"
REFERENCE = "candidates/pooler_triton_split8.py"
PARENT = "candidates/projection_pooler_column_batch2_hybrid.py"
FIXED = {
    TIMING_WORKER: "042c395d20ffa6b191629aa79d9b68b9766cb92ea94df56d7763e741d46cdd59",
    RECEIPT: "2ae054ec40d6f39a119dfe0c97fd330dfde09e315ba51d45b0cffb6cc71cd30f",
    CORRECTNESS_WORKER: "445ab7d44e36cdceafdf1ecdbe19d7fafa247d8033ab580ab38f74b98f3533db",
    REFERENCE: "76f202f5fe9efb23ce0e10c6739871fab74512464a0e9f1229cda53c0c625406",
    PARENT: "001ea12e8764564d507de51e433769f5bd9a9916bfc4c014f0f87522c278545e",
    "official/reference.py": "e62491abc9ddcad33f5faba8b0ee74cf6370b7be5dd2dc05031742f3ebf7147d",
    "official/checks.py": "9a046c3c38db28b325dcd1557baecdf6bc9ef941d2175f9a900ba4c230bb8055",
}
PAYLOAD = (WORKER, TIMING_WORKER, CORRECTNESS_WORKER, REFERENCE, PARENT, RECEIPT,
    "projection_p046_confirmation.py", "benchmark.py", "harness.py",
    "official/reference.py", "official/checks.py", "official/muse_bf16_proj.json",
    "LICENSE", "THIRD_PARTY_NOTICES.md", "licenses/Triton-MIT.txt",
    "licenses/June-9-Researcher-Reciprocity.txt")
CASES = (8, 7)
RUNNERS = ("control", "weight_first", "p046")
SERIES = ("partial", "whole", "selected_partial", "selected_whole")
SOURCE_NAMES = tuple(name for name in PAYLOAD[:12] if name != RECEIPT)
ARTIFACTS = frozenset(("summary.json", "timing.json", "activity.json", "timing.log", "activity.log")) | frozenset(
    "activity.pooler_m" + str(m) + "." + runner + ".repeat" + str(repeat) + ".trace.json"
    for m in CASES for runner in RUNNERS for repeat in range(3))


def validate_sources(report, expected_source_hashes=None):
    reported = report.get("sources_sha256", {})
    require(set(reported) == set(SOURCE_NAMES) and
            all(reported.get(name) == value for name, value in FIXED.items() if name != RECEIPT),
            "Source identity mismatch")
    require(report.get("inherited_p052", {}).get("sha256") == FIXED[RECEIPT], "Inherited P052 receipt mismatch")
    if expected_source_hashes is not None:
        require(all(reported[name] == expected_source_hashes[name] for name in reported),
                "Sources differ from admitted payload")


def validate_timing(report, expected_source_hashes=None):
    require(report.get("mode") == "timing" and report.get("complete") is True and not report.get("failure")
            and report.get("oracle_checks") == 580, "Activity requires completed 580-check timing receipt")
    validate_sources(report, expected_source_hashes)
    cases = report.get("cases", [])
    require([(case.get("m"), case.get("passed")) for case in cases] == [(8, True), (7, True)],
            "Timing case coverage mismatch")
    for case in cases:
        checks = case.get("correctness", [])
        require(len(checks) == 50 and all(check.get("passed") is True for check in checks),
                "Explicit correctness checks incomplete")
        series = case.get("series", {})
        require(set(series) == set(SERIES), "Timing series coverage mismatch")
        for data in series.values():
            require(len(data.get("samples", [])) == 24 and len(data.get("warmup", [])) == 6,
                    "Timing sample coverage mismatch")
            require(all(row.get("correctness") == {"parent": True, "candidate": True}
                        for row in data["samples"] + data["warmup"]), "Replay correctness incomplete")
    return True


def reserve(rate=RATE_FLOOR):
    # 45s startup + 230s function hard bound + 30s teardown/resolution margin.
    return Decimal(305) * max(rate, RATE_FLOOR) * Decimal("1.1") + Decimal(".02")


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
    for name in (*PAYLOAD, "projection_p053_modal.py", "projection_p044_modal.py"):
        path = root / name
        require(path.is_file() and not path.is_symlink(), "Payload missing or symlink: " + name)
        raw = path.read_bytes()
        require(raw == subprocess.check_output(["git", "show", head + ":" + name], cwd=root),
                "Uncommitted or changed payload: " + name)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    require(all(hashes[name] == value for name, value in FIXED.items()), "Frozen source or inherited receipt mismatch")
    return {name: root / name for name in PAYLOAD}, hashes


def unpack_artifacts(manifest, chunks, dest, expected_source_hashes=None):
    require(not dest.exists() and not dest.is_symlink(), "Artifact destination exists; reconcile instead of replacing")
    require(0 < manifest["chunks"] <= MAX_PACKED // CHUNK and
            len(chunks) == manifest["chunks"], "Invalid chunk count")
    require(all(isinstance(c, bytes) and 0 < len(c) <= CHUNK for c in chunks), "Invalid chunk size")
    packed = b"".join(chunks)
    require(len(packed) == manifest["packed_bytes"] <= MAX_PACKED, "Packed size mismatch")
    require(hashlib.sha256(packed).hexdigest() == manifest["sha256"], "Packed hash mismatch")
    maximum_encoded = 12 * 1024 * 1024
    decoder = zlib.decompressobj()
    raw = decoder.decompress(packed, maximum_encoded + 1)
    require(decoder.eof and not decoder.unconsumed_tail and not decoder.unused_data,
            "Invalid or oversized compressed stream")
    require(len(raw) == manifest["encoded_bytes"] <= maximum_encoded, "Encoded size mismatch")
    content = json.loads(raw)
    require(isinstance(content, dict) and set(content) == set(manifest["artifact_sha256"]),
            "Artifact manifest mismatch")
    require("summary.json" in content and set(content) <= ARTIFACTS, "Unsafe artifact path")
    decoded = {n: base64.b64decode(v, validate=True) for n, v in content.items()}
    require(sum(map(len, decoded.values())) == manifest["artifact_bytes"] <= MAX_RAW, "Raw size mismatch")
    require(all(hashlib.sha256(v).hexdigest() == manifest["artifact_sha256"][n]
                for n, v in decoded.items()), "Artifact hash mismatch")
    summary = json.loads(decoded["summary.json"])
    if summary.get("complete"):
        require("timing.json" in decoded and "activity.json" in decoded, "Completed reports missing")
        require([(stage.get("mode"), stage.get("exit_code")) for stage in summary.get("stages", [])]
                == [("timing", 0), ("activity", 0)], "Incorrect dependent stage history")
        timing = json.loads(decoded["timing.json"])
        validate_timing(timing, expected_source_hashes)
        activity = json.loads(decoded["activity.json"])
        require(activity.get("mode") == "activity" and activity.get("complete") is True
                and not activity.get("failure") and activity.get("oracle_checks") == 18,
                "Activity did not complete 18 oracle checks")
        validate_sources(activity, expected_source_hashes)
        require(activity.get("timing_receipt", {}).get("sha256") == hashlib.sha256(decoded["timing.json"]).hexdigest(),
                "Activity is not bound to timing receipt")
        profiles = activity.get("profiles", [])
        expected = {(m, runner, repeat) for m in CASES for runner in RUNNERS for repeat in range(3)}
        require(len(profiles) == 18 and {(p.get("m"), p.get("runner"), p.get("repeat")) for p in profiles} == expected,
                "Activity profile coverage mismatch")
        for profile in profiles:
            name = "activity.pooler_m" + str(profile["m"]) + "." + profile["runner"] + ".repeat" + str(profile["repeat"]) + ".trace.json"
            require(profile.get("passed") is True and profile.get("trace_file") == name and name in decoded
                    and profile.get("trace_sha256") == hashlib.sha256(decoded[name]).hexdigest(),
                    "Missing or mismatched activity trace")
    dest.mkdir(parents=True, exist_ok=False)
    for name, value in decoded.items():
        (dest / name).write_bytes(value)
    return summary


def make_remote_entry():
    # No control-root, credentials, SDK client, config, or local state closure.
    def entry():
        import base64, hashlib, json, os, signal, subprocess, sys, time, zlib
        from pathlib import Path
        root = Path("/opt/projection-p053-phase")
        dest = Path("/tmp/projection-p053-phase")
        dest.mkdir(exist_ok=False)
        started = time.monotonic()
        stages = []
        gate_error = None
        # This gate uses only JSON and hashes, never imports a GPU worker.
        def timing_passed():
            path = dest / "timing.json"
            if not path.is_file() or path.is_symlink():
                return False
            report = json.loads(path.read_text())
            sources = ("projection_p053_phase_screen.py", "projection_p052_orientation_screen.py",
                "projection_p051_orientation.py", "candidates/pooler_triton_split8.py",
                "candidates/projection_pooler_column_batch2_hybrid.py", "projection_p046_confirmation.py",
                "benchmark.py", "harness.py", "official/reference.py", "official/checks.py", "official/muse_bf16_proj.json")
            expected_sources = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources}
            inherited_sha = hashlib.sha256((root / "results/remote-P052-screen/timing.json").read_bytes()).hexdigest()
            if not (report.get("mode") == "timing" and report.get("complete") is True and not report.get("failure")
                    and report.get("oracle_checks") == 580 and report.get("sources_sha256") == expected_sources
                    and report.get("inherited_p052", {}).get("sha256") == inherited_sha):
                return False
            cases = report.get("cases", [])
            if [(case.get("m"), case.get("passed")) for case in cases] != [(8, True), (7, True)]:
                return False
            for case in cases:
                checks = case.get("correctness", [])
                if len(checks) != 50 or not all(check.get("passed") is True for check in checks):
                    return False
                series = case.get("series", {})
                if set(series) != {"partial", "whole", "selected_partial", "selected_whole"}:
                    return False
                if any(len(data.get("samples", [])) != 24 or len(data.get("warmup", [])) != 6 for data in series.values()):
                    return False
                if not all(row.get("correctness") == {"parent": True, "candidate": True}
                           for data in series.values() for row in data["samples"] + data["warmup"]):
                    return False
            return True
        for mode in ("timing", "activity"):
            if mode == "activity":
                try:
                    gate = timing_passed()
                except (ValueError, TypeError, KeyError, OSError):
                    gate = False
                if not gate:
                    gate_error = "timing_receipt_rejected"
                    break
            command = [sys.executable, "-B", str(root / "projection_p053_phase_screen.py"),
                       "--mode", mode, "--output", str(dest / (mode + ".json")),
                       "--deadline-seconds", "90", "--outer-timeout-seconds", "105"]
            if mode == "activity":
                command.extend(["--timing-receipt", str(dest / "timing.json")])
            stage_start = time.monotonic()
            child = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            try:
                output, _ = child.communicate(timeout=105)
                code = child.returncode
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                output, _ = child.communicate(timeout=2)
                code = 124
            (dest / (mode + ".log")).write_bytes(output[-4000:])
            stages.append(dict(mode=mode, exit_code=code, wall_seconds=time.monotonic() - stage_start))
            if code:
                break  # Preserve failure; no replacement run or dependent stage.
        (dest / "summary.json").write_text(json.dumps(dict(stages=stages, gate_error=gate_error,
            complete=len(stages) == 2 and all(s["exit_code"] == 0 for s in stages),
            worker_wall_seconds=time.monotonic() - started)))
        files = sorted(dest.glob("*"))
        # Omit only exact regular atomic-save leftovers; reject symlinks/unknowns.
        files = [p for p in files if not (p.name in {"timing.json.tmp", "activity.json.tmp"}
                 and p.is_file() and not p.is_symlink())]
        def allowed(name):
            names = {"summary.json", "timing.json", "activity.json", "timing.log", "activity.log"}
            names.update("activity.pooler_m" + str(m) + "." + runner + ".repeat" + str(repeat) + ".trace.json"
                for m in (8, 7) for runner in ("control", "weight_first", "p046") for repeat in range(3))
            return name in names
        if any(not p.is_file() or p.is_symlink() or not allowed(p.name) for p in files):
            yield {"kind": "failure", "reason": "artifact_path_guard"}
            return
        raw_bytes = sum(p.stat().st_size for p in files)
        if raw_bytes > 8 * 1024 * 1024:
            yield {"kind": "failure", "reason": "artifact_raw_size_guard"}
            return
        encoded = json.dumps({p.name: base64.b64encode(p.read_bytes()).decode() for p in files}, sort_keys=True).encode()
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
    record_path = ROOT / "experiments/projection-p053-phase-modal.json"
    artifacts = ROOT / "local-notes/projection-p053-phase-artifacts"
    require(not record_path.exists() and not record_path.is_symlink()
            and not artifacts.exists() and not artifacts.is_symlink(),
            "Existing diagnostic state; reconcile, never relaunch")
    subprocess.run(["git", "check-ignore", "-q", "local-notes/projection-p053-phase-artifacts/summary.json"],
                   cwd=ROOT, capture_output=True, check=True)
    files, hashes = frozen_payload(head)
    auth, expiry = configure_existing_auth(control_root)
    pre_path = control_root / "local-notes/projection-p053-phase-preflight.local.json"
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
        image = image.add_local_file(path, "/opt/projection-p053-phase/" + name, copy=False)
    app = modal.App("okc-projection-p053-phase-diagnostic", include_source=False)
    fn = app.function(image=image, gpu="H200", cpu=(2, 2), memory=(8192, 8192),
        min_containers=0, max_containers=1, buffer_containers=0, scaledown_window=2,
        timeout=230, startup_timeout=45, retries=None, serialized=True,
        include_source=False, block_network=True, restrict_modal_access=True)(make_remote_entry())
    record = dict(experiment="P053 timing/activity split-K orientation diagnostic", state="reserved", app_id=None,
        created_utc=now(), source_commit=head, worker_sha256=hashes[WORKER], correctness_worker_sha256=hashes[CORRECTNESS_WORKER], previous_timing_worker_sha256=hashes[TIMING_WORKER], previous_receipt_sha256=hashes[RECEIPT], parent_source_sha256=hashes[PARENT], provenance_source_sha256=hashes[REFERENCE], base_image=IMAGE,
        payload_sha256={n: hashes[n] for n in PAYLOAD}, controller_sha256=hashes["projection_p053_modal.py"],
        cost_reserve_usd=str(cost_reserve), verified_free_credit_floor_usd=str(free_floor),
        resource_rate_usd_per_second=str(rate), paid_limit_usd="0", billing_before=billing,
        authentication_expiry_utc=expiry, preflight_sha256=hashlib.sha256(pre_path.read_bytes()).hexdigest(),
        shared_controller_sha256=hashes["projection_p044_modal.py"],
        resources=dict(gpu="H200", cpu=[2, 2], memory_mib=[8192, 8192], max_containers=1,
            min_containers=0, buffer_containers=0, scaledown_window=2, function_timeout=230,
            startup_timeout=45, child_internal_timeout=90, child_hard_timeout=105, stages=2,
            app_watchdog=275, stop_allowance=15, retry_policy=None, explicit_invocations=1),
        compile_only_requested=False, performance_measurement_requested=True, activity_separate_from_timing=True,
        no_hardware_counters=True, not_official_score=True)
    # The exclusive record is the durable no-duplicate reservation. If interrupted
    # between its write and the slot update, it still prevents another launch.
    with record_path.open("x") as stream:
        stream.write(json.dumps(record, indent=2) + "\n")
    save(slot, dict(state="reserved", provider="modal", experiment="P053 timing/activity", app_id=None))
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

    timer = threading.Timer(275, watchdog)
    timer.daemon = True
    try:
        timer.start()
        with app.run(detach=False, environment_name="main"):
            record.update(state="accepted", app_id=app.app_id, accepted_utc=now())
            save(record_path, record)
            save(slot, dict(state="accepted", provider="modal", experiment="P053 timing/activity", app_id=app.app_id))
            print("Accepted P053 timing/activity app " + app.app_id, flush=True)
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
            summary = unpack_artifacts(manifest, chunks, artifacts, hashes)
            record.update(state="function-terminal", artifact_retrieval_complete=True,
                          diagnostic_complete=summary["complete"], stages=summary["stages"],
                          worker_wall_seconds=summary["worker_wall_seconds"])
            save(record_path, record)
            stop()
    except Exception as error:
        record.update(state="needs-reconciliation", app_id=app.app_id, error_type=type(error).__name__)
        print("P053 timing/activity needs reconciliation of the existing request; no relaunch.", flush=True)
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
        save(slot, dict(state="terminal" if terminal else "uncertain", provider="modal", experiment="P053 timing/activity", app_id=app.app_id))
        print(json.dumps({k: record.get(k) for k in ("app_id", "state", "diagnostic_complete", "shutdown_verified", "paid_after_zero")}), flush=True)
    return 0 if terminal and record.get("diagnostic_complete") and record.get("artifact_retrieval_complete") else 1



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("self-test", "launch"))
    parser.add_argument("--frozen-head")
    parser.add_argument("--control-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "self-test":
        require(len(PAYLOAD) == 16 and len(set(PAYLOAD)) == 16, "Payload allowlist changed")
        require(all("local-notes" not in p and "auth" not in p for p in PAYLOAD), "Unsafe payload")
        require(all((ROOT / name).is_file() and not (ROOT / name).is_symlink()
                    for name in PAYLOAD), "Missing or symlink payload")
        require(len(ARTIFACTS) == 23, "Artifact allowlist changed")
        print("4 local guards passed; no Modal SDK import or external call.")
        return 0
    require(args.frozen_head is not None, "Launch requires reviewed committed freeze")
    return launch(args.frozen_head, args.control_root)


if __name__ == "__main__":
    raise SystemExit(main())
